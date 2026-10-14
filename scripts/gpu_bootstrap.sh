#!/usr/bin/env bash
# One-time setup on a 1x H100 pod, then a 6-step smoke train. Everything goes on the network
# volume at /workspace, so a 4x pod mounting the same volume can train right away
# (scripts/run_parallel.sh).
#
# On the pod first (RunPod images may lack rsync): apt-get update && apt-get install -y rsync
# Then from the Mac: code from the branch worktree, data/ from the main checkout (no .env):
#   rsync -a --exclude .git --exclude demo --exclude __pycache__ --exclude .DS_Store \
#       ~/repo/moshi-silence/.claude/worktrees/docs-progress/ box:/workspace/moshi-silence/
#   rsync -a --exclude .DS_Store ~/repo/moshi-silence/data/ box:/workspace/moshi-silence/data/  # ~6.2 GB
#   ssh box 'bash /workspace/moshi-silence/scripts/gpu_bootstrap.sh'
# `box` = an ssh config alias for the pod's exposed SSH port (or rsync -e "ssh -p PORT").
# Re-syncing data/ later puts relative jsonl paths back; rerun this script afterwards.
#
# Needs: Linux + CUDA GPU, python3 (stdlib only), git, curl; volume >= 150 GB (weights ~16 GB, data ~6 GB,
# env ~10 GB, phase-1 checkpoints ~41 GB = 105 x ~0.39 GB). Safe to rerun.
set -euo pipefail

W=${W:-/workspace}
REPO=$(cd "$(dirname "$0")/.." && pwd)
MF=$W/moshi-finetune
MF_COMMIT=2acc879fe7c48f885a18f6cc9548bccb2674d87b      # upstream main the patches were made on
MOSHI_REV=${MOSHI_REV:-2bfc9ae6e89079a5cc7ed2a68436010d91a3d289}  # kyutai/moshiko-pytorch-bf16, pinned 2026-10-14
MOSHI_DIR=$W/moshiko
PY=$MF/.venv/bin

echo "== 1. moshi-finetune @ ${MF_COMMIT:0:7} + 3 patches, env from its uv.lock"
[ -d "$MF" ] || git clone https://github.com/kyutai-labs/moshi-finetune "$MF"
git -C "$MF" checkout -q "$MF_COMMIT"
apply() {  # $1 = patch, $2 = a string the patched tree contains
  if grep -rq -- "$2" "$MF/train.py" "$MF/finetune"; then echo "  $1: already applied"
  else git -C "$MF" apply "$REPO/patches/$1" && echo "  $1: applied"; fi
}
apply moshi-finetune-audio-mask.patch mask_main_speech_audio
apply moshi-finetune-eval-reload.patch make_eval_loader
apply moshi-finetune-text-tail.patch "Past the end of the file the audio is zero padding"
# pyproject pins sphn==0.1.12 but takes moshi from git main, which now needs sphn>=0.2, so a
# plain `pip install -e .` can't resolve. uv.lock pins the tested set (moshi 061cc4c =
# 0.2.4a1, sphn 0.1.12, torch 2.6.0, huggingface-hub 0.28.1).
# uv, its managed Python and its cache all live on the volume, so the env survives moving the
# volume to a new pod (1x for setup, then 4x for training); the home dir doesn't persist.
# moshi-finetune only says requires-python >= 3.10, so uv would pick the newest Python (3.14),
# for which torch 2.6.0 has no wheels: pin 3.12.
export UV_INSTALL_DIR=$W/bin UV_PYTHON_INSTALL_DIR=$W/uv-python UV_CACHE_DIR=$W/uv-cache \
       UV_PYTHON_PREFERENCE=only-managed UV_PYTHON=3.12 UV_NO_MODIFY_PATH=1 PATH="$W/bin:$PATH"
[ -x "$W/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
(cd "$MF" && uv sync --frozen)
(cd "$MF" && uv pip freeze) > "$W/pip-freeze.txt"   # exact environment, for the report

echo "== 2. weights @ ${MOSHI_REV:0:7}"
"$PY/huggingface-cli" download kyutai/moshiko-pytorch-bf16 --revision "$MOSHI_REV" --local-dir "$MOSHI_DIR"
echo "$MOSHI_REV" > "$MOSHI_DIR/REVISION"

echo "== 3. absolute wav paths in the jsonls (sphn also resolves relative ones; this is belt and braces)"
cd "$REPO"
python3 -m silence.make_splits data/train --hours 1 10 --abs > /dev/null
python3 - <<'EOF'
import json
from pathlib import Path
for f in Path("data/dev").glob("dev_*.jsonl"):
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    for r in rows:
        r["path"] = str((f.parent / r["path"]).resolve())
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{f}: {len(rows)} episodes")
EOF

echo "== 4. phase-1 configs (+ R0_devloss: the base model's dev loss, no training)"
python3 -m silence.make_configs --phase 1 --train "$REPO/data/train/train_1h.jsonl" \
  --eval "$REPO/data/dev/dev_200.jsonl" --moshi-dir "$MOSHI_DIR" --runs "$W/runs" --out "$REPO/configs"

echo "== 5. smoke train: 6 steps of R3; dev loss + checkpoint at steps 3 and 6"
# Two evaluations, so the eval-reload patch is exercised (unpatched, the 2nd one is NaN).
SMOKE=$W/runs/_smoke
rm -rf "$SMOKE"
sed -e "s#^max_steps: .*#max_steps: 6#" -e "s#^eval_freq: .*#eval_freq: 3#" -e "s#^ckpt_freq: .*#ckpt_freq: 3#" \
    -e "s#^run_dir: .*#run_dir: $SMOKE#" "$REPO/configs/R3_1h_lr5e-05.yaml" > "$W/smoke.yaml"
cd "$MF"
# moshi-finetune reads CUDA_VISIBLE_DEVICES with no default (finetune/distributed.py), so set it.
CUDA_VISIBLE_DEVICES=0 "$PY/torchrun" --nproc-per-node 1 -m train "$W/smoke.yaml" 2>&1 | tee "$W/smoke.log" \
  || { echo "SMOKE FAIL: train.py exited non-zero (see $W/smoke.log)"; exit 1; }
n_ckpt=$(find "$SMOKE/checkpoints" -maxdepth 1 -name 'checkpoint_*' 2>/dev/null | wc -l || true)
if grep -qiw "nan" "$W/smoke.log"; then echo "SMOKE FAIL: NaN in the log"; exit 1; fi
if [ "$n_ckpt" -lt 2 ]; then echo "SMOKE FAIL: $n_ckpt checkpoints (want 2)"; exit 1; fi
if ! grep -q "text_eval_loss" "$W/smoke.log"; then echo "SMOKE FAIL: no text_eval_loss logged"; exit 1; fi
n_err=$(grep -ci "error" "$W/smoke.log" || true)
[ "$n_err" -eq 0 ] || echo "WARNING: $n_err lines mention 'error' (sphn skips unreadable wavs with only a log line)"
echo "SMOKE OK ($n_ckpt checkpoints, no NaN). Peak memory and s/step are in $W/smoke.log."
echo "Next: stop this pod, start a 4x H100 pod on the same network volume, then (no re-setup needed):"
echo "  cd $REPO && mkdir -p $W/logs && nohup bash scripts/run_parallel.sh configs/R0_devloss.yaml configs/R?_1h_*.yaml > $W/logs/parallel.log 2>&1 &"
echo "  tail -f $W/logs/parallel.log"
