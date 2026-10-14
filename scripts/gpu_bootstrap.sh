#!/usr/bin/env bash
# One-time setup on the rented GPU box, then a 6-step smoke train.
#
#   rsync -a --exclude .venv --exclude models --exclude cache moshi-silence/ box:/workspace/moshi-silence/
#   ssh box 'bash /workspace/moshi-silence/scripts/gpu_bootstrap.sh'
#
# Needs: Linux + CUDA GPU (H100 80 GB planned), python >= 3.10 with pip, git, ~25 GB disk for
# weights + ~50 GB for checkpoints. Idempotent: reruns skip finished steps.
set -euo pipefail

W=${W:-/workspace}
REPO=$(cd "$(dirname "$0")/.." && pwd)
MF=$W/moshi-finetune
MF_COMMIT=2acc879fe7c48f885a18f6cc9548bccb2674d87b      # upstream main the patches were made on
MOSHI_REV=${MOSHI_REV:-2bfc9ae6e89079a5cc7ed2a68436010d91a3d289}  # kyutai/moshiko-pytorch-bf16, pinned 2026-10-14
MOSHI_DIR=$W/moshiko
# The weights repo has no config.json, so moshi builds Moshi 7B from the defaults in the
# installed `moshi` package: pin it too (moshi-finetune's pyproject takes moshi's main, unpinned).
MOSHI_PKG_COMMIT=${MOSHI_PKG_COMMIT:-e6a55d2722a65870ef52a6c9f6ecfc0e90f38362}  # kyutai-labs/moshi main, 2026-05-16

echo "== 1. moshi-finetune @ ${MF_COMMIT:0:7} + patches"
if [ ! -d "$MF" ]; then
  git clone https://github.com/kyutai-labs/moshi-finetune "$MF"
  git -C "$MF" checkout -q "$MF_COMMIT"
  git -C "$MF" apply "$REPO/patches/moshi-finetune-audio-mask.patch"
  git -C "$MF" apply "$REPO/patches/moshi-finetune-eval-reload.patch"
fi
pip install -q -e "$MF"
pip install -q --force-reinstall --no-deps \
  "moshi @ git+https://github.com/kyutai-labs/moshi.git@$MOSHI_PKG_COMMIT#subdirectory=moshi"
pip freeze > "$W/pip-freeze.txt"   # exact environment, for the report

echo "== 2. weights @ ${MOSHI_REV:0:7}"
hf download kyutai/moshiko-pytorch-bf16 --revision "$MOSHI_REV" --local-dir "$MOSHI_DIR"
echo "$MOSHI_REV" > "$MOSHI_DIR/REVISION"

echo "== 3. absolute wav paths in the jsonls"
cd "$REPO"
python -m silence.make_splits data/train --hours 1 10 --abs > /dev/null
python - <<'EOF'
import json
from pathlib import Path
for f in Path("data/dev").glob("dev_*.jsonl"):
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    for r in rows:
        r["path"] = str((f.parent / r["path"]).resolve())
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{f}: {len(rows)} episodes")
EOF

echo "== 4. phase-1 configs"
python -m silence.make_configs --phase 1 --train "$REPO/data/train/train_1h.jsonl" \
  --eval "$REPO/data/dev/dev_200.jsonl" --moshi-dir "$MOSHI_DIR" --runs "$W/runs" --out "$REPO/configs"

echo "== 5. smoke train: 6 steps of R3; dev loss + checkpoint at steps 3 and 6"
# Two evaluations, so the eval-reload patch is exercised (unpatched, the 2nd one is NaN).
SMOKE=$W/runs/_smoke
rm -rf "$SMOKE"
sed -e "s#^max_steps: .*#max_steps: 6#" -e "s#^eval_freq: .*#eval_freq: 3#" -e "s#^ckpt_freq: .*#ckpt_freq: 3#" \
    -e "s#^run_dir: .*#run_dir: $SMOKE#" "$REPO/configs/R3_1h_lr5e-05.yaml" > "$W/smoke.yaml"
cd "$MF"
torchrun --nproc-per-node 1 -m train "$W/smoke.yaml" 2>&1 | tee "$W/smoke.log"
n_ckpt=$(find "$SMOKE/checkpoints" -maxdepth 1 -name 'checkpoint_*' 2>/dev/null | wc -l)
if grep -qiw "nan" "$W/smoke.log"; then echo "SMOKE FAIL: NaN in the log"; exit 1; fi
if [ "$n_ckpt" -lt 2 ]; then echo "SMOKE FAIL: $n_ckpt checkpoints (want 2)"; exit 1; fi
echo "SMOKE OK ($n_ckpt checkpoints, no NaN). Peak memory and s/step are in $W/smoke.log."
echo "Run R1..R5 one at a time (train.py can't resume; use an on-demand instance):"
echo "  cd $MF && for c in $REPO/configs/R?_1h_*.yaml; do torchrun --nproc-per-node 1 -m train \$c; done"
