#!/usr/bin/env bash
# Generation eval of training runs on a dev subset: every checkpoint of each run (plus the base
# model as R0), one run per GPU, then scoring. Resumable: finished checkpoints are skipped.
#
#   cd /workspace/moshi-silence
#   nohup bash scripts/eval_dev.sh dev_80 R0 R1_1h_lr2e-06 R2_1h_lr1e-05 R3_1h_lr5e-05 \
#       R4_1h_lr0.0002 R5_1h_lr0.0005 > /workspace/logs/eval_dev.log 2>&1 &
#   /workspace/moshi-finetune/.venv/bin/python -m silence.eval_report /workspace/eval \
#       --configs configs --runs /workspace/runs --out /workspace/report/phase1
#
# R0 = the base model (no checkpoints). Other names are run names under $W/runs. Runs are dealt
# round-robin to the GPUs (GPUS="0 1" overrides); each GPU works through its runs in order.
# Output: $W/eval/<run>/<set>/<checkpoint_XXXXXX>/{<id>.json, codes.npz, meta.json, scores.jsonl}.
set -uo pipefail

W=${W:-/workspace}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-$W/moshi-finetune/.venv/bin/python}
[ $# -ge 2 ] || { echo "usage: $0 <dev_80|dev_200> <run> [<run> ...]"; exit 2; }
SET=$1; shift
EPS=$REPO/data/dev/$SET.jsonl
[ -f "$EPS" ] || { echo "no $EPS"; exit 2; }
LOGS=$W/logs/eval
GEN_ARGS="--moshi-dir $W/moshiko --finetune $W/moshi-finetune ${EXTRA:-}"  # EXTRA="--batch 80" etc.
export HF_HUB_OFFLINE=1  # weights are local; the (404) config.json probe must not hit a flaky hub
mkdir -p "$LOGS"
read -r -a GPU_LIST <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
cd "$REPO"

score() {  # score every generated dir that has no scores yet
  for d in "$@"; do
    [ -f "$d/meta.json" ] && [ ! -f "$d/scores.jsonl" ] && "$PY" -m silence.eval_score "$d" > "$d/score.log" 2>&1
  done
}

job() {  # $1 = gpu, then run names
  local g=$1; shift
  for run in "$@"; do
    local out=$W/eval/$run/$SET log=$LOGS/${run}_$SET.log
    echo "$(date +%T) GPU $g: $run on $SET (log: $log)"
    if [ "$run" = R0 ]; then
      CUDA_VISIBLE_DEVICES=$g "$PY" -m silence.eval_gen run --episodes "$EPS" --out "$out" $GEN_ARGS --ckpt base >> "$log" 2>&1
    else
      local ck=("$W/runs/$run/checkpoints"/checkpoint_*)
      [ -d "${ck[0]}" ] || { echo "  $run: no checkpoints under $W/runs/$run"; continue; }
      CUDA_VISIBLE_DEVICES=$g "$PY" -m silence.eval_gen run --episodes "$EPS" --out "$out" $GEN_ARGS --ckpt "${ck[@]}" >> "$log" 2>&1
    fi
    local rc=$?
    score "$out"/*/
    echo "$(date +%T) GPU $g: $run done (exit $rc)"
  done
}

declare -A per_gpu
i=0
for run in "$@"; do
  g=${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}
  per_gpu[$g]="${per_gpu[$g]:-} $run"
  i=$((i + 1))
done
for g in "${!per_gpu[@]}"; do
  # shellcheck disable=SC2086
  job "$g" ${per_gpu[$g]} &
done
wait
echo "$(date +%T) all evals finished"
