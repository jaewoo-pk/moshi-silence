#!/usr/bin/env bash
# Run moshi-finetune configs one per GPU, starting the next queued config as soon as a GPU
# frees up. Each run is an independent single-GPU job (batch 16 stays batch 16).
#
#   cd /workspace/moshi-silence
#   nohup bash scripts/run_parallel.sh configs/R0_devloss.yaml configs/R?_1h_*.yaml \
#       > /workspace/logs/parallel.log 2>&1 &
#   tail -f /workspace/logs/parallel.log            # progress; per-run logs next to it
#
# Use nohup (or tmux): an SSH disconnect would otherwise kill the runs, and train.py can't
# resume. GPUS="0 1" overrides the GPU list (default: every GPU nvidia-smi sees).
# A config is skipped if it finished before (logs/<name>.done). A run_dir left over from an
# interrupted attempt makes that config fail fast: train.py refuses an existing run_dir,
# and a half-finished run can't be resumed, so delete the run_dir to rerun it.
set -uo pipefail

W=${W:-/workspace}
MF=${MF:-$W/moshi-finetune}
LOGS=${LOGS:-$W/logs}
TORCHRUN=$MF/.venv/bin/torchrun
mkdir -p "$LOGS"
[ $# -gt 0 ] || { echo "usage: $0 config.yaml [config.yaml ...]"; exit 2; }
[ -x "$TORCHRUN" ] || { echo "no $TORCHRUN: run scripts/gpu_bootstrap.sh first"; exit 2; }
read -r -a GPU_LIST <<< "${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
echo "$(date +%T) ${#GPU_LIST[@]} GPUs [${GPU_LIST[*]}], ${#} configs"

queue=()
failed=()
for cfg in "$@"; do
  cfg=$(readlink -f "$cfg")
  name=$(basename "$cfg" .yaml)
  run_dir=$(sed -n 's/^run_dir: *//p' "$cfg")
  if [ -f "$LOGS/$name.done" ]; then echo "  skip $name (done)"; continue; fi
  if [ -e "$run_dir" ]; then
    echo "  FAIL $name: $run_dir exists from an earlier attempt (rm -rf it to rerun)"; failed+=("$name"); continue
  fi
  queue+=("$cfg")
done

declare -A pid_gpu pid_name pid_t0
free=("${GPU_LIST[@]}")

start() {  # $1 = config, $2 = gpu
  local name; name=$(basename "$1" .yaml)
  rm -f "$LOGS/$name.rc"
  ( cd "$MF" && CUDA_VISIBLE_DEVICES=$2 "$TORCHRUN" --nproc-per-node 1 \
      --master_port $((29500 + 10 * $2)) -m train "$1" > "$LOGS/$name.log" 2>&1
    echo $? > "$LOGS/$name.rc" ) &
  pid_gpu[$!]=$2; pid_name[$!]=$name; pid_t0[$!]=$(date +%s)
  echo "$(date +%T) start $name on GPU $2 (log: $LOGS/$name.log)"
}

while [ ${#queue[@]} -gt 0 ] || [ ${#pid_gpu[@]} -gt 0 ]; do
  while [ ${#queue[@]} -gt 0 ] && [ ${#free[@]} -gt 0 ]; do
    start "${queue[0]}" "${free[0]}"
    queue=("${queue[@]:1}"); free=("${free[@]:1}")
  done
  wait -n  # bash >= 4.3; which job ended is read from the .rc files below
  for p in "${!pid_gpu[@]}"; do
    name=${pid_name[$p]}
    [ -f "$LOGS/$name.rc" ] || continue
    kill -0 "$p" 2> /dev/null && continue
    rc=$(cat "$LOGS/$name.rc"); mins=$(( ($(date +%s) - ${pid_t0[$p]}) / 60 ))
    if [ "$rc" = 0 ] && ! grep -qiw nan "$LOGS/$name.log"; then
      touch "$LOGS/$name.done"; echo "$(date +%T) done  $name (GPU ${pid_gpu[$p]}, ${mins} min)"
    else
      failed+=("$name"); echo "$(date +%T) FAIL  $name (exit $rc, ${mins} min): tail $LOGS/$name.log"
    fi
    free+=("${pid_gpu[$p]}")
    unset "pid_gpu[$p]" "pid_name[$p]" "pid_t0[$p]"
  done
done

echo "$(date +%T) all finished; failed: ${failed[*]:-none}"
[ ${#failed[@]} -eq 0 ]
