#!/usr/bin/env bash
# Phase 1b orchestration (2026-10-15): R5 dropped; R6 (5e-6) and R7 (2e-5) added; dev_80
# generation evals interleaved so no GPU idles. R1-R3 run on GPUs 1-3, R4 on GPU 0.
set -uo pipefail
W=/workspace; L=$W/logs; cd $W/moshi-silence
PY=$W/moshi-finetune/.venv/bin/python
GEN="--moshi-dir $W/moshiko --finetune $W/moshi-finetune"
export HF_HUB_OFFLINE=1
log() { echo "$(date +%T) [1b] $*" | tee -a $L/parallel.log; }
mark() {  # old scheduler is gone: write .done ourselves (same rule: rc 0 and no NaN)
  local n=$1
  if [ "$(cat $L/$n.rc 2>/dev/null)" = 0 ] && ! grep -qiw nan $L/$n.log; then touch $L/$n.done; log "done  $n"
  else log "FAIL  $n (rc $(cat $L/$n.rc 2>/dev/null))"; fi
}
waitrc() { for n in "$@"; do until [ -f $L/$n.rc ]; do sleep 15; done; done; }

waitrc R1_1h_lr2e-06 R2_1h_lr1e-05 R3_1h_lr5e-05
for n in R1_1h_lr2e-06 R2_1h_lr1e-05 R3_1h_lr5e-05; do mark $n; done
log "start R6/R7 on GPUs 1 2; evals R0 R1 R2 on GPU 3"
GPUS="1 2" nohup bash scripts/run_parallel.sh configs/R6_1h_lr5e-06.yaml configs/R7_1h_lr2e-05.yaml > $L/parallel_67.log 2>&1 &
GPUS="3" nohup bash scripts/eval_dev.sh dev_80 R0 R1_1h_lr2e-06 R2_1h_lr1e-05 > $L/eval_dev_gpu3.log 2>&1 &

waitrc R4_1h_lr0.0002; mark R4_1h_lr0.0002
log "evals R3 R4 on GPU 0"
GPUS="0" nohup bash scripts/eval_dev.sh dev_80 R3_1h_lr5e-05 R4_1h_lr0.0002 > $L/eval_dev_gpu0.log 2>&1 &

waitrc R6_1h_lr5e-06 R7_1h_lr2e-05
sleep 20  # let run_parallel write .done
log "R6/R7 finished; evaluating their checkpoints split over 4 GPUs (after the GPU 0/3 evals free up)"
evalhalf() {  # $1 gpu, $2 run, $3 parity (0 = even-indexed checkpoints, 1 = odd)
  local ck=($W/runs/$2/checkpoints/checkpoint_*) sel=() i
  for i in "${!ck[@]}"; do [ $((i % 2)) = $3 ] && sel+=("${ck[$i]}"); done
  CUDA_VISIBLE_DEVICES=$1 $PY -m silence.eval_gen run --episodes data/dev/dev_80.jsonl --out $W/eval/$2/dev_80 $GEN \
      --ckpt "${sel[@]}" >> $L/eval/${2}_dev_80_p$3.log 2>&1
  for d in $W/eval/$2/dev_80/*/; do
    [ -f $d/meta.json ] && [ ! -f $d/scores.jsonl ] && $PY -m silence.eval_score $d > $d/score.log 2>&1
  done
  log "eval $2 part $3 on GPU $1 done"
}
mkdir -p $L/eval
evalhalf 1 R6_1h_lr5e-06 0 & evalhalf 2 R7_1h_lr2e-05 0 &
# GPUs 0 and 3 join when their first eval batch is done
( while pgrep -f "eval_dev.sh dev_80 R0" > /dev/null; do sleep 20; done; evalhalf 3 R6_1h_lr5e-06 1 ) &
( while pgrep -f "eval_dev.sh dev_80 R3" > /dev/null; do sleep 20; done; evalhalf 0 R7_1h_lr2e-05 1 ) &
wait
log "phase-1 generation evals all done"
