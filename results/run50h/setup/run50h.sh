#!/usr/bin/env bash
# 50 h data-scaling runs (2026-10-15), EPAD x10, lr 5e-5, a FIXED 345 steps (= R21's compute), checkpoint +
# dev loss every 17 steps, then dev_80 generation evals of every checkpoint.
#   R22_50h_lr5e-05_epad10      train50/train_50h.jsonl (5,457 eps)  345 steps ~ 1.01 epochs   vs R21 (old 10 h)
#   R23_10h_lr5e-05_v2_epad10   train50/train_10h.jsonl (1,084 eps)  345 steps ~ 5.09 epochs   = new recipe at 10 h
# Usage (one run per 1-GPU pod sharing the volume): bash scripts/run50h.sh 22|23
#   cd /workspace/moshi-silence && nohup bash scripts/run50h.sh 22 > /workspace/logs/run50h_22.log 2>&1 &
set -uo pipefail
W=/workspace; L=$W/logs; MF=$W/moshi-finetune; cd $W/moshi-silence
export HF_HUB_OFFLINE=1
mkdir -p $L/eval
log() { echo "$(date +%T) [run50h] $*" | tee -a $L/parallel.log; }
grep -q "text_epad_weight" $MF/finetune/args.py || { log "loss-weights patch MISSING"; exit 1; }
log "volume use: $(du -sh $W 2>/dev/null | cut -f1) of 250 GB"
D=$W/data50/train50; E=$W/moshi-silence/data/dev/dev_200.jsonl
C="--phase 2 --steps 345 --every 17 --eval $E --moshi-dir $W/moshiko --runs $W/runs --out configs --lrs 5e-5 --epad-weight 5"
case ${1:?22 or 23} in
  22) python3 -m silence.make_configs $C --train $D/train_50h.jsonl --first 22 --suffix _epad10; RUN=R22_50h_lr5e-05_epad10 ;;
  23) python3 -m silence.make_configs $C --train $D/train_10h.jsonl --first 23 --suffix _v2_epad10; RUN=R23_10h_lr5e-05_v2_epad10 ;;
  *) echo "22 or 23"; exit 2 ;;
esac
[ -f configs/$RUN.yaml ] || { log "missing configs/$RUN.yaml"; exit 1; }
log "training $RUN"
GPUS="0" bash scripts/run_parallel.sh configs/$RUN.yaml > $L/parallel_run50h_$1.log 2>&1
log "training finished: $(grep -c ' done ' $L/parallel_run50h_$1.log) done, $(grep -c FAIL $L/parallel_run50h_$1.log) failed"
log "dev_80 generation evals"
GPUS="0" bash scripts/eval_dev.sh dev_80 $RUN > $L/eval_dev_run50h_$1.log 2>&1
log "run50h $1: all training and evals done"
