#!/usr/bin/env bash
# R24 (2026-10-15): v3 data (v2 plan + long silences in cue-released holds) rebalanced to B1 15%, EPAD x10, lr 5e-5,
# 2 epochs; checkpoint + dev loss every 31 steps (~20 checkpoints, pick any = early stopping), dev_80 evals of all,
# then the 7 test suites on the final checkpoint.
#   cd /workspace/moshi-silence && nohup bash scripts/run_v3.sh > /workspace/logs/run_v3.log 2>&1 &
set -uo pipefail
W=/workspace; L=$W/logs; cd $W/moshi-silence; export HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
log() { echo "$(date +%T) [run_v3] $*" | tee -a $L/parallel.log; }
D=$W/data50/train_v3
if [ ! -f $D/train_v3_b1-15.jsonl ]; then
  log "check.py + filter + rebalance"
  $W/moshi-finetune/.venv/bin/python -m silence.check $D --finetune $W/moshi-finetune --workers 20 > $D/check.log 2>&1
  tail -1 $D/check.log | tee -a $L/parallel.log
  $W/buildenv/bin/python -m silence.filter_checked $D | head -1 | tee -a $L/parallel.log
  $W/buildenv/bin/python -m silence.rebalance $D --family B1 --share 0.15 --out train_v3_b1-15.jsonl | tee -a $L/parallel.log
fi
N=$(grep -c "" $D/train_v3_b1-15.jsonl); STEPS=$(( (2 * N + 15) / 16 ))
C="--phase 2 --steps $STEPS --every 31 --eval $W/moshi-silence/data/dev/dev_200.jsonl --moshi-dir $W/moshiko --runs $W/runs --out configs --lrs 5e-5 --epad-weight 5"
python3 -m silence.make_configs $C --train $D/train_v3_b1-15.jsonl --first 24 --suffix _epad10
RUN=R24_v3_b1-15_lr5e-05_epad10
[ -f configs/$RUN.yaml ] || RUN=$(ls configs | grep "^R24_" | grep yaml$ | head -1 | sed 's/.yaml$//')
log "training $RUN: $N episodes, $STEPS steps (2 epochs)"
GPUS="0" bash scripts/run_parallel.sh configs/$RUN.yaml > $L/parallel_run_v3.log 2>&1
log "training finished: $(grep -c ' done ' $L/parallel_run_v3.log) done, $(grep -c FAIL $L/parallel_run_v3.log) failed"
GPUS="0" bash scripts/eval_dev.sh dev_80 $RUN > $L/eval_dev_run_v3.log 2>&1
log "dev_80 evals done"
LAST=$(ls -d $W/runs/$RUN/checkpoints/checkpoint_* | sort | tail -1)
OUT=$W/eval/test_R24 bash scripts/test_suites.sh $LAST > $L/test_R24.log 2>&1
log "run_v3: all training and evals done"
