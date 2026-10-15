#!/usr/bin/env bash
# Phase 2 (2026-10-15): train_10h (1,101 episodes), 5 epochs = 345 steps, checkpoint + dev loss
# every 17 steps (~1/4 epoch, 21 checkpoints), four peak LRs in parallel on one 4x H100 pod:
#   GPU 0: R8_10h_lr2e-05   GPU 1: R9_10h_lr5e-05   GPU 2: R10_10h_lr0.0002   GPU 3: R11_10h_lr0.0005
# (2e-5, 5e-5, 2e-4 have 1 h counterparts in phase 1; 5e-4 brackets the top.) Then dev_80
# generation evals of every checkpoint, one run per GPU.
#   cd /workspace/moshi-silence && nohup bash scripts/phase2.sh > /workspace/logs/phase2.log 2>&1 &
set -uo pipefail
W=/workspace; L=$W/logs; cd $W/moshi-silence
export HF_HUB_OFFLINE=1
mkdir -p $L/eval
log() { echo "$(date +%T) [p2] $*" | tee -a $L/parallel.log; }

log "configs"
python3 -m silence.make_configs --phase 2 --epochs 5 --ckpts-per-epoch 4 --lrs 2e-5 5e-5 2e-4 5e-4 \
  --train $W/moshi-silence/data/train/train_10h.jsonl --eval $W/moshi-silence/data/dev/dev_200.jsonl \
  --moshi-dir $W/moshiko --runs $W/runs --out configs
RUNS=(R8_10h_lr2e-05 R9_10h_lr5e-05 R10_10h_lr0.0002 R11_10h_lr0.0005)
for c in "${RUNS[@]}"; do [ -f configs/$c.yaml ] || { log "missing configs/$c.yaml"; exit 1; }; done

log "training: ${RUNS[*]} on GPUs 0 1 2 3"
CFGS=(); for c in "${RUNS[@]}"; do CFGS+=("configs/$c.yaml"); done
GPUS="0 1 2 3" bash scripts/run_parallel.sh "${CFGS[@]}" > $L/parallel_p2.log 2>&1
log "training finished: $(grep -c ' done ' $L/parallel_p2.log) done, $(grep -c FAIL $L/parallel_p2.log) failed"

log "dev_80 generation evals, one run per GPU"
GPUS="0 1 2 3" bash scripts/eval_dev.sh dev_80 "${RUNS[@]}" > $L/eval_dev_p2.log 2>&1
log "phase 2: all training and evals done"
