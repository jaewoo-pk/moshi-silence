#!/usr/bin/env bash
# EPAD weight at 10 h (2026-10-15): the 10 h models are over-silent (96-98% silent, ~70% speaks), so
# does up-weighting the "start talking" frame move them to a better balance? train_10h, lr 5e-5,
# 5 epochs = 345 steps, checkpoint + dev loss every 17 steps; identical to phase 2 except EPAD.
# Baseline: R9_10h_lr5e-05 (EPAD x1).
#   R20_10h_lr5e-05_epad3    EPAD weight 1.5 (x3 PAD's 0.5)
#   R21_10h_lr5e-05_epad10   EPAD weight 5   (x10)
# Usage: bash scripts/epad10h.sh [3|10|both]   (both = one run per GPU on a 2-GPU pod; 3 or 10 = one run
# on GPU 0, e.g. on two separate 1-GPU pods sharing the volume)
#   cd /workspace/moshi-silence && nohup bash scripts/epad10h.sh 3 > /workspace/logs/epad10h_3.log 2>&1 &
set -uo pipefail
W=/workspace; L=$W/logs; MF=$W/moshi-finetune; cd $W/moshi-silence
export HF_HUB_OFFLINE=1
mkdir -p $L/eval
log() { echo "$(date +%T) [epad10h] $*" | tee -a $L/parallel.log; }

grep -q "text_epad_weight" $MF/finetune/args.py && log "loss-weights patch present" || { log "loss-weights patch MISSING"; exit 1; }
log "free space: $(du -sh $W/runs | cut -f1) in runs/ (volume 250 GB)"

T=$W/moshi-silence/data/train/train_10h.jsonl; E=$W/moshi-silence/data/dev/dev_200.jsonl
C="--phase 2 --epochs 5 --ckpts-per-epoch 4 --train $T --eval $E --moshi-dir $W/moshiko --runs $W/runs --out configs --lrs 5e-5"
WHICH=${1:-both}
python3 -m silence.make_configs $C --first 20 --epad-weight 1.5 --suffix _epad3
python3 -m silence.make_configs $C --first 21 --epad-weight 5 --suffix _epad10
case $WHICH in
  3) RUNS=(R20_10h_lr5e-05_epad3); G="0" ;;
  10) RUNS=(R21_10h_lr5e-05_epad10); G="0" ;;
  *) RUNS=(R20_10h_lr5e-05_epad3 R21_10h_lr5e-05_epad10); G="0 1" ;;
esac
TAG=epad10h_$WHICH
CFGS=(); for c in "${RUNS[@]}"; do [ -f configs/$c.yaml ] || { log "missing configs/$c.yaml"; exit 1; }; CFGS+=("configs/$c.yaml"); done

log "training: ${RUNS[*]} on GPUs $G"
GPUS="$G" bash scripts/run_parallel.sh "${CFGS[@]}" > $L/parallel_$TAG.log 2>&1
log "training finished: $(grep -c ' done ' $L/parallel_$TAG.log) done, $(grep -c FAIL $L/parallel_$TAG.log) failed"

log "dev_80 generation evals, one run per GPU"
GPUS="$G" bash scripts/eval_dev.sh dev_80 "${RUNS[@]}" > $L/eval_dev_$TAG.log 2>&1
log "EPAD 10 h ($WHICH): all training and evals done"
