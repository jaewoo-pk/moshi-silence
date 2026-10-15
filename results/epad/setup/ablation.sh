#!/usr/bin/env bash
# EPAD-weight sweep (2026-10-15): does up-weighting the "start talking now" frame (EPAD) fix the
# quiet-model bias? train_1h, peak LR 5e-5, 20 epochs = 142 steps, checkpoint + dev loss every
# 7 steps (~1 epoch); everything else = phase 1 (baseline: R3_1h_lr5e-05, EPAD x1 = weight 0.5).
#   GPU 0: R16_1h_lr5e-05_epad3     EPAD weight 1.5  (x3 PAD's 0.5)
#   GPU 1: R17_1h_lr5e-05_epad10    EPAD weight 5    (x10)
#   GPU 2: R18_1h_lr5e-05_epad30    EPAD weight 15   (x30)
#   GPU 3: R19_1h_lr5e-05_epad100   EPAD weight 50   (x100)
# Audio loss weight stays 1.0. The dev loss keeps the original weighting (comparable).
#   cd /workspace/moshi-silence && nohup bash scripts/ablation.sh > /workspace/logs/ablation.log 2>&1 &
set -uo pipefail
W=/workspace; L=$W/logs; MF=$W/moshi-finetune; cd $W/moshi-silence
export HF_HUB_OFFLINE=1
mkdir -p $L/eval
log() { echo "$(date +%T) [epad] $*" | tee -a $L/parallel.log; }

if grep -q "text_epad_weight" $MF/finetune/args.py; then log "loss-weights patch already applied"
else git -C $MF apply $W/moshi-silence/patches/moshi-finetune-loss-weights.patch && log "loss-weights patch applied" || { log "PATCH FAILED"; exit 1; }; fi

log "configs"
T=$W/moshi-silence/data/train/train_1h.jsonl; E=$W/moshi-silence/data/dev/dev_200.jsonl
C="--phase 1 --no-base --train $T --eval $E --moshi-dir $W/moshiko --runs $W/runs --out configs --lrs 5e-5"
python3 -m silence.make_configs $C --first 16 --epad-weight 1.5 --suffix _epad3
python3 -m silence.make_configs $C --first 17 --epad-weight 5 --suffix _epad10
python3 -m silence.make_configs $C --first 18 --epad-weight 15 --suffix _epad30
python3 -m silence.make_configs $C --first 19 --epad-weight 50 --suffix _epad100
RUNS=(R16_1h_lr5e-05_epad3 R17_1h_lr5e-05_epad10 R18_1h_lr5e-05_epad30 R19_1h_lr5e-05_epad100)
CFGS=(); for c in "${RUNS[@]}"; do [ -f configs/$c.yaml ] || { log "missing configs/$c.yaml"; exit 1; }; CFGS+=("configs/$c.yaml"); done

log "training: ${RUNS[*]} on GPUs 0 1 2 3"
GPUS="0 1 2 3" bash scripts/run_parallel.sh "${CFGS[@]}" > $L/parallel_epad.log 2>&1
log "training finished: $(grep -c ' done ' $L/parallel_epad.log) done, $(grep -c FAIL $L/parallel_epad.log) failed"

log "dev_80 generation evals, one run per GPU"
GPUS="0 1 2 3" bash scripts/eval_dev.sh dev_80 "${RUNS[@]}" > $L/eval_dev_epad.log 2>&1
log "EPAD sweep: all training and evals done"
