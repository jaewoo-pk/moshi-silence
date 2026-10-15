#!/usr/bin/env bash
# Human-recorded test sets (2026-10-15): generation eval of 5 models on test_human (your voice, scripted),
# test_human_free (own wording) and test_human_tts (same scripts, TTS voice = paired control), 3 samples / episode.
#   cd /workspace/moshi-silence && nohup bash scripts/human_eval.sh > /workspace/logs/human_eval.log 2>&1 &
set -uo pipefail
W=/workspace; cd $W/moshi-silence; PY=$W/moshi-finetune/.venv/bin/python; export HF_HUB_OFFLINE=1
R=$W/runs
CK="base $R/R17_1h_lr5e-05_epad10/checkpoints/checkpoint_000042 $R/R21_10h_lr5e-05_epad10/checkpoints/checkpoint_000153
    $R/R22_50h_lr5e-05_epad10/checkpoints/checkpoint_000345 $R/R24_v3_b1-15_lr5e-05_epad10/checkpoints/checkpoint_000527"
for s in test_human test_human_free test_human_tts; do
  echo "$(date +%T) $s"
  CUDA_VISIBLE_DEVICES=0 $PY -m silence.eval_gen run --episodes data/$s/manifest.jsonl --out $W/eval/human/$s \
    --moshi-dir $W/moshiko --finetune $W/moshi-finetune --samples 3 --ckpt $CK >> $W/logs/eval/human_eval.log 2>&1
  for d in $W/eval/human/$s/*/; do [ -f $d/meta.json ] && [ ! -f $d/scores.jsonl ] && $PY -m silence.eval_score $d > $d/score.log 2>&1; done
done
echo "$(date +%T) HUMAN EVAL DONE"
