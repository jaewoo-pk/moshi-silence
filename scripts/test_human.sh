#!/usr/bin/env bash
# Human test set (2026-10-15): generation eval on the three paired suites built by silence/human.py
# (test_human_tts = TTS twins, test_human = human voice verbatim, test_human_free = own-words instruction / cue),
# 3 samples / episode (final-test protocol), same seed per episode across models.
#   OUT=/workspace/eval/human bash scripts/test_human.sh base /workspace/runs/R21_.../checkpoints/checkpoint_000153 ...
# Output: $OUT/<suite>/<checkpoint_XXXXXX|base>/, scored with eval_score. Checkpoint dir names must differ.
set -uo pipefail
W=/workspace; cd $W/moshi-silence; PY=$W/moshi-finetune/.venv/bin/python; export HF_HUB_OFFLINE=1
OUT=${OUT:?set OUT}; mkdir -p $W/logs/eval
for s in test_human_tts test_human test_human_free; do
  echo "$(date +%T) $s: $*"
  CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -m silence.eval_gen run --episodes data/$s/manifest.jsonl --out $OUT/$s \
    --moshi-dir $W/moshiko --finetune $W/moshi-finetune --samples ${SAMPLES:-3} ${EXTRA:-} --ckpt "$@" \
    >> $W/logs/eval/test_human.log 2>&1
  for d in $OUT/$s/*/; do [ -f $d/meta.json ] && [ ! -f $d/scores.jsonl ] && $PY -m silence.eval_score $d > $d/score.log 2>&1; done
done
echo "$(date +%T) human test suites done"
