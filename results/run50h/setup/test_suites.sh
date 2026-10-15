#!/usr/bin/env bash
# Generation eval of given checkpoints on the 7 held-out test suites (2026-10-15), 1 sample / episode.
#   OUT=/workspace/eval/test bash scripts/test_suites.sh base /workspace/runs/R21_.../checkpoints/checkpoint_000153 ...
# One eval_gen process per suite (labels are read from that suite's dir); LoRA tensors are swapped per checkpoint.
# Output: $OUT/<suite>/<checkpoint_XXXXXX|base>/, scored with eval_score. EXTRA="--batch 20" when sharing the GPU.
set -uo pipefail
W=/workspace; cd $W/moshi-silence; PY=$W/moshi-finetune/.venv/bin/python; export HF_HUB_OFFLINE=1
OUT=${OUT:?set OUT}; mkdir -p $W/logs/eval
for s in test_script test_word test_voice test_distract test_cond test_ordinary test_all; do
  echo "$(date +%T) $s: $*"
  CUDA_VISIBLE_DEVICES=${GPU:-0} $PY -m silence.eval_gen run --episodes data/$s/manifest.jsonl --out $OUT/$s \
    --moshi-dir $W/moshiko --finetune $W/moshi-finetune ${EXTRA:-} --ckpt "$@" >> $W/logs/eval/test_$(basename $OUT).log 2>&1
  for d in $OUT/$s/*/; do [ -f $d/meta.json ] && [ ! -f $d/scores.jsonl ] && $PY -m silence.eval_score $d > $d/score.log 2>&1; done
done
echo "$(date +%T) test suites done"
