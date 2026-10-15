#!/usr/bin/env bash
# dev_200 generation eval of single candidate checkpoints (2026-10-15). Args: run/checkpoint_XXXXXX ...
# Output: /workspace/eval/<run>/dev_200/<checkpoint>/ (same layout as eval_dev.sh), then scored.
set -uo pipefail
W=/workspace; cd $W/moshi-silence; PY=$W/moshi-finetune/.venv/bin/python; export HF_HUB_OFFLINE=1
for rc in "$@"; do
  run=${rc%%/*}; ck=${rc#*/}; out=$W/eval/$run/dev_200
  echo "$(date +%T) dev_200: $run $ck"
  CUDA_VISIBLE_DEVICES=0 $PY -m silence.eval_gen run --episodes data/dev/dev_200.jsonl --out $out \
    --moshi-dir $W/moshiko --finetune $W/moshi-finetune --ckpt $W/runs/$run/checkpoints/$ck >> $W/logs/eval/dev200_confirm.log 2>&1
  [ -f $out/$ck/meta.json ] && $PY -m silence.eval_score $out/$ck > $out/$ck/score.log 2>&1
done
echo "$(date +%T) dev_200 confirm done"
