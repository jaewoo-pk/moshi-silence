#!/usr/bin/env bash
# 50 h build, final step on a pod (2026-10-15): assemble every planned episode from the clips the
# render-only shards put in /workspace/cache50, run check.py, drop failures, write nested subsets.
#   cd /workspace/moshi-silence && nohup bash scripts/build50_final.sh > /workspace/logs/build50/final.log 2>&1 &
set -euo pipefail
W=/workspace; cd $W/moshi-silence; export PYTHONPATH=$PWD
OUT=$W/data50/train50; N=${N:-20}
echo "$(date -u +%T) assemble"
nice -n 5 $W/buildenv/bin/python -m silence.build train --tts multi --mix v2 --packs packs/packs_v2_screened.jsonl \
  --exclude data/dev --n 5600 --out $OUT --cache $W/cache50 --model-dir $W/models/pocket-tts --workers $N
echo "$(date -u +%T) check.py"
$W/moshi-finetune/.venv/bin/python -m silence.check $OUT --finetune $W/moshi-finetune --workers $N > $OUT/check.log 2>&1 || true
tail -1 $OUT/check.log
echo "$(date -u +%T) filter"
$W/buildenv/bin/python -m silence.filter_checked $OUT
$W/buildenv/bin/python -m silence.make_splits $OUT --hours 1 10 25 50 --abs | grep -E "^train_|only"
du -sh $OUT
echo "$(date -u +%T) BUILD50 DONE"
