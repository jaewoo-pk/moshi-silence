#!/usr/bin/env bash
# Pull one EPAD-10h run's artifacts from a pod (read-only on the pod) into results/epad10h/.
# Usage (repo root): bash results/epad10h/setup/pull_run.sh <ssh alias> <run> <tag 3|10>
set -euo pipefail
H=$1; RUN=$2; TAG=$3
D=results/epad10h
S="ssh -o ConnectTimeout=10"
rsync -rlt -e "$S" "$H:/workspace/moshi-silence/configs/$RUN.yaml" "$H:/workspace/moshi-silence/configs/$RUN.epochs.json" $D/configs/
mkdir -p $D/runs/$RUN
rsync -rlt -e "$S" "$H:/workspace/runs/$RUN/args.yaml" "$H:/workspace/runs/$RUN/metrics.train.jsonl" "$H:/workspace/runs/$RUN/metrics.eval.jsonl" $D/runs/$RUN/
rsync -rlt -e "$S" "$H:/workspace/logs/$RUN.log" "$H:/workspace/logs/parallel_epad10h_$TAG.log" "$H:/workspace/logs/eval_dev_epad10h_$TAG.log" \
  "$H:/workspace/logs/epad10h_$TAG.log" "$H:/workspace/logs/parallel.log" $D/logs/
rsync -rlt -e "$S" "$H:/workspace/logs/eval/${RUN}_dev_80.log" $D/logs/eval/
mkdir -p $D/eval_raw/$RUN
rsync -rlt -e "$S" --exclude 'codes.npz' --exclude '*.wav' "$H:/workspace/eval/$RUN/dev_80" $D/eval_raw/$RUN/
# checkpoint manifest: sizes + sha256, computed on the pod
$S $H "cd /workspace/runs && for c in $RUN/checkpoints/checkpoint_*; do printf '%s\t%s\t%s\n' $RUN \$(basename \$c) \$(stat -c %s \$c/consolidated/lora.safetensors); done; sha256sum $RUN/checkpoints/checkpoint_*/consolidated/lora.safetensors" > $D/setup/checkpoints_manifest_$RUN.txt
# verify: remote vs local file counts for the eval dir (excluding npz / wav)
R=$($S $H "find /workspace/eval/$RUN/dev_80 -type f ! -name codes.npz ! -name '*.wav' | wc -l")
L=$(find $D/eval_raw/$RUN/dev_80 -type f ! -name codes.npz ! -name '*.wav' | wc -l | tr -d ' ')
NS=$(ls $D/eval_raw/$RUN/dev_80/*/scores.jsonl | wc -l | tr -d ' ')
NC=$(grep -c safetensors $D/setup/checkpoints_manifest_$RUN.txt)
echo "$RUN: eval files remote $R local $L; scores.jsonl $NS; checkpoints hashed $NC"
