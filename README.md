# moshi-silence

Teaching Moshi (`kyutai/moshiko-pytorch-bf16`) when to stay silent on instruction and when to
speak again, with synthetic fine-tuning data and [moshi-finetune](https://github.com/kyutai-labs/moshi-finetune).

## For evaluators

| What | Where |
|---|---|
| **Code** | This repo: data generator and eval harness in `silence/`, GPU scripts in `scripts/`, moshi-finetune patches in `patches/`, results in `results/`. |
| **LoRA weights** | Not in the repo (GitHub's 100 MB file cap). Contact the owner. |
| **Demo** | https://jaewoo-pk.github.io/moshi-demo/ |
| **Process document** | [`docs/decisions.md`](docs/decisions.md) |

## Setup (data generation runs on a CPU)

```
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
# Pocket TTS weights (ungated repo; needs Hugging Face access once):
hf download kyutai/pocket-tts-without-voice-cloning --include "languages/english_2026-09/*" --local-dir models/pocket-tts
cp .env.example .env    # put your ANTHROPIC_API_KEY here (gitignored)
# for check.py: a moshi-finetune checkout next to this repo
git clone https://github.com/kyutai-labs/moshi-finetune ../moshi-finetune
```

## Building the data

Only one process should write to `data/` at a time.

```
# 1. content packs (Anthropic API; rerun with the same --out to top families up)
python -m silence.gen_packs --out packs/packs.jsonl --per-family 500 --rejects packs/rejects.jsonl

# 2. train: 1,150 episodes = 10.5 h; 1 h / 10 h prefixes + ledger
python -m silence.build train --packs packs/packs.jsonl --n 1150 --out data/train
python -m silence.make_splits data/train --hours 1 10

# 3. dev: 200 episodes from train-split packs unused by data/train and screened for overlap
python -m silence.build_dev --exclude data/train --out data/dev

# 4. test suites: from test packs screened against train + dev
python -m silence.screen_packs --split test --against data/train data/dev --out packs/packs_test_screened.jsonl
for s in test_script test_word test_voice test_distract test_cond test_ordinary test_all; do
  python -m silence.build $s --packs packs/packs_test_screened.jsonl --n 90 --out data/$s; done

# 5. checks
python -m silence.check data/train --finetune ../moshi-finetune          # --plots for index.html
python -m silence.check_overlap data/train data/dev data/test_*
```

## Training (rented GPU)

RunPod on-demand (not spot: runs can't resume), with a network volume at `/workspace`.
Set up on a 1× H100 pod, then train on a 4× H100 pod using the same volume.

```
# 1x pod: upload code + data, install, smoke-test. `box` = ssh alias for the pod.
rsync -rlpt --exclude .git --exclude demo --exclude __pycache__ --exclude .DS_Store ./ box:/workspace/moshi-silence/
rsync -rlpt --exclude .DS_Store data/ box:/workspace/moshi-silence/data/
ssh box 'bash /workspace/moshi-silence/scripts/gpu_bootstrap.sh'

# 4x pod, same volume: one run per GPU; the next config starts as a GPU frees up
cd /workspace/moshi-silence && mkdir -p /workspace/logs
nohup bash scripts/run_parallel.sh configs/R0_devloss.yaml configs/R?_1h_*.yaml > /workspace/logs/parallel.log 2>&1 &
tail -f /workspace/logs/parallel.log
```

The bootstrap clones moshi-finetune at `2acc879f`, applies the patches in `patches/`, builds
the env from its `uv.lock`, downloads the weights at revision `2bfc9ae6`, writes the configs
(`python -m silence.make_configs --help`) and runs a 6-step smoke train. `run_parallel.sh`
gives each run its own GPU and log, marks finished runs `.done`, and refuses to start over a
leftover `run_dir` (delete it to redo a run).

## Evaluation (generated audio)

```
PY=/workspace/moshi-finetune/.venv/bin/python
$PY -m silence.eval_gen selftest --episodes data/dev/dev_80.jsonl          # alignment check vs training
nohup bash scripts/eval_dev.sh dev_80 R0 R1_1h_lr2e-06 R2_1h_lr1e-05 R3_1h_lr5e-05 \
    R4_1h_lr0.0002 R5_1h_lr0.0005 > /workspace/logs/eval_dev.log 2>&1 &   # all checkpoints, one run per GPU
$PY -m silence.eval_report /workspace/eval --configs configs --runs /workspace/runs --out /workspace/report/phase1
python -m silence.eval_score --oracle data/dev --jsonl data/dev/dev_80.jsonl   # scorer check on the ground truth
```
