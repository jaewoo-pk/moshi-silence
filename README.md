# moshi-silence

## For evaluators: where to look

| What | Where |
|---|---|
| **Code** | This repo. Data generator in `silence/` (`specs.py`, `build.py`, `assemble.py`, `check.py`), eval harness (`eval_gen.py`, `eval_score.py`, `eval_report.py`), training configs (`make_configs.py`), GPU scripts in `scripts/`, patches to moshi-finetune in `patches/`. |
| **LoRA weights** | Not in this repo (GitHub's 100 MB file cap). All checkpoints are on the RunPod network volume `np7c8l94dh` under `/workspace/runs/<run>/checkpoint_<step>/`. Best model: `R24_v3_b1-15_lr5e-05_epad10/checkpoint_000527` (45.6 h set, EPAD x10, lr 5e-5). Runner-up: `R21_10h_lr5e-05_epad10/checkpoint_000153` (10 h). Each is ~0.37 GB. A Hugging Face upload is pending; ask and we'll share the link. |
| **Demo website** | **https://jaewoo-pk.github.io/moshi-demo/** (also listed under Projects on https://jaewoo-pk.github.io). Base Moshi vs. the 45 h model on ten held-out conversations: audio, timelines, verdicts, two failure cases. Source folder: `demo45h/site/` (self-contained; built by `demo45h/build.py`). Also: `results/demo_curated/index.html` (curated clips across all runs) and the human test set listening room `results/human/listen/index.html`. |
| **Process document** | `docs/decisions.md`: the work-trial report, section by section (dataset, evals, 1 h LR sweep, 10 h, EPAD, 50 h) with the decisions made at each step. Supporting detail: `docs/training_record.md` (every run, numbers), `docs/report.md` (draft write-up), `results/*/summary.md` (per-phase results). |

How much synthetic fine-tuning data does it take to teach Moshi (`kyutai/moshiko-pytorch-bf16`)
when to stay silent on instruction and when to speak again? This repo generates the data
(in the format [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune)
trains on), audits it, and sets up the fine-tuning runs. See `CLAUDE.md` for the full design,
decisions and status.

## Status (2026-10-14)

| | |
|---|---|
| Packs | 2,500 LLM content packs (`packs/packs.jsonl`); 348 screened test packs |
| Train | `data/train`: 1,150 episodes / 10.49 h, all pass checks. `train_1h.jsonl` = 113 episodes / 1.01 h, `train_10h.jsonl` = 1,101 / 10.01 h (1 h ⊂ 10 h) |
| Dev | `data/dev`: 200 episodes / 1.97 h, `dev_80` ⊂ `dev_200`, balanced per family, no pack or script overlap with train |
| Test | `data/test_*`: 7 suites × 90 = 630 episodes / 6.1 h, all pass checks, no script overlap with train |
| Training | Configs + GPU bootstrap ready and reviewed, not yet run. |
| Evaluation | Generation harness (forced script + free windows), scorer and report built and CPU-tested; not yet run with the 7B weights. |

## Episodes

Four behavior families, mixed 15 / 25 / 20 / 40 in training:

- **B1 timed hold:** "give me ten seconds, then ask me again". Variants: std, double hold,
  early answer, rescind.
- **B2 side conversation:** "stay quiet until I'm back", with distractors.
- **B3 hear me out:** "let me describe all three options". Variants: std, finishes early,
  rescind.
- **Ordinary Q&A**, plus lexical decoys.

`episodes/<id>.wav` is stereo 24 kHz: **left = Moshi, right = user + friend + noise**.
Train episodes are ≤ 60 s, i.e. exactly one training window. `episodes/<id>.json` holds:

- `alignments`: Moshi's words only, `[word, [start, end], "SPEAKER_MAIN"]`. This is the
  only key the trainer reads.
- `events`: eval labels: `instr_end`, holds and releases (kind, reference time, Moshi
  onset), distractors (code + span), response points, all turns with times.
- `meta`: pack, voices, templates, mixing params, TTS clip ids.

`python -m silence.transcript data/train/episodes/tr_000000.json` prints an episode as a timed
script.

## Layout

```
silence/phrasebank.py      instruction / cue / ack phrasings, train vs test lists
silence/splits.py          which axis each test suite holds out
silence/voices.py          voice pools (Pocket TTS: 16 train, 4 test, `michael` for Moshi)
silence/specs.py           episode builders, TRAIN_MIX, timing
silence/tts.py             TTS backends (pocket | kyutai | espeak) + clip cache
silence/assemble.py        timeline -> stereo wav + {alignments, events, meta} json
silence/build.py           CLI: specs -> TTS -> episodes + manifest
silence/check.py           checks via moshi-finetune's own Interleaver (--plots: plots + index.html)
silence/transcript.py      episodes as timed transcripts
silence/gen_packs.py       LLM content packs (Anthropic API) + schema / speakability filters
silence/make_splits.py     nested 1 h / 10 h subsets + data ledger + exposure math
silence/build_dev.py       dev set (train distribution, unseen and screened packs)
silence/screen_packs.py    test packs minus any overlapping train / dev
silence/check_overlap.py   train vs held-out overlap audit
silence/make_configs.py    moshi-finetune yamls for the run matrix
silence/eval_gen.py        GPU generation eval: script forced outside the scored windows
silence/eval_score.py      behavior scores from VAD + episode events
silence/eval_report.py     rates with bootstrap CIs, tables, per-epoch curves
silence/list_kyutai_voices.py, calibrate_offset.py   only for the optional 1.6B Kyutai TTS
scripts/gpu_bootstrap.sh   GPU box setup + smoke train + eval selftest/benchmark
scripts/run_parallel.sh    one training run per GPU
scripts/eval_dev.sh        generation eval of whole runs, one run per GPU
patches/                   three moshi-finetune patches (see below)
packs/                     generated packs (+ rejects, smoke test, log)
demo/                      14 example episodes (one per type; wavs not in git)
```

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

TTS is Kyutai Pocket TTS (~100M params, CPU) through the `pocket-tts-timestamped` fork, which
gives word timestamps from the model's own attention. It runs at ~12× realtime with 4 workers
on an Apple Silicon CPU: 10 h of episodes in ~40 min. `--tts espeak` is a robotic fallback
(Linux only). `--tts kyutai` uses the 1.6B Kyutai TTS on a GPU; calibrate its word offset
first with `python -m silence.calibrate_offset`.

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

### Test suites

Each suite changes one axis versus training:

| Suite | What's new |
|---|---|
| `test_script` | scripts |
| `test_word` | scripts + the user's instruction / cue wording |
| `test_voice` | scripts + user / friend voices |
| `test_distract` | scripts + dense distractors + the friend saying "I'm back" |
| `test_cond` | scripts + new waiting conditions (keyword release, phone call, minute-long holds) |
| `test_ordinary` | scripts; ordinary questions and decoys only (regression) |
| `test_all` | scripts + wording + voices |

Each suite draws its own episode types and pack order, so the mix-based suites share only
42–68 of their 90 scripts. Comparing models on one suite is paired; comparing suites is not.

## Training (rented GPU)

Plan: RunPod Secure Cloud, on-demand (not spot: runs can't resume), with a ≥ 150 GB network
volume at `/workspace`. Set up on a 1× H100 pod, then train on a 4× H100 pod using the same
volume.

```
# 1x pod: upload code (from the branch worktree) + data (~6.2 GB, from the main checkout),
# install, smoke-test. `box` = ssh alias for the pod's exposed port. The RunPod PyTorch image
# has rsync; -rlpt instead of -a because the network volume refuses chown.
rsync -rlpt --exclude .git --exclude demo --exclude __pycache__ --exclude .DS_Store \
    ~/repo/moshi-silence/.claude/worktrees/docs-progress/ box:/workspace/moshi-silence/
rsync -rlpt --exclude .DS_Store ~/repo/moshi-silence/data/ box:/workspace/moshi-silence/data/
ssh box 'bash /workspace/moshi-silence/scripts/gpu_bootstrap.sh'

# 4x pod, same volume: one run per GPU; the next config starts as a GPU frees up
cd /workspace/moshi-silence && mkdir -p /workspace/logs
nohup bash scripts/run_parallel.sh configs/R0_devloss.yaml configs/R?_1h_*.yaml > /workspace/logs/parallel.log 2>&1 &
tail -f /workspace/logs/parallel.log
```

`run_parallel.sh`:
- gives each run its own GPU, `torchrun` port and log (`/workspace/logs/<run>.log`)
- marks finished runs `.done` and skips them on a rerun
- refuses to start over a leftover `run_dir`, since an interrupted run can't resume (delete
  the `run_dir` to redo it)

Measured on an H100 SXM (2026-10-14): ~10 s/step for 16 × 60 s, ~47 s per dev-loss pass over
200 episodes, ~43 min per 1 h run. Phase-1 training takes ~1.5 h on 4× (vs ~4 h on 1×) for
the same GPU-hours. Generation eval: 28 ms/step at batch 40, ≈ 70 s per checkpoint on dev_80.

The bootstrap:
1. clones moshi-finetune at upstream `2acc879f` and applies the three patches
2. builds its environment from moshi-finetune's `uv.lock` (`pip install -e .` doesn't resolve
   upstream)
3. downloads the weights pinned at revision `2bfc9ae6`
4. writes the configs
5. runs a 6-step smoke train, which must produce two checkpoints, no NaN, and a logged text
   dev loss

The patches:

| Patch | What it does |
|---|---|
| `moshi-finetune-audio-mask.patch` | `mask_main_speech_audio`: no audio loss while Moshi speaks, so the synthetic TTS voice can't pull Moshi's voice. The text loss is unchanged. |
| `moshi-finetune-eval-reload.patch` | Upstream builds the finite eval loader once, so every evaluation after the first logs NaN. Builds a fresh one per evaluation, keeps the last partial batch, and logs text / audio dev loss separately. |
| `moshi-finetune-text-tail.patch` | Text past the end of a short file was PAD and counted in the text loss (~46% of text frames at 1 h). Now excluded. |

Run plan: fixed epochs, LoRA r=64, batch 16, 60 s windows, every checkpoint kept.

| Phase | Runs | Data | Peak LR | Epochs |
|---|---|---|---|---|
| 0 | `R0_devloss` | base model | 0 | — (dev loss only) |
| 1 | R1–R5 | 1 h (113 episodes, 142 steps) | 2e-6, 1e-5, 5e-5, 2e-4, 5e-4 | 20, checkpoint + dev loss every epoch |
| 2 | R6–R7 | 10 h (1,101 episodes) | top 2 from phase 1 | E chosen from phase 1 |

The dev set is for choosing E, LRs and checkpoints. The test suites are run once, on the
final checkpoints. That's roughly 7–10 H100-hours of training. Configs are written by
`python -m silence.make_configs` (see its `--help`).

## Evaluation (generated audio)

Each episode is replayed to Moshi: the user channel is fed frame by frame, and Moshi's own
stream is forced to the script except inside the scored windows (from the end of each hold
instruction to the next scripted human turn after the release; for ordinary episodes,
everything after the first user turn). Inside them Moshi runs free. Its decoded audio is run
through Silero VAD and scored against the episode's `events`: hold violations, resume on time
/ latency, B1 timer error, distractor reactions per code, ordinary responses, barge-in.

```
PY=/workspace/moshi-finetune/.venv/bin/python
$PY -m silence.eval_gen selftest --episodes data/dev/dev_80.jsonl          # alignment check vs training
nohup bash scripts/eval_dev.sh dev_80 R0 R1_1h_lr2e-06 R2_1h_lr1e-05 R3_1h_lr5e-05 \
    R4_1h_lr0.0002 R5_1h_lr0.0005 > /workspace/logs/eval_dev.log 2>&1 &   # all checkpoints, one run per GPU
$PY -m silence.eval_report /workspace/eval --configs configs --runs /workspace/runs --out /workspace/report/phase1
python -m silence.eval_score --oracle data/dev --jsonl data/dev/dev_80.jsonl   # scorer check on the ground truth
```

Note: training conditions Moshi's frame-t output on user audio up to frame t-1, while
`LMGen` fed a wav directly uses frame t. `eval_gen` shifts the user stream by one frame to
match training (verified exactly on a small model; `selftest` checks the real one).

## Known gaps

- Pocket TTS has 20 English voices plus Moshi's fixed `michael`, so the voice test has only
  4 unseen speakers.
- Turn-gap timing is a placeholder distribution; background noise is synthetic.
- The demo can't be rebuilt from the current code (its hand-written packs were removed).
- Whisper ASR and the LLM judge for the final test evals, and the human-recorded test set,
  are still to do.
