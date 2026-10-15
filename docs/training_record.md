# Training record (for the write-up)

A running, factual record of how the models were trained: setup, hyperparameters, data,
compute, cost, issues and results. Kept up to date as runs finish. Raw artifacts are under
`results/` (layout at the end). All times UTC (ET = UTC − 4 on these dates).

## 1. Question and design in one paragraph

How much synthetic duplex data does it take to teach Moshi to stay silent on request and to
resume at the right moment, and does the behavior generalize? Every model is a LoRA fine-tune
of the same public checkpoint. Phase 1 sweeps the peak learning rate on 1 h of training data
(113 episodes, 20 epochs) with a dev evaluation at every epoch. It chooses the number of epochs
E and the two best LRs. Phase 2 trains those two LRs on 10 h (1,101 episodes) for E epochs.
Final models are scored once on 7 held-out test suites (630 episodes) plus a human-recorded set.

## 2. Base model and code (pinned)

| Item | Version |
|---|---|
| Base checkpoint | `kyutai/moshiko-pytorch-bf16` @ `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289` (7B Moshi LM + Mimi codec + SentencePiece 32k text tokenizer) |
| Fine-tuning code | `kyutai-labs/moshi-finetune` @ `2acc879fe7c48f885a18f6cc9548bccb2674d87b` + 3 local patches (below) |
| moshi package | git `061cc4c630d9` (0.2.4a1), from moshi-finetune's `uv.lock` |
| torch | 2.6.0+cu124 (torchaudio 2.6.0) |
| Other pins | sphn 0.1.12, numpy 2.1.3, safetensors 0.5.3, sentencepiece 0.2.0, huggingface-hub 0.28.1, silero-vad 6.2.3 (eval only) |
| Python / env | CPython 3.12.15 managed by uv 0.13.1 (`uv sync --frozen` on moshi-finetune's lockfile) |
| Full freeze | `results/phase1/setup/pip-freeze.txt` |

Patches to moshi-finetune (`patches/`):
1. **audio-mask**: new option `mask_main_speech_audio`. No audio loss on Moshi's own speech
   frames (pre 0.08 s / post 0.24 s margin), because that audio is synthetic TTS we don't want
   the model to imitate. Audio loss stays on Moshi's silent frames.
2. **eval-reload**: upstream builds the dev loader once as a finite generator, so every
   evaluation after the first sees no batches and logs NaN. It also drops the last partial batch and
   never logs the text/audio split. The patch builds a fresh loader per evaluation and logs
   `text_eval_loss` / `audio_eval_loss`.
3. **text-tail**: text frames past the end of a short file were PAD and counted in the text
   loss (≈ 46% of all text frames at 1 h). The patch sets them to ignore (−1), like the audio.

## 3. Hardware and platform

| Item | Value |
|---|---|
| Provider | RunPod Secure Cloud, on-demand, data center US-NE-1 |
| GPUs | NVIDIA H100 80GB HBM3 (SXM); driver 580.126.09, host CUDA 13.0 |
| Setup pod | 1× H100, 26 vCPU, 251 GB RAM |
| Training pod | 4× H100, 104 vCPU, 1 TB RAM (one run per GPU; no multi-GPU data parallelism) |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (Ubuntu 24.04.3); the training env is the uv env above, not the image's torch |
| Storage | 150 GB Standard network volume (`np7c8l94dh`) at `/workspace`: env, weights, data, runs |
| Price | $3.99 per H100-hour (4× pod: $15.96/hr); volume ≈ $0.07/GB-month |

## 4. Data used in training (phase 1)

Synthetic two-channel episodes (left = Moshi, right = user + friend + noise; 24 kHz), built by
`silence/` with Pocket TTS. Full description in `README.md` and `CLAUDE.md`.

| Set | Episodes | Unique packs | TTS clips | Hours | Speech h | Moshi speech h | Hold-silence h |
|---|---|---|---|---|---|---|---|
| train_1h (phase 1) | 113 | 116 | 885 | 1.01 | 0.70 | 0.25 | 0.39 |
| train_10h (phase 2) | 1,101 | 1,135 | 8,463 | 10.01 | 7.05 | 2.48 | 3.73 |
| dev_200 (dev loss) | 200 | — | — | 1.97 | — | — | — |
| dev_80 (generation eval, phase 1) | 80 (first 80 of dev_200) | — | — | 0.77 | — | — | — |

- train_1h by type: B1 timed hold 22 (std 17 / double 3 / early 1 / rescind 1), B2 side
  conversation 24, B3 hear-me-out 22 (std 18 / early 2 / rescind 2), ORD 31, DECOY 14. It's a
  random draw from the training mix, so there are only 1–2 contrast episodes per type at 1 h.
- train_1h ⊂ train_10h (first-K prefix). No pack is reused within train. Dev shares no pack or
  episode with train (closest script Jaccard 0.47).
- **Exposure** (each episode is one 60 s window; shorter episodes are padded and the padding
  is masked): phase 1 = 142 steps × 16 = 2,272 episode presentations = 20.1 epochs ≈ 20.3 h of
  audio seen, from 1.01 h unique. Unique data and repetition are reported separately.

## 5. Training configuration

Identical for all runs except the peak LR and the data size / number of steps (`silence/make_configs.py`;
exact files in `results/phase1/configs/`):

| Setting | Value |
|---|---|
| Method | LoRA on every linear layer (main transformer, depformer, heads); base weights frozen; embeddings not tuned |
| LoRA | rank 64, scaling 2 (≈ 194M trainable parameters; 0.39 GB per bf16 adapter checkpoint) |
| Window | `duration_sec` 60 (one episode per window; context 750 frames at 12.5 Hz) |
| Batch | 16 windows per step, 1 GPU per run |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 |
| LR schedule | OneCycle: warmup from peak/25 over the first 5% of steps, cosine decay to ≈ 0 at the last step |
| Loss | text CE (Moshi's words only; PAD/EPAD weight 0.5) + audio CE over Moshi's 8 codebooks (first codebook × 100); no audio loss on Moshi's speech frames (patch 1) |
| Precision | bf16, gradient checkpointing |
| Seed | 0 (same data order across runs: step-1 loss is identical in every run, 1.027) |
| Dev loss | all 200 dev episodes, at every checkpoint |
| Checkpoints | every 7 steps (≈ 1 epoch) and the last step; LoRA adapters only; none deleted |

**Phase 1 run matrix** (train_1h, 20 epochs = ⌈20 × 113 / 16⌉ = 142 steps):

| Run | Peak LR | Steps | Checkpoints (= dev evals) |
|---|---|---|---|
| R0_devloss | 0 (base model, 2 no-op steps) | 2 | 0 (1 dev eval) |
| R1_1h_lr2e-06 | 2e-6 | 142 | 21 |
| R2_1h_lr1e-05 | 1e-5 | 142 | 21 |
| R3_1h_lr5e-05 | 5e-5 | 142 | 21 |
| R4_1h_lr0.0002 | 2e-4 | 142 | 21 |
| ~~R5_1h_lr0.0005~~ | 5e-4 | — | dropped before it started (see below) |
| R6_1h_lr5e-06 | 5e-6 | 142 | 21 (added) |
| R7_1h_lr2e-05 | 2e-5 | 142 | 21 (added) |

**Sweep change (2026-10-15, 00:51 UTC, decided mid-run).** By epoch ~13, 5e-5 and 2e-4 had
memorized the 1 h training scripts: train text loss fell to 0.014 / 0.009, while dev text loss
bottomed at epoch 4 / 3 (0.341 / 0.331) and then rose to ~0.75 / ~0.70, above the base
model's 0.537. 5e-4 would only repeat this faster, so it was dropped. Two LRs were added where
the decision sits instead: 2e-5 (between 1e-5, which overfits only mildly, and 5e-5) and 5e-6
(between 2e-6, not converged in 20 epochs, and 1e-5). The recipe is identical (20 epochs,
same schedule and seed), so all six LRs lie on one curve.

Checkpoint → epoch: step × 16 / 113, i.e. 0.991, 1.982, …, 19.823 (step 140), 20.106 (step 142)
(`configs/<run>.epochs.json`). Caveat for the write-up: the epoch-E checkpoint is a snapshot of
a 20-epoch OneCycle run (taken at a higher LR than a run that ends at E), not a run trained to E.

**Phase 2 run matrix** (launched 2026-10-15 02:35 UTC; train_10h = 1,101 episodes / 10.01 h, 5 epochs =
⌈5 × 1101 / 16⌉ = 345 steps, checkpoint + dev loss every 17 steps ≈ ¼ epoch → 21 checkpoints; recipe,
seed and eval identical to phase 1):

| Run | Peak LR | 1 h counterpart |
|---|---|---|
| R8_10h_lr2e-05 | 2e-5 | R7 |
| R9_10h_lr5e-05 | 5e-5 | R3 |
| R10_10h_lr0.0002 | 2e-4 | R4 |
| R11_10h_lr0.0005 | 5e-4 | — (brackets the top; R5 at 1 h was dropped) |

Why this design: at 1 h, behavior plateaus after ~50–60 optimizer steps at 2e-4, i.e. within the
first epoch of 10 h. So phase 2 uses a short schedule (5 epochs, ≈ 6× the steps where 1 h plateaued)
with quarter-epoch checkpoints to see *when* the plateau arrives in data exposure, and four LRs at the
deliverable data size rather than inferring the 10 h LR from 1 h. Exposure: 345 × 16 = 5,520 episode
presentations ≈ 5.0 epochs ≈ 50 h of audio seen, from 10.0 h unique.

## 6. Evaluation protocol (dev)

- **Dev loss** (above): teacher-forced, text and audio separately.
- **Generation eval** (`silence/eval_gen.py`): the model generates Moshi's channel against the
  recorded user channel. Outside the scored windows Moshi's text and audio are forced to the
  script, so every hold is scored, not just the first. Mimi decodes the result, Silero VAD
  gives speech segments, and Moshi's text stream gives its words. 1 sample per episode, the same
  seed per episode for every model. Scored on dev_80 for every checkpoint in phase 1
  (`silence/eval_score.py`, `silence/eval_report.py`; 95% bootstrap CIs over episodes).
- Metrics: hold violated (≥ 1 s of speech or > 3 words inside the hold, after a 2.5 s ack
  grace), resumed on time (onset within max(1.5 s, 0.2 N) of the release, ≤ 3 s late), cue
  latency, B1 timer error, ordinary questions answered, reactions to distractors.
- **Alignment check on the real model** (`eval_gen selftest`, bf16, H100): the inputs the
  streaming generator feeds the model are integer-identical to the training forward's inputs
  (100% of columns) with the user stream delayed one frame. Text-logit argmax agreement is
  99.7% (max |Δlogit| median 0.094, p99 0.232: bf16 streaming vs full-sequence attention).
  Script text CE 0.491 streaming vs 0.490 full forward.

## 7. Compute and cost (measured)

| Quantity | Value |
|---|---|
| Training speed | ~10 s/step (≈ 9.3–10 s; ≈ 11.3k words/s), batch 16 × 60 s, 1× H100 |
| Peak GPU memory (training) | 29.1 GB |
| Model load + first step | ≈ 2.6 min per run |
| Dev loss, 200 episodes | ≈ 47 s per evaluation |
| Generation | 28 ms/step at batch 40 → 40 dev episodes in 35 s; dev_80 ≈ 70 s per checkpoint |
| Phase 1 per run | ≈ 43 min (24 min training + ~16 min dev evals + load) |
| Phase 1 actual | 35 min per run on the 4× pod (≈ 8.9 s/step with 4 runs in parallel); generation eval ≈ 19 min per run (21 checkpoints × 80 episodes, ≈ 57 s each) |

Timeline and spend:

| When (UTC) | What | Cost |
|---|---|---|
| 2026-10-14 23:47 – ~23:56 | First setup pod on the wrong (personal) account; failed at `uv sync` (see §8); terminated | ≈ $0.60 (personal account) + a few cents of volume |
| 2026-10-14 23:58 – 2026-10-15 00:18 | 1× H100 setup pod: env, weights, data upload (6.1 GB), smoke train, selftest, benchmark | ≈ $1.35 |
| 2026-10-15 00:20 – 01:52 | 4× H100 pod (`z77njns1kqy38r`): phase 1 training (R0–R4, R6, R7; R5 dropped), all dev_80 generation evals (127 models × 80 episodes), copy back; terminated | ≈ $25 (1.55 h × $15.96) |
| 2026-10-15 02:33 – 04:08 | 4× H100 pod (`sq9r1dzsa8l6rc`): phase 2 (R8–R11, 10 h × 4 LRs), dev_80 evals of 84 checkpoints, copy back; terminated | ≈ $25 (1.6 h × $15.96) |
| 2026-10-15 02:40 – 03:07 | 1× H100 chat pod (`ywpfqsbqerh53c`): live chat with base + 1 h 2e-4 ep 16; terminated | ≈ $1.80 |
| 2026-10-15 05:25 – 06:30 | 1× H100 chat pod (`vc42i3dtvzv7s5`), live chat with checkpoints; terminated at the user's request | ≈ $4.3 |
| 2026-10-15 12:31 – 12:43 | 1× H100 pod (`oehusbm18t7m75`): R17 + R24 on the 3 human-recorded sets (base / R21 / R22 had been run by another session); terminated | ≈ $0.8 |
| 2026-10-15 09:43 – 12:29 | 1× H100 pod (`5obz5mapyti207`): live chat R21 vs R22, v3 build (CPU), R24 training + dev_80 evals + test suites; terminated | ≈ $11 |
| 2026-10-15 07:29 – 09:31 | 1× H100 pod (`i1wn60oi6h7n8g`): R22 (50 h) training + dev_80 evals, test suites for base / R17 / R21 / R22; terminated | ≈ $8.1 |
| 2026-10-15 05:33 – 07:14 | two 1× H100 pods (`086j0wex48cmlp`, `xk9kzxl2eev9of`): EPAD at 10 h (R20, R21) + dev_80 evals; their CPUs also rendered/assembled the 50 h set (§12) | ≈ $13.4 |
| 2026-10-15 04:19 – 05:16 | 4× H100 pod (`7f5yuy4acmhxzp`): EPAD sweep (R16–R19), dev_80 evals of 72 checkpoints; disk-full crash at step 132–133; volume grown 150 → 250 GB; terminated | ≈ $15 |

## 8. Issues found and how they were handled

- **Upstream install doesn't resolve with pip** (pyproject pins sphn 0.1.12 but takes moshi
  from git main, which now needs sphn ≥ 0.2): installed from moshi-finetune's `uv.lock`.
- **`uv sync` failed building openai-whisper 20240930** (a dependency of whisper-timestamped,
  unused here): its setup.py imports `pkg_resources`, which setuptools ≥ 81 dropped, and build
  isolation picks the newest setuptools. Fix: sync everything else first (installs the locked
  setuptools 78.1.0), then build whisper without isolation.
- **Eval loader / NaN dev loss, PAD tail in the text loss**: patches 2 and 3 (§2).
- **Checkpoint retention**: upstream keeps only the last 3 checkpoints; set to keep all.
- **Hub dependency at load**: moshi probes the hub for a config.json the repo doesn't have;
  every training and eval job runs with `HF_HUB_OFFLINE=1` and local weight paths.
- **User-stream alignment in generation**: feeding a wav straight into moshi's streaming
  generator hands the model the user's audio one frame (80 ms) earlier than training does;
  `eval_gen` delays it by one frame (verified on the real model, §6).
- Platform: the network volume refuses `chown`, so files are copied with `rsync -rlpt`.

## 9. Observations so far

- **Base model (R0)**: dev loss text 0.537, audio 0.393. On 40 dev episodes (benchmark
  sample, not the dev_80 result) it doesn't hold silence: 83% of holds violated, 42% of
  resumes on time, ordinary questions answered 43%, distractors trigger speech 67–100% of the time.
- **Audio dev loss collapses almost immediately.** Moshi's speech frames carry no audio loss
  (patch 1), so the audio loss is essentially on Moshi's *silent* frames. The base model
  expects to be talking there. After one epoch, audio dev loss is 0.373 (2e-6), 0.251 (1e-5)
  and 0.124 (5e-5), vs 0.393 for the base model. So audio dev loss mostly measures "emit
  silence codes" and saturates early; text dev loss and the behavior metrics carry the signal
  after that.
- Text dev loss after one epoch: 0.536 (2e-6), 0.527 (1e-5), 0.478 (5e-5) vs 0.537 base.
  2e-6 is indistinguishable from the base model at epoch 1, as expected.

- **Live chat, R21 vs R22 (user, 2026-10-15 ~10:05 UTC; pod `5obz5mapyti207`, slot A = R21 `checkpoint_000153`,
  slot B = R22 `checkpoint_000345`; logged in `results/chat/notes.jsonl`).** In conversation with the user's own voice
  the **50 h model (R22) is "just so much better empirically"**, and the 10 h model (R21) "can tend to be a bit quiet".
  This is the reverse of the automated ranking (R21 ≥ R22 by 2–6 points on 6 of 7 test suites). Likely reasons: (1) every
  dev / test suite uses Pocket TTS voices only, while a real voice is out of distribution; R22 trained on 63 voices from
  3 engines (incl. the more natural ElevenLabs / Fish), R21 on 16 Pocket voices, so R22 is plausibly more robust to real
  speech, which none of our metrics measures; (2) R22 speaks more readily (speaks when due 93–97% vs 91–96%, ordinary
  answered 100%), which helps free conversation, while the test suites penalize the flip side (speaking after a pause
  inside hear-me-out / double / rescinded holds), situations casual chat rarely creates. Implication: the human-recorded
  test set is needed before calling 10 h vs 50 h, and R24 (v3) should be judged on real-voice chat as well as the suites.
  Anecdotal (one listener, unscripted, no counts).

## 10. Results

### Base model on dev_80 (R0; generation eval, 1 sample per episode, seed per episode)

80 episodes, 63 holds (18 timed, 45 cue-released), 42 ordinary responses (hold-free episodes):

| Metric | Base model |
|---|---|
| Hold violated (≥ 1 s or > 3 words inside the hold) | **86.2%** (n = 58) |
| Talking through the release | 20.6% (n = 63) |
| Resumed on time | **39.7%** (n = 63); no onset in window 58.7% |
| Cue latency (on-time resumes) | median +0.14 s, p90 1.78 s (n = 21) |
| B1 timer error | median +0.38 s, p90 19.1 s (n = 11) |
| Words inside holds (incl. ack) | mean 15.6 |
| Ordinary questions answered | **42.9%** (n = 42); first turn 20.0% (n = 20) |
| Barge-in into mid-turn pauses | 50.0% (n = 6) |
| Unprompted onsets inside human turns | 56 in 80 episodes |
| Distractor triggered speech | D1a 71%, D1b 80%, D1c 67%, D2a 67%, D2b 83%, D2c 40%, D3a 80%, D3b 73% |

The unmodified model neither holds silence nor reliably answers: it talks through most holds
and misses over half of the ordinary questions. Generation speed: 80 episodes in 57 s (29 ms/step).

### Phase 1 sweep

**First behavior results (R3, 5e-5, dev_80; scored 2026-10-15 ~01:20 UTC while the rest ran):**

| Checkpoint (epoch) | Hold violated | Resumed on time | Ordinary answered | Words in holds | Interrupt onsets |
|---|---|---|---|---|---|
| base model | 86.2% | 39.7% | 42.9% | 15.6 | 56 |
| ep 1 | 44.8% | 20.6% | 11.9% | 5.4 | 19 |
| ep 2 | 27.6% | 22.2% | 33.3% | 2.8 | 5 |
| ep 3 | 15.5% | 33.3% | 61.9% | 2.2 | 0 |
| ep 4 | 3.4% | 39.7% | 78.6% | 1.2 | 0 |
| ep 5 | 8.6% | 54.0% | 73.8% | 1.6 | 0 |
| ep 6 | 17.2% | 58.7% | 71.4% | 2.4 | 0 |
| ep 7 | 5.2% | 60.3% | 85.7% | 1.9 | 2 |
| ep 8 | 5.2% | 54.0% | 71.4% | 1.1 | 1 |

(n = 58 scored holds, 63 resumes, 42 ordinary responses; 1 sample; ±10–12 pp at 95%.)
R1 (2e-6) moves the same way, slowly: hold violated 84.5% (ep 1) → 62.1% (ep 7).

Takeaways so far:
- Real behavior learning from 1 h: hold violations fall from 86% to ≈ 5%, while ordinary
  questions answered *rise* (43% → 71–86%) and unprompted interruptions go to ~0. So it is
  not a model that simply went quiet.
- **Dev text loss is a poor checkpoint selector.** R3's dev text loss is lowest at epoch 4
  and doubles by epoch 10, but behavior at epochs 5–8 stays as good or better (resume timing
  keeps improving). The loss blowup is memorization of the training scripts' *wording*, not
  loss of the *timing* behavior. Checkpoints are chosen on behavior metrics.
- Resuming on time is the weak side (≤ 60%); holding silence is close to solved on dev.

**Too low an LR gives a silent model (R1, 2e-6, full eval):** hold violations improve
(86% → 33–45%), but resume-on-time *falls* (40% → 22% [12, 33] at the end), ordinary questions
answered collapse (43% [29, 58] → 10% [0, 23]), and "no onset" rises from 59% to 78%. Meanwhile
R1's dev text loss falls monotonically (0.54 → 0.46). The model learns the dominant, easy part of
the signal (Moshi's channel is silent most of the time) but not when to start talking again,
and the loss rewards it. At 5e-5 (R3) the model learns both halves. Report both rates for
ordinary answers: all questions (n = 42, headline) and first turn only (n = 20).



### Phase 1 final (all 6 LRs × 21 checkpoints + base, dev_80; completed 2026-10-15 01:50 UTC)

Behavior averaged over the plateau (epochs 10–20, 11 checkpoints) and at the best checkpoint
under the provisional equal-weight score mean(1 − violated, on time, ordinary answered):

| Peak LR | Plateau: hold violated | resumed on time | ordinary answered | words in holds | Best (dev_80): epoch, viol / on time / ord | Dev text loss: best (epoch) → final |
|---|---|---|---|---|---|---|
| base model | 86% | 40% | 43% | 15.6 | — | 0.537 |
| 2e-6 (R1) | 40% | 22% | 10% | 5.1 | ep 5: 76% / 51% / 36% | 0.457 (20) → 0.457 |
| 5e-6 (R6) | 23% | 26% | 39% | 3.5 | ep 19: 17% / 27% / 43% | 0.373 (19) → 0.373 |
| 1e-5 (R2) | 15% | 42% | 80% | 2.5 | ep 19: 14% / 51% / 86% | 0.351 (10) → 0.370 |
| 2e-5 (R7) | 12% | 62% | 83% | 2.3 | ep 15: 17% / 73% / 90% | 0.347 (6) → 0.642 |
| 5e-5 (R3) | 16% | 64% | 87% | 2.9 | ep 15: 14% / 70% / 95% | 0.341 (4) → 0.803 |
| **2e-4 (R4)** | **11%** | **74%** | **98%** | 3.2 | **ep 16: 10% [3, 19] / 76% [64, 87] / 100%** | 0.331 (3) → 0.741 |

The top 8 checkpoints by that score are all R4 (2e-4), at epochs 9–20 (scores 0.864–0.886).
Distractor reactions (share that triggered speech, plateau mean): base D1a 71%, D1b 80%, D2a 67%,
D2b 83%, D3a 80%, D3b 73% → R4 1%, 4%, 15%, 8%, 5%, 9% (R3: 1%, 1%, 6%, 5%, 2%, 7%). D2c (fillers
during timed holds) stays high (R4 60%, R3 25%; only 5 instances on dev_80).

Findings:
1. **Two phases of learning.** Every LR first learns to be silent (violations and answers both
   drop; e.g. R4 at epoch 2: 0% violated, 3% on time, ~8% answered) and only later learns
   when to speak again (on-time and answered rise; R4 from epoch ~5). Low LRs (2e-6, 5e-6)
   never leave the silent phase within 20 epochs: a quiet model, not an instruction follower.
2. **Behavior improves monotonically with LR across the whole sweep;** 2e-4 is best on every
   behavior metric. Within this sweep there's no sign of a behavioral optimum.
3. **Dev loss ranks the LRs almost backwards.** At the end, the LRs with the lowest dev text
   loss (5e-6, 1e-5) behave worst among the trained models, and the highest-loss runs
   (5e-5, 2e-4) behave best. Dev text loss measures memorization of the scripts' wording.
   It doesn't measure the timing decision. **Checkpoints must be selected on behavior.**
4. **Hindsight on dropping 5e-4 (R5):** it was dropped on dev-loss evidence. Since behavior is
   still improving at 2e-4, the 5e-4 point would have shown whether behavior peaks or
   degrades. It's worth running (one 43-min run + ~20 min eval) if phase 2's LR choice
   should be bracketed.
5. Resume timing is the hardest part (best ≈ 75% on time, vs ≈ 90%+ for holds kept and ~100%
   for ordinary answers).

**Two-axis re-scoring (no regeneration; `results/phase1/analysis/`).** Silent when due = holds
kept; speaks when due = cue releases + timed releases + ordinary questions, with a lenient
"spoke" (any new onset from the release until the scripted user moves on, ~4 s). Plateau
epochs 10–20:

| Peak LR | Silent when due | Speaks when due | Cue release: spoke / on time | Timed release: spoke / on time | Ordinary answered |
|---|---|---|---|---|---|
| base | 14% | 45% | 51% / 47% | 33% / 22% | 43% |
| 2e-6 | 60% | 21% | 29% / 26% | 26% / 13% | 10% |
| 5e-6 | 77% | 33% | 32% / 29% | 20% / 18% | 39% |
| 1e-5 | 85% | 58% | 55% / 54% | 13% / 13% | 80% |
| 2e-5 | 88% | 70% | 77% / 77% | 24% / 23% | 83% |
| 5e-5 | 84% | 73% | 83% / 83% | 17% / 16% | 87% |
| 2e-4 | 89% | 83% | 96% / 96% | 19% / 19% | 98% |

- **Timed holds ("give me N seconds") are not learned:** after the timer, every model re-asks
  on its own only 13–26% of the time (base 33%), and when it does the timing is fine (median
  error ≈ 0–5% of N). The models treat a timed hold as "wait until the user speaks". It's the
  only speak event in the data with no human utterance right before it, and train_1h has only
  22 B1 episodes. Only 18 timed releases on dev_80. Next-experiment item: more B1 data.
- Cue-released holds (side conversation, hear-me-out) are essentially solved at 2e-4 (96%).
- A silence metric alone rewards quiet models; report both axes. Proposed selection score:
  mean(silent when due, speaks when due).
- 2e-4's dev audio loss rises late (0.078 → 0.131), which could mean degraded voice quality,
  invisible to the behavior metrics. Check by listening (demo) before phase 2.

Caveats: dev_80 is in-distribution (train voices and wording, unseen scripts); 1 sample per
episode; 58 holds, 63 resumes and 42 ordinary questions, so ±10–12 pp per checkpoint (95%).
Picking the max over 126 checkpoints on the same data inflates the winner, so the
candidates are re-scored on dev_200 before any choice. Final claims come from the test suites.

Plots (regenerate with the commands in §11):
- `results/phase1/plots/loss_curves.png`: train and dev loss (total / text / audio) vs epoch, per LR
- `results/phase1/plots/lr_schedule.png`: the LR each run followed
- `results/phase1/plots/dev_loss_vs_lr.png`: best and final dev loss vs peak LR
- `results/phase1/eval/curves_dev_80.png`: behavior vs epoch per LR, with CIs
- `results/demo/index.html`: **listening demo**. Base model + every 1 h and 10 h run at its lowest-dev-loss and final checkpoint (20 models), on 9 dev conversations (one per behavior type, chosen among the 1 h models where they disagree most), with audio, timelines, scored outcomes, generated words and a 1 h / 10 h filter (`python -m silence.demo_phase1`)

### Loss per LR, 1 h vs 10 h (all runs; 2026-10-15 03:45 UTC)

Cross-entropy; best = lowest over all checkpoints (epoch in brackets); train = mean over the final epoch.

| Data | Peak LR | Dev text: best (epoch) | Dev text: final | Dev audio: best (epoch) | Dev audio: final | Train text: final epoch | Train audio: final epoch |
|---|---|---|---|---|---|---|---|
| — | base model | 0.537 | — | 0.393 | — | — | — |
| 1 h (20 ep) | 2e-6 | 0.457 (20.1) | 0.457 | 0.104 (19.8) | 0.104 | 0.492 | 0.109 |
| 1 h (20 ep) | 5e-6 | 0.373 (18.8) | 0.373 | 0.084 (19.8) | 0.084 | 0.355 | 0.082 |
| 1 h (20 ep) | 1e-5 | 0.351 (9.9) | 0.370 | 0.075 (18.8) | 0.076 | 0.198 | 0.064 |
| 1 h (20 ep) | 2e-5 | 0.347 (5.9) | 0.642 | 0.074 (8.9) | 0.078 | 0.044 | 0.042 |
| 1 h (20 ep) | 5e-5 | 0.341 (4.0) | 0.803 | 0.074 (5.9) | 0.101 | 0.008 | 0.016 |
| 1 h (20 ep) | 2e-4 | 0.331 (3.0) | 0.741 | 0.078 (6.9) | 0.131 | 0.002 | 0.004 |
| 10 h (5 ep) | 2e-5 | 0.280 (3.0) | 0.299 | 0.061 (4.7) | 0.061 | 0.187 | 0.054 |
| 10 h (5 ep) | 5e-5 | 0.275 (2.0) | 0.392 | 0.059 (4.0) | 0.060 | 0.082 | 0.043 |
| 10 h (5 ep) | 2e-4 | 0.273 (1.0) | 0.479 | 0.061 (3.2) | 0.065 | 0.027 | 0.027 |
| 10 h (5 ep) | 5e-4 | 0.292 (1.0) | 0.473 | 0.060 (4.2) | 0.063 | 0.024 | 0.027 |

- 10 h beats 1 h on best dev loss at every LR (text 0.273–0.292 vs 0.331–0.457; audio ≈ 0.06 vs ≈ 0.075).
- Higher LR reaches its best dev loss sooner, then the wording memorization sets in; much milder at
  10 h (2e-4 final train text 0.027 vs 0.002 at 1 h).
- 5e-4 at 10 h is past the useful range on loss (best text 0.292, worse than 2e-5 to 2e-4).
- The late audio-loss drift seen at 1 h (2e-4: 0.078 → 0.131) is gone at 10 h (every LR ends within
  0.006 of its best).
- Loss is not the selection criterion (see the phase-1 finding); 10 h behavior results follow.

### Phase 2 behavior (10 h, dev_80; completed 2026-10-15 04:07 UTC)

Score = mean(silent when due, speaks when due), by epoch (quarter-epoch checkpoints, every 17 steps); base 0.29.
±0.06 noise per checkpoint.

| Peak LR | ep 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 | Plateau (ep 2–5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2e-5 | .43 | .61 | .71 | .73 | .77 | .79 | .83 | .81 | .84 | .87 | .86 | .87 | 0.84 |
| 5e-5 | .48 | .71 | .78 | .81 | .83 | .82 | .87 | .86 | .85 | .85 | .84 | .85 | 0.85 |
| 2e-4 | .59 | .77 | .82 | .83 | .79 | .81 | .82 | .79 | .81 | .85 | .81 | .82 | 0.82 |
| 5e-4 | .64 | .53 | .60 | .64 | .57 | .64 | .54 | .56 | .53 | .52 | .58 | .59 | 0.57 |

Plateau breakdown and the 1 h comparison at the same LR (1 h plateau = epochs 10–20; 10 h = epochs 2–5):

| LR | Data | Score | Silent when due | Speaks when due | Cue release on time | Timed release: spoke | Ordinary answered |
|---|---|---|---|---|---|---|---|
| 2e-5 | 1 h | 0.79 | 88% | 70% | 77% | 24% | 83% |
| 2e-5 | 10 h | 0.84 | 96% | 72% | 76% | 20% | 89% |
| 5e-5 | 1 h | 0.79 | 84% | 73% | 83% | 17% | 87% |
| 5e-5 | 10 h | 0.85 | 96% | 74% | 89% | 15% | 82% |
| 2e-4 | 1 h | 0.86 | 89% | 83% | 96% | 19% | 98% |
| 2e-4 | 10 h | 0.82 | 98% | 66% | 85% | 8% | 72% |
| 5e-4 | 10 h | 0.57 | 100% | 14% | 22% | 0% | 12% |

Best single checkpoints on dev_80: 10 h 2e-5 ep 4.2 (0.88: 97% silent / 79% speaks, cue 87%, ordinary 95%),
1 h 2e-4 ep 15.9 (0.89), 10 h 5e-5 ep 3.7 (0.87). All within noise of each other.

Findings:
1. **10 h makes the models better at holding silence, not at speaking again.** Silent-when-due rises to
   96–98% (vs 84–89% at 1 h), but speaks-when-due stays at 66–74%. The 10 h models are more conservative.
2. **The best LR shifts down with more data.** 1 h is best at 2e-4; 10 h is best at 2e-5–5e-5, where
   2e-4 is slightly worse and more conservative (ordinary answered 72%). 5e-4 at 10 h collapses into a
   **silent model** (100% silent, 14% speaks), the same failure mode as too low an LR at 1 h, from the other end.
3. **When behavior plateaus is set by optimizer steps, not epochs.** At both data sizes the score reaches
   ≈ 0.8 after ~70 steps (≈ 1,100 episode presentations): about 1 epoch at 10 h, ~10 epochs at 1 h. After
   that, 2e-5 at 10 h keeps creeping up slowly (0.87 at epochs 4–5).
4. **Timed holds remain unlearned at 10 h** (re-asked on its own 8–20%). More of the same data doesn't
   fix it; it needs targeted data (see next experiment).
5. On in-distribution dev, the best 1 h and 10 h checkpoints tie. 10 h's expected advantage is
   generalization (unseen wording, voices, conditions), which only the test suites can show.

### Is the training signal biased toward silence? (analysis, 2026-10-15)

Measured over the training sets (text targets at 12.5 Hz; Moshi words tokenized with the model's tokenizer):

| | train_1h | train_10h |
|---|---|---|
| Text targets = PAD ("stay silent") | 89.7% of frames | 89.8% |
| Text targets = EPAD ("start talking now") | 0.64% (1 per 140 PAD frames) | 0.62% (1 per 144) |
| Share of text-loss weight on EPAD (EPAD shares text_padding_weight 0.5 with PAD) | 0.58% | 0.57% |
| Audio loss covers | only Moshi's silent frames (72%); speech audio masked | same |
| Time inside a requested hold / Moshi speaking | 38% / 28% | 37% / 28% |
| Speak-onset events per hour (response / release / ack / topic) | 115 / 70 / 53 / 49 | 111 / 70 / 51 / 48 |

So the objective strongly favors silence: the decision that matters (when to start talking) carries
~0.6% of the text-loss weight, and every audio gradient says "emit silence codes" (the speech mask was
deliberate, to avoid imitating the TTS voice). This matches the observations: every LR first passes
through a silent phase; 10 h ends up more cautious than 1 h; 5e-4 at 10 h collapses into silence; timed
holds (the only release with no external cue) stay unlearned.

**EPAD-weight sweep** (launched 2026-10-15 04:19 UTC; 4th patch `moshi-finetune-loss-weights.patch` adds
`text_epad_weight` and `audio_loss_weight`, training only; the dev loss keeps the original weighting):
1 h, peak LR 5e-5, 20 epochs, checkpoint every epoch, everything else as phase 1. Baseline R3 (×1).

| Run | EPAD × PAD weight | EPAD weight | EPAD share of text-loss weight |
|---|---|---|---|
| R3 (baseline) | ×1 | 0.5 | 0.6% |
| R16_1h_lr5e-05_epad3 | ×3 | 1.5 | ~1.7% |
| R17_1h_lr5e-05_epad10 | ×10 | 5 | ~5.5% |
| R18_1h_lr5e-05_epad30 | ×30 | 15 | ~15% |
| R19_1h_lr5e-05_epad100 | ×100 | 50 | ~37% |

Prediction: speaks-when-due and timed-hold re-asks rise with the EPAD weight; at some multiple
silent-when-due starts to drop (Moshi talks during holds). The grid deliberately extends past the
expected optimum.

**Results** (completed 2026-10-15 05:14 UTC; 4× pod `7f5yuy4acmhxzp`, ≈ $15). All four runs crashed at
step 132–133 of 142 when the 150 GB volume filled up (the volume was then grown to 250 GB), so each has
18 checkpoints, epochs 1–17.8. Plateau = epochs 10–17.8 for every run, including the baseline.
Figure: `results/epad/analysis/epad_sweep.png`.

| EPAD × | Plateau score | Silent when due | Speaks when due | Cue release on time | Timed hold re-asked | Ordinary answered | Best checkpoint (epoch: score, silent/speaks, timed re-ask) |
|---|---|---|---|---|---|---|---|
| ×1 (R3) | 0.80 | 86% | 73% | 84% | 15% | 86% | ep 15: 0.83 (86% / 80%, 28%) |
| ×3 | 0.81 | 80% | 82% | 93% | 24% | 97% | ep 14: 0.85 (88% / 83%, 17%) |
| ×10 | 0.80 | 78% | 83% | 94% | 34% | 95% | **ep 6: 0.86 (81% / 91%, 61%)** |
| ×30 | 0.77 | 71% | 83% | 95% | 28% | 98% | ep 18: 0.81 (81% / 81%, 17%) |
| ×100 | 0.80 | 75% | 86% | 94% | 40% | 99% | ep 10: 0.85 (79% / 90%, 56%) |

Findings:
1. **The EPAD weight is a dial along a speak/silence trade-off, not a free gain at the plateau.**
   Speaking goes up (73% → 82–86%; cue releases 84% → 93–95%; ordinary 86% → 95–99%), silence goes down
   (86% → 71–80%), and the balanced score stays ≈ 0.80. So the loss weighting explains *which side* of
   the trade-off the model lands on, but not the whole gap.
2. **It changes the learning trajectory most.** With EPAD up-weighted, the model learns to speak first
   and to hold silence later (the reverse of the baseline). Mid-training (epochs 4–9), the ×10/×30/×100
   runs re-ask after timed holds 60–80% of the time, the first time this behavior was clearly learned.
   But it **decays back to 20–40% by the plateau** as training continues: with more epochs the models
   drift back toward waiting for the user.
3. **The best balanced checkpoint is mid-training:** ×10 at epoch 6 (81% silent, 91% speaks, 61%
   timed re-ask, score 0.86), on par with the best checkpoints overall but with far better timed-hold behavior.
4. Dev text loss is again misleading: ×100 has the lowest late dev text loss of the sweep but no better behavior.

Implications: combine a moderate EPAD weight (×3–×10) with a **shorter schedule** that ends near the
mid-training peak (~epoch 6 at 1 h ≈ 40 steps), or with the **10 h data**, whose models are over-silent
(96–98% silent, ~70% speaks), so shifting them toward speaking should raise the balanced score. Timed
holds still need more targeted data to be stable.

### EPAD weight at 10 h (R20 ×3, R21 ×10; 2026-10-15)

train_10h, lr 5e-5, 5 epochs (345 steps), checkpoints every 17 steps; identical to R9 except `text_epad_weight`
(R20 1.5 = ×3, R21 5 = ×10). Dev_80, 1 sample. Plateau = epochs 2–5 (14 checkpoints). Full tables, figure and
per-checkpoint numbers: `results/epad10h/summary.md`, `results/epad10h/analysis/epad10h.png`.

| Run | Plateau score | Silent | Speaks | Cue release on time | Timed re-ask | Ordinary answered | Best checkpoint (epoch): score |
|---|---|---|---|---|---|---|---|
| R9 ×1 | 0.85 | 96% | 74% | 89% | 15% | 82% | `checkpoint_000255` (3.7): 0.87 |
| R20 ×3 | 0.87 | 90% | 85% | 93% | 40% | 95% | `checkpoint_000153` (2.2): 0.91 |
| R21 ×10 | 0.88 | 81% | 94% | 96% | 73% | 100% | `checkpoint_000153` (2.2): 0.91 |
| R8 (2e-5) ×1 | 0.84 | 96% | 72% | 76% | 20% | 89% | `checkpoint_000289` (4.2): 0.88 |

- The same silence ↔ speaking trade as at 1 h, but at 10 h the baseline was over-silent, so the plateau score rises.
- Timed re-asks persist at 10 h (×10: 61–83% at every checkpoint past epoch 2, timing error ≈ 0); at 1 h they faded.
- The lost silence is almost all in timed holds (violated 10% / 37% / 57% for ×1 / ×3 / ×10; reactions to the user's
  mutters during them 21% / 41% / 53%); side-conversation and hear-me-out holds stay 97–99% silent.
- Text dev loss again ranks backwards (R21 worst, 0.323 vs R9 0.275).
- 0.91 is the best dev_80 score so far but within per-checkpoint noise (±0.06): confirm on dev_200.

### 50 h (R22) and the held-out test suites (2026-10-15)

R22 = 50 h v2 set, EPAD ×10, lr 5e-5, 345 steps (1 epoch; same compute as R21). Dev text loss fell to 0.305 without turning up;
dev_80 behavior plateaued by step ~150 at ≈ 0.83 (R21 0.88). Test suites (1 sample; R17 / R21 at their best dev_80 checkpoint,
R22 final):

| Suite | base | 1 h ×10 | 10 h ×10 | 50 h ×10 |
|---|---|---|---|---|
| test_script | 0.32 | 0.93 | **0.97** | 0.93 |
| test_word | 0.35 | 0.82 | **0.87** | 0.84 |
| test_voice | 0.37 | 0.86 | **0.92** | 0.89 |
| test_distract | 0.29 | 0.93 | **0.98** | 0.92 |
| test_cond | 0.32 | 0.67 | **0.72** | 0.70 |
| test_ordinary | 0.59 | 0.96 | 0.98 | **1.00** |
| test_all | 0.32 | 0.79 | **0.84** | 0.80 |

Generalization: unseen scripts ≈ dev; new wording / voices cost 5–10 points, almost all in timed holds; distractors incl. the
friend's "I'm back" are ignored at 10 h; new conditions only partly transfer (keyword / phone holds kept but releases often
missed; "a minute" holds fail). 50 h did not beat 10 h; confounded by the v2 recipe (more, mutter-heavy timed holds on top of
EPAD ×10), checkpoint selection (R21 best-of-21, R22 final) and exposure (5 vs 1 passes). Details: `results/run50h/summary.md`,
`results/test/summary.md`.

### Data fix v3 → R24, and the human-recorded sets (2026-10-15)

R22's losses traced to a shortcut: "speak after a few seconds of user silence" (B1-heavy v2 mix + EPAD ×10, and no long
silences that mean "keep waiting"). v3 keeps the v2 plan (same lines, voices, clips) and only lengthens silences inside
cue-released holds; then B1 is capped at 15%. R24 (v3, ~45 h, EPAD ×10, stopped at step 527 of 614) recovered the lost
holds (dev B3 rescind 15% → 94%, B1 std 36% → 64%) and is the best model on dev (0.94 at its final checkpoint), on the
generalization suites (test_script 0.97, test_word 0.90, test_voice 0.94, test_all 0.86) and on the user's own recorded
voice (0.83, vs TTS control 0.84; R21 0.79, R22 0.74, R17 0.67, base 0.33; own wording: R24 0.73). Regressions: the friend's
"I'm back" (44% reacted vs 0% for R21), keyword releases. Details: `results/run_v3/summary.md`, `results/human/summary.md`.

| Human set (3 samples) | base | 1 h ×10 | 10 h ×10 | 50 h v2 ×10 | 45 h v3 ×10 |
|---|---|---|---|---|---|
| TTS voice, same scripts (control) | 0.41 | 0.79 | 0.83 | 0.79 | **0.84** |
| your voice, scripted | 0.33 | 0.67 | 0.79 | 0.74 | **0.83** |
| your voice, own wording | 0.28 | **0.73** | 0.68 | 0.67 | **0.73** |

## 11. Artifacts and how to regenerate

```
results/phase1/
  configs/   the exact yaml per run + <run>.epochs.json
  runs/      <run>/args.yaml, metrics.train.jsonl (every step), metrics.eval.jsonl (every checkpoint)
  logs/      per-run stdout (train.py) + parallel.log (scheduler)
  setup/     bootstrap.log, smoke.log, eval_selftest.log, eval_bench.log, pip-freeze.txt
  plots/     training plots + training_summary.{csv,md}
  eval/      generation-eval summary.{csv,md} + behavior curves (after the evals)
```

```
# pull from the box (no checkpoints; they stay on the volume)
rsync -rlpt --include '*/' --include 'metrics.*.jsonl' --include 'args.yaml' --exclude '*' \
    box:/workspace/runs/ results/phase1/runs/ && find results/phase1/runs -type d -empty -delete
rsync -rlpt box:/workspace/logs/ results/phase1/logs/
python -m silence.plot_training results/phase1/runs --configs results/phase1/configs --out results/phase1/plots
```

LoRA checkpoints (105 × 0.39 GB) stay on the network volume:
`/workspace/runs/<run>/checkpoints/checkpoint_XXXXXX/consolidated/{lora.safetensors, config.json}`.

## 12. 50 h train set (built 2026-10-15, 06:05–07:14 UTC)

**Goal.** Test data scaling beyond 10 h, with more voice diversity than the 16 Pocket training
voices. The user asked for it in under ~1 h of wall clock. The set lives on the network volume
only (`/workspace/data50/train50`; clip cache `/workspace/cache50`).

**Final numbers (build finished 2026-10-15 07:14 UTC; metadata in `results/build50/`):**

| Item | Value |
|---|---|
| Episodes planned | 5,600 |
| Episodes kept (after drops + check.py filter) | **5,585** |
| Hours (total / speech / Moshi speech / hold silence) | **51.24** / 35.85 / 13.20 / 17.57 h; mean episode 33.0 s |
| Subsets (first-K prefixes, nested) | `train_1h` 110 eps / 1.01 h, `train_10h` 1,084 / 10.00 h, `train_25h` 2,724 / 25.00 h, `train_50h` 5,457 / 50.00 h; plus `train_all.jsonl` (5,585) |
| Type mix (kept) | ORD 1,704, B2_std 1,083, B1_std 975, B3_std 617, DECOY 551, B1_double 214, B3_early 121, B3_rescind 113, B1_rescind 104, B1_early 103 (B1 25.0%, B2 19.4%, B3 15.3%, ORD+DECOY 40.4%) |
| Dropped at assembly | 1 failed hosted clip (Fish) + 2 over 60 s (B2_std 1, B3_std 1; the over-length redraw needed a new hosted clip, which the assembly box can't render) → 5,597 assembled |
| check.py failures (`rejected.json`), 12 = 0.2% | Moshi word times vs audio onset off by ~−1.0 s: 7 (6 B1_double, 1 B1_std; plus 1 DECOY at −0.26 s); user/Moshi overlap: 2 (B1_std 0.06 s, B1_rescind 0.29 s); human turn < 6 dB over the floor: 2 (B2_std). The B1_double −1.0 s pattern is systematic (6 of 214 = 2.8%) and worth a look before the next build |
| Hosted-engine clips failed | 1 of 12,076 Fish lines (`tts_failed.jsonl`); 0 of 3,584 ElevenLabs |
| Unique packs / TTS clips used | 5,799 packs / 40,921 clips (45,350 clip uses) |
| Wall clock | packs 06:05–06:35, rendering 06:27–07:03, assembly + check + filter 07:03–07:14 (≈ 70 min end to end; the "< 1 h" target was missed by ~10 min) |
| Disk | 17 GB (`/workspace/data50/train50`) + clip cache `/workspace/cache50` |

**Cost of the build:**

| Item | Cost |
|---|---|
| Claude Sonnet 5.5, pack generation (~822 calls) | ≈ $20 (estimate: second run 0.56M in / 0.52M out tokens ≈ $6.3; the crashed first run ≈ 2.4× that) |
| Fish Audio s2-pro (≈ 622k characters) | **$9.39** (credit $25.83 → $16.44) |
| ElevenLabs Flash (≈ 209k characters) | 42,161 of 116,845 plan credits (≈ 0.2 credit per character for Flash); no marginal cost on the Starter plan |
| Pods | none rented for the build: it used the CPUs of the two EPAD-10h H100 pods (already running, ≈ 8% training slowdown) plus the Mac; tr3/tr10 were terminated at 07:14 UTC, ≈ 2 min after their evals |
| **Total marginal** | **≈ $30** |

QC (sampled ASR, faster-whisper base.en, 366 clips, seed 0; full table
`~/.claude/jobs/12a5eb83/tmp/asrqc/results.csv`, temporary): WER mean / median / p90 — ElevenLabs
0.021 / 0 / 0.098, Fish 0.013 / 0 / 0.005, Pocket control 0.036 / 0 / 0.089; no voice above 0.15 mean
WER; no truncation, repeats or long silences. Most residual errors are ASR artifacts (dropped "uh"/"um",
"Moshi" → "Moshe", digits). Speaking rate (speech span per word, median): ElevenLabs 0.249 s,
Pocket 0.286 s, Fish 0.322 s (Fish ~13% slower than the Pocket-fitted timing estimates).

### Content

- `packs/packs_v2.jsonl` = the original `packs.jsonl` (2,500) + **5,470 new packs** written by
  Claude Sonnet 5.5 (`gen_packs --seed 1`, then `--seed 2`, both `--test-frac 0` so every new
  pack is train; `--per-family-json` targets B1 2120 / B2 1580 / B3 1240 / ORD 2250 / DECOY 780).
  ≈ 822 API calls, ≈ $20 (estimate). Log `packs/gen_packs_v2.log`, rejects `packs/rejects_v2.jsonl`.
  The seed-1 run crashed at call 578 on a `StopIteration` raised inside pack validation; it is now
  caught (counted as "schema: malformed") and the run resumed with seed 2.
- **Reverse screen** (`silence/screen_train_packs.py`), the mirror of `screen_packs`: a NEW
  train pack is dropped if it contains a line actually spoken in `data/dev` or any of the 7 test
  suites (normalized), or if its key text has content-word Jaccard ≥ 0.5 with any test pack or
  any pack used in dev (same family). Original packs pass through.

  | Family | Dropped: shared spoken line | Dropped: similar key text |
  |---|---|---|
  | B1 | 15 | 75 |
  | B2 | 172 | 4 |
  | B3 | 13 | 18 |
  | ORD | 2 | 7 |
  | **Total** | **306** | |

  Kept 5,164 new → `packs/packs_v2_screened.jsonl` (7,664 packs), the build's input. Verified: 0
  planned lines contain a held-out (test-list) phrase-bank wording. Caveat: the exact-line rule
  is strict on short generic lines ("Okay."), which explains most of the B2 drops.
- Dev packs are excluded with the new `build.py --exclude data/dev`.

### Mix

`--mix v2` = `specs.TRAIN_MIX_V2`, by episode count:

| Family | v1 (1 h / 10 h) | v2 (50 h) |
|---|---|---|
| B1 timed hold | 15 (std 9, double 2, early 2, rescind 2) | **25** (std 17, double 4, early 2, rescind 2) |
| B2 side conversation | 25 | 20 |
| B3 hear me out | 20 (std 16, early 2, rescind 2) | 15 (std 11, early 2, rescind 2) |
| ORD / DECOY | 30 / 10 | 30 / 10 |

Plus `specs.MUTTER_P` 0.7 (was 0.5): the share of B1 std holds (N ≥ 5) with thinking fillers (D2c)
inside the hold. Reason: timed holds were the one behavior no 1 h / 10 h run learned stably (§10).

### Voices

`--tts multi` (`silence/voices.py` `MULTI`): the user and the friend each get an engine drawn with
`ENGINE_SHARES` pocket 0.40 / ElevenLabs 0.15 / Fish 0.45, then a voice from that engine's pool.
Moshi is always Pocket `michael` (same as before, so Moshi's text/audio targets are unchanged in kind).

| Engine | Model | Train voices | Held-out voices |
|---|---|---|---|
| Pocket TTS | english_2026-09 | 16 | 4 (eve, george, jane, peter_yearsley) |
| ElevenLabs | Flash v2.5 | 17 premade (the account's one cloned voice excluded) | 4: Lily, Matilda, Daniel, Bill |
| Fish Audio | s2-pro | 30 public-library voices | 4 |

Fish library voices were vetted by title, tags and description as generic voices; the library has
many voices named after or cloned from real people or fictional characters, and all of those were
excluded. 63 training voices in all (vs 16 for 1 h / 10 h). Caveat: the held-out ElevenLabs / Fish
voices don't appear in any existing test suite (the suites were built with Pocket only).

### Code changes (all in `silence/`)

- `tts.py`: `ElevenLabsTTS` and `FishTTS` on a shared `_HTTPTTS` base (raw 24 kHz PCM).
  - Concurrency 3 (EL Starter limit) / 5 (Fish limit below $100 prepaid). Limits are per account,
    so each hosted engine is rendered from one process only.
  - 429 / 5xx: retry with backoff, honoring Retry-After capped at 60 s. 401 / 402 / 403 (key,
    quota, balance) or 25 consecutive failures raise `FatalEngineError`, which stops the engine.
  - Audio checks: reject responses starting with RIFF / ID3 / `{` (not raw PCM), silent or
    < 0.1 s audio, and a speaking rate outside 0.12–1.0 s/word over `speech_span` (≥ 3 words;
    Pocket's median is 0.35, p99 0.70). Up to 3 resamples, then the line fails.
  - 0.15 s of silence padding around each clip so `assemble.gate`'s margins and fades fit.
  - Hosted engines return no word times: word starts are spread over the speech span in
    proportion to word length. Only the onset / offset of human turns is used downstream, and
    only Moshi's words (always Pocket) become text tokens.
  - `MultiCache` routes `el:` / `fish:` voice ids to `<cache>/<engine>/`; keys are
    engine|voice-id|text, so Pocket clips keep their old keys (reused from the 10 h build).
  - Atomic cache writes: uuid-named temp file + `os.replace`, audio before json, so "json exists"
    implies a complete clip even with several writers on one volume. Hosted clips are stored as
    16-bit FLAC (lossless for 16-bit PCM, half the bytes to upload); `SILENCE_CACHE_FLAC=1` does
    the same for Pocket clips rendered off the volume.
  - Hosted clips are written one by one as they arrive; failures are recorded per clip
    (`tts_failed.jsonl`) instead of discarding a whole batch.
- `build.py`: `--tts multi`, `--mix v2`, `--exclude <sets>`, `--render-only <engines>`,
  `--shard i/n` (clip key hash mod n), `--reverse` (take over another shard from the far end),
  `--max-drop` (default 3%). Every run prints a **plan hash** (hash of the clip set) that all
  shards must share. The final (assembly) run renders nothing: episodes with a failed hosted clip
  are dropped (bounded by `--max-drop`), and a missing Pocket clip aborts. In multi mode an episode
  over 60 s only gets the B2 side-conversation shrink (which reuses rendered lines); otherwise it is
  dropped and its files deleted (no new packs at assembly, since they'd need new clips).
- `check.py --workers N` (parallel checks). New `silence/filter_checked.py`: moves failing episodes
  to `rejected/`, rewrites `manifest.jsonl` / `episodes.jsonl` in order (originals kept as
  `*.all.jsonl`, so first-K prefixes stay nested), writes `rejected.json` and `train_all.jsonl`
  (absolute paths).

### Plan and execution

- `build.py train --tts multi --mix v2 --n 5600`, seed 0, estimated ~50 h. **41,070 unique
  clips**: Pocket ~24.1k (≈ 23.2 h of speech), Fish ~12.7k (≈ 10.3 h), ElevenLabs ~4.2k (≈ 3.4 h).
- The Mac (macOS libm) and Linux plans differ by 1 ulp in 25 timing floats; the clip sets are
  identical (verified), so rendering can be split across machines, but only one machine assembles.
- Pocket: shard 0/3 on pod tr3 (`086j0wex48cmlp`), 1/3 on tr10 (`xk9kzxl2eev9of`), both on the
  pods' CPUs next to the R20 / R21 training; 2/3 on the Mac, into the old Pocket cache via a
  symlink (reusing the 10 h build's `michael` clips). The pods then took over the Mac's shard from
  the far end (`--reverse`).
- ElevenLabs and Fish ran from the Mac only, so the API keys never left it. A sync script uploaded
  Mac clips as FLAC to `/workspace/cache50` with `rsync --ignore-existing` (a pod's own render is
  never overwritten).
- Assembly, `check.py` and `filter_checked` on tr3. tr10 was terminated right after its EPAD eval
  and copy-back; tr3 stays until the dataset is built (~07:20), then is terminated (user's choice).
- Incidents: shard 0 crashed once at 06:41 (BrokenPipe after the ssh session that launched it was
  killed) and was relaunched detached (`ssh -f -n` + `setsid`). A 4× RTX PRO 6000 pod as a CPU
  burst was attempted but blocked by the permission classifier (CPU-only pods aren't offered in
  US-NE-1). The chat pod `vc42i3dtvzv7s5` was terminated ~06:30 UTC at the user's request.

### Measured throughput and cost

| Item | Measured |
|---|---|
| Fish s2-pro | ~20× realtime at 5 concurrent requests (~0.8 s latency) |
| ElevenLabs Flash v2.5 | ~20× realtime at 3 concurrent; ≈ 0.5 credit per character |
| ElevenLabs budget | Starter plan, 116,845-credit limit; plan ~209k chars ≈ 105k credits |
| Fish budget | $15 per 1M chars; plan ~622k chars ≈ $9 (account credit $25.83) |
| Pocket on an H100 pod's CPUs | 15–21× realtime per pod (22-vCPU quota, nice 19, 18 workers) |
| Pocket on the Mac (4 performance cores) | ~11× realtime |
| Side effect on training | Pocket on the training pods slowed R20 / R21 ~8% (10.2–10.4k vs ~11.2k words/s) |
| Mac → pod upload | ~1.2 MB/s for one big file; the clip sync ran ~25 clips/s |

### Code reviews before the paid run

Two adversarial reviews of the new code (a subagent, then the Fable model). Main findings and fixes:

| Finding | Fix |
|---|---|
| Concurrency was per process, but the limits are per account | Hosted engines render from one process only; concurrency = account limit (3 / 5) |
| One failed line discarded a paid batch; an engine thread could die silently | Clips written as they arrive, failures recorded per clip; thread errors collected and the process exits non-zero |
| No validation of returned audio | Format / silence / speaking-rate checks with resampling (above) |
| Durations fitted to Pocket → the over-length redraw needed new hosted clips (crash), and mutters could overlap the re-ask | Redraw errors caught; over-length episodes get only the B2 shrink or are dropped; `check.py` + `filter_checked` remove episodes that fail (e.g. overlap) |
| New packs could leak dev / test scripts into train | Reverse screen (`screen_train_packs`) + `--exclude data/dev` |
| Non-atomic cache writes | uuid temp + `os.replace`, audio before json |
| No circuit breaker on auth / quota errors | `FatalEngineError` (401/402/403, or 25 consecutive failures) |
| Orphan files left when an episode is dropped | Its wav / json / spec deleted on drop |
| The final run would re-render missing clips | Assembly run renders nothing; missing Pocket clips abort, failed hosted clips drop episodes (≤ `--max-drop`) |
| Concurrent writers of the same key (shards, sync) | Atomic replace with unique temp names; `rsync --ignore-existing` |

### Caveats for the report

- **50 h differs from 1 h / 10 h in recipe, not just size:** v2 mix (more B1), `MUTTER_P` 0.7,
  3 engines / 63 training voices (vs 16 Pocket), newly generated packs, the dev exclusion. A
  1 h → 10 h → 50 h curve therefore confounds data size with recipe.
- Its first-K prefixes (e.g. a "10 h" prefix of train50) are **not** the old `train_1h` /
  `train_10h`. Ideally also train train50's own 10 h prefix, to separate size from recipe.
- Episode timing (durations, mutter placement) is planned from Pocket-fitted duration estimates;
  hosted clips can be longer or shorter, which is why some episodes are dropped or fail check.py.
- Human word times in hosted-engine clips are synthetic (only turn onset / offset are real). This
  doesn't touch the training targets (only Moshi's words become text tokens), but any analysis of
  human word timing in train50 should not trust them.
- The held-out ElevenLabs / Fish voices are not covered by any test suite yet.
