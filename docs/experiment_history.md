# Experiment history (sketch)

What we ran, in order, and why. Scores are on dev_80: mean of "silent when due" and "speaks when due",
±0.06 per checkpoint. Base model = 0.29. Minute-by-minute version: `docs/experiment_history_detailed.md`.

## 0. Data and harness (10-14)

- **Synthetic data**: LLM-written scripts ("packs") + Pocket TTS. 10 h train (1,150 episodes, nested 1 h ⊂ 10 h),
  dev_80 ⊂ dev_200, 7 held-out test suites × 90.
- Changes made along the way:
  - **Distractors**: a hand-written list first; moved into the LLM scripts because the listed lines made no
    sense in context.
  - **The user always speaks first.**
  - **Audio loss masked on Moshi's speech**, so Moshi doesn't learn to sound like the TTS voice.
  - **Three moshi-finetune bugs patched**: NaN dev loss, padding counted in the text loss, checkpoints deleted.
- **Eval**: generate Moshi's side against the recorded user, score with VAD. The script is forced outside the
  scored windows, and the user audio is shifted one frame so the model sees exactly what training fed it.

## 1. Phase 1: LR sweep at 1 h (R1–R7)

**Why:** does 1 h teach the behavior at all, and at which LR?

- 1 h, 20 epochs, LRs 2e-6 … 2e-4, checkpoint every epoch.
- 5e-4 was dropped and 5e-6 / 2e-5 added mid-run, after 5e-5 and 2e-4 memorized the scripts (dev loss blew up).

**Found:**
- It learns from 1 h: 2e-4 plateaus at 0.86 (holds kept 89%, cue releases 96% on time, questions answered 98%).
- Too low an LR gives a quiet model: it stays silent and stops answering.
- Dev loss ranks the LRs backwards, so checkpoints are chosen on behavior.
- **Timed holds ("give me 10 s") are unlearned**: the models wait for the user instead of re-asking.

## 2. Phase 2: LR sweep at 10 h (R8–R11)

**Why:** does 10× more data help, and does the best LR move?

- 10 h, 5 epochs, LRs 2e-5 … 5e-4, checkpoint every ¼ epoch.

**Found:**
- 10 h holds silence better (96–98%) but doesn't speak again better: plateau 0.84–0.85, a tie with 1 h.
- The best LR drops to 2e-5–5e-5; 5e-4 collapses into silence.
- Behavior plateaus after ~70 optimizer steps at both sizes.
- Timed holds are still unlearned.

## 3. EPAD-weight sweep at 1 h (R16–R19)

**Why:** the loss is silence-biased by construction. The "start talking now" token (EPAD) is ~0.6% of the
text loss, and the audio loss only sees silence. So we up-weighted EPAD ×3 / ×10 / ×30 / ×100 at 1 h,
lr 5e-5 (new loss-weights patch).

**Found:**
- At the plateau it just trades silence for speaking: the score stays ~0.80.
- Mid-training, timed-hold re-asks reach 60–80% for the first time. The best is ×10 at epoch 6 (0.86).
- (The volume filled up and cut these runs at epoch 17.8; it was grown to 250 GB.)

## 4. EPAD at 10 h (R20 ×3, R21 ×10)

**Why:** the 10 h models were over-silent, so pushing toward speech should be a net gain there.

**Found:**
- Yes: plateau 0.87 / **0.88** vs 0.85; best checkpoints 0.91.
- With ×10 the timed re-ask sticks (~73%), at the cost of some silence inside timed holds.
- **R21 became the best model.**

## 5. 50 h, new recipe (R22)

**Why:** test scaling past 10 h with more voice diversity.

- New 50 h set:
  - 5,164 new packs, screened against dev/test;
  - user/friend voices from Pocket + ElevenLabs + Fish (63 voices);
  - more timed holds (25%) with more fillers in them;
  - built in ~70 min for ~$30.
- R22: EPAD ×10, same compute as R21 (345 steps = 1 epoch).

**Found:** 50 h did **not** beat 10 h (dev ~0.83, 2–6 points lower on the test suites). Likely cause: the recipe
taught the shortcut "after a few seconds of silence, start talking". Recipe and size are confounded.

## 6. Held-out test suites (base, 1 h ×10, 10 h ×10, 50 h ×10)

**Why:** does it generalize beyond the training distribution?

| Suite | base | 1 h | **10 h** | 50 h |
|---|---|---|---|---|
| new scripts | 0.32 | 0.93 | **0.97** | 0.93 |
| new wording | 0.35 | 0.82 | **0.87** | 0.84 |
| new voices | 0.37 | 0.86 | **0.92** | 0.89 |
| dense distractors | 0.29 | 0.93 | **0.98** | 0.92 |
| new conditions | 0.32 | 0.67 | **0.72** | 0.70 |
| ordinary only | 0.59 | 0.96 | 0.98 | **1.00** |
| everything new | 0.32 | 0.79 | **0.84** | 0.80 |

**Found:**
- New scripts score like dev; new wording and voices cost 5–10 points, mostly in timed holds.
- Distractors are ignored, including the friend saying "I'm back" (10 h: 18/18).
- New conditions only partly transfer: "a minute" holds fail, keyword/phone releases are often missed.

## 7. Next (in progress)

- **v3 data**: timed holds back to 15%, plus long silences inside cue-released holds, to break the "talk after a
  pause" shortcut. Timing changes only, no new audio.
- **R24**: v3, EPAD ×10, 2 epochs.
- **Still open:**
  - R23 (separates recipe from size)
  - dev_200 confirmation of the best checkpoints
  - human-recorded test set
  - report

Spend so far ≈ $135 (GPU ≈ $95, APIs ≈ $40).
