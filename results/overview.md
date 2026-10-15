# Overview of all results so far (2026-10-15)

Everything is on dev_80 (80 unseen dev conversations: training voices and wording, new scripts), 1 sample per episode,
±0.06 on a single checkpoint's score. Score = mean(silent when due, speaks when due). Plateau = the flat part of each
run (1 h: epochs ≈ 10–20; 10 h: epochs 2–5). 14 trained runs, 282 checkpoints, all scored the same way.

## Experiments run

| Session | What | Cost |
|---|---|---|
| Phase 1 | 1 h, 6 LRs (2e-6 … 2e-4), 20 epochs, checkpoint every epoch | ≈ $25 |
| Phase 2 | 10 h, 4 LRs (2e-5 … 5e-4), 5 epochs, checkpoint every ¼ epoch | ≈ $25 |
| EPAD sweep | 1 h, lr 5e-5, EPAD weight ×3 / ×10 / ×30 / ×100 (loss-weights patch) | ≈ $15 |
| Setup, live chat | bootstrap pod, chat pod | ≈ $3.75 |

## Every run (plateau averages)

| Run | Score | Silent when due | Speaks when due | "I'm back" on time | "N seconds": re-asked | Ordinary answered | Best checkpoint: epoch (ckpt): score, silent/speaks, timed re-ask |
|---|---|---|---|---|---|---|---|
| base model | 0.29 | 14% | 45% | 47% | 33% | 43% | — |
| 1 h · 2e-6 | 0.40 | 60% | 21% | 26% | 26% | 10% | ep 11.9 (000084): 0.45, 67%/22%, 22% |
| 1 h · 5e-6 | 0.55 | 77% | 33% | 29% | 20% | 39% | ep 18.8 (000133): 0.59, 83%/34%, 17% |
| 1 h · 1e-5 | 0.71 | 85% | 58% | 54% | 13% | 80% | ep 18.8 (000133): 0.75, 86%/65%, 22% |
| 1 h · 2e-5 | 0.79 | 88% | 70% | 77% | 24% | 83% | ep 16.9 (000119): 0.83, 91%/74%, 33% |
| 1 h · 5e-5 | 0.79 | 84% | 73% | 83% | 17% | 87% | ep 14.9 (000105): 0.83, 86%/80%, 28% |
| 1 h · 2e-4 | 0.86 | 89% | 83% | 96% | 19% | 98% | ep 10.9 (000077): 0.90, 100%/79%, 28% |
| 10 h · 2e-5 | 0.84 | 96% | 72% | 76% | 20% | 89% | ep 4.2 (000289): 0.88, 97%/79%, 22% |
| 10 h · 5e-5 | 0.85 | 96% | 74% | 89% | 15% | 82% | ep 3.7 (000255): 0.87, 98%/75%, 6% |
| 10 h · 2e-4 | 0.82 | 98% | 66% | 85% | 8% | 72% | ep 4.9 (000340): 0.86, 100%/72%, 6% |
| 10 h · 5e-4 | 0.57 | 100% | 14% | 22% | 0% | 12% | ep 1.2 (000085): 0.69, 100%/38%, 0% |
| 1 h · 5e-5 · EPAD×3 | 0.81 | 80% | 82% | 93% | 24% | 97% | ep 13.9 (000098): 0.85, 88%/83%, 17% |
| 1 h · 5e-5 · EPAD×10 | 0.80 | 78% | 83% | 94% | 34% | 95% | ep 5.9 (000042): 0.86, 81%/91%, 61% |
| 1 h · 5e-5 · EPAD×30 | 0.77 | 71% | 83% | 95% | 28% | 98% | ep 17.8 (000126): 0.81, 81%/81%, 17% |
| 1 h · 5e-5 · EPAD×100 | 0.80 | 75% | 86% | 94% | 40% | 99% | ep 9.9 (000070): 0.85, 79%/90%, 56% |

## What we learned

1. **Fine-tuning works from 1 h.** Base 0.29 → best checkpoints ≈ 0.87–0.90: holds kept 86% → ≈ 90–100%, cue releases
   47% → ≈ 90–100% on time, ordinary questions 43% → ≈ 90–100% answered, distractor reactions ≈ 70–80% → ≈ 1–15%.
2. **LR is the main lever.** Too low → a quiet model (stays silent, stops answering). At 1 h, behavior improves up to 2e-4;
   at 10 h the best LR shifts down to 2e-5–5e-5, and 5e-4 collapses into a quiet model.
3. **Behavior plateaus after ≈ 70 optimizer steps** (≈ 1,100 conversations seen) at both data sizes: ≈ 10 epochs at 1 h, ≈ 1 at 10 h.
4. **10 h vs 1 h:** better at holding silence (96–98%) and at ignoring return-cue lookalikes, not better at speaking again.
   The best 1 h and 10 h checkpoints tie on dev; 10 h's expected advantage (generalization) needs the test suites.
5. **Dev loss is a poor selector:** it ranks LRs almost backwards and rises (memorization of wording) while behavior holds.
6. **The training signal is silence-biased by construction** (EPAD = 0.6% of the text-loss weight; audio loss only on silence).
   Up-weighting EPAD shifts models toward speaking (dial, score flat at the plateau) and teaches the timed re-ask mid-training
   (60–80% at epochs 4–9), which then fades. Best balanced: EPAD ×10 at epoch 6 (0.86, 61% timed re-ask).

## Failure modes seen in the transcripts (`results/transcripts.md`)

- **Timed holds**: the quiet models never re-ask; the strong models often answer the user's own filler ("Well now, that's a
  toss-up") during the hold. Only the EPAD ×10 checkpoint stays silent and re-asks on its own.
- **Return-cue lookalikes**: "Okay, I'm done for today anyway" said to the friend triggers "Welcome back!" from the 1 h
  models (incl. EPAD); the 10 h model waits for the real cue.
- **Text quality is mostly fine** ("Welcome back! About your sister's clay sculpture, what shape are you thinking of?");
  small artifacts like "I' take it" come from rebuilding words from text tokens for the transcript.

## In progress: 50 h train set (2026-10-15)

A 50 h training set (5,585 episodes / 51.24 h, `/workspace/data50/train50` on the volume; built 2026-10-15) tests
data scaling past 10 h. It is not just more of the same data: a B1-heavier mix (v2) with more fillers in timed holds,
5,164 new screened packs, and user/friend voices from three engines (Pocket, ElevenLabs, Fish; 63 training voices vs
16). No models trained on it yet. Details, caveats and final numbers: `docs/training_record.md` §12; metadata `results/build50/`.

**EPAD at 10 h (R20 ×3, R21 ×10, 2026-10-15):** plateau score 0.87 / 0.88 vs 0.85 for R9 (×1): less over-silent
(silent 90% / 81% vs 96%), speaks when due 85% / 94% vs 74%, timed re-ask 40% / 73% vs 15%; best checkpoints
`R20_10h_lr5e-05_epad3/checkpoint_000153` and `R21_10h_lr5e-05_epad10/checkpoint_000153` at 0.91 (to confirm on dev_200).
Costs: timed holds lose silence (mutter reactions up to 53%). Details: `results/epad10h/summary.md`.

**Live chat (user, 2026-10-15):** R22 (50 h) felt much better than R21 (10 h) in real conversation, and R21 a bit
quiet, the reverse of the TTS-voiced test suites. Likely real-voice robustness from R22's 63 voices / 3 engines, which no suite
measures. See `docs/training_record.md` §9.

## Where things are

- Listening demo (20 models × 9 conversations, audio + timelines): `results/demo/index.html`
- Representative models only (10 × 9 conversations, with good/bad notes, incl. the 10 h EPAD ×10 best and final checkpoints and the 50 h run R22): `results/demo_curated/index.html`
- Transcripts (the 10 representative models): `results/transcripts.md` (5 conversations) and the interactive viewer `results/transcripts.html` (9 conversations, audio from demo_curated); `python -m silence.transcript_gen --html results/transcripts.html`
- Plots: `results/phase1/eval/curves_dev_80.png`, `results/phase2/eval/curves_dev_80.png`, `results/epad/analysis/epad_sweep.png`, loss curves under `results/*/plots/`
- Full tables + method: `docs/training_record.md`; checkpoints: volume `np7c8l94dh` (manifests in `results/*/setup/`)
