# R22: 50 h, EPAD ×10, lr 5e-5, 345 steps (1 epoch), 2026-10-15

Trained 07:31–09:06 UTC on 1× H100 pod `i1wn60oi6h7n8g` (terminated 09:31; ≈ $8.1 incl. evals). Data: the v2 50 h set
(`/workspace/data50/train50/train_50h.jsonl`, 5,457 episodes; v2 mix, 3 TTS engines). Identical recipe and step count to
R21 (old 10 h set, 5 epochs), so R21 vs R22 = data at equal training compute. The first hour shared the GPU with the
test-suite evals of base / R17 / R21 (training ran at ~7k instead of ~11k words/s then).

## Dev (dev_80, 1 sample; score = mean of silent-when-due and speaks-when-due)

- Text dev loss fell monotonically to 0.305 at step 345 (R21: min 0.323 at epoch 1.5, then up to 0.40). No overfitting in 1 epoch.
- Behavior plateaued by ~step 150 (epoch 0.45): score 0.81–0.86 from there to the end (mean of the last 10 checkpoints ≈ 0.83;
  silent 69–78%, speaks 90–94%, timed re-ask 61–83%). R21's plateau: 0.88 (81% / 94%). The curve is flat at the end, so a
  second epoch was judged not worth it (2.5 h) and not run.

## Held-out test suites (1 sample; R22 = final checkpoint, R21 / R17 = best dev_80 checkpoint)

| Suite | base | 1 h ×10 (R17) | 10 h ×10 (R21) | 50 h ×10 (R22) |
|---|---|---|---|---|
| test_script | 0.32 | 0.93 (96% / 90%) | **0.97** (98% / 96%) | 0.93 (90% / 97%) |
| test_word | 0.35 | 0.82 (74% / 90%) | **0.87** (81% / 93%) | 0.84 (72% / 95%) |
| test_voice | 0.37 | 0.86 (82% / 90%) | **0.92** (89% / 95%) | 0.89 (82% / 96%) |
| test_distract | 0.29 | 0.93 (87% / 100%) | **0.98** (99% / 97%) | 0.92 (89% / 94%) |
| test_cond | 0.32 | 0.67 (46% / 89%) | **0.72** (66% / 79%) | 0.70 (49% / 91%) |
| test_ordinary | 0.59 | 0.96 | 0.98 | **1.00** |
| test_all | 0.32 | 0.79 (70% / 88%) | **0.84** (77% / 91%) | 0.80 (67% / 93%) |

**50 h did not beat 10 h**: R22 speaks a little more (+1–12 pp) and holds silence less (−7 to −17 pp), ending 2–6 points
below R21 on six suites (and above it on test_ordinary). Losses are in timed holds (B1) and, on test_all, B3 holds (78% vs 100%);
side-conversation holds stay at 100%.

Not a clean data-size result. Confounds, largest first:
1. **Recipe.** The 50 h set has 25% timed holds (vs 15%) with mutters in 70% (vs 50%), new scripts and 3 TTS engines. With
   EPAD ×10 (which already traded silence in timed holds for re-asks) more B1 data pushes further toward speaking.
   R23 (the v2 recipe at 10 h, same 345 steps) would separate recipe from size; not run.
2. **Checkpoint selection.** R21 / R17 are their best-of-21 dev_80 checkpoints (an optimistic pick, ±0.06 noise); R22 is
   its final checkpoint (no selection). R21's final checkpoint scored 0.90 on dev_80 vs 0.91 for the pick, so this is a
   small effect, but it favors R21.
3. **Exposure.** At equal steps R21 saw each 10 h episode ~5 times; R22 saw each 50 h episode once. 
4. Noise: 1 sample, 7–30 timed holds per suite.

Next: R23 (v2 recipe at 10 h) to separate recipe from size; or R22 with EPAD ×3 (the new recipe already pushes toward speaking).

Files: `configs/`, `runs/R22_*/` (args, train/eval metrics), `eval_raw/` (R22 dev_80 + R21 copy), `analysis/` (rescore),
`logs/`, `setup/` (scripts, checkpoint manifest with sha256; checkpoints stay on volume `np7c8l94dh`). Test-suite outputs:
`results/test/raw_R22/`, table `results/test/summary.md`.
