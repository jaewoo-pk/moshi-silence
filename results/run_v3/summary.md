# R24: v3 data (long silences in cue-released holds, B1 15%), EPAD ×10, lr 5e-5, 2026-10-15

Why: R22 (50 h v2) learned "speak after a few seconds of user silence" (B1-heavy mix + EPAD ×10, no long silences that
mean "keep waiting"); it lost hear-me-out / double / rescinded holds. v3 = the v2 plan with longer silences drawn from a
separate RNG (identical lines, voices and clips; only timing): side-conversation stalls up to 20 s (half 8–20 s), pauses
between options 3–12 s, a 2.5–10 s silence before the B3 cue in 40% (std) / 60% (early, rescind) of B3 episodes. Built
from the cache in 8 min (5,578 episodes, 52.4 h; 21 over 60 s dropped; 5,567 pass check.py), then rebalanced to B1 15%
(`silence/rebalance.py`): `train_v3_b1-15.jsonl` = 4,907 episodes / 45.6 h.

Training: planned 614 steps (2 epochs, OneCycle over 614), checkpoint + dev loss every 31 steps; **stopped by the user at
step ~531** (1.73 epochs; LR then ≈ 6% of peak); last checkpoint `checkpoint_000527`. Pod `5obz5mapyti207` (09:43–12:29 UTC,
incl. the R21/R22 chat demo and the v3 build; ≈ $11). Text dev loss 0.296–0.298 from step 310 (R22 best 0.301, R21 min 0.323).

## Dev (dev_80, 1 sample)

| | R21 10 h v1 | R22 50 h v2 | **R24 45 h v3** |
|---|---|---|---|
| score, mean of last 8 checkpoints | 0.89 | 0.82 | **0.91** |
| final checkpoint | 0.90 (84% / 95%) | 0.82 (69% / 94%) | **0.94 (97% / 92%)** (= its best) |
| timed re-ask (last 8) | 76% | 74% | 70% |
| silent: B3 rescind / B3 std | 85% / 99% | 15% / 91% | **94% / 100%** |
| silent: B1 std / early / rescind / double | 41 / 7 / 95 / 70% | 36 / 3 / 40 / 20% | **64 / 79 / 81 / 56%** |

## Held-out test suites (1 sample; R21 best-dev checkpoint, R22 / R24 final)

| Suite | R21 | R22 | **R24** |
|---|---|---|---|
| test_script | **0.97** | 0.93 | **0.97** (100% / 95%) |
| test_word | 0.87 | 0.84 | **0.90** |
| test_voice | 0.92 | 0.89 | **0.94** |
| test_distract | **0.98** | 0.92 | 0.91 (friend's "I'm back" reacted to 44% vs R21 0%) |
| test_cond | **0.72** | 0.70 | 0.68 |
| test_ordinary | 0.98 | **1.00** | 0.99 |
| test_all | 0.84 | 0.80 | **0.86** |

## Human-recorded sets (speaker s01 = the user; 3 samples) — see `results/human/summary.md`

R24 is the most robust to the real voice: TTS control 0.84 → your voice 0.83 (R21 0.83 → 0.79, R22 0.79 → 0.74,
R17 0.79 → 0.67). With your own wording all models drop (R24 0.73, tied with R17).

Conclusion: the data fix worked. The v3 recipe at ~45 h is the best model on dev, on 4 of 7 test suites (all the
generalization axes: scripts, wording, voices, all three) and on the human recordings. Remaining weaknesses: the
friend saying "I'm back" (never trained; v3's long side-conversation silences may make it look like a release), keyword
releases, "a minute" holds, timed holds with the user's own wording.
