# EPAD weight at 10 h (R20 ×3, R21 ×10) vs R9 (×1) and R8 (2e-5, ×1)

train_10h (1,101 episodes), lr 5e-5, 5 epochs = 345 steps, checkpoint + dev loss every 17 steps (21 checkpoints);
identical to phase-2 R9 except `text_epad_weight` (R20: 1.5 = ×3 PAD's 0.5; R21: 5 = ×10). Trained 05:33–06:49 UTC on
two 1× H100 pods (`tr3`, `tr10`), dev_80 generation evals 06:49–07:13 UTC. Dev_80, 1 sample per episode; per checkpoint
n = 58 silence-scored holds, 45 cue releases, 18 timer releases, 42 ordinary questions (±10–12 pp at 95% per checkpoint).

Metrics (same definitions as phase 2 / the 1 h EPAD sweep, `analysis/rescore_speak_silence.py`):
silent when due = 1 − hold violated; speaks when due = lenient "spoke" over cue releases + timed releases + ordinary
questions; timed re-ask = spoke after a B1 timed hold without being prompted; score = mean(silent, speaks).
Plateau = epochs 2–5 (14 checkpoints, ep 1.98–5.01), as in phase 2. Figure: `analysis/epad10h.png`.

## Plateau (epochs 2–5) and best checkpoint by score (ties → earlier epoch)

| Run | Plateau score | Silent | Speaks | Cue release on time | Timed re-ask | Timed on time | Ordinary answered | Best checkpoint (epoch): score, silent / speaks, timed re-ask, ordinary | Min text dev loss (epoch) |
|---|---|---|---|---|---|---|---|---|---|
| R9 ×1 (lr 5e-5) | 0.85 | 96% | 74% | 89% | 15% | 11% | 82% | `checkpoint_000255` (3.7): 0.87, 98% / 75%, 6%, 88% | 0.275 (2.0) |
| **R20 ×3** | **0.87** | 90% | 85% | 93% | 40% | 35% | 95% | `checkpoint_000153` (2.2): **0.91**, 98% / 84%, 28%, 100% | 0.286 (1.5) |
| **R21 ×10** | **0.88** | 81% | 94% | 96% | 73% | 65% | 100% | `checkpoint_000153` (2.2): **0.91**, 90% / 93%, 61%, 100% | 0.323 (1.5) |
| R8 ×1 (lr 2e-5) | 0.84 | 96% | 72% | 76% | 20% | 17% | 89% | `checkpoint_000289` (4.2): 0.88, 97% / 79%, 22%, 95% | 0.280 (3.0) |
| base model | 0.29 | 14% | 45% | 47% | 33% | 22% | 43% | | 0.537 |

Other plateau means (eval_report, `eval/summary.csv`): R9 / R20 / R21: hold violated 4% / 10% / 19%; resumed on time
67% / 77% / 87%; cue latency median 1.14 / 0.88 / 0.58 s; timed-release onset error (median, × N) +0.21 / +0.02 / −0.01;
barge-in into mid-turn pauses 11% / 27% / 23%; B1-filler (D2c) reactions 21% / 41% / 53%; all other distractors ≤ 5%
(D1a–D3b) for every run.

R21 final checkpoint (`checkpoint_000345`, ep 5.0) is nearly as good as its best: 0.90, 84% silent / 95% speaks,
78% timed re-ask, 100% ordinary.

## Where the silence goes: violations by hold type (plateau, epochs 2–5)

| Hold type (holds / ckpt) | R9 ×1 | R20 ×3 | R21 ×10 |
|---|---|---|---|
| B2 side conversation (20) | 4% | 3% | 2% |
| B3 hear me out, std (16) | 1% | 3% | 2% |
| B1 timed hold, std (11) | 10% | 37% | 57% |
| B1 early answer (3) | 2% | 14% | 90% |
| B1 double (2) / rescind (2) | 0% / 0% | 0% / 0% | 25% / 7% |

The whole silence loss from EPAD comes from **timed holds**: B2/B3 cue-released holds stay 97–99% silent at ×10.
In B1 holds the up-weighted model often speaks before N is up; its reactions to the user's muttered fillers during B1 holds (D2c) rise in step (21% → 41% → 53%).

## Findings

1. **EPAD at 10 h moves the balance the same way as at 1 h, and here it's a net gain.** Speaking rises (74% → 85% → 94%;
   cue release on time 89% → 93% → 96%; ordinary 82% → 95% → 100%) and silence drops (96% → 90% → 81%). At 1 h the score stayed flat
   (≈ 0.80); at 10 h the over-silent baseline has room, so the plateau score rises 0.85 → 0.87 (×3) / 0.88 (×10) and the best
   checkpoints reach 0.91 vs 0.87–0.88 for R9/R8. That's the highest dev_80 score so far, but it's within the ±0.06
   per-checkpoint noise of the previous best (0.88–0.89); confirm on dev_200.
2. **Timed re-asks are learned and stay learned at 10 h with ×10.** R21 re-asks after a timed hold 73% of the time at the
   plateau (65% on time, median onset error ≈ 0·N), vs 15% for R9. At 1 h the ×10 timed-hold gain (60–80% mid-training)
   decayed to 34% by the plateau; at 10 h it holds through epoch 5 (61–83% at every checkpoint after epoch 2). ×3 is in between (40%).
3. **The cost of ×10 is premature speech in timed holds** (B1 violated 57%, D2c filler reactions 53%), not in
   side conversations or hear-me-out holds. ×3 is the safer trade (90% silent, 98% at its best checkpoint, with 84% speaks);
   ×10 is the one that actually re-asks after timed holds.
4. Dev text loss is again not the selection signal: R21 has the highest text dev loss (min 0.323 vs 0.275 for R9; the dev loss
   uses the original unweighted EPAD) yet the best behavior. Text dev loss also rises after epoch ~2 in all three lr 5e-5 runs.

Next experiment suggestion: an EPAD weight between ×3 and ×10 (e.g. ×5) at 10 h, or ×10 with B1 filler-robustness data; rescore the
candidates `R20/checkpoint_000153`, `R21/checkpoint_000153`, `R21/checkpoint_000345` on dev_200.

## Per checkpoint

Cells: **score** (silent / speaks / timed re-ask / ordinary, text dev loss).

| epoch | EPAD ×1 (R9) | EPAD ×3 (R20) | EPAD ×10 (R21) | lr 2e-5, EPAD ×1 (R8) |
|---|---|---|---|---|
| 0 (base) | 0.29 (14%/45%/33%/43%, 0.537) | 0.29 (14%/45%/33%/43%, 0.537) | 0.29 (14%/45%/33%/43%, 0.537) | 0.29 (14%/45%/33%/43%, 0.537) |
| 0.25 | **0.48** (72%/23%/22%/14%, 0.399) | **0.58** (48%/68%/39%/90%, 0.411) | **0.41** (12%/70%/17%/100%, 0.488) | **0.43** (71%/16%/22%/2%, 0.467) |
| 0.49 | **0.71** (97%/46%/6%/64%, 0.310) | **0.68** (50%/87%/61%/100%, 0.324) | **0.65** (40%/90%/94%/100%, 0.384) | **0.61** (81%/42%/17%/50%, 0.356) |
| 0.74 | **0.78** (93%/62%/11%/79%, 0.289) | **0.79** (78%/81%/28%/98%, 0.304) | **0.76** (66%/86%/44%/100%, 0.362) | **0.71** (84%/57%/6%/83%, 0.317) |
| 0.99 | **0.81** (90%/71%/17%/90%, 0.282) | **0.74** (59%/90%/61%/98%, 0.298) | **0.67** (45%/89%/78%/100%, 0.350) | **0.73** (86%/59%/6%/79%, 0.301) |
| 1.24 | **0.86** (98%/74%/28%/98%, 0.280) | **0.82** (76%/88%/44%/98%, 0.291) | **0.74** (57%/90%/89%/100%, 0.332) | **0.75** (84%/66%/17%/88%, 0.292) |
| 1.48 | **0.83** (100%/67%/6%/81%, 0.279) | **0.89** (93%/86%/33%/98%, 0.286) | **0.86** (81%/90%/61%/98%, 0.323) | **0.77** (93%/60%/11%/74%, 0.286) |
| 1.73 | **0.82** (97%/67%/6%/81%, 0.277) | **0.88** (91%/85%/22%/100%, 0.287) | **0.87** (78%/96%/83%/100%, 0.336) | **0.79** (95%/63%/17%/88%, 0.283) |
| 1.98 | **0.82** (88%/75%/22%/86%, 0.275) | **0.85** (81%/89%/44%/100%, 0.288) | **0.81** (71%/90%/67%/100%, 0.338) | **0.79** (91%/68%/22%/88%, 0.281) |
| 2.22 | **0.78** (98%/57%/11%/81%, 0.302) | **0.91** (98%/84%/28%/100%, 0.306) | **0.91** (90%/93%/61%/100%, 0.329) | **0.83** (97%/69%/22%/83%, 0.283) |
| 2.47 | **0.87** (93%/80%/11%/95%, 0.304) | **0.85** (84%/85%/44%/95%, 0.313) | **0.87** (79%/94%/72%/100%, 0.336) | **0.83** (97%/69%/28%/88%, 0.281) |
| 2.72 | **0.85** (95%/75%/28%/81%, 0.303) | **0.88** (90%/86%/39%/93%, 0.311) | **0.89** (84%/93%/67%/100%, 0.338) | **0.80** (100%/59%/6%/86%, 0.281) |
| 2.96 | **0.86** (98%/74%/22%/79%, 0.297) | **0.87** (90%/84%/39%/93%, 0.307) | **0.86** (79%/93%/67%/100%, 0.338) | **0.81** (91%/71%/17%/90%, 0.280) |
| 3.21 | **0.86** (95%/77%/22%/86%, 0.360) | **0.87** (84%/90%/50%/100%, 0.369) | **0.83** (74%/92%/78%/98%, 0.385) | **0.84** (93%/75%/11%/93%, 0.289) |
| 3.46 | **0.85** (97%/74%/6%/83%, 0.347) | **0.90** (93%/88%/44%/98%, 0.351) | **0.88** (81%/95%/83%/100%, 0.367) | **0.84** (97%/71%/28%/86%, 0.293) |
| 3.71 | **0.87** (98%/75%/6%/88%, 0.353) | **0.90** (93%/87%/39%/95%, 0.358) | **0.88** (83%/93%/67%/100%, 0.367) | **0.82** (98%/67%/17%/86%, 0.288) |
| 3.95 | **0.85** (97%/74%/17%/81%, 0.352) | **0.89** (91%/86%/33%/95%, 0.364) | **0.90** (84%/96%/78%/100%, 0.379) | **0.87** (97%/78%/22%/90%, 0.289) |
| 4.20 | **0.86** (98%/73%/17%/76%, 0.366) | **0.88** (93%/83%/28%/95%, 0.376) | **0.89** (83%/94%/83%/100%, 0.381) | **0.88** (97%/79%/22%/95%, 0.293) |
| 4.45 | **0.84** (97%/72%/11%/79%, 0.385) | **0.87** (91%/82%/44%/88%, 0.393) | **0.89** (83%/96%/78%/100%, 0.394) | **0.86** (98%/74%/22%/90%, 0.297) |
| 4.69 | **0.86** (98%/74%/6%/81%, 0.391) | **0.83** (88%/79%/39%/88%, 0.398) | **0.86** (79%/93%/72%/100%, 0.400) | **0.86** (95%/76%/28%/93%, 0.299) |
| 4.94 | **0.85** (98%/72%/11%/76%, 0.392) | **0.86** (88%/84%/44%/93%, 0.398) | **0.88** (84%/91%/67%/100%, 0.400) | **0.84** (97%/72%/17%/88%, 0.299) |
| 5.01 | **0.85** (97%/74%/22%/76%, 0.392) | **0.88** (93%/84%/44%/93%, 0.398) | **0.90** (84%/95%/78%/100%, 0.400) | **0.87** (98%/75%/17%/90%, 0.299) |


## Files

- `configs/`: R20/R21 yaml + epochs.json (R8/R9/R0 copied from `results/phase2/configs` for the report).
- `runs/<run>/`: args.yaml, metrics.train.jsonl (345 steps), metrics.eval.jsonl (21 dev losses). R0/R8/R9 copied from phase 2.
- `logs/`: training stdout, `parallel_epad10h_{3,10}.log`, `eval_dev_epad10h_{3,10}.log`, `epad10h_{3,10}.log`, `parallel.log`, `eval/<run>_dev_80.log`.
- `eval_raw/<run>/dev_80/<checkpoint>/`: per-episode json, scores.jsonl, meta.json, score.log (no codes.npz / wav for R20/R21;
  R0/R8/R9 copied from phase 2 incl. codes.npz).
- `setup/`: `epad10h.sh`, loss-weights patch, `pull_run.sh` (the copy script), `checkpoints_manifest.txt` (size + sha256 of all
  42 R20/R21 `lora.safetensors`, computed on the pod; checkpoints stay on volume `np7c8l94dh` under `/workspace/runs/`).
- `eval/summary.{csv,md}`, `eval/curves_dev_80.png`: eval_report output. `analysis/`: rescore + plot + tables.

Regenerate (repo root):
```
python results/epad10h/analysis/rescore_speak_silence.py results/epad10h
python -m silence.eval_report results/epad10h/eval_raw --configs results/epad10h/configs --runs results/epad10h/runs --out results/epad10h/eval
PYTHONPATH=. python results/epad10h/analysis/analyze_epad10h.py
```
