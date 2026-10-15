# Human test set: behavior by model

One speaker (s01, ~20 min of recording) voices the user side of 24 held-out `test_all` episodes; the friend and Moshi's scripted lines stay TTS. Three paired suites (same episode ids, so the same noise and friend RIR): **TTS** = the original episodes, **human** = every user line recorded verbatim, **own words** = the 16 hold episodes with the instruction and the release cue / rescind said in the speaker's own words. 3 samples per episode, VAD scoring, same definitions as `results/test/summary.md`. Cells: **score** (silent / speaks).

## All 24 episodes

| Model | TTS | human (verbatim) |
|---|---|---|
| base | **0.41** (24% / 58%) | **0.33** (16% / 50%) |
| 10 h ×10 (R21 ep 2.2) | **0.83** (75% / 91%) | **0.79** (73% / 86%) |
| 50 h ×10 (R22 step 345) | **0.79** (61% / 97%) | **0.74** (57% / 91%) |

## The 16 hold episodes

| Model | TTS | human (verbatim) | human (own words) |
|---|---|---|---|
| base | **0.40** (24% / 57%) | **0.29** (16% / 43%) | **0.28** (14% / 43%) |
| 10 h ×10 (R21 ep 2.2) | **0.81** (75% / 88%) | **0.75** (73% / 78%) | **0.68** (65% / 71%) |
| 50 h ×10 (R22 step 345) | **0.77** (61% / 94%) | **0.73** (57% / 88%) | **0.67** (53% / 80%) |

## Paired differences (95% bootstrap CI over episodes)

| Model | Comparison | episodes | score | silent when due | speaks when due |
|---|---|---|---|---|---|
| base | human − TTS | 24 | -0.08 [-0.18, +0.02] | -8 pp [-23, +8] | -9 pp [-18, +1] |
| base | own words − verbatim | 16 | -0.01 [-0.13, +0.10] | -2 pp [-15, +9] | +0 pp [-17, +14] |
| base | own words − TTS | 16 | -0.12 [-0.22, +0.00] | -10 pp [-19, -2] | -14 pp [-31, +6] |
| 10 h ×10 (R21 ep 2.2) | human − TTS | 24 | -0.04 [-0.13, +0.05] | -2 pp [-15, +10] | -6 pp [-17, +4] |
| 10 h ×10 (R21 ep 2.2) | own words − verbatim | 16 | -0.08 [-0.22, +0.06] | -8 pp [-24, +8] | -8 pp [-29, +14] |
| 10 h ×10 (R21 ep 2.2) | own words − TTS | 16 | -0.14 [-0.25, -0.02] | -10 pp [-21, -2] | -18 pp [-40, +4] |
| 50 h ×10 (R22 step 345) | human − TTS | 24 | -0.05 [-0.17, +0.07] | -4 pp [-22, +14] | -6 pp [-16, +3] |
| 50 h ×10 (R22 step 345) | own words − verbatim | 16 | -0.06 [-0.17, +0.06] | -4 pp [-16, +6] | -8 pp [-25, +8] |
| 50 h ×10 (R22 step 345) | own words − TTS | 16 | -0.11 [-0.25, +0.04] | -8 pp [-29, +11] | -14 pp [-33, +4] |

## Details

### test_human_tts

| Metric | base | 10 h ×10 (R21 ep 2.2) | 50 h ×10 (R22 step 345) |
|---|---|---|---|
| silent when due (n=51) | 24% | 75% | 61% |
| speaks when due (n=105) | 58% | 91% | 97% |
| cue release: spoke (n=33) | 55% | 91% | 94% |
| cue release: on time (n=33) | 55% | 91% | 94% |
| timed re-ask (n=18) | 61% | 83% | 94% |
| ordinary answered (n=54) | 59% | 94% | 100% |
| reacted to D1a (n=27) | 85% | 0% | 0% |
| reacted to D1b (n=12) | 100% | 0% | 0% |
| reacted to D1c (n=6) | 67% | 0% | 0% |
| reacted to D2a (n=12) | 83% | 0% | 0% |
| reacted to D2b (n=3) | 67% | 0% | 0% |
| reacted to D2c (n=3) | 67% | 33% | 0% |
| reacted to D3a (n=18) | 83% | 0% | 0% |

### test_human

| Metric | base | 10 h ×10 (R21 ep 2.2) | 50 h ×10 (R22 step 345) |
|---|---|---|---|
| silent when due (n=51) | 16% | 73% | 57% |
| speaks when due (n=105) | 50% | 86% | 91% |
| cue release: spoke (n=33) | 48% | 73% | 85% |
| cue release: on time (n=33) | 48% | 73% | 88% |
| timed re-ask (n=18) | 33% | 89% | 94% |
| ordinary answered (n=54) | 56% | 93% | 94% |
| reacted to D1a (n=27) | 70% | 0% | 0% |
| reacted to D1b (n=12) | 75% | 0% | 0% |
| reacted to D1c (n=6) | 83% | 0% | 0% |
| reacted to D2a (n=12) | 83% | 0% | 0% |
| reacted to D2b (n=3) | 67% | 0% | 33% |
| reacted to D2c (n=3) | 0% | 33% | 67% |
| reacted to D3a (n=18) | 94% | 0% | 0% |

### test_human_free

| Metric | base | 10 h ×10 (R21 ep 2.2) | 50 h ×10 (R22 step 345) |
|---|---|---|---|
| silent when due (n=51) | 14% | 65% | 53% |
| speaks when due (n=51) | 43% | 71% | 80% |
| cue release: spoke (n=33) | 45% | 67% | 73% |
| cue release: on time (n=33) | 48% | 67% | 79% |
| timed re-ask (n=18) | 39% | 78% | 94% |
| reacted to D1a (n=27) | 85% | 0% | 7% |
| reacted to D1b (n=12) | 58% | 0% | 0% |
| reacted to D1c (n=6) | 83% | 0% | 33% |
| reacted to D2a (n=12) | 75% | 0% | 0% |
| reacted to D2b (n=3) | 67% | 0% | 33% |
| reacted to D2c (n=3) | 33% | 67% | 33% |
| reacted to D3a (n=18) | 78% | 0% | 0% |
