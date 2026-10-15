# Held-out test suites: behavior by model

7 suites × 90 episodes, 1 sample per episode, VAD-based scoring (same definitions as dev). Cells: **score** (silent / speaks). Score = mean of silent-when-due and speaks-when-due (test_ordinary has no holds: score = speaks).

| Suite | base | 1 h ×10 (R17 ep 5.9) | 10 h ×10 (R21 ep 2.2) | 50 h v2 ×10 (R22 step 345) | 45 h v3 ×10 (R24 step 527) |
|---|---|---|---|---|---|
| test_script | **0.32** (6% / 57%) | **0.93** (96% / 90%) | **0.97** (98% / 96%) | **0.93** (90% / 97%) | **0.97** (100% / 95%) |
| test_word | **0.35** (11% / 59%) | **0.82** (74% / 90%) | **0.87** (81% / 93%) | **0.84** (72% / 95%) | **0.90** (83% / 97%) |
| test_voice | **0.37** (16% / 57%) | **0.86** (82% / 90%) | **0.92** (89% / 95%) | **0.89** (82% / 96%) | **0.94** (96% / 92%) |
| test_distract | **0.29** (3% / 56%) | **0.93** (87% / 100%) | **0.98** (99% / 97%) | **0.92** (89% / 94%) | **0.91** (90% / 92%) |
| test_cond | **0.32** (6% / 58%) | **0.67** (46% / 89%) | **0.72** (66% / 79%) | **0.70** (49% / 91%) | **0.68** (56% / 81%) |
| test_ordinary | **0.59** (– / 59%) | **0.96** (– / 96%) | **0.98** (– / 98%) | **1.00** (– / 100%) | **0.99** (– / 99%) |
| test_all | **0.32** (14% / 50%) | **0.79** (70% / 88%) | **0.84** (77% / 91%) | **0.80** (67% / 93%) | **0.86** (75% / 96%) |

## Details per suite

### test_script

| Metric | base | 1 h ×10 (R17 ep 5.9) | 10 h ×10 (R21 ep 2.2) | 50 h v2 ×10 (R22 step 345) | 45 h v3 ×10 (R24 step 527) |
|---|---|---|---|---|---|
| silent when due (n=49) | 6% | 96% | 98% | 90% | 100% |
| speaks when due (n=136) | 57% | 90% | 96% | 97% | 95% |
| cue release: spoke (n=45) | 53% | 93% | 98% | 98% | 91% |
| cue release: on time (n=45) | 60% | 91% | 96% | 98% | 87% |
| timed re-ask (n=7) | 57% | 29% | 43% | 57% | 86% |
| ordinary answered (n=84) | 60% | 93% | 100% | 100% | 98% |
| reacted to D1a (n=47) | 91% | 0% | 0% | 0% | 0% |
| reacted to D1b (n=16) | 88% | 6% | 0% | 0% | 0% |
| reacted to D1c (n=2) | 50% | 0% | 0% | 0% | 0% |
| reacted to D2a (n=12) | 83% | 0% | 0% | 0% | 0% |
| reacted to D2b (n=11) | 55% | 0% | 0% | 0% | 0% |
| reacted to D2c (n=1) | 100% | 100% | 100% | 100% | 0% |
| reacted to D3a (n=25) | 84% | 0% | 0% | 0% | 0% |
| reacted to D3b (n=8) | 75% | 0% | 0% | 0% | 0% |
| silent in B1_double holds (n=1) | 0% | 100% | 100% | 100% | 100% |
| silent in B1_early holds (n=1) | 100% | 100% | 100% | 0% | 100% |
| silent in B1_std holds (n=3) | 33% | 67% | 67% | 33% | 100% |
| silent in B2_std holds (n=25) | 0% | 96% | 100% | 96% | 100% |
| silent in B3_early holds (n=3) | 0% | 100% | 100% | 100% | 100% |
| silent in B3_rescind holds (n=2) | 50% | 100% | 100% | 50% | 100% |
| silent in B3_std holds (n=14) | 0% | 100% | 100% | 100% | 100% |

### test_word

| Metric | base | 1 h ×10 (R17 ep 5.9) | 10 h ×10 (R21 ep 2.2) | 50 h v2 ×10 (R22 step 345) | 45 h v3 ×10 (R24 step 527) |
|---|---|---|---|---|---|
| silent when due (n=54) | 11% | 74% | 81% | 72% | 83% |
| speaks when due (n=135) | 59% | 90% | 93% | 95% | 97% |
| cue release: spoke (n=46) | 59% | 85% | 85% | 89% | 96% |
| cue release: on time (n=46) | 63% | 87% | 85% | 91% | 91% |
| timed re-ask (n=8) | 50% | 75% | 62% | 75% | 100% |
| ordinary answered (n=81) | 59% | 95% | 100% | 100% | 98% |
| reacted to D1a (n=44) | 73% | 5% | 0% | 0% | 0% |
| reacted to D1b (n=14) | 86% | 0% | 0% | 0% | 0% |
| reacted to D1c (n=3) | 67% | 0% | 0% | 0% | 0% |
| reacted to D2a (n=10) | 70% | 10% | 0% | 0% | 0% |
| reacted to D2b (n=13) | 85% | 0% | 0% | 15% | 0% |
| reacted to D3a (n=23) | 91% | 0% | 0% | 0% | 0% |
| reacted to D3b (n=7) | 86% | 14% | 0% | 14% | 14% |
| silent in B1_double holds (n=4) | 50% | 50% | 50% | 25% | 25% |
| silent in B1_early holds (n=4) | 25% | 50% | 25% | 50% | 75% |
| silent in B1_rescind holds (n=1) | 100% | 100% | 100% | 100% | 100% |
| silent in B1_std holds (n=4) | 25% | 25% | 0% | 0% | 0% |
| silent in B2_std holds (n=23) | 0% | 87% | 100% | 100% | 100% |
| silent in B3_early holds (n=3) | 0% | 67% | 67% | 33% | 100% |
| silent in B3_rescind holds (n=1) | 0% | 100% | 100% | 100% | 100% |
| silent in B3_std holds (n=14) | 7% | 79% | 100% | 71% | 93% |

### test_voice

| Metric | base | 1 h ×10 (R17 ep 5.9) | 10 h ×10 (R21 ep 2.2) | 50 h v2 ×10 (R22 step 345) | 45 h v3 ×10 (R24 step 527) |
|---|---|---|---|---|---|
| silent when due (n=57) | 16% | 82% | 89% | 82% | 96% |
| speaks when due (n=129) | 57% | 90% | 95% | 96% | 92% |
| cue release: spoke (n=46) | 63% | 93% | 100% | 98% | 93% |
| cue release: on time (n=46) | 59% | 91% | 100% | 98% | 91% |
| timed re-ask (n=15) | 53% | 53% | 67% | 80% | 73% |
| ordinary answered (n=68) | 54% | 96% | 99% | 99% | 96% |
| reacted to D1a (n=57) | 82% | 2% | 0% | 0% | 0% |
| reacted to D1b (n=17) | 100% | 0% | 0% | 0% | 0% |
| reacted to D1c (n=3) | 100% | 0% | 0% | 0% | 0% |
| reacted to D2a (n=14) | 86% | 7% | 0% | 0% | 0% |
| reacted to D2b (n=6) | 33% | 0% | 0% | 0% | 0% |
| reacted to D2c (n=4) | 0% | 50% | 0% | 25% | 0% |
| reacted to D3a (n=29) | 83% | 3% | 0% | 0% | 0% |
| reacted to D3b (n=8) | 38% | 0% | 0% | 0% | 0% |
| silent in B1_double holds (n=4) | 50% | 75% | 50% | 50% | 50% |
| silent in B1_early holds (n=1) | 100% | 100% | 100% | 100% | 100% |
| silent in B1_rescind holds (n=1) | 100% | 100% | 100% | 0% | 100% |
| silent in B1_std holds (n=7) | 43% | 14% | 43% | 29% | 100% |
| silent in B2_std holds (n=29) | 0% | 93% | 100% | 100% | 100% |
| silent in B3_early holds (n=1) | 0% | 100% | 100% | 100% | 100% |
| silent in B3_rescind holds (n=2) | 100% | 100% | 100% | 100% | 100% |
| silent in B3_std holds (n=12) | 0% | 92% | 100% | 83% | 100% |

### test_distract

| Metric | base | 1 h ×10 (R17 ep 5.9) | 10 h ×10 (R21 ep 2.2) | 50 h v2 ×10 (R22 step 345) | 45 h v3 ×10 (R24 step 527) |
|---|---|---|---|---|---|
| silent when due (n=90) | 3% | 87% | 99% | 89% | 90% |
| speaks when due (n=90) | 56% | 100% | 97% | 94% | 92% |
| cue release: spoke (n=90) | 56% | 100% | 97% | 94% | 92% |
| cue release: on time (n=90) | 48% | 99% | 96% | 94% | 91% |
| reacted to D1a (n=115) | 71% | 2% | 0% | 1% | 0% |
| reacted to D1b (n=33) | 94% | 3% | 0% | 0% | 3% |
| reacted to D1c (n=9) | 100% | 0% | 0% | 0% | 0% |
| reacted to D2a (n=61) | 72% | 13% | 0% | 2% | 3% |
| reacted to D2b (n=29) | 55% | 0% | 0% | 7% | 0% |
| reacted to D3a (n=61) | 64% | 5% | 0% | 3% | 3% |
| reacted to D3b (n=29) | 52% | 0% | 0% | 10% | 3% |
| reacted to D4 (n=18) | 78% | 11% | 0% | 11% | 44% |
| silent in B2_D4 holds (n=18) | 0% | 56% | 100% | 89% | 61% |
| silent in B2_std holds (n=43) | 2% | 91% | 98% | 93% | 98% |
| silent in B3_std holds (n=29) | 7% | 100% | 100% | 83% | 97% |

### test_cond

| Metric | base | 1 h ×10 (R17 ep 5.9) | 10 h ×10 (R21 ep 2.2) | 50 h v2 ×10 (R22 step 345) | 45 h v3 ×10 (R24 step 527) |
|---|---|---|---|---|---|
| silent when due (n=90) | 6% | 46% | 66% | 49% | 56% |
| speaks when due (n=90) | 58% | 89% | 79% | 91% | 81% |
| cue release: spoke (n=60) | 55% | 92% | 68% | 87% | 73% |
| cue release: on time (n=60) | 58% | 88% | 67% | 93% | 67% |
| timed re-ask (n=30) | 63% | 83% | 100% | 100% | 97% |
| reacted to D1a (n=56) | 75% | 0% | 2% | 2% | 2% |
| reacted to D1b (n=14) | 86% | 0% | 0% | 7% | 7% |
| reacted to D1c (n=2) | 100% | 0% | 0% | 0% | 0% |
| reacted to D2a (n=12) | 75% | 25% | 0% | 25% | 8% |
| reacted to D2c (n=24) | 33% | 50% | 42% | 62% | 54% |
| reacted to D3a (n=29) | 76% | 0% | 0% | 14% | 10% |
| silent in B1_minutes holds (n=30) | 10% | 0% | 7% | 0% | 0% |
| silent in COND_keyword holds (n=29) | 3% | 83% | 97% | 83% | 83% |
| silent in COND_phone holds (n=31) | 3% | 55% | 94% | 65% | 84% |

### test_ordinary

| Metric | base | 1 h ×10 (R17 ep 5.9) | 10 h ×10 (R21 ep 2.2) | 50 h v2 ×10 (R22 step 345) | 45 h v3 ×10 (R24 step 527) |
|---|---|---|---|---|---|
| speaks when due (n=192) | 59% | 96% | 98% | 100% | 99% |
| ordinary answered (n=192) | 59% | 96% | 98% | 100% | 99% |

### test_all

| Metric | base | 1 h ×10 (R17 ep 5.9) | 10 h ×10 (R21 ep 2.2) | 50 h v2 ×10 (R22 step 345) | 45 h v3 ×10 (R24 step 527) |
|---|---|---|---|---|---|
| silent when due (n=57) | 14% | 70% | 77% | 67% | 75% |
| speaks when due (n=137) | 50% | 88% | 91% | 93% | 96% |
| cue release: spoke (n=42) | 43% | 93% | 88% | 90% | 93% |
| cue release: on time (n=42) | 48% | 93% | 88% | 93% | 90% |
| timed re-ask (n=15) | 53% | 47% | 73% | 80% | 87% |
| ordinary answered (n=80) | 52% | 92% | 96% | 98% | 99% |
| reacted to D1a (n=48) | 90% | 2% | 0% | 0% | 0% |
| reacted to D1b (n=14) | 100% | 0% | 0% | 0% | 0% |
| reacted to D1c (n=3) | 100% | 0% | 0% | 0% | 0% |
| reacted to D2a (n=15) | 100% | 7% | 0% | 0% | 0% |
| reacted to D2b (n=4) | 75% | 0% | 0% | 0% | 0% |
| reacted to D2c (n=3) | 33% | 67% | 0% | 33% | 0% |
| reacted to D3a (n=25) | 80% | 0% | 0% | 0% | 0% |
| reacted to D3b (n=2) | 0% | 0% | 0% | 0% | 0% |
| silent in B1_double holds (n=10) | 70% | 40% | 30% | 20% | 0% |
| silent in B1_rescind holds (n=4) | 25% | 25% | 50% | 0% | 75% |
| silent in B1_std holds (n=5) | 0% | 0% | 20% | 0% | 40% |
| silent in B2_std holds (n=25) | 0% | 92% | 100% | 100% | 100% |
| silent in B3_early holds (n=2) | 0% | 50% | 100% | 100% | 100% |
| silent in B3_rescind holds (n=2) | 0% | 100% | 100% | 100% | 100% |
| silent in B3_std holds (n=9) | 0% | 100% | 100% | 78% | 100% |
