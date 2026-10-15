# Human-recorded test sets: behavior by model

Speaker s01 (the user), one mic; friend lines TTS. 3 samples per episode, common seeds. test_human_tts = the same 24 scripts with the TTS user voice (paired control). Cells: **score** (silent when due / speaks when due).

| Set | base | 1 h ×10 (R17) | 10 h ×10 (R21) | 50 h v2 ×10 (R22) | 45 h v3 ×10 (R24) |
|---|---|---|---|---|---|
| test_human_tts: TTS voice, same scripts (control) | **0.41** (24% / 58%) | **0.79** (69% / 90%) | **0.83** (75% / 91%) | **0.79** (61% / 97%) | **0.84** (82% / 86%) |
| test_human: your voice, scripted | **0.33** (16% / 50%) | **0.67** (51% / 84%) | **0.79** (73% / 86%) | **0.74** (57% / 91%) | **0.83** (78% / 87%) |
| test_human_free: your voice, own wording | **0.28** (14% / 43%) | **0.73** (71% / 75%) | **0.68** (65% / 71%) | **0.67** (53% / 80%) | **0.73** (69% / 76%) |

## Details

### test_human_tts

| Metric | base | 1 h ×10 (R17) | 10 h ×10 (R21) | 50 h v2 ×10 (R22) | 45 h v3 ×10 (R24) |
|---|---|---|---|---|---|
| silent when due (n=51) | 24% | 69% | 75% | 61% | 82% |
| speaks when due (n=105) | 58% | 90% | 91% | 97% | 86% |
| cue release: spoke (n=33) | 55% | 88% | 91% | 94% | 82% |
| timed re-ask (n=18) | 61% | 83% | 83% | 94% | 67% |
| ordinary answered (n=54) | 59% | 93% | 94% | 100% | 94% |
| silent in B1_double holds (n=6) | 83% | 50% | 33% | 0% | 0% |
| silent in B1_rescind holds (n=3) | 67% | 100% | 100% | 100% | 100% |
| silent in B1_std holds (n=12) | 33% | 0% | 25% | 8% | 75% |
| silent in B2_std holds (n=18) | 0% | 94% | 100% | 100% | 100% |
| silent in B3_early holds (n=3) | 0% | 100% | 100% | 100% | 100% |
| silent in B3_rescind holds (n=3) | 33% | 100% | 100% | 67% | 100% |
| silent in B3_std holds (n=6) | 0% | 100% | 100% | 67% | 100% |
| reacted to D1a (n=27) | 85% | 0% | 0% | 0% | 0% |
| reacted to D1b (n=12) | 100% | 0% | 0% | 0% | 0% |
| reacted to D1c (n=6) | 67% | 17% | 0% | 0% | 0% |
| reacted to D2a (n=12) | 83% | 0% | 0% | 0% | 0% |
| reacted to D2b (n=3) | 67% | 0% | 0% | 0% | 0% |
| reacted to D2c (n=3) | 67% | 100% | 33% | 0% | 0% |
| reacted to D3a (n=18) | 83% | 0% | 0% | 0% | 0% |

### test_human

| Metric | base | 1 h ×10 (R17) | 10 h ×10 (R21) | 50 h v2 ×10 (R22) | 45 h v3 ×10 (R24) |
|---|---|---|---|---|---|
| silent when due (n=51) | 16% | 51% | 73% | 57% | 78% |
| speaks when due (n=105) | 50% | 84% | 86% | 91% | 87% |
| cue release: spoke (n=33) | 48% | 73% | 73% | 85% | 76% |
| timed re-ask (n=18) | 33% | 83% | 89% | 94% | 83% |
| ordinary answered (n=54) | 56% | 91% | 93% | 94% | 94% |
| silent in B1_double holds (n=6) | 33% | 50% | 33% | 33% | 17% |
| silent in B1_rescind holds (n=3) | 33% | 33% | 33% | 67% | 100% |
| silent in B1_std holds (n=12) | 33% | 17% | 58% | 8% | 58% |
| silent in B2_std holds (n=18) | 0% | 67% | 89% | 94% | 100% |
| silent in B3_early holds (n=3) | 0% | 100% | 100% | 33% | 100% |
| silent in B3_rescind holds (n=3) | 33% | 100% | 67% | 100% | 100% |
| silent in B3_std holds (n=6) | 0% | 33% | 100% | 50% | 83% |
| reacted to D1a (n=27) | 70% | 0% | 0% | 0% | 0% |
| reacted to D1b (n=12) | 75% | 25% | 0% | 0% | 0% |
| reacted to D1c (n=6) | 83% | 0% | 0% | 0% | 0% |
| reacted to D2a (n=12) | 83% | 17% | 0% | 0% | 0% |
| reacted to D2b (n=3) | 67% | 0% | 0% | 33% | 33% |
| reacted to D2c (n=3) | 0% | 0% | 33% | 67% | 33% |
| reacted to D3a (n=18) | 94% | 6% | 0% | 0% | 0% |

### test_human_free

| Metric | base | 1 h ×10 (R17) | 10 h ×10 (R21) | 50 h v2 ×10 (R22) | 45 h v3 ×10 (R24) |
|---|---|---|---|---|---|
| silent when due (n=51) | 14% | 71% | 65% | 53% | 69% |
| speaks when due (n=51) | 43% | 75% | 71% | 80% | 76% |
| cue release: spoke (n=33) | 45% | 76% | 67% | 73% | 73% |
| timed re-ask (n=18) | 39% | 72% | 78% | 94% | 83% |
| silent in B1_double holds (n=6) | 67% | 67% | 33% | 33% | 17% |
| silent in B1_rescind holds (n=3) | 0% | 0% | 67% | 33% | 100% |
| silent in B1_std holds (n=12) | 25% | 33% | 17% | 8% | 8% |
| silent in B2_std holds (n=18) | 0% | 94% | 100% | 89% | 100% |
| silent in B3_early holds (n=3) | 0% | 100% | 67% | 0% | 100% |
| silent in B3_rescind holds (n=3) | 0% | 100% | 100% | 100% | 100% |
| silent in B3_std holds (n=6) | 0% | 83% | 67% | 67% | 100% |
| reacted to D1a (n=27) | 85% | 4% | 0% | 7% | 0% |
| reacted to D1b (n=12) | 58% | 0% | 0% | 0% | 0% |
| reacted to D1c (n=6) | 83% | 17% | 0% | 33% | 0% |
| reacted to D2a (n=12) | 75% | 0% | 0% | 0% | 0% |
| reacted to D2b (n=3) | 67% | 33% | 0% | 33% | 0% |
| reacted to D2c (n=3) | 33% | 33% | 67% | 33% | 100% |
| reacted to D3a (n=18) | 78% | 0% | 0% | 0% | 0% |
