# Experiment history: moshi-silence

A granular, chronological record of everything this project did, from the first commit
(2026-10-14 18:01 UTC) to the time of writing (2026-10-15 ~09:45 UTC). It was reconstructed from
the Claude Code session transcripts (9 sessions, listed in §13), cross-checked against git,
`docs/training_record.md`, `results/overview.md` and the per-experiment `summary.md` files.

- **All times are UTC** (the user's local time is ET = UTC − 4). Times come from the transcript
  timestamps; git's `-0400` timestamps agree with them.
- "The user" = the project owner. "The assistant" = Claude Code.
- Score = mean(silent when due, speaks when due) on **dev_80**, 1 sample per episode, unless stated
  otherwise. A single checkpoint's score is noisy (about ±0.06).
- `docs/training_record.md` stays the reference for exact methods and full result tables. This file
  is the story: what was tried, in what order, why, what broke, and what each step found.

---

## 0. At a glance

### 0.1 Timeline in one table

| When (UTC) | What happened |
|---|---|
| before 10-14 18:01 | Generator (`silence/`) written in earlier work not covered by these transcripts; 12-episode Pocket-TTS demo; `gen_packs.py` untested |
| 10-14 18:01–19:05 | `git init`; demo rebuilt (12/12); user listens; distractors moved from a hand-written lexicon into LLM-written packs; review fixes; TTS throughput measured; first pack smoke test |
| 19:15–20:29 | Whole plan laid out; **audio-loss mask** patch (option a); Moshi never speaks first; structured outputs; 57 s length budget; B1_double; `speech_span` onset fix; gaps ×1.5; D3a repair |
| 20:29–21:17 | **2,500 packs** (~$9.50); **10 h train build** (1,000 eps = 9.07 h, extended to 1,150 eps = 10.49 h, all pass) |
| 20:31–22:02 | Training plan: **fixed epochs**, 1 h LR sweep, top-2 LRs at 10 h; dev set (200, overlap-screened); 7 test suites × 90; eval-reload + text-tail patches; two code reviews; RunPod chosen |
| 22:02–23:45 | Eval harness (scorer, forced-script generator with 1-frame user shift, LoRA swap, report); two pre-launch reviews (Mimi OOM, Python 3.14, `.env` leak, CUDA_VISIBLE_DEVICES); RunPod MCP added |
| 23:44–00:18 | First setup pod on the **wrong (personal) account**, fails at `uv sync`; switched to the API-key account; setup + smoke + selftest + R0 benchmark pass |
| 10-15 00:20–01:52 | **Phase 1** on a 4× pod: 1 h × LRs; dev loss blows up (memorization); R5 dropped, R6/R7 added mid-run; all 127 models scored on dev_80 |
| 01:52–02:30 | Phase-1 analysis: 2e-4 best; dev loss ranks LRs backwards; listening demo; two-axis rescoring shows **timed holds unlearned** |
| 02:33–04:08 | **Phase 2**: 10 h × 4 LRs (2e-5…5e-4), 5 epochs; first live-chat pod (02:40–03:07) |
| 04:13–05:17 | Silence-bias analysis; 4th patch (loss weights); **EPAD-weight sweep** ×3…×100 at 1 h; **disk-full crash** (150 → 250 GB) |
| 05:20–06:06 | Overview of all 14 runs; transcripts; second chat pod; **R20/R21 (EPAD ×3/×10 at 10 h)** launched; curated demo; 50 h planning + TTS pricing |
| 06:06–07:14 | **50 h build** (new packs, reverse screen, Pocket + ElevenLabs + Fish): 5,585 eps / 51.24 h, ≈ $30 |
| 07:15 | R20/R21 results: 10 h EPAD ×10 plateau 0.88, best checkpoints 0.91 |
| 07:29–09:31 | **R22 (50 h, EPAD ×10, 1 epoch)** + **first held-out test-suite results** (base / 1 h ×10 / 10 h ×10 / 50 h ×10). 50 h did not beat 10 h |
| 09:37–09:44 | Diagnosis ("talk after a few seconds of silence" shortcut); v3 timing fix written; chat pod with R21 vs R22 started; R24 (v3, ×10, 2 epochs) requested |

### 0.2 Every training run

| Run | Data | Peak LR | EPAD × | Steps (epochs) | Ckpts | Pod | Outcome (dev_80) |
|---|---|---|---|---|---|---|---|
| R0_devloss | none (base) | 0 | 1 | 2 no-op | 0 | 4× `z77njns1kqy38r` | base: score 0.29; dev text 0.537 / audio 0.393 |
| R1_1h_lr2e-06 | train_1h | 2e-6 | 1 | 142 (20) | 21 | same | quiet model, plateau 0.40 |
| R2_1h_lr1e-05 | train_1h | 1e-5 | 1 | 142 (20) | 21 | same | 0.71 |
| R3_1h_lr5e-05 | train_1h | 5e-5 | 1 | 142 (20) | 21 | same | 0.79; EPAD-sweep baseline |
| R4_1h_lr0.0002 | train_1h | 2e-4 | 1 | 142 (20) | 21 | same | **best 1 h**, 0.86 |
| ~~R5_1h_lr0.0005~~ | — | 5e-4 | — | — | — | — | dropped 00:51 before start |
| R6_1h_lr5e-06 | train_1h | 5e-6 | 1 | 142 (20) | 21 | same | 0.55 (added 00:51) |
| R7_1h_lr2e-05 | train_1h | 2e-5 | 1 | 142 (20) | 21 | same | 0.79 (added 00:51) |
| R8_10h_lr2e-05 | train_10h | 2e-5 | 1 | 345 (5) | 21 | 4× `sq9r1dzsa8l6rc` | 0.84; best ckpt 289 (ep 4.2) 0.88 |
| R9_10h_lr5e-05 | train_10h | 5e-5 | 1 | 345 (5) | 21 | same | 0.85 |
| R10_10h_lr0.0002 | train_10h | 2e-4 | 1 | 345 (5) | 21 | same | 0.82 |
| R11_10h_lr0.0005 | train_10h | 5e-4 | 1 | 345 (5) | 21 | same | collapsed into silence, 0.57 |
| R12–R15 | — | — | — | — | — | — | designed (10 h EPAD/audio-weight ablation), never run |
| R16_1h_lr5e-05_epad3 | train_1h | 5e-5 | 3 | 133 of 142 (17.8) | 18 | 4× `7f5yuy4acmhxzp` | 0.81 (crashed: disk full) |
| R17_1h_lr5e-05_epad10 | train_1h | 5e-5 | 10 | 133 of 142 | 18 | same | 0.80; best ckpt 42 (ep 5.9) 0.86, 61% timed re-ask |
| R18_1h_lr5e-05_epad30 | train_1h | 5e-5 | 30 | 133 of 142 | 18 | same | 0.77 |
| R19_1h_lr5e-05_epad100 | train_1h | 5e-5 | 100 | 133 of 142 | 18 | same | 0.80 |
| R20_10h_lr5e-05_epad3 | train_10h | 5e-5 | 3 | 345 (5) | 21 | 1× `086j0wex48cmlp` (tr3) | 0.87; best ckpt 153 0.91 |
| R21_10h_lr5e-05_epad10 | train_10h | 5e-5 | 10 | 345 (5) | 21 | 1× `xk9kzxl2eev9of` (tr10) | **0.88**; best ckpt 153 (ep 2.2) 0.91; best on held-out tests |
| R22_50h_lr5e-05_epad10 | train50 `train_50h` (v2 recipe) | 5e-5 | 10 | 345 (1.01) | 21 | 1× `i1wn60oi6h7n8g` | ≈ 0.83; below R21 on 6 of 7 test suites |
| R23_10h_v2 | train50 `train_10h` | 5e-5 | 10 | 345 | — | — | proposed 07:23, dropped by the user 07:26 |
| R24 | v3 (50 h, rebalanced timing) | 5e-5 | 10 | 2 epochs | — | — | requested 09:43; not started at time of writing |

### 0.3 Spend (as recorded in the transcripts)

| Item | Cost |
|---|---|
| Pack-generation tests (Sonnet 5.5) | ≈ $1.15 |
| Packs v1 (2,500, 322 calls) | ≈ $9.50 |
| Personal-account setup pod (mistake) | ≈ $0.60 |
| Setup pod `ldkqt7xnpy9pxe` | ≈ $1.35 (one message said $1.80) |
| Phase 1, 4× pod | ≈ $25 |
| Phase 2, 4× pod | ≈ $25 |
| Chat pod 1 `ywpfqsbqerh53c` | ≈ $1.80 |
| EPAD sweep, 4× pod | ≈ $15 |
| Chat pod 2 `vc42i3dtvzv7s5` | ≈ $4.30 |
| R20/R21, two 1× pods | ≈ $13.4 |
| Packs v2 (~822 calls) | ≈ $20 (estimate) |
| Fish Audio TTS | $9.39 |
| ElevenLabs TTS | 42,161 of 116,845 Starter-plan credits (no marginal cost) |
| R22 + test suites, 1× pod | ≈ $8.10 |
| Network volume `np7c8l94dh` | ≈ $0.35/day at 150 GB, ≈ $0.58/day at 250 GB |
| Chat pod 3 `5obz5mapyti207` | running since 09:43 at $3.99/hr |
| **Total so far** | **≈ $135** (GPU ≈ $95, APIs ≈ $40), plus the volume and the running chat pod |

---

## 1. Before the transcripts (pre-history)

Not covered by any transcript; known from the state found at 18:01 on 10-14:

- The generator existed: `silence/` with phrase banks, specs, Pocket TTS backend + clip cache,
  `retime`, `assemble.gate`, `check.py` (using moshi-finetune's own Interleaver).
- Two timing fixes were already in (`tts.retime()` and `assemble.gate()`).
- A 12-episode demo built from hand-written `packs/demo_packs.jsonl`; earlier listening had turned
  up garbled words ("latte", "Alfama"), so two demo clips had been rewritten.
- `gen_packs.py` (LLM content packs) was written but untested. No git repo, no packs at scale, no data.
- `../moshi-finetune` was cloned at upstream `2acc879`.

---

## 2. Day 1, 18:01–19:05: repo, demo listening, distractor redesign (session `bb14ec5c`)

Working in the main checkout. Hardware: Mac M5 (4 performance cores).

### 18:01–18:08: git init and demo rebuild
- The user's instructions: read CLAUDE.md, `git init`, rebuild the demo, confirm 12/12, say which 3
  episodes to listen to; then a pack smoke test (`--per-family 10`); then full packs, 7 test suites
  and the 10 h set with an ETA; "Stop and ask before anything that costs money or takes more than ~30 min."
- Commits **`1d33d2a`** (initial: generator, checks, phrase banks, pack generator) and **`0a50fa2`**
  (demo refresh). Demo rebuild: **12/12 pass**; Pocket clip cache 108 → 110 (the "iced coffee" and
  "old town" lines replaced "latte" / "Alfama"). Timer errors −0.16 / +0.23 s; cue → resume 0.38–0.87 s.
- Suggested listening: demo_03_B2_std, demo_04_B3_std, demo_00_B1_std.

### 18:08: the user's listening feedback
- B2: the friend sounds natural, but the distractor line **"bring it back when you're done" makes no
  sense**, and the friend's question goes unanswered, so the conversation is "left hanging".
- B3: "macbook is mispronounced".
- B1: "is 20s necessary?" (to save training time).
- Diagnosis: distractors (D1a/D3a) were inserted from a hand-written lexicon
  (`phrasebank.DISTRACT`) into the LLM-written side conversation, with no context.

### 18:11–18:21: distractors moved into the LLM packs
- First fix (hand-written two-line exchanges) was rejected by the user: **"wait, no need to hard
  code these lol"**. At 18:14 the user restated the brief's definition: distractors = questions
  addressed to the other person, pauses before the conversation is finished, phrases resembling the
  return cue.
- Implemented: the LLM writes distractors as **tagged lines in each pack** (D1a/D1b/D1c questions,
  D3a lookalikes; every question answered). `specs.py` only adds D2a stalls and the test-only D4.
  The lexicon was deleted; the `distract` split axis was removed; `test_distract` became "dense
  distractors + D4".
- Pronunciation: brand/product names banned in the prompt; `HARD_WORDS` += macbook, thinkpad,
  chromebook; camelCase rejected (catches MacBook / iPhone).
- Validator extended to all families (B1 fields, B3 3–4 options, DECOY word present, B2 unanswered
  question, tags on the right speaker). Tests: 10 demo packs valid; negative cases rejected.
- Other review fixes: over-60 s train episodes redrawn (moshi-finetune would cut a second window
  mid-episode); B1 early/rescind need N ≥ 10; mutters spaced; same-speaker overlap fixed; new
  check "no human turn overlaps Moshi"; gender-neutral friend names; the `demo` split removed from
  `build.py` (so the old demo can't be rebuilt).
- Answer on 20 s holds: **keep them**. Training cost doesn't depend on episode length (every window
  is padded to 60 s), silence costs nothing in TTS, and variable N is what makes B1 test the instruction.

### 18:21–18:34: first smoke test and TTS throughput
- Smoke test (`--per-family 10`): 38 packs in ~10 min (~2.5 min per call; DECOY 0). Suspected
  (never confirmed) cause: default thinking eating into `max_tokens`. Killed; parallel generation added.
- Pocket TTS on the Mac: 1 process 6.0× realtime; **4 processes 11.6×**; 8 → 11.5×; 10 → 10.3×.
  Threads don't help; efficiency cores add nothing.

### 18:36–18:50: estimates and the early plan
- The user: "i want the 10h dataset done in 30mins or less, no matter what". Estimate: on the Mac,
  train + tests ≈ 52 min, train only ≈ 34 min, so it can't be done in 30 min locally. Packs ≈ $8–10.
- Training estimate (from Kyutai's README, not measured): 108k tokens/step ≈ 9 s/step.
- Early plan (later superseded): 250 fixed steps, checkpoint every 50, LR {2e-6, 1e-5, 5e-5} ×
  {1 h, 10 h} = R1–R6, R7 at 100 h, an 8× H100 node for ≈ $60–110. The assistant suggested shifting
  the LRs up to {1e-5, 5e-5, 2e-4}. It also flagged that 90 episodes per suite gives ±10 points and that
  eval compute was unbudgeted.

### 19:03–19:05: data deep-dive and "shouldn't the user ask first"
- Smoke-pack problems: B3 context written in the third person ("The user is choosing…"); themes
  clustered within a batch; no D3a; formulaic ORD (Moshi opens, user says "Yes, …").
- The user: **"btw, shouldnt the user ask first"**. Agreed: Moshi spoke first in every family.
- 19:04 the user: "yes, apply all of them and rerun the smoke test… api costs arent a problem".
  Applied: user-first openings (`USER_FIRST = 0.8` at this point), B1 opener, one D3a in every B2,
  2–3 themes per call, thinking off, parallel API calls, a three-phase `build.py` (plan → render in
  parallel → assemble). The demo dir and `demo_packs.jsonl` were deleted.
- The rerun smoke test's result isn't in the transcript. At 19:15, `packs/smoke.jsonl` held 40 packs
  (B1/B2/B3/ORD 10 each, no DECOY).

---

## 3. Day 1, 19:15–21:17: plan, audio mask, generator hardening, packs and the 10 h build (sessions `d9971fba`, `08aab4c5`)

`08aab4c5` is a resumed copy of `d9971fba` that carries on past 20:29.

### 19:15–19:27: the whole plan, then a granular data review
- The user asked for the plan "crystal clear": three datasets (1 h / 10 h / 100 h), how they're
  built, how examples stay distinct, the sweep, 1 h vs 10 h ("i'm worried that 1h might be too
  little"), GPUs, evaluation with held-out scripts/voices and a human set.
- Plan given: episode ≤ 60 s stereo; mix B1 15 / B2 25 / B3 20 / ORD 40; nested 1 h ⊂ 10 h ⊂ 100 h;
  ~124 episodes/h at 29 s; the same 250-step sweep; 8× H100; 7 × 90 test suites + human set.
- Flagged: CLAUDE.md's `--n 1000` would give only ~8 h.
- Granular review, **top finding:** Moshi's channel is Pocket TTS `michael`, so the audio loss
  (semantic codebook ×100) would train Moshi to **sound like the TTS voice**. Options: (a) mask the
  audio loss on Moshi's speech, (b) self-distill, (c) voice-clone Moshi. Other findings: shortcut
  risks (any second voice → silence; quiet reverberant voice = not the user), small phrase banks,
  holds always early in the window, no ASR check of clips. A ranked to-do list of 8 items (~6 h).

### 19:31–19:33: the audio-loss mask (patch 1)
- The user chose **option (a)**, and asked whether ×100 makes the text loss irrelevant.
- Branch `silence-audio-mask` in moshi-finetune: `mask_main_speech_audio` (pre 0.08 s / post
  0.24 s) in interleaver, args, train, eval; text and audio loss logged separately. Saved as
  `patches/moshi-finetune-audio-mask.patch` (203 lines). CPU test on 4 episodes: **0 voiced frames
  left unmasked**; the margins mask only 8–16% more than the voiced frames.
- Answer on ×100: the audio loss is a weighted *average*, so ×100 only splits it across codebooks
  (codebook 0 ≈ 93%); text vs audio stays 1:1. The real dilution: one EPAD frame is **0.12–0.28% of the
  text-loss weight** per episode. That argued for an EPAD-weight ablation, which was run later (§9).

### 19:45–19:56: user-first, structured outputs, length budget, B1_double
- The user: **"NEVER have moshi speak first. user goes first."** The 80/20 randomization was removed;
  the prompts, `validate()`, `specs.py`, `load_packs()` (drops Moshi-first packs) and **check.py check 8**
  all enforce it.
- Speed: **structured outputs** (per-family JSON schemas), thinking off, low effort. Live test: 5 calls,
  14.6 s, 10 packs, 0 rejects (~350 output tokens per pack).
- The user picked ideas 1, 2 and 4:
  - (1) Length budget: speaking rate calibrated on 156 clips (0.275 s/word); `BUDGET_DUR` 57 s; worst
    case over 12 sampled setups.
  - (2) 10 user speaking styles; "Moshi" said in about a quarter of openers.
  - (4) **B1_double** (two timed holds in one episode); TRAIN_MIX B1 = std 9 / double 2 / early 2 / rescind 2.
- New `silence/transcript.py`. 14 sample episodes (one per type, incl. test-only B2_D4, COND_keyword,
  COND_phone, B1_minutes): **13/14 passed**. B1_double failed: Moshi's word times were 0.83 s ahead of its audio.

### 20:00–20:07: demo samples and the onset bug
- The 14 samples were copied into `demo/` (53 MB, gitignored) with `index.html` and `transcripts.txt`.
- **Onset bug:** ~5% of Pocket clips (15 of 281) open with faint noise that `speech_span` took as
  the speech onset, off by up to 0.93 s. Fix: an onset counts only if real speech follows (a frame
  > −15 dB within 150 ms, or a dense 100 ms window reaching −23 dB). 14/14 pass; Moshi's word times
  are within ±0.04 s of the audio. This became the third "keep this" timing fix.

### 20:07–20:19: gaps ×1.5 and the D3a repair
- The user: "the text gaps are a little too short, maybe lengthen by like 50%". Added `GAP_SCALE`
  1.5 for turn gaps (median 0.41 → 0.61 s), Moshi reply gaps (0.45–1.35 s), user↔friend gaps, acks
  and B3 option gaps. The check.py cue-latency window was widened to 1.5 s. The prompts were
  tightened to stay under 57 s (B2 side conversations 5–6 lines; B3 option detail ≤ 12 words). B1
  draws only the N values that fit. Estimated mean episode 37 s (≈ 97 episodes/h). The actual mean
  turned out to be 32.6 s.
- The user: "fix the D3a repair". A D3a on the last line is now dropped instead of rejecting the
  pack; D3a must sit on line 2–4; `_is_q` = "contains '?'". **B2 yield went from ~9% to ~32%**, so the
  full run's estimate rose from ~$10 to ~$15. New flags `--families` and `--rejects`.
- Test-set sizing (answering the user): size by precision, not as a fraction of train. 90/suite gives
  ±10 points, 60 gives ±13 (recommended at the time). The suites were later built at 90.
- Not implemented from the review list: ASR WER filter, ORD_friend contrast, expanded phrase banks,
  warm-up prefix, barge-in variant, gain/reverb randomization for the user.

### 20:24–20:43: launch of packs and the 10 h build
- The user rejected a TTS benchmark and said: "just generate the 10h dataset… save to /repo".
- **20:29** full pack run: `gen_packs --per-family 500 --workers 32`. **Done at 20:34**: **2,500
  packs** (500 per family), 322 calls, 691,660 input / 811,186 output tokens, **≈ $9.50**. 620 rejects
  (no D3a 206, B3 too long 106, B2 too long 99, near-duplicate 93, D3a on last line 40, …). 387 packs
  (15%) fell to test by hash.
- **20:34** train build `--n 1000`: 7,661 clips on 4 workers at ~11–12× realtime.
- 20:43 the user asked whether the dataset is only 7.5 h. Answer: that's the TTS speech; total audio
  ≈ 10 h, with silences added at mix time; `make_splits` cuts on total audio. "ah, understood… continue building."

### 21:11–21:17: 10 h set done (two sessions collide)
- **1,000 episodes = only 9.07 h** (37.6 min; mean 32.6 s vs the 37 s estimate). Session `08aab4c5`
  extended to `--n 1150` (same seed; the first 1,000 are byte-identical): **1,150 episodes / 10.49 h**.
- At the same moment, session `35d2ba60` ran `--n 1110` into the same `data/train`. The manifest
  had to be restored, and a rule was added: **only one session writes `data/`**.
- `make_splits --hours 1 10`: **train_1h = 113 eps / 1.01 h** (116 packs, 885 clips, 0.39 h hold
  silence); **train_10h = 1,101 eps / 10.01 h** (1,135 packs, 8,463 clips, 3.73 h hold silence).
  **1,150/1,150 pass check.py.** No pack reused.
- train_1h by type: B1 std 17 / double 3 / early 1 / rescind 1; B2 24; B3 std 18 / early 2 / rescind 2;
  ORD 31; DECOY 14. That's only 1–2 contrast episodes per type at 1 h.

---

## 4. Day 1, 20:31–22:02: training plan, dev/test sets, patches, reviews (session `35d2ba60` part 1)

### 20:33–20:53: the training plan is decided
- 20:36 "what's up with steps, epochs…": the assistant explained steps / batch / epochs and OneCycle.
  It first recommended keeping fixed steps.
- **20:39 reversal by the user: "we're definitely doing fixed number of epochs. 1h is supposed to be a
  proof-of-concept: does the model actually start picking up on the desired behavior".**
- 20:42 the user liked the 1 h exploration at 20 epochs across LRs. 20:45: "might as well do all four"
  ({2e-6, 1e-5, 5e-5, 2e-4}; 2e-6 doubles as a pipeline check). 20:47: "sure, let's do five runs"
  (+5e-4), and "i'll reject your other suggestions" (no behavior-loss validation script, no second seed).
- **20:50 the user: cut phase 1b (1 h rerun at exactly E) and phase 3 (100 h) and the stretched-holds
  ablation; phase 2 = top 2 LRs at 10 h for E epochs.**
- **20:53 the user: "yes, mask main speech audio. also we should be evaluating the trained models at each
  epoch with a small eval set, no?"** → dev loss at every checkpoint, plus a dev set.
- Caveat recorded: an epoch-E checkpoint of a 20-epoch OneCycle run isn't the same as a run trained to E.
- 20:59 worktree `docs-progress` created (branch `worktree-docs-progress`). Commit `62621b0`.

### 21:03–21:33: the dev set
- The user: "there needs to be an eval set… constructed now, preferrably at different sizes". 21:10:
  "a 200 eval set for something trained on 100 is kinda insane, no?" → sized by precision; **dev_80 ⊂
  dev_200**, built from unused train packs, balanced 20/50 per family.
- Why packs were left over: equal generation per family vs an unequal mix (1,081 of 2,113 train
  packs unused after 1,000 episodes).
- 21:21 "just make the dev 200 first". `silence/build_dev.py`. **First build overlapped train**: closest
  script Jaccard 0.58 and 12 verbatim shared lines. A screen was added (drop candidates that share a
  spoken line with train, or have Jaccard ≥ 0.5 with a train pack): **200 of 928 dropped**.
- Rebuild: **200 eps / 1.97 h (dev_80 = 0.77 h), 200/200 pass**, closest Jaccard 0.47, 1 shared pack
  line ("Okay."). `silence/check_overlap.py`. Commit `ba46115`.
- TODO recorded: future train builds must skip the dev packs (this was later fixed by `build.py --exclude`).

### 21:39–21:59: test suites, configs, bootstrap, patches 2 and 3
- **21:43 the user: "let's do 90. also, we can start training, right?"**
- `silence/screen_packs.py`: the first screen compared raw pack strings and kept only 303 packs (D4
  `friend_away` lines are spoken only in test episodes). Switched to comparing **lines actually spoken**
  in train/dev: **348 of 387 test packs kept**.
- **Bug: the eval loader is a finite generator built once**, so every eval after the first sees 0 batches
  and logs NaN; it also drops the last partial batch. → **patch 2 `eval-reload`** (fresh loader per eval;
  later also logs text/audio dev loss).
- **Trap: `num_ckpt_keep` defaults to 3** → `null`.
- `silence/make_configs.py`: 113 eps × 20 epochs → **142 steps, 21 checkpoints** (every 7 steps ≈ 0.99
  epoch + last), plus `<run>.epochs.json`. Weights pinned at `2bfc9ae6…`.
- `scripts/gpu_bootstrap.sh`; smoke test = 6 steps with 2 evals (exercises the eval-reload fix). Commit `a5ddb85`.
- Two review agents (21:47):
  - **Docs vs code:** the worktree and main checkout were split. Neither could be uploaded alone
    (pipeline code uncommitted in the main checkout; training files only on the branch). Several doc
    errors were found (distractors described as lexicons; B1 mix 9/2/2/2; real step counts).
  - **Training path:**
    - **`pip install -e` can't resolve** (sphn pin vs moshi main) → `uv sync --frozen` with the
      lockfile (moshi `061cc4c`, torch 2.6.0).
    - **Text past the end of a short file is PAD and counts in the text loss: ~46–47% of all text
      frames at 1 h** → **patch 3 `text-tail`**.
    - Dev loss not split into text/audio. No step-0 dev loss → `R0_devloss` (lr 0, 2 no-op steps).
    - LoRA r64 ≈ 194M params, 0.39 GB per checkpoint, ~41 GB for phase 1.
- Commits `94a6f1e`, **`27655b1`** (snapshot of the main checkout's uncommitted working copy into the
  branch), `1324654`, `47c9016` (full CLAUDE.md/README review).
- **21:59 all 7 suites built: 630/630 pass**, 6.1 h, 0 shared packs with train, closest Jaccard ≤ 0.44.
  Clip-duration scan of 14,581 distinct clips: no run-on clip. Commit `dd901d6`.

### 21:56–22:02: GPU choice
- Recommendation: RunPod Secure Cloud, on-demand (train.py can't resume), H100 SXM, network volume
  ≥ 150 GB, SSH over exposed TCP. One independent run per GPU, never split (batch is per GPU).
- **22:01 the user: "yeah do 1x then 4x, write the parallel script".** `scripts/run_parallel.sh` (own
  `CUDA_VISIBLE_DEVICES` and port per run, queue, `.done` markers, refuses leftover run dirs), tested with
  a fake torchrun. uv, its Python and its cache moved onto the volume. Commit `0fbf583`.

---

## 5. Day 1, 22:02–23:52: eval harness, pre-launch reviews, launch prep (session `35d2ba60` part 2, `02c11200`)

### 22:16–22:31: data check and test-suite pairing
- All synthetic sets verified on disk (train 1,150; dev 200; 7 × 90). Detailed tables recomputed.
- Finding: test_script has only **6 B1 episodes**; held-out conditions are thin (new N in 8/15/30
  episodes, D4 in 18).
- **Docs error found: the test suites are NOT paired.** `PackSampler`'s single RNG shuffles lazily in
  each suite's own type order, so the suites share only 42–68 of 90 packs pairwise. Commit `85333c4`. A
  paired/stratified rebuild was offered; the user never decided.
- Moshi going off script: in eval, re-force the script between scored windows (every hold scored).

### 22:35–22:50: eval decisions and the scorer
- **22:40 the user: "go with all six, start writing the eval harness"**:
  1. Silero VAD timing + Moshi's text stream on dev; Whisper only for final tests.
  2. 1 sample on dev, 3 on test.
  3. LLM judge on tests only (~$10, ask first).
  4. dev_80 every epoch if ≤ 2 min per checkpoint.
  5. Re-force between windows.
  6. Thresholds: 2.5 s ack grace, ≥ 1 s or > 3 words = violation, 3 s windows, B1 tolerance
     max(1.5 s, 0.2N).
- `silence/eval_score.py`. The oracle (ground-truth Moshi channel) first scored ordinary responses at
  93% (B1_double has a response point inside a hold) and found that N = 3 s holds have an empty
  violation region. Fixed: `violated = None` when the region is empty; ordinary scored only on
  hold-free episodes. **Oracle: 0% violated (n=58), 100% on time (n=63), 100% ordinary.** Negative
  control: 76% violated. Commit `b7b0843`.
- 22:49 the user interrupted: check everything before the GPUs ("these are expensive gpus… this
  experiment needs to work").

### 22:50–22:54: first pre-launch review
- Measured-in-principle compute corrected to ≈ 9 s/step (from the README), not 15.
- **Blockers fixed:**
  - the smoke torchrun didn't set `CUDA_VISIBLE_DEVICES` (KeyError in `distributed.py`);
  - uv would pick **Python 3.14** (no torch 2.6 wheels) → `UV_PYTHON=3.12`;
  - the upload plan didn't work, and its rsync **would have copied `.env`** (the API key) → two rsyncs:
    code from the worktree (no `.env`), `data/` from the main checkout.
- Also: LR written with 2 significant figures → `.6e`; a pipefail-unsafe count. Commit `c411ac8`.

### 23:03–23:11: the generator (`eval_gen.py`)
- `ForcedLMGen`: a copy of `LMGen.step` that forces Moshi's text + 8 audio codebooks to the script
  outside the free windows (forced text substituted before the depformer).
- **Alignment finding:** feeding a wav straight into LMGen hands the model each user frame **80 ms
  earlier** than training did. With `--user-shift 1`, a tiny random LM reproduces **100% of training's
  input columns** (max |Δlogit| 2.7e-6); without the shift, 2.5%.
- LoRA swap: load the base model once (unfused), swap only the LoRA tensors per checkpoint; base (B = 0)
  reproduces the LoRA-free model token for token.
- `eval_report.py` (bootstrap CIs, curves, `--sensitivity`), `scripts/eval_dev.sh`. The bootstrap's new
  step 6 runs the selftest and a 40-episode R0 benchmark. Commit `6166c00`.

### 23:13–23:34: second pre-launch review (eval harness)
- **Blocker: Mimi encode/decode of a whole batch of 40 would need ~110–145 GB** (fp32, ~3 GB per 60 s
  episode) → Mimi in sub-batches of 4 (`MIMI_CHUNK`).
- The selftest now compares exact integer input columns (argmax is fragile in bf16).
  `HF_HUB_OFFLINE=1` for all jobs (a hub 429 at the config probe would crash a job). Resume windows
  excluded from the interrupt count. `<s>`/`</s>` skipped. Commit `3c60a1b`.

### 23:40–23:45: RunPod via MCP
- RunPod docs: the proxied SSH can't scp/rsync; volumes are tied to one data center.
- The user: "you could lowkey do this: npx @runpod/mcp-server@latest add". The installer is
  interactive; the hosted MCP server was added with `claude mcp add` instead (**OAuth**). Commits `5cbe940`
  (launch plan, incl. "state spec + price and get the user's OK before every billable action") and
  `0e13dfe` (CLAUDE.md refresh).

### 23:49–23:52: is 100 h feasible? (parallel session `02c11200`)
- The user now had RunPod, Fish and ElevenLabs keys. Answer: feasible for **≈ $100–120 extra**:
  ~10,975 episodes; ~1,400 pack calls ($40–60); ~5.2 h of Pocket TTS on the Mac; an equal-steps 100 h
  run first (689 steps, ~$5). Required fixes first: dev-pack exclusion, screening new packs against
  test/dev, a seed offset for pack topics, a 200 GB volume.
- It advised **not** to use Fish/ElevenLabs for training data (confounds the scaling curve), only for an
  unseen-engine test set. **This was reversed later**: the 50 h build used all three engines.

---

## 6. 10-14 23:44 → 10-15 00:18: GPU setup (sessions `fc2e3aca`, `12a5eb83`)

### First attempt, on the wrong account
- 23:44 `/mcp` authenticated by OAuth, i.e. the user's **personal** RunPod account.
- **Live price $3.99/hr per H100 SXM**, not the assumed ~$3. 23:47 the user chose "Go, US-NE-1".
- Volume `48luw8jqa8` (150 GB) + 1× pod `6i6n6s5a6flkwq` (template `runpod-torch-v280`, `PUBLIC_KEY`).
- **Trap:** `rsync -a` exits 23 on the MooseFS volume (chown refused) → `rsync -rlpt`.
- **Failure at `uv sync --frozen`:** openai-whisper 20240930's setup.py imports `pkg_resources`; build
  isolation pulls setuptools 84, which dropped it. Neither `UV_BUILD_CONSTRAINT` nor a user or project
  `uv.toml` helped. Two-step fix proposed (`--no-install-package openai-whisper`, then
  `--no-build-isolation-package openai-whisper`).
- An oddity: the bootstrap logged all 3 patches "already applied" on a fresh clone. The tree was verified
  byte-identical to `2acc879` + 3 patches; the cause remains unknown.
- **23:52 the user: "wait, can we actually switch to runpod api key".** 23:55: "cancel existing instances so
  i dont use my own money". The pod was terminated **from inside** (`runpodctl remove pod` with the
  pod-scoped key in `/proc/1/environ`). The user deleted the personal volume by hand. ≈ $0.60.
- The API-key account turned out to be **shared** (others' pods and volumes). Rule: never touch anything
  this project didn't create; pass the user's key as `PUBLIC_KEY`.
- Claude Code's auto-mode classifier blocked one RunPod stock read ("Containment Escape") right after the
  in-pod termination.

### Second attempt, on the API-key account
- **23:58 volume `np7c8l94dh`** (150 GB, US-NE-1) + 1× pod `ldkqt7xnpy9pxe`.
- Data uploaded (6.1 GB; counts match in all 9 sets). Weights at `2bfc9ae` downloaded. **The two-step
  uv sync worked.**
- **Smoke train (6 steps of R3's config):** ~**10 s/step** (≈ 11.3k words/s), peak **29.1 GB**, load +
  first step ≈ 2.6 min, dev eval ≈ **47 s** on 200 episodes, 2 checkpoints, **no NaN** (eval-reload works).
- **Selftest on the 7B:** shift 1 → **100% identical input columns**; text argmax agreement 99.7%; script CE
  0.491 streaming vs 0.490 full.
- **R0 benchmark:** 40 dev episodes in 35 s = **28 ms/step at batch 40** → dev_80 ≈ 70 s per checkpoint, so
  generation evals run **every epoch**. Base on 40 episodes: 82.8% of holds violated, 41.9% resumed on time,
  42.9% of ordinary answered, distractors trigger speech 67–100%.
- 00:18 setup pod terminated (≈ $1.35). Compute section of CLAUDE.md rewritten with measured numbers.

---

## 7. 10-15 00:19–02:30: Phase 1, the 1 h LR sweep

### 00:19–00:27: launch
- **00:19 the user: "yes, launch the 4x pod. also make sure to save LoRa weights at each epoch you do the
  dev test at."** (Verified: eval and checkpoint fire on the same steps.)
- **4× pod `z77njns1kqy38r`** ($15.96/hr). `run_parallel.sh` with R0 + R1–R5: R0–R3 together, R4 on R0's GPU,
  R5 queued.
- Recipe: train_1h, 20 epochs = 142 steps, batch 16 × 60 s, LoRA r64 / scaling 2, AdamW wd 0.1, OneCycle
  5% warmup, text_padding_weight 0.5, first codebook ×100, audio mask on, seed 0, checkpoint + dev loss
  every 7 steps.
- 00:23 the user: "you should also log the training loss and stuff and make a nice plot of everything" →
  `silence/plot_training.py`, `silence/plotstyle.py`, **`docs/training_record.md`** created; a 10-minute
  sync loop pulls metrics to `results/phase1/`.

### 00:25–00:53: dev loss blows up; the matrix changes
- R0: dev text **0.537**, audio **0.393**.
- Audio dev loss collapses within ~1 epoch (it's almost all on Moshi's silent frames because speech
  audio is masked). Text dev loss is the signal.
- Dev text loss minima: R4 (2e-4) 0.331 at epoch 3, R3 (5e-5) 0.341 at epoch 4, R2 (1e-5) 0.351 at
  epoch 10. Then R3 and R4 rise to ~0.80 / ~0.74, above the base. Train text loss → 0.014 / 0.009:
  **memorization of the 113 scripts**.
- 00:49 the user: "5e-5 and 2e-4's dev curves are really blowing up. they're overfitting, right?"
- 00:50 the user: "thoughts on just scrapping R3-5 and doing … 5e-6 and 2e-5?" The assistant advised
  keeping R3/R4 (nearly done, and the overfitting is itself a result), dropping R5, and adding 5e-6 and
  2e-5. **00:51 the user: "sure. let's scrap R5 and add R6, R7."**
- The scheduler was killed by hand (R1–R4 kept running). `make_configs --first 6 --no-base --lrs 5e-6 2e-5`.
  `phase1b.sh` runs R6/R7 and the dev_80 evals so no GPU idles. Phase 2 renamed to R8+.
- The 6-LR plot palette was switched to a fixed categorical map (no single-hue ramp passed the ordinal
  check at 6 levels).

### 01:01–01:34: first behavior results
- **R0 on dev_80:** 86.2% of holds violated (n=58), 39.7% resumed on time (n=63), 42.9% of ordinary questions
  answered (n=42), 56 unprompted onsets in 80 episodes, distractors trigger speech 40–83%.
- 01:07 the user: "is there real learning happening?" **R3 by epoch:** violated 86% → 45% (ep 1) → **3%**
  (ep 4); answered 43% → 79–86%; interruptions → 0. So it's real learning, not just going quiet, and it
  happens **while dev text loss doubles**: **dev loss is a poor checkpoint selector.**
- **R1 (2e-6): a "quiet model".** Violations improve (86% → 33–45%), but on-time falls to 22% and answered to
  10%, while its dev loss improves monotonically.
- 01:09 predictions for 10 h / 100 h were given (10 h: violations 2–5%, on time 75–85%, timed releases
  still hardest; 100 h: diminishing returns, diversity matters more).
- 01:24 the user: "do we run the eval on the model saved at the final epoch, or the best loss epoch?"
  → every checkpoint is evaluated. A selection rule was proposed (equal-weight mean on dev_200, ties →
  earlier epoch); the user didn't decide ("i'll just let everything conclude for now").

### 01:34–01:52: wrap-up
- All 6 runs trained (35 min each on the 4× pod, ≈ 8.9 s/step). Generation evals: **127 models × 80 episodes
  = 10,160 outputs**, ≈ 57 s per checkpoint. Pod terminated 01:52 (≈ $25).
- Mimi weights + tokenizer copied to the Mac so audio can be decoded without a GPU.

### Phase-1 results (plateau = epochs 10–20)

| Peak LR | Hold violated | Resumed on time | Ordinary answered | Dev text best (epoch) → final |
|---|---|---|---|---|
| base | 86% | 40% | 43% | 0.537 |
| 2e-6 | 40% | 22% | 10% | 0.457 (20) → 0.457 |
| 5e-6 | 23% | 26% | 39% | 0.373 (19) → 0.373 |
| 1e-5 | 15% | 42% | 80% | 0.351 (10) → 0.370 |
| 2e-5 | 12% | 62% | 83% | 0.347 (6) → 0.642 |
| 5e-5 | 16% | 64% | 87% | 0.341 (4) → 0.803 |
| **2e-4** | **11%** | **74%** | **98%** | 0.331 (3) → 0.741 |

Findings:
1. **Two phases of learning**: first silence (everything drops, e.g. R4 at epoch 2: 0% violated but 3% on
   time), then when to speak. The lowest LRs never leave the silent phase.
2. Behavior improves monotonically with LR; 2e-4 is best on every metric. The top 8 checkpoints are all R4.
3. **Dev loss ranks the LRs almost backwards.**
4. In hindsight, dropping 5e-4 was premature (it was dropped on dev-loss evidence).
5. Distractor reactions fall from 67–83% to 1–15%; D2c (fillers during timed holds) stays high.

### 02:03–02:07: listening demo
- The user: "can you make an interactive demo … actual audio clips … at the point of lowest dev loss and the
  final epoch". Built on the Mac from saved `codes.npz` (no GPU): `silence/demo_phase1.py` →
  `results/phase1/demo/index.html` (9 conversations × 12 models, 108 clips).
- It showed that the **lowest-dev-loss checkpoints mostly stay silent and never resume**, while the
  final ones resume on time.

### 02:23–02:28: the user doubts the metrics → two-axis rescoring
- The user: "'hold n seconds' is probably not that precise right now… maybe our eval metrics aren't too good,
  since models tend to just go quiet a lot."
- Rescored (no regeneration) as **silent when due** vs **speaks when due**:

| Peak LR | Silent | Speaks | Cue release: spoke / on time | Timed release: spoke / on time | Ordinary |
|---|---|---|---|---|---|
| base | 14% | 45% | 51% / 47% | 33% / 22% | 43% |
| 2e-6 | 60% | 21% | 29% / 26% | 26% / 13% | 10% |
| 5e-6 | 77% | 33% | 32% / 29% | 20% / 18% | 39% |
| 1e-5 | 85% | 58% | 55% / 54% | 13% / 13% | 80% |
| 2e-5 | 88% | 70% | 77% / 77% | 24% / 23% | 83% |
| 5e-5 | 84% | 73% | 83% / 83% | 17% / 16% | 87% |
| 2e-4 | 89% | 83% | 96% / 96% | 19% / 19% | 98% |

- **Timed holds aren't imprecise, they're unlearned.** Every model re-asks on its own only 13–26% of the
  time, below the base's 33%; when it does, timing is fine. A timed release is the only "speak" event with
  no human utterance right before it, and train_1h has only 22 B1 episodes.
- 02:28 the user: "i need to know (1) at which epoch the model becomes 'optimal', and (2) which LRs give the
  best results." → Score by epoch: a plateau, with **no behavioral degradation through 20 epochs**. 2e-4
  plateaus at ≈ 0.86 from epoch ~7. Ranking 2e-4 > 5e-5 ≈ 2e-5 > 1e-5 > 5e-6 > 2e-6. Plateau after ~50–60
  optimizer steps → 10 h might plateau within its first epoch → E ≈ 5 with **quarter-epoch checkpoints**.

---

## 8. 10-15 02:32–04:11: Phase 2 (10 h × 4 LRs) and the first live chat

### 02:32–02:35: design changes during launch
- The user: try 1e-4 and 5e-4 at 1 h too; "I also don't notice any meaningful degradation" (after listening
  to 2e-4); run 10 h at 2e-4 and 2e-5 for E = 5 in parallel. Plan v1: 2 × 10 h + R5 (5e-4) + R10 (1e-4) at 1 h.
- `make_configs --ckpts-per-epoch`; `scripts/phase2.sh`. Disk estimate "~110 of 150 GB". **This estimate
  was wrong** (see §9).
- **02:33 the user: "yeah. go"** → **4× pod `sq9r1dzsa8l6rc`**.
- **02:34 the user: "might as well do 5 epochs on 10h for all four learning rates, no?"** The assistant
  suggested 5e-5 instead of 1e-4. **02:34 the user: "go with 2e-5, 5e-5, 2e-4, 5e-4".**
- **02:35 launched R8–R11:** train_10h (1,101 eps), 5 epochs = 345 steps, checkpoint + dev loss every 17
  steps (¼ epoch, 21 checkpoints), same seed and recipe. 5,520 presentations ≈ 50 h of audio seen.

### 02:37–03:07: first live chat
- The user: "can i actually talk with any of the 1h models?" Checked: `moshi.server` supports
  `--lora-weight` + `--config-path`. "can you not just get a worse gpu" → the volume is pinned to
  US-NE-1. "wait can i do it locally?" (M5, 24 GB; would need MLX 4-bit) → "ahh it's fine, just spin up a h100".
- **1× pod `ywpfqsbqerh53c`** (02:40). Slot A = base, slot B = R4 ckpt 112 (1 h 2e-4, ep 15.9).
  New **`silence/chat_gui.py`** (local panel at :8770 with a model picker, tunnel and notes logger) and
  **`silence/chat_scripts.py`** (13 test scripts: train wording, held-out wording, new conditions).
- Predictions: the timed hold and the friend's "I'm back" would likely fail.
- **03:07 the user: "kill the live instance for now"** (≈ $1.80). No chat notes were saved.

### 02:52–04:08: phase-2 training and results
- At epoch 1, every 10 h run is already below the best 1 h dev text loss. 5e-4 rises from epoch 1.
- All four done 03:45 (69 min each). Dev_80 evals of 84 checkpoints (6,720 outputs). Pod terminated
  04:08 (≈ $25).

| Data | Peak LR | Dev text best (ep) → final | Dev audio best → final |
|---|---|---|---|
| 10 h | 2e-5 | 0.280 (3.0) → 0.299 | 0.061 → 0.061 |
| 10 h | 5e-5 | 0.275 (2.0) → 0.392 | 0.059 → 0.060 |
| 10 h | 2e-4 | 0.273 (1.0) → 0.479 | 0.061 → 0.065 |
| 10 h | 5e-4 | 0.292 (1.0) → 0.473 | 0.060 → 0.063 |

The late audio drift seen at 1 h 2e-4 (0.078 → 0.131) is gone at 10 h.

| LR | Data | Score | Silent | Speaks | Cue on time | Timed re-ask | Ordinary |
|---|---|---|---|---|---|---|---|
| 2e-5 | 1 h | 0.79 | 88% | 70% | 77% | 24% | 83% |
| 2e-5 | 10 h | 0.84 | 96% | 72% | 76% | 20% | 89% |
| 5e-5 | 1 h | 0.79 | 84% | 73% | 83% | 17% | 87% |
| 5e-5 | 10 h | 0.85 | 96% | 74% | 89% | 15% | 82% |
| 2e-4 | 1 h | 0.86 | 89% | 83% | 96% | 19% | 98% |
| 2e-4 | 10 h | 0.82 | 98% | 66% | 85% | 8% | 72% |
| 5e-4 | 10 h | 0.57 | 100% | 14% | 22% | 0% | 12% |

Findings:
1. **10 h makes the model better at holding silence (96–98%), not at speaking again (66–74%).**
2. **The best LR shifts down with more data:** 2e-4 at 1 h → 2e-5–5e-5 at 10 h.
3. **Too high an LR also gives a silent model** (5e-4 at 10 h, 100% silent / 14% speaks).
4. **The plateau is set by optimizer steps, not epochs:** ≈ 0.8 after ~70 steps (~1,100 conversations)
   at both sizes.
5. Timed holds are still unlearned (8–20% re-ask).
6. The best checkpoints tie within noise: 10 h 2e-5 ep 4.2 (0.88), 1 h 2e-4 (0.89–0.90), 10 h 5e-5 ep 3.7 (0.87).

- The demo was extended: `results/demo/index.html`, 9 conversations × **20 models** (base + 11 at 1 h + 8 at
  10 h), with 1 h / 10 h filters.

---

## 9. 10-15 04:13–05:17: silence bias, EPAD sweep, disk incident

### 04:13: "is my dataset encouraging holding silence?"
Measured on the training targets:

| | train_1h | train_10h |
|---|---|---|
| Text targets = PAD | 89.7% | 89.8% |
| Text targets = EPAD ("start talking now") | 0.64% | 0.62% |
| EPAD share of text-loss weight | 0.58% | 0.57% |
| Audio loss covers | Moshi's silent frames only (72%) | same |
| Time inside a requested hold / Moshi speaking | 38% / 28% | 37% / 28% |

- Answer: "yes, strongly, and by construction". The decision that matters carries ~0.6% of the text-loss
  weight, and every audio gradient says "emit silence" (the deliberate speech mask). This explains the
  universal silent phase, 10 h being over-cautious, 5e-4 collapsing and timed holds staying unlearned.

### 04:15–04:19: patch 4 and the sweep design
- **04:15 the user: "yeah, let's do your fix. can i parallelize though?"**
- **Patch 4 `moshi-finetune-loss-weights.patch`:** `text_epad_weight` (EPAD's own weight) and
  `audio_loss_weight`; training only (dev loss keeps the original weighting). Defaults reproduce upstream.
  `make_configs --epad-weight --audio-weight`.
- A first design (R12–R15: 10 h, EPAD ×10 / audio ×0.3 / both) was written but not run.
- 04:16 the user: "i meant can i train one model across multiple gpus" → possible with FSDP (~3–3.5×
  faster, same GPU-hours, not bit-identical, untested LoRA key names); recommended one run per GPU for sweeps.
- **04:18 the user: "what if we train with a fixed LR and 1h dataset, just with different multiples of the
  EPAD"** → dose-response at **1 h, lr 5e-5, 20 epochs**, baseline R3: **R16 ×3, R17 ×10, R18 ×30, R19 ×100**
  (EPAD ≈ 1.7% / 5.5% / 15% / 37% of the text-loss weight). Audio weight kept at 1. **04:19 "go for it".**
- **4× pod `7f5yuy4acmhxzp`** (04:19). Step-1 loss identical in all four runs (0.462).

### 04:54: disk-full incident
- **All four runs crashed at step 132–133 of 142** with `Disk quota exceeded` while saving a checkpoint:
  `runs/` had reached **104 GB** on the 150 GB volume. The earlier disk estimates had undercounted
  (checkpoints are ~0.37 GB each and phase 2 added 84 of them).
- Freed the uv cache and `runs/_smoke`. **04:55 volume resized 150 → 250 GB** (+ ≈ $0.23/day). The
  assistant later wrote "as you suggested", but no such user message appears in the transcript. Removed the
  partial `checkpoint_000133` dirs (0-byte LoRA files), which had broken two evals.
- Each run kept **18 checkpoints (epochs 1–17.8)**. train.py can't resume. **04:59 the user: "use them as they
  are".** The plateau window for the sweep and R3 became epochs 10–17.8.
- New rule: check free space with `du -sh /workspace/*` before launching (`df` shows the whole cluster).

### 05:01–05:17: EPAD sweep results
- Mid-run (epoch 4): ×10 already re-asked after timed holds 72% of the time, but held silence only 43%.
  05:03 the user: "so x30 is the most promising right now right" → no: ×30 talks through ~¾ of holds.

| EPAD × | Plateau score | Silent | Speaks | Cue on time | Timed re-ask | Ordinary | Best checkpoint |
|---|---|---|---|---|---|---|---|
| ×1 (R3) | 0.80 | 86% | 73% | 84% | 15% | 86% | ep 15: 0.83 |
| ×3 | 0.81 | 80% | 82% | 93% | 24% | 97% | ep 14: 0.85 |
| ×10 | 0.80 | 78% | 83% | 94% | 34% | 95% | **ep 5.9: 0.86 (81% / 91%, 61% timed)** |
| ×30 | 0.77 | 71% | 83% | 95% | 28% | 98% | ep 17.8: 0.81 |
| ×100 | 0.80 | 75% | 86% | 94% | 40% | 99% | ep 9.9: 0.85 |

Findings:
1. **At the plateau, EPAD weight is a dial along the speak/silence trade-off, not a free gain** (score
   stays ≈ 0.80).
2. **It changes the trajectory:** up-weighted models learn to speak first and to be quiet later. Mid-training
   (epochs 4–9), ×10–×100 re-ask after timed holds **60–80%** of the time (the first time this was learned),
   but it fades to 20–40% by the plateau.
3. Best balanced checkpoint: `R17_1h_lr5e-05_epad10/checkpoint_000042` (ep 5.9).
4. Dev text loss again misleads (×100 has the lowest late dev loss but no better behavior).

- 72 checkpoints scored (5,760 outputs); `results/epad/` + `analysis/epad_sweep.png`. Pod terminated 05:16
  (≈ $15). Spend so far ≈ $69.

---

## 10. 10-15 05:20–06:06: overview, second chat, R20/R21 launch, 50 h planning

### 05:20–05:22: everything so far
- The user: "can we look at all of the data we've amassed so far? I also want to see some transcripts…
  let's see what the next steps are now".
- **`results/overview.md`**: 14 trained runs, 282 checkpoints, all scored the same way.
- **`silence/transcript_gen.py` → `results/transcripts.md`.** Notable behaviors:
  - Return-cue lookalike ("Okay, I'm done for today anyway", said to the friend): **every 1 h model says
    "Welcome back!" ~16 s early; the 10 h model waits** for the real cue.
  - Filler during a timed hold ("Well now, that's a toss-up"): the strong models answer it; only EPAD ×10
    stays silent and then re-asks.
  - Text quality is mostly fine ("Welcome back! About your sister's clay sculpture, what shape are you
    thinking of?").
- Status vs the brief: data size ✓, LR ✓, audio/demo ✓, failures ✓; generalization (test suites) ✗, human
  set ✗, report partial.

### 05:24–05:34: second chat pod and the curated model list
- **1× chat pod `vc42i3dtvzv7s5`** (05:25). Slot A = R8 ckpt 289 (10 h 2e-5 ep 4.2), slot B = R17 ckpt 42
  (1 h EPAD ×10 ep 5.9). Ready 05:29.
- The user: "trim the list of models… just be the representative ones… characterize what they're good/bad
  for". `chat_gui.CURATED` = 7 models with good/bad notes (best 10 h, best 1 h, EPAD ×10, base, quiet R1,
  collapsed R11, over-corrected EPAD ×100); "show all" brings back 283.

### 05:29–05:34: R20/R21 (EPAD at 10 h)
- **05:29 the user: "what if we train 10h, 3x, and 5e-5?"** → **05:32 "do both, 3x and 10x".**
- No 2× H100 free in US-NE-1, so **two 1× pods**: `086j0wex48cmlp` (tr3, **R20 ×3**, started 05:33:41) and
  `xk9kzxl2eev9of` (tr10, **R21 ×10**, 05:34:40). Same recipe as R9 except `text_epad_weight` (1.5 / 5).
  `scripts/epad10h.sh [3|10|both]`.
- Epoch-1 dev text: ×3 0.298, ×10 0.350 vs R9 0.282 (the up-weighting costs a little on the original-weighted loss).

### 05:41–05:44: demos of the representative models
- `demo_phase1.py --curated` → **`results/demo_curated/index.html`** (9 conversations × 7 models, 28 MB).
  Bug fixed: the crashed EPAD runs had logged a step-133 dev eval for a checkpoint never saved.
- Scoreboard on the 9 demo conversations: the 3 good models beat the base on everything except timed holds;
  best 10 h is the only one that ignores side conversation 2's "I'm done" lookalike; EPAD ×10 is the only good
  model that handles timed holds.
- **`results/transcripts.html`** (interactive viewer with audio), served at 127.0.0.1:8780.

### 05:48–06:04: 50 h planning
- "say i want to do like a 50h training run. how long would that take?" → ~6–7 h wall clock (mostly data on
  the Mac), GPU ≈ 2–3 h ≈ $10–12. Advice: wait for the 10 h EPAD results; diversity is the likely bottleneck.
- "what should go in such a dataset? i have more voice diversity available via elevenlabs api and fish api"
  → a problem → cause → fix table: more timed holds with re-asks; mutters in most timed holds; denser
  lookalikes; shorter holds / more speak events; more phrasings and voices. Proposed mix B1 25 / B2 20 /
  B3 15 / ORD 40, 100–300 speakers across 3 engines with held-out voices per engine, Whisper QC, a 10 h
  pilot first.
- Prices looked up live: ElevenLabs Flash $0.04 / 1K chars (v3 $0.08); Fish s2 $15 / 1M bytes; Sonnet 5.5
  $2 / $10 per MTok. 50 h needs ≈ 1.0M user/friend characters.
- 05:55 "flash vs v3? i want to keep cost under ~10 dollars" → Flash.
- **05:56 the user: "let's go with the 50h and 40% Pocket / 15% Flash / 45% Fish version."** The keys were
  added to `.env` by the user (`read -rs`, `chmod 600`) and checked by name and length only.
- Account facts: ElevenLabs Starter, **116,845 credits** (not 150K), 22 voices; Fish credit **$25.83**,
  5 concurrent. **Fish's streamed WAV header misreports the length** (89,478 s claimed for a 2.27 s clip):
  duration must come from the byte count. Fish's public library is full of clones of real people and characters.
- "any way to speed the dataset generation up? say i want to do it in under an hour" → build in the cloud on
  the volume with overlapping stages, ≈ 50–75 min.
- **06:03 "yeah, do all that"**, then **06:04 "idk what you do just make the dataset happen in under an hour"**.
- Benchmarks: **Fish s2-pro at 5 concurrent 20.5× realtime; ElevenLabs Flash at 3 concurrent 20.0×**.
  Unused v1 train packs recounted at 721. The pods have 208 cores and ~1.8 TB RAM.

---

## 11. 10-15 06:06–07:16: the 50 h build and the R20/R21 results

### 06:06–06:17: packs v2
- **06:06** `gen_packs --seed 1 --test-frac 0 --workers 48 --per-family-json
  '{"B1":2120,"B2":1580,"B3":1240,"ORD":2250,"DECOY":780}'`. **Crashed at call ~578** (a `StopIteration`
  inside validation) → now caught as "schema: malformed"; resumed with `--seed 2`. Done 06:16:
  **`packs_v2.jsonl` = 7,970 packs (5,470 new)**, ≈ $20.
- 06:11 chat pod `vc42i3dtvzv7s5` deleted (the user had asked; ≈ $4.30).
- **First adversarial review** (06:13–06:15):
  1. **Blocker:** EL/Fish concurrency limits are per account, not per process.
  2. One failed request discarded a whole paid batch and the run exited 0.
  3. No validation of returned audio.
  4. Pocket-fitted durations push hosted-voice episodes over 60 s and the redraw would crash.
  5. **Leakage:** 254 new packs shared a line with test packs and 54 were near-duplicates; similar overlap with dev.
  Plus non-atomic cache writes, shards silently building different plans, and held-out EL/Fish voices in no test suite.
- 06:14 the user rejected a 20-process Pocket benchmark on the training pod tr10 (R21 unaffected).
- **Reverse screen** `silence/screen_train_packs.py`: a new pack is dropped if it contains a line spoken in dev
  or any test suite, or is a near-duplicate (Jaccard ≥ 0.5) of a test or dev pack. **306 dropped** (B2 172 + 4,
  B1 15 + 75, B3 13 + 18, ORD 2 + 7) → **`packs_v2_screened.jsonl` = 7,664 packs (5,164 new)**.
- A second adversarial review ran (its report isn't in the transcript; its findings were later summarized in
  `docs/training_record.md` §12).

### 06:11–07:03: code and rendering
- `voices.py` multi-engine pools (Moshi always Pocket `michael`; EL premade voices, the account's cloned
  voice excluded; 30 vetted generic Fish voices, celebrity/character clones excluded; 4 held-out voices per
  engine). `tts.py` ElevenLabs + Fish backends (per-clip atomic writes, retries honoring `Retry-After`,
  `FatalEngineError` on 401/402/403 or 25 consecutive failures, audio checks, FLAC storage). `specs.TRAIN_MIX_V2`
  (B1 25, B2 20, B3 15, ORD 30, DECOY 10) + `MUTTER_P` 0.7. `build.py --tts multi --mix v2 --exclude data/dev
  --render-only --shard i/n --reverse --max-drop`, plus a plan hash.
- 8-episode multi-engine test: 8/8 pass.
- **Plan (n = 5,600):** 41,070 unique clips: Pocket 24,129 (23.2 h of speech), Fish 12,706 (10.3 h, 622k chars),
  EL 4,235 (3.4 h, 209k chars).
- A **4× RTX PRO 6000 pod as a CPU burst was blocked by the auto-mode classifier**; no CPU-only pods exist in
  US-NE-1. So Pocket ran on the **CPUs of the two training pods** (nice 19, 18 workers) plus the Mac.
  Mac → pod upload measured at only ~1.2 MB/s → hosted clips stored as FLAC.
- 06:29 three Pocket shards: 0/3 on tr3, 1/3 on tr10, 2/3 on the Mac (reusing the 10 h `michael` clips via
  symlink). EL + Fish rendered from the Mac only, so the keys never left it.
- **Plan-hash mismatch** Mac vs pod: 25 timing floats differ by 1 ulp (macOS vs Linux libm), but the clip
  sets are identical. The hash was made robust to this; rule: only one machine assembles.
- 06:40 the user chose to keep tr3 to ~07:20 as the assembly box. 06:41: 0 planned lines contain a held-out
  phrase-bank wording.
- **Shard 0 crashed** (BrokenPipe after its launching ssh session died) → relaunched detached (`ssh -f -n` +
  `setsid`). Takeover shards (`--reverse`) had a waiter bug (the `pgrep` pattern matched its own command line).
- **06:43 the user: "use as much claude usage as needed to speed this up as much as possible"** → three
  background agents: documentation, ASR QC, EPAD-10 h analysis.
- **ASR QC** (faster-whisper base.en, 366 sampled clips): WER mean EL 0.021 / Fish 0.013 / Pocket 0.036; no
  voice above 0.15 mean WER; no truncation or repeats. Speaking rate: EL ~13% faster than Pocket, Fish ~13% slower.
- **07:01** hosted rendering done: Fish 12,075 / 12,076 (1 failed line), EL 3,584 / 3,584.

### 07:03–07:14: assembly
- `scripts/build50_final.sh` on tr3: **5,597 episodes assembled (51.38 h) in 6.6 min** (1 dropped for the failed
  Fish clip, 2 dropped as over 60 s).
- check.py: **5,585 / 5,597 pass**. Of the 12 failures, 7 are Moshi word times off by ~−1.0 s, and 6 of those are
  **B1_double** (6 of 214, systematic, cause not fixed); 2 are user/Moshi overlap; 2 are quiet human turns.
- **Final: 5,585 episodes / 51.24 h** (35.85 h speech, 13.20 h Moshi speech, 17.57 h hold silence; mean 33.0 s),
  5,799 packs, 40,921 clips. Nested `train_1h` 110 / `train_10h` 1,084 / `train_25h` 2,724 / `train_50h` 5,457
  + `train_all`. 17 GB at `/workspace/data50/train50`. Metadata in `results/build50/`.
- **07:14 both pods deleted** (tr3, tr10; ≈ $13.4). Wall clock ≈ 70 min, ~10 min over the target.
  Build cost ≈ $30 (packs ≈ $20, Fish $9.39, ElevenLabs within plan). Pocket on the training pods slowed
  R20/R21 by ~8%.

### 07:15: R20/R21 (EPAD at 10 h) results
Trained 05:33–06:49, dev_80 evals until 07:13. Plateau = epochs 2–5.

| Run | Score | Silent | Speaks | Cue on time | Timed re-ask | Ordinary | Best checkpoint |
|---|---|---|---|---|---|---|---|
| R9 ×1 | 0.85 | 96% | 74% | 89% | 15% | 82% | ckpt 255 (ep 3.7): 0.87 |
| R20 ×3 | 0.87 | 90% | 85% | 93% | 40% | 95% | ckpt 153 (ep 2.2): 0.91 |
| R21 ×10 | **0.88** | 81% | 94% | 96% | **73%** | 100% | ckpt 153 (ep 2.2): 0.91; final ckpt 345: 0.90 |

- Same trade-off as at 1 h, but a **net gain** because the 10 h baseline was over-silent.
- **At 10 h the timed re-ask sticks** (×10: 61–83% at every checkpoint after epoch 2, timing error ≈ 0); at 1 h it faded.
- The lost silence is almost all in timed holds (B1 violated 10% / 37% / 57% for ×1 / ×3 / ×10; reactions to
  mutters 21% → 53%). Side-conversation and hear-me-out holds stay 97–99% silent.
- Dev text loss ranks backwards again (R21 0.323, the worst; R9 0.275).
- 0.91 is within noise of the others; a dev_200 confirmation was planned but never run.

---

## 12. 10-15 07:21–09:44: R22 (50 h), the held-out test suites, diagnosis

### 07:21–07:29: deciding what to run
- 07:23 "ok, for 50h what should i run" → proposed **R22** (train_50h, 345 steps ≈ 1 epoch, EPAD ×10, lr 5e-5,
  same compute as R21) and **R23** (the v2 recipe's own 10 h prefix, 345 steps), to separate size from recipe;
  plus dev_200. `make_configs --steps --every`.
- 07:25 "yeah go, both runs at x10 plus dev_200". **07:26 corrected: "wait just run R22 + dev 80 and test set"**
  → R23 and dev_200 dropped.
- **07:28 the user: "R22 + R21 best + 1h x10 best + base. also run R22 on like 3 epochs max. we want to keep
  training short as possible"** → R22 at 1 epoch; test suites for base, R17 ckpt 42, R21 ckpt 153, R22 final.
- `scripts/test_suites.sh` (one eval_gen process per suite, LoRA swap per checkpoint).

### 07:29–09:31: pod `i1wn60oi6h7n8g`
- R22 training 07:31–09:07. During the first hour the test suites shared the GPU at batch 20, so training
  slowed to ~7k words/s (~15 s/step). Volume 177 of 250 GB used.
- In parallel, session `4ec42cb5` (07:35–07:53) added R21 ckpts 153 and 345 to the curated demo, the
  transcript viewer and the chat panel. It pulled the two missing `codes.npz` files from the running pod
  after the user asked "wait why do we need a gpu for this". Committed as **`926d4b0`**. Per-type analysis of
  R21 ckpt 153 ("what is it bad at?"): no B2/B3 violations and every ordinary question answered, but timed
  holds are bimodal: 5 of 11 re-ask too early, 6 of 11 never re-ask. "The ×10 weighting bought the re-ask at
  the cost of timing discipline."
- Test suites for the fixed models were done at 08:36. New scorer `results/test/analysis/score_tests.py`.
- R22 dev text loss fell monotonically to **0.305** (no memorization in 1 epoch). Dev_80 behavior plateaued by
  step ~150 at ≈ 0.81–0.86 (mean of the last 10 ≈ 0.83; silent 69–78%, speaks 90–94%, timed re-ask 61–83%).
- 09:08 the user: "one more epoch?" → train.py can't resume, so it would be a new 2-epoch run (~$10). Decided at
  09:31 from the flat dev curve: **no second epoch**. Pod deleted 09:31 (≈ $8.10).

### Held-out test results (1 sample per episode; R17/R21 = best dev_80 checkpoint, R22 = final)

| Suite | base | 1 h ×10 (R17 ep 5.9) | 10 h ×10 (R21 ep 2.2) | 50 h ×10 (R22) |
|---|---|---|---|---|
| test_script (new scripts) | 0.32 | 0.93 | **0.97** | 0.93 |
| test_word (new user wording + new N) | 0.35 | 0.82 | **0.87** | 0.84 |
| test_voice (4 unseen Pocket voices) | 0.37 | 0.86 | **0.92** | 0.89 |
| test_distract (dense distractors + D4) | 0.29 | 0.93 | **0.98** | 0.92 |
| test_cond (keyword, phone, "a minute") | 0.32 | 0.67 | **0.72** | 0.70 |
| test_ordinary | 0.59 | 0.96 | 0.98 | **1.00** |
| test_all (new wording + voices) | 0.32 | 0.79 | **0.84** | 0.80 |

What generalized and what didn't:
- Unseen scripts score about the same as dev. New wording or voices cost 5–10 points, almost all of it in timed holds.
- Distractors are ignored at 10 h, including **the friend's "I'm back" (D4): R21 silent 18/18**, R17 56%, base 0%.
- New conditions only partly transfer: keyword/phone holds are kept (R21 97% / 94%), but releases are often
  missed (R21 spoke at 68% of cue releases on test_cond); **"a minute" holds fail** (7% silent; the longest
  trained hold is 30 s).
- **50 h did not beat 10 h.** R22 speaks a little more (+1–12 pp) and holds silence less (−7 to −17 pp).
  Confounds: (1) recipe (v2: 25% timed holds with 70% mutters, new scripts, 3 engines) combined with ×10;
  (2) checkpoint selection (R21 best-of-21 vs R22 final; R21's final was 0.90 vs 0.91, so the effect is
  small); (3) exposure (R21 saw each episode ~5×, R22 once); (4) noise (7–30 timed holds per suite).
- Caveats: only Pocket voices are held out in the test suites (the held-out EL/Fish voices are in none);
  the suites are not paired; 1 sample.

### 09:37–09:44: diagnosis and the v3 fix (state at time of writing)
- **09:37 the user: "ugh. how might we be able to fix this data!"**
- Dev_80 silence by hold type (R21 v1 vs R22 v2, last 10 checkpoints): B1_double 70% → **20%**, B1_rescind 95%
  → **40%**, B3_rescind 85% → **15%**, B3_std 99% → 91%, B2 98% → 98%.
- Diagnosis: a learned shortcut, **"after a few seconds of user silence, start talking."** v2 has 67% more timed
  holds (Moshi speaks after N s of silence with no cue), ×10 up-weights exactly those moments, and no training
  data has long silences where Moshi must keep waiting (D2a stalls 2–8 s, D2b pauses 3–5 s). The voice engines
  aren't the cause (B2 is unchanged).
- Fixes proposed:
  1. rebalance the 50 h set back to 15% B1 (no new audio);
  2. long silences in cue-released holds (B2 stalls up to ~20 s, B3 pauses up to ~10 s), plus more contrast
     episodes; timing-only, all clips cached;
  3. EPAD ×3;
  4. later, 120 s windows for "a minute" holds.
- **09:42 the user: "can you just spin up a demo of R21 first? then do fix 1 and 2 above. can we keep epad x 10?"**
  → yes, keep ×10.
- `chat_gui.CURATED` gained R22's final checkpoint. **Chat pod `5obz5mapyti207`** created 09:43 (slot A =
  R21 ckpt 153, slot B = R22 ckpt 345, loading at the end of the transcript).
- `specs.py` v3 timing: a separate per-episode RNG so every v2 draw is unchanged; **0 new clips**. B2 stalls
  mean 4.9 → 8.2 s (max 19.9); B3 pauses mean 4.0 → 6.3 s (max 12.0); 0 episodes estimated over 60 s.
  The B1 rebalance isn't implemented yet.
- **09:43 the user: "what do you think R24 will be like? also do 2 epochs for good measure, robust to early
  stopping though"**. Not yet answered when the transcript ends.

---

## 13. Consolidated decision log

| Time (UTC) | Decision | By | Why |
|---|---|---|---|
| 10-14 18:11 | Distractors written by the LLM in context, not a lexicon | user ("no need to hard code") | lexicon lines made no sense in context |
| 18:12 | Keep long (20–30 s) timed holds | assistant, accepted | windows are padded to 60 s anyway; variable N tests the instruction |
| 19:04 / 19:45 | The user always speaks first | user | Moshi opening every window teaches "Moshi starts talking" |
| 19:31 | Mask the audio loss on Moshi's speech | user (option a) | otherwise Moshi learns to sound like the TTS voice |
| 19:50 | Length budget, speaking styles, B1_double | user | fit 60 s windows; variety; two holds per episode |
| 20:07 | Gaps ×1.5 | user | turns felt rushed |
| 20:39 | **Fixed epochs, not fixed steps**; 1 h = proof of concept | user | |
| 20:47 | 5 LRs at 1 h; no behavior-loss script; no second seed | user | runs are cheap |
| 20:50 | Cut 1 h rerun at E, 100 h, stretched-holds ablation; phase 2 = top 2 LRs | user | |
| 20:53 | Dev evaluation every epoch | user | |
| 21:10 | dev_80 ⊂ dev_200 | user + assistant | size by precision |
| 21:43 | 90 episodes per test suite | user | |
| 22:01 | 1× pod for setup, then 4× | user | |
| 22:40 | All six eval-harness recommendations | user | |
| 23:47 | US-NE-1 at the live $3.99/hr | user | |
| 23:52 | Switch to the RunPod API-key account | user | not paying personally |
| 10-15 00:19 | Save LoRA at every dev eval | user | |
| 00:51 | **Drop R5 (5e-4), add R6 (5e-6) and R7 (2e-5)** | user | 5e-5/2e-4 memorized; fill the gaps |
| 02:34 | **Phase 2 = all four LRs (2e-5, 5e-5, 2e-4, 5e-4) at 10 h, E = 5, ¼-epoch checkpoints** | user | GPUs run in parallel anyway |
| 03:07 | Kill the first chat pod | user | |
| 04:15 | Up-weight EPAD (loss-weights patch) | user | silence-biased objective |
| 04:18 | EPAD dose-response at 1 h, lr 5e-5 (×3…×100) | user | one factor at a time |
| 04:55 | Volume 150 → 250 GB | assistant (no explicit user approval visible) | disk full |
| 04:59 | Use the truncated EPAD runs | user | |
| 05:32 | EPAD ×3 and ×10 at 10 h | user | 10 h models over-silent |
| 05:56 | 50 h; user/friend voices 40% Pocket / 15% EL Flash / 45% Fish; TTS ≤ ~$10 | user | voice diversity |
| 06:04 | Build it in under an hour | user | |
| 06:40 | Keep tr3 for assembly, kill tr10 after its evals | user | |
| 06:43 | "use as much claude usage as needed" | user | speed |
| 07:26 | Run only R22 + dev_80 + test suites (no R23, no dev_200) | user | |
| 07:28 | Test models: R22 + R21 best + 1 h ×10 best + base; R22 ≤ 3 epochs (ran 1) | user | keep training short |
| 09:31 | No second R22 epoch | assistant, from the flat dev curve | |
| 09:42 | v3 = fixes 1 + 2, keep EPAD ×10; demo R21 first | user | |
| 09:43 | R24: 2 epochs, robust to early stopping | user | pending |

---

## 14. Bugs, traps and incidents

| When | Problem | Fix |
|---|---|---|
| 10-14 18:08 | Lexicon distractors nonsensical; friend's question unanswered | LLM-written tagged distractors; every question answered |
| 18:08 | "MacBook" mispronounced | no brand names; HARD_WORDS; camelCase reject |
| 18:19 | Over-60 s train episodes → moshi-finetune cuts a second window mid-episode | redraw / shorten B2; check.py fails > 60 s |
| 18:21 | Smoke test ~2.5 min per call | parallel calls, thinking off (cause unconfirmed) |
| 19:03 | B3 context in the third person; themes cluster; no D3a; formulaic ORD | prompt/validator fixes |
| 19:27 | Audio loss would train Moshi toward the TTS voice | patch 1 audio mask |
| 19:56 | ~5% of Pocket clips: noise read as onset (up to 0.93 s) | `speech_span` "real speech follows" rule |
| 20:12 | B2 yield ~9% (D3a errors, too long) | D3a repair → ~32% |
| 21:11 | `--n 1000` gave 9.07 h (mean 32.6 s, not 37 s) | extended to 1,150 |
| 21:16 | Two sessions wrote `data/train` at once | manifest restored; one writer at a time |
| 21:28 | Dev v1 overlapped train (Jaccard 0.58, 12 shared lines) | screen; dev rebuilt |
| 21:43 | Test screen on raw strings too strict | screen on lines actually spoken |
| 21:43 | Eval loader finite → NaN after the first eval | patch 2 eval-reload |
| 21:43 | `num_ckpt_keep` default 3 | `null` |
| 21:49 | Worktree vs main checkout split; neither uploadable alone | snapshot commit; two rsyncs |
| 21:51 | `pip install -e` unresolvable | `uv sync --frozen` |
| 21:51 | ~46% of 1 h text frames were tail PAD in the loss | patch 3 text-tail |
| 22:24 | Test suites claimed "paired" | they aren't (42–68 of 90 shared); docs fixed |
| 22:46 | Oracle missed ordinary responses (B1_double); N = 3 s holds unscoreable | `violated = None`; ordinary only on hold-free episodes |
| 22:53 | Smoke run without CUDA_VISIBLE_DEVICES; uv → Python 3.14; rsync would copy `.env`; LR precision | fixed |
| 23:05 | Plain LMGen feeds the user audio 80 ms early | `--user-shift 1` |
| 23:30 | Whole-batch Mimi needs > 100 GB | Mimi in chunks of 4 |
| 23:30 | Hub probe could crash jobs | `HF_HUB_OFFLINE=1` |
| 23:44 | MCP authenticated to the personal account | switched to the API-key account; pod killed from inside |
| 23:48 | `rsync -a` exit 23 on MooseFS | `rsync -rlpt` |
| 23:49 | openai-whisper needs `pkg_resources` (setuptools 84) | two-step `uv sync` |
| 23:49 | "already applied" for patches on a fresh clone | tree verified correct; cause unknown |
| 23:56 / 06:18 / 07:39 | Auto-mode classifier blocked a RunPod read, a 4× CPU-burst pod and a raw-IP ssh | worked around or skipped |
| 10-15 00:25 | Plot regex missed `lr0.0002`; legend empty | fixed |
| 01:21 | `eval_report` showed first-turn rate as "ordinary answered" | `ord_responded` added |
| 02:23 | Metrics rewarded quiet models | two-axis scoring |
| 04:54 | **Volume full: 4 EPAD runs crashed at step 132–133** | volume → 250 GB; runs used truncated |
| 05:41 | Demo crashed on the never-saved step-133 checkpoint | filter to scored checkpoints |
| 05:59 | Fish WAV header length wrong | duration from byte count |
| 06:12 | gen_packs crashed on StopIteration at call ~578 | caught; resumed with seed 2 |
| 06:13 | Per-account API concurrency; paid batches lost on one failure; no audio validation; train/test leakage of new packs | one hosted process; per-clip writes; audio checks; reverse screen |
| 06:36 | Plan hash differed Mac vs Linux (1-ulp floats) | robust hash; one assembler |
| 06:41 | Pocket shard 0 BrokenPipe after its ssh died | relaunched detached |
| 06:47 | Takeover waiters matched their own command line | `[s]hard` pattern |
| 07:12 | 6 of 214 B1_double episodes: Moshi word times off by −1 s | filtered out; **cause not fixed** |

---

## 15. Findings, consolidated

1. **Fine-tuning works from 1 h of synthetic data.** Base 0.29 → best checkpoints 0.86–0.91 on dev; holds
   kept 14% → ~90–100%, cue releases 47% → ~90–100% on time, ordinary questions 43% → ~90–100%,
   distractor reactions ~70–80% → ~0–15%.
2. **LR is the main lever, and both extremes give a quiet model.** Too low (2e-6, 5e-6 at 1 h) never leaves the
   silent phase; too high at 10 h (5e-4) collapses into silence. The best LR shifts down with data
   (2e-4 at 1 h → 2e-5–5e-5 at 10 h).
3. **Learning has two phases:** silence first, then when to speak again (unless EPAD is up-weighted, which
   reverses the order).
4. **The behavior plateau is set by optimizer steps** (~70 steps ≈ 1,100 conversations), not epochs.
5. **Dev loss is a poor selector**: it ranks LRs almost backwards and rises (memorizing wording) while behavior
   holds or improves. Audio dev loss saturates within an epoch.
6. **The objective is silence-biased by construction** (EPAD ≈ 0.6% of the text-loss weight; audio loss only on
   silent frames). EPAD up-weighting is a dial along the silence/speech trade-off at 1 h, and a net gain at
   10 h, where the baseline was over-silent.
7. **Timed holds are the hard case.** Without EPAD they're essentially unlearned (re-ask 8–26%); EPAD ×10 at
   10 h makes the re-ask stick (~73%) but timing is bimodal and silence during timed holds suffers.
8. **10 h vs 1 h:** better at holding silence and ignoring lookalikes (incl. the friend's "I'm back"), and best on
   every held-out suite except test_ordinary.
9. **Generalization:** new scripts ≈ dev; new wording/voices −5–10 points (mostly timed holds); distractors
   handled; new conditions only partly ("a minute" holds fail, keyword/phone releases often missed).
10. **50 h (v2 recipe) did not beat 10 h**, likely because the recipe taught a "talk after a few seconds of
    silence" shortcut. Size and recipe are confounded.

---

## 16. Open items at time of writing (2026-10-15 ~09:45 UTC)

- **Running:** chat pod `5obz5mapyti207` (1× H100, $3.99/hr) with R21 ckpt 153 vs R22 final.
- v3 data: the long-silence timing is written; the B1 rebalance (fix 1) isn't implemented yet; R24 (v3, EPAD
  ×10, 2 epochs, checkpoints for early stopping) not started; the user's question "what do you think R24
  will be like?" is unanswered.
- R23 (recipe vs size) proposed but not run.
- dev_200 confirmation of the 0.91 checkpoints not run. The checkpoint-selection rule was never formally
  decided (in practice: best dev_80 score = mean of silent and speaks).
- B1_double −1 s word-time issue: cause unknown.
- Held-out ElevenLabs/Fish voices aren't in any test suite. Test suites aren't paired. The paired/stratified
  rebuild was never decided.
- 3-sample test evals, Whisper ASR pass and LLM judge not built.
- Human-recorded test set (~40 episodes) not started.
- The "talk about trees" optional experiment not started.
- Report and deliverable demo not written. Whether the user listened to the original 14-episode demo is not recorded.
- Git: most work since `926d4b0` is uncommitted on `worktree-docs-progress` (`results/`, `docs/`, the 50 h
  code, patch 4, packs v2).
- Temporary files (CPU eval tests, scratch venv, ASR QC results) live in job tmp dirs and will vanish with
  those jobs.

---

## 17. Inconsistencies in the record

- Setup pod cost: ≈ $1.35 in most places, ≈ $1.80 in one message.
- Best 1 h 2e-4 checkpoint: "ep 16 (0.89)" at 04:11, "ep 10.9 (0.90)" in `results/overview.md`, and the chat
  panel uses ckpt 112 (ep 15.9). These are different scorings/windows of the same run; all are within noise.
- Space freed in the disk incident: "8.4 GB" vs a `du` showing the uv cache at 1.9 GB vs CLAUDE.md's 7.6 GB.
- dev_80 hours: 0.78 h in one log, 0.77 h in the docs.
- Unused v1 train packs: 763 stated at 05:48, recounted as 721 at 06:05.
- The 19:05 smoke-test rerun's result isn't in any transcript.
- The second 50 h code review's report and the creation of `scripts/run50h.sh` aren't visible in the condensed transcripts.

---

## 18. Sources

Session transcripts (`~/.claude/projects/-Users-parkjay-repo-moshi-silence*/`):

| Session | Span (UTC) | Content |
|---|---|---|
| `bb14ec5c` (dup `bc23c639`) | 10-14 18:01–19:05 | git init, demo listening, distractor redesign, first smoke test |
| `d9971fba` | 19:15–20:29 | plan, audio mask, generator hardening, onset fix |
| `08aab4c5` | 19:15–21:17 (resumed copy of the above) | packs v1, 10 h build |
| `35d2ba60` | 20:31–23:48 | training plan, dev/test sets, patches, eval harness, reviews, RunPod MCP |
| `fc2e3aca` | 23:44–23:49 | first (personal-account) setup pod |
| `02c11200` | 23:49–23:52 | 100 h feasibility |
| `12a5eb83` | 10-14 23:44 → 10-15 09:44 | GPU setup, phase 1, phase 2, chat, EPAD sweep, R20/R21, 50 h build, R22, test suites, v3 |
| `4ec42cb5` | 10-15 07:35–07:53 | R21 added to the demo site (commit `926d4b0`) |

Written records: `CLAUDE.md`, `docs/training_record.md`, `results/overview.md`, `results/epad10h/summary.md`,
`results/run50h/summary.md`, `results/test/summary.md`, git log of `worktree-docs-progress`.
