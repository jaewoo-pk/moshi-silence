# CLAUDE.md: moshi-silence

Project memory for a one-day research sprint (started 2026-10-14). Read this fully before
touching code or data. Last full review: 2026-10-14, ~6 pm ET.

## The task

Investigate **how much synthetic fine-tuning data it takes to teach Moshi when to stay
silent and when to speak again**. Use a public duplex checkpoint (`kyutai/moshiko-pytorch-bf16`)
and Kyutai's fine-tuning code (https://github.com/kyutai-labs/moshi-finetune), with public
or self-generated data. The brief, condensed:

1. Concrete behaviors: "Give me ten seconds to think, then ask me again." / "I'm going to
   talk to my friend. Stay quiet until I tell you I'm back." / "Let me describe all three
   options before you comment."
2. The interesting question: does it learn the underlying instruction and generalize to
   new wording and conditions?
3. Synthetic conversations: TTS, varied speakers, realistic pauses. Both humans on the
   incoming stream, silence on Moshi's output while it should wait, then an appropriate
   response. Include ordinary conversations so it keeps responding normally. Silence must
   come from the model itself (no external mute rule).
4. Measure learning vs. data size and learning rate. Baseline the unmodified model; start
   with 1 h and 10 h and a small LR sweep; extend promising settings to 100 h; 1,000 h is a
   stretch goal. Every comparison starts from the same checkpoint. Track unique examples,
   hours, and training exposure separately (repeating examples or adding longer silences
   must not look like more data).
5. Test generalization incl. failure cases: unseen scripts, voices, and some waiting
   conditions held out; a small human-recorded test set; check interruptions during the
   requested silence, resume timing, and that ordinary requests still get answered.
   Distractors: questions addressed to the other person, pauses before the conversation is
   finished, phrases resembling the return cue. Evaluate generated audio; listen to
   representative successes and failures.
6. Deliverable: short experimental report + demo. Behavior vs. data size, LR effect,
   reproducible code, a few audio examples, what generalized, what failed, next experiment.
   Optional if time allows: a "talk about trees more often" experiment as a second, simpler
   test of how fast fine-tuning changes behavior.

## Status at a glance (2026-10-14, ~6 pm ET)

| Piece | State |
|---|---|
| Generator (`silence/`) | Working. Demo 14/14 pass check.py. |
| Content packs | Done: 2,500 (`packs/packs.jsonl`), 348 screened test packs |
| Train set `data/train` | Done: 1,150 episodes / 10.49 h, all pass check.py; `train_1h` = 113 eps, `train_10h` = 1,101 eps |
| Dev set `data/dev` | Done: 200 episodes / 1.97 h (`dev_80` ⊂ `dev_200`), all pass, no overlap with train |
| Test suites `data/test_*` | Done: 7 × 90 = 630 episodes / 6.1 h, all pass check.py, no script overlap with train |
| moshi-finetune patches | 3 patches, tested to stack on upstream main `2acc879f` |
| Training configs + GPU bootstrap | Written; pre-launch review done 2026-10-14 (configs load with the real `TrainArgs`, patches apply to pristine upstream, data audited); **not yet run on a GPU** |
| Eval harness | Details agreed with the user (see "Eval harness"). `eval_score.py` written and oracle-tested; `eval_gen.py` (GPU) and `eval_report.py` not written yet |
| Human-recorded test set | Not started (the user records ~40 episodes later) |
| Training runs, report, deliverable demo | Not started |
| Demo listening pass | Not recorded anywhere; ask the user whether it happened |

**Git:**
- Local repo, no remote.
- `master` has only the first 2 commits.
- **Branch `worktree-docs-progress` is the complete, current state**: a snapshot of the main
  checkout's (then uncommitted) pipeline code, plus everything built since (dev/test
  tooling, configs, bootstrap, patches, generated packs).
- The main checkout's working copy holds the same pipeline code as uncommitted changes, plus
  the built `data/`.
- Neither `master` nor the main checkout's working copy has everything
  `scripts/gpu_bootstrap.sh` needs (no `scripts/`, 1 of 3 patches, no `make_configs.py`).
  **Upload the code from the worktree** (`.claude/worktrees/docs-progress`, which has no
  `.env` and no `data/`) **and `data/` from the main checkout**: two rsyncs, commands in the
  `scripts/gpu_bootstrap.sh` header. No git operation needed. (Checking the branch out in the
  main checkout fails: the branch is already checked out in the worktree.)

**Only one session at a time may write to `data/`.** Two sessions once extended `data/train`
at the same moment (`--n 1110` / `--n 1150`); the manifest had to be restored.

## Next steps, in order

1. ~~Build and check the test suites~~ (done, see "Test suites").
2. Rent the GPU. **Agreed plan** (user, 2026-10-14):
   - RunPod **Secure Cloud, on-demand** (never spot: `train.py` can't resume). ~$3/hr per
     H100 SXM.
   - Create a **network volume ≥ 150 GB** first, mounted at `/workspace`.
   - **1× H100 SXM pod for setup**, with SSH over exposed TCP so rsync works.
   - Then a **4× H100 SXM pod on the same volume** for the training burst.
   - Drop back to 1× (or stop) while waiting on harness work: idle 4× costs ~$12/hr.
3. **On the 1× pod:** `apt-get install -y rsync` on the pod, then the two rsyncs (worktree code,
   main-checkout `data/`, ~6.2 GB; commands in the `scripts/gpu_bootstrap.sh` header), then
   `bash scripts/gpu_bootstrap.sh`. Everything (uv, its Python 3.12, the env, weights,
   configs) lands on the volume. Note the measured s/step and peak memory from `smoke.log`
   and update the compute table below. Stop the pod.
4. **On the 4× pod:** `nohup bash scripts/run_parallel.sh configs/R0_devloss.yaml
   configs/R?_1h_*.yaml > /workspace/logs/parallel.log 2>&1 &`. That's one run per GPU,
   with the next queued config starting as a GPU frees up: R0–R3 start together, R4 takes
   R0's GPU a few minutes later, R5 starts when the first of R1–R3 finishes.
5. **Finish the eval harness** (`eval_gen.py`, `eval_report.py`; see "Eval harness"). Stop the
   4× pod once phase 1 has trained; the checkpoints stay on the volume, and harness debugging
   needs only 1×. Then: R0 generation eval on dev, phase-1 dev evals, then choose E and the
   top 2 LRs. After that R6–R7, test suites on the final checkpoints, listening log, report.
6. Human-recorded test set: label each episode with the same `events` schema.

## Environment on this Mac

- `.venv/` (uv, Python 3.12): torch 2.9.1, moshi 0.2.13, pocket-tts-timestamped 1.2.0,
  anthropic 1.14.0, sphn, sentencepiece, matplotlib. `source .venv/bin/activate`.
- `../moshi-finetune/`: clone of kyutai-labs/moshi-finetune on local branch
  `silence-audio-mask`, i.e. upstream main `2acc879f` plus the audio-mask patch as uncommitted
  changes. check.py imports its Interleaver. Training happens on a rented GPU, not here.
- `models/pocket-tts/languages/english_2026-09/` (372 MB): Pocket TTS weights from the
  ungated `kyutai/pocket-tts-without-voice-cloning`. They load from disk, so no HF access is
  needed at run time.
- `.env`: `ANTHROPIC_API_KEY` (gitignored). Never print or log it.
- `cache/tts/pocket/`: rendered clips, keyed by `sha1(backend|voice|text)[:16]`. They're
  stored raw; retime and gate run at assembly. Rebuilds reuse them.
- Pocket TTS throughput: **~12× realtime with 4 workers** (up to 17× on small extensions).
  The 10 h build was 7,661 clips ≈ 7.8 h of speech in 37.6 min.
- `--tts espeak` loads `libespeak-ng.so.1`, which is Linux-only, so it fails on this Mac.
- The Moshi weights (~15 GB) aren't here; the GPU bootstrap downloads them.

## Data: what's built and how

All builds use seed 0. Episodes are stereo 24 kHz PCM_16 wavs (**left = Moshi, right = user +
friend + noise**), each with a JSON (`alignments`, `events`, `meta`) and a `.spec.json`, plus
`manifest.jsonl`, `episodes.jsonl` and `checks.json` per set. `python -m silence.transcript
<ep.json>` prints an episode as a timed script.

### Content packs (`packs/`)

- `packs.jsonl`: 2,500 packs, 500 per family (B1, B2, B3, ORD, DECOY), written by
  `claude-sonnet-5-5`. That took 322 API calls (~0.69M input / ~0.81M output tokens).
  - DECOY packs are stored as `family: "ORD"` with a `decoy` field.
  - `split` is decided by hash: `sha1(key_text) % 1000 < 150` → test. That gave 387 test
    packs: B1 94, B2 75, B3 61, ORD 79, DECOY 78.
- 620 candidates were rejected, logged with reasons in `rejects.jsonl`:
  - schema checks: no D3a 206; D3a on the last line 40; decoy word missing 26; unanswered
    question 16
  - too long for 57 s: B3 106, B2 99
  - near-duplicates (content-word Jaccard ≥ 0.6 within a family): 93
  - a few speakability failures
- `smoke.jsonl`: the 40-pack smoke test.
- `gen_packs.log`: the run log.
- Top-up: rerun with the same `--out`. Each family is topped up to `--per-family`. The
  topic/style sequence is `random.Random(0)` and replays on every run, so only the avoid
  list differs.
- `packs_test_screened.jsonl`: the 348 test packs the test suites are built from (see below).

### Train (`data/train`)

`python -m silence.build train --packs packs/packs.jsonl --n 1150 --out data/train`

- `--n 1000` reached only 9.07 h (mean episode 32.6 s, not ~36 s), so it was extended to 1150
  with the same seed. Types and packs come out in a fixed order, so the first 1,000 episodes
  are byte-identical.
- `python -m silence.make_splits data/train --hours 1 10` writes first-K prefixes
  (1 h ⊂ 10 h) and `ledger.json`:

| Subset | Episodes | Packs | TTS clips | Hours | Speech h | Moshi speech h | Hold-silence h |
|---|---|---|---|---|---|---|---|
| train_1h | 113 | 116 | 885 | 1.01 | 0.70 | 0.25 | 0.39 |
| train_10h | 1,101 | 1,135 | 8,463 | 10.01 | 7.05 | 2.48 | 3.73 |
| full build | 1,150 | 1,185 | 8,851 | 10.49 | 7.36 | 2.59 | 3.92 |

- `train_1h` by type: B1 std 17 / double 3 / early 1 / rescind 1; B2 24; B3 std 18 / early 2 /
  rescind 2; ORD 31; DECOY 14. That's only 1–2 contrast episodes per type at 1 h. It's a
  random draw from TRAIN_MIX; say so in the report.
- No pack is used twice in train.
- Logs: `data/build_train.log` (first 1,000) and `data/build_train_1150.log`.

### Dev (`data/dev`), for model selection only

`python -m silence.build_dev --exclude data/train --out data/dev`

- 200 episodes / 1.97 h. `dev_80.jsonl` (0.77 h) is the first 80 of `dev_200.jsonl`.
- Exactly balanced: 20 / 50 per family (B1, B2, B3, ORD+DECOY), each family split by its
  TRAIN_MIX sub-mix, families round-robin.
- Train distribution (train voices, phrasings, conditions) on **unseen scripts**. Episodes
  are ≤ 60 s.
- **No overlap with train** (the user's requirement):
  - It excludes every pack used anywhere in `data/train`, including B1 `pack2`.
  - A screen also drops candidates that share any pack string with a training pack, or are
    near-duplicates (Jaccard ≥ 0.5). That dropped 200 of 928 candidates. It's stricter than
    needed, because it compares raw pack strings.
- `check_overlap` result: 0 shared packs or episodes; closest script Jaccard 0.47 (median
  0.22); 1 shared pack line ("Okay."). 407 phrase-bank line instances (137 distinct) are
  shared by design.
- Use: validation loss on all 200 every epoch. Generation eval on `dev_80` in phase 1 and on
  all 200 in phase 2 and for the final checkpoint choice. Never trained on, never reported
  as the result.
- `episodes.jsonl` says `split: "dev"`; each episode's `meta.split` says `"train"` (the
  phrase / voice pools it used).
- **Trap:** `build.py train` doesn't know about dev. Extending train past 1,150 with the same
  seed draws exactly the leftover packs dev took. Any future train build must skip the dev
  packs (`meta.pack` / `pack2` in `data/dev`).

### Test suites (`data/test_*`), reported once on final checkpoints

```
python -m silence.screen_packs --split test --against data/train data/dev --out packs/packs_test_screened.jsonl
for s in test_script test_word test_voice test_distract test_cond test_ordinary test_all; do
  python -m silence.build $s --packs packs/packs_test_screened.jsonl --n 90 --out data/$s; done
```

- The screen drops test packs containing a line **actually spoken** in train/dev episodes,
  or near-duplicating a train/dev pack (Jaccard ≥ 0.5). It dropped 39 of 387, leaving B1 81,
  B2 60, B3 51, ORD 79, DECOY 77. It compares spoken lines, not raw pack strings, because
  e.g. `friend_away` "Okay, I'm back" lines are spoken only in test-only D4 episodes.

| Suite | Packs | Wording | Voices | Mix / extra |
|---|---|---|---|---|
| test_script | test | train | train | TRAIN_MIX |
| test_word | test | **test** | train | TRAIN_MIX |
| test_voice | test | train | **test** | TRAIN_MIX |
| test_distract | test | train | train | B2_std 50 / B2_D4 20 / B3_std 30; dense: every B2 gets a D2a stall, every B3 a D2b pause + D3b |
| test_cond | test | train | train | COND_keyword 35 / COND_phone 35 / B1_minutes 30 |
| test_ordinary | test | train | train | ORD 70 / DECOY 30 |
| test_all | test | **test** | **test** | TRAIN_MIX; no dense distractors, D4 or new conditions |

- Test episodes have no 60 s cap; B1 "a minute" runs past 60 s. That's fine for generation
  eval, since the context is 240 s.
- **The suites are NOT paired** (checked 2026-10-14). `PackSampler` seeds on
  `packs:{seed}:test`, but it uses one RNG for all families and shuffles each family's queue
  lazily, in the order the suite's own type sequence (`types:{seed}:{split}`) first asks for
  it. So each suite draws its own episode types and its own pack order. test_script /
  test_word / test_voice / test_all share only 42–68 of their 90 packs pairwise, and their
  type mixes differ (B1: 6 / 11 / 14 / 14 episodes). Differences between suites therefore
  mix the held-out axis with script and type-mix differences. Comparisons between models on
  the same suite are fully paired.
- test_word and test_all use the test B1 durations (8 / 12 / 25 s, never trained), so B1
  timing there confounds new wording with new N.
- test_distract and test_cond need ~63 B2 packs from 60. The sampler silently reshuffles and
  reuses (`uses` is counted but never printed), so ~3 B2 scripts repeat in each.
- test_word holds out the **user's** wording. Moshi's acks ("Sure.", "Okay, I'll wait.") come
  from a shared list, and a couple of B3 option lines coincide ("The first one is taking the
  train."). That's expected.
- **Built and checked (2026-10-14, ~6 pm):** all 630 episodes pass check.py.

| Suite | Hours | Disk | Test packs | Closest train script (Jaccard) | Phrase-bank lines shared with train | Pack lines shared |
|---|---|---|---|---|---|---|
| test_script | 0.80 | 266 MB | 92 | 0.44 | 144 (82 distinct) | 0 |
| test_word | 0.82 | 271 MB | 92 | 0.43 | 43 (11; Moshi acks + 2 B3 option lines) | 0 |
| test_voice | 0.83 | 275 MB | 93 | 0.43 | 172 (94) | 1 ("Sure.") |
| test_distract | 1.15 | 382 MB | 89 | 0.43 | 262 (91) | 1 (D4 friend "Okay, I'm back.", by design) |
| test_cond | 1.14 | 378 MB | 90 | 0.43 | 60 (8) | 0 |
| test_ordinary | 0.54 | 179 MB | 90 | 0.43 | 0 | 0 |
| test_all | 0.82 | 271 MB | 95 | 0.43 | 40 (12) | 1 ("Take your time.") |

  0 shared packs or episodes with train in every suite. test_voice and test_all use only the
  4 test voices; the others use train voices. `data/` totals 6.2 GB (train 3.6, dev 0.65,
  tests 2.0).
- Pocket TTS logged "Maximum generation length reached without EOS" twice (test_script,
  test_distract). A duration scan of all 14,581 distinct clips used in train/dev/test found
  no run-on clip: median 0.35 s/word, p99 0.70, max 1.76 (one-word acks, raw clips before
  gating).

### Human test set (not started)

~40 episodes: the user plus 1–2 friends on one mic, labeled with the same `events` schema.
Final models only, reported separately.

## Generator design (agreed with the user; don't relitigate without asking)

**Training mix by episode count, family shares 15 / 25 / 20 / 40** (`specs.TRAIN_MIX`):

| Family | Share | Variants (weights) |
|---|---|---|
| B1 timed hold ("give me N seconds") | 15 | std 9, double 2 (two holds in one episode), early answer 2, rescind 2 |
| B2 side conversation ("stay quiet until I'm back") | 25 | std 25 |
| B3 hear me out ("let me describe all three options") | 20 | std 16, finishes early 2, rescind 2 |
| Ordinary | 40 | ORD 30 (⅓ of user turns have a 0.8–2 s mid-turn pause), DECOY 10 |

- B2 rescind was deliberately cut (a cue release is already the release).
- Contrast episodes (early answer, rescind, finishes early) teach following the instruction
  rather than "go deaf for a while".
- Lexical decoys use hold words with no hold intent ("I just got back from Boston", "took
  ten seconds").
- **B1 N:**
  - Train N ∈ {3, 5, 10, 15, 20, 30} s, capped so the episode fits the 57 s budget.
  - early / rescind use N ≥ 10; double uses N ≤ 12.
  - Test-only: {8, 12, 25} s; "half a minute" and "a minute" only in test_cond.
- **Distractor codes:**

  | Code | What it is |
  |---|---|
  | D1a | friend → user question |
  | D1b | user → friend question |
  | D1c | friend question that could be for Moshi |
  | D2a | 2–8 s stall in a side conversation (50% of B2; always in dense suites) |
  | D2b | 3–5 s pause between options |
  | D2c | fillers during a B1 hold (50% of B1 std with N ≥ 5) |
  | D3a | return-cue lookalike |
  | D3b | completion lookalike |
  | **D4** (test only) | the *friend* says "Okay, I'm back!"; Moshi must stay quiet |

  Distractor lines are **written by the LLM in context**: they're tagged lines in each
  pack's `side`, D3b is `aside`, D2c is `mutter`, D4 is `friend_away`. So they're held out
  together with the scripts. They are not inserted from a lexicon.
- **Voices (Pocket TTS):** Moshi is always `michael`. Test-only: eve, george, jane,
  peter_yearsley. Train: the other 16 English voices. Only 4 unseen voices is a known
  limitation; say so in the report.
- **Phrase bank** (`phrasebank.py`): instructions, return cues, rescinds, early answers,
  completions, B1 durations, friend arrivals and names each have train vs test lists.
  Moshi's acks are one shared list. `STORY_INSTR` is unused.
- **Timing and mixing** (`specs.py`, `assemble.py`):
  - Turn gap: lognormal(−0.9, 0.45), clipped to [0.12, 1.2] s, × `GAP_SCALE` 1.5. This is a
    placeholder; ideally fit it on DailyTalkContiguous.
  - Moshi gap 0.45–1.35 s.
  - Side-conversation gaps overlap by up to 0.25 s 15% of the time.
  - Moshi acks the instruction in 70% of holds.
  - Friend gain −9 to −4 dB with a synthetic RIR (rt60 0.2–0.6 s).
  - Noise (pink / brown / white_lp) on the user channel only, at SNR 15–35 dB against a
    −20 dBFS speech target. Fillers at −6 dB; peak limit 0.99.
- **Length:** episode budget `BUDGET_DUR` 57 s. Train episodes must be ≤ `MAX_TRAIN_DUR`
  60.0 s, so each is exactly one 60 s window. Over-length B2 episodes get a shorter side
  conversation, otherwise the next pack.
- **No episode opens with Moshi** (`load_packs` drops packs that do).

**Timing fixes in tts.py / assemble.py. Keep all three:**
- `tts.retime()`: Pocket's attention timestamps put the first word at ~0 s when a clip opens
  with silence. Words stamped before the energy onset move onto the onset.
- `assemble.gate()`: zeroes each clip outside [onset − 50 ms, offset + 100 ms] with 10 ms
  fades. This removes Pocket's ~−42 dBFS noise floor from Moshi's channel during holds.
- `speech_span()` onset must be followed by real speech (a frame > −15 dB within 150 ms, or a
  dense 100 ms reaching −23 dB), so breath or noise at the start of ~5% of clips isn't taken
  as the onset.

**check.py** (moshi-finetune's own Interleaver; a stand-in tokenizer unless `--spm`; writes
`checks.json`, plus plots and `index.html` with `--plots`). Fails on:
- hold-silence amplitude > 1e-3, outside the gate margins
- word vs audio onset misalignment > ±0.15 s
- B1 timer error > 0.3 s
- cue latency outside [0.25, 1.5] s
- a human turn < 6 dB above the floor
- human/Moshi overlap > 0.05 s
- a train episode > 60 s
- Moshi speaking first
- no EPAD right before the release, or non-PAD text inside a hold

It globs `episodes/*.wav`, not the manifest, so orphans from an earlier larger `--n` get
counted.

## Pronunciation (Pocket TTS garbles unusual words; the user heard "latte" break)

- The `gen_packs.py` prompt asks for plain, common English: no loan words, unusual brands,
  acronyms, rare place names or uncommon first names.
- `gen_packs.is_speakable()` drops packs with digits, non-ASCII characters, all-caps
  acronyms (except I / OK / TV), or a word in `HARD_WORDS`. Add to `HARD_WORDS` when
  listening turns up a bad word.
- Phrase banks pass the filter today, but nothing enforces it.
- Not built: ASR every rendered clip (e.g. whisper) and drop clips above ~10% WER. Worth
  adding if listening keeps turning up garbled words.

## Layout

```
silence/phrasebank.py      instruction / cue / ack phrasings, train vs test lists
silence/splits.py          which axis each test suite holds out
silence/voices.py          voice pools (pocket default; kyutai, espeak)
silence/specs.py           episode builders, TRAIN_MIX, timing
silence/tts.py             TTS backends (pocket | kyutai | espeak) + ClipCache + retime
silence/assemble.py        timeline -> stereo wav + {alignments, events, meta} json; gate, mixing
silence/build.py           CLI: specs -> TTS (parallel) -> episodes, manifest.jsonl, episodes.jsonl
silence/check.py           checks via moshi-finetune's Interleaver; --plots for plots + index.html
silence/transcript.py      print episodes as timed transcripts
silence/gen_packs.py       LLM content packs (Anthropic API, reads .env); schema + speakability filters
silence/make_splits.py     nested 1h ⊂ 10h jsonl subsets + ledger.json; `exposure` subcommand
silence/build_dev.py       dev set: unused + screened train packs, balanced, dev_80 ⊂ dev_200
silence/screen_packs.py    held-out pack file minus packs overlapping train/dev
silence/check_overlap.py   train vs held-out audit (packs, near-dup scripts, shared lines, voices)
silence/make_configs.py    moshi-finetune yamls per run (fixed epochs) + checkpoint -> epoch maps
silence/list_kyutai_voices.py, calibrate_offset.py   only for the optional 1.6B Kyutai TTS
scripts/gpu_bootstrap.sh   GPU box: moshi-finetune + 3 patches (uv.lock env), pinned weights, configs, smoke
scripts/run_parallel.sh    run configs one per GPU (own CUDA device + master port + log); skips .done, refuses leftover run_dirs
patches/moshi-finetune-audio-mask.patch    mask_main_speech_audio (no audio loss on Moshi's TTS speech)
patches/moshi-finetune-eval-reload.patch   fresh eval loader per eval + text/audio dev loss logged
patches/moshi-finetune-text-tail.patch     no text loss on the zero-padded tail of a window
packs/                     packs.jsonl, packs_test_screened.jsonl, rejects.jsonl, smoke.jsonl, gen_packs.log
demo/                      14 example episodes (json, plots, index.html, transcripts.txt; wavs gitignored)
data/                      gitignored: train, dev, test_*, build logs
```

**Demo:** 14 episodes (one per type, plus B1 double, phone call, B1 minutes), 14/14 pass.
It's a listening / plumbing check only. It **can't be rebuilt**: `build.py` has no `demo`
split anymore, `packs/demo_packs.jsonl` is deleted, and its pack ids aren't in `packs.jsonl`.
The committed `index.html` points at wavs that only exist in the main checkout.

## How moshi-finetune consumes the data (verified in the source)

- JSON next to each wav (`splitext(path)[0] + ".json"`) with
  `{"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...]}`. Extra keys are ignored.
- jsonl rows are `{path, duration}`. sphn resolves relative paths against the jsonl's own
  directory; the bootstrap writes absolute ones anyway.
- `keep_main_only=True`: **only Moshi's words become text tokens**. User speech is
  audio-only input.
- Text stream: word tokens at their start frame, PAD (3) elsewhere, EPAD (0) on the frame
  before a word that follows padding. zero token = −1. 12.5 Hz frames.
- Loss = text CE + audio CE over Moshi's 8 codebooks. The user's codebooks are input only.
  - `text_padding_weight` 0.5 down-weights **both** PAD and EPAD.
  - `first_codebook_weight_multiplier` 100 makes the audio loss ≈ 93% semantic codebook.
  - A separate higher EPAD weight would be a worthwhile ablation, since EPAD is the "start
    talking now" decision.
- Files are cut into consecutive `duration_sec` windows from t=0, each an independent
  context. Hence the ≤ 60 s episode cap.
- Moshi's context is 3000 frames = 240 s.
- Training iterates **real epochs**: an infinite stream, reshuffled without replacement each
  pass (seed +1 per epoch), with no drop_last loss. So epoch = steps × batch / n.
- `InterleavedTokenizer` passes a frame count where `prepare_item` expects seconds. It works
  only because the result is cropped; don't "fix" it.
- Optimizer: AdamW, betas (0.9, 0.95). Adam normalizes the gradient scale, so loss weights
  change emphasis, not step size.
- LR schedule: `OneCycleLR(max_lr=lr, total_steps=max_steps, pct_start=0.05)`: warmup from
  lr/25, cosine decay to ~0. The swept "LR" is the peak. **A mid-run checkpoint isn't a
  shorter run**: it was taken at high LR, before annealing.
- LoRA: `W' = W + scaling·B·A` on every `nn.Linear` (main transformer, depformer, heads). B
  starts at zero, so step 0 is the base model. r = 64 is ~194M parameters, ~0.39 GB per bf16
  checkpoint.
- **`train.py` has no resume** (no optimizer state saved) and refuses an existing `run_dir`.
- YAML is loaded strictly: unknown keys raise. An unpatched tree fails loudly on
  `mask_main_speech_audio`.
- Recommended upstream: LoRA r=128, lr 2e-6 for 2000 steps (the code default is 1e-4).
  ~40 GB peak at batch 16 × 100 s on one H100.

**Traps found in upstream and how we handle them:**
- `pip install -e .` can't resolve: pyproject pins `sphn==0.1.12` but takes moshi from git
  main, which now needs sphn ≥ 0.2. Use `uv sync --frozen`. uv.lock pins moshi `061cc4c`
  (0.2.4a1), torch 2.6.0, huggingface-hub 0.28.1 (it has `huggingface-cli`; there's no `hf`).
- **The eval loader is built once but is a finite generator.** Every evaluation after the
  first sees zero batches and logs NaN. It also drops the last partial batch, and never logs
  the text/audio split. → eval-reload patch. Upstream has a `fix_eval` branch, not checked.
- **Text past the end of a short file is PAD and counts in the text loss.** The audio there is
  −1 and masked. At 1 h that's ~46% of all text frames (113 × 750 window frames vs ~45k real).
  → text-tail patch sets them to −1.
- `num_ckpt_keep` defaults to 3 and deletes older checkpoints. → `num_ckpt_keep: null`.
- No revision argument for the weights. → the bootstrap downloads a snapshot at
  `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289` and the configs point at the files. The repo has
  no `config.json`, so moshi uses its built-in 7B config: the architecture comes from the
  uv-locked moshi package. `hf_repo_id` still makes one (404) request, so the box needs
  network access to HF.
- sphn skips unreadable wavs with only a log line (`on_error=log`). The smoke check warns on
  "error" lines.
- PyYAML reads `2e-06` (no dot) as a string. make_configs writes `2.000000e-06` (`.6e`, so
  phase-2 LRs keep full precision).
- `finetune/distributed.py` reads `os.environ['CUDA_VISIBLE_DEVICES']` with no default, so a
  torchrun without it set crashes with KeyError. The bootstrap smoke run and
  `run_parallel.sh` both set it.
- moshi-finetune says only `requires-python >= 3.10` and has no `.python-version`, so uv
  picks the newest Python (3.14), for which torch 2.6.0 has no wheels. The bootstrap pins
  `UV_PYTHON=3.12`.
- The dev loss averages per-batch means, and 200 = 12 × 16 + 8, so the last 8 dev episodes
  count about double (~4% of the weight). It's the same at every checkpoint, so comparisons
  are unaffected; mention it if absolute dev-loss values are reported.

**Loading a checkpoint for inference** (eval harness):
`CheckpointInfo.from_hf_repo(repo, moshi_weights=…, mimi_weights=…, tokenizer=…,
config_path=ckpt/config.json, lora_weights=ckpt/lora.safetensors).get_moshi(...)`, then
`LMGen`. **Pass the checkpoint's `config.json`**: it carries `lora: true`, `lora_rank: 64`,
`lora_scaling: 2`. Without it, moshi asserts or defaults to rank 128 and fails on shapes.
`fuse_lora=True` for speed. Checkpoints live in
`run_dir/checkpoints/checkpoint_XXXXXX/consolidated/{lora.safetensors, config.json}`.

## Run matrix (agreed 2026-10-14)

**Epochs are fixed per phase, not steps.** 1 h is a proof of concept: does the model start
picking up the behavior at all?

Fixed for every run (`silence/make_configs.py`):
- the pinned weights
- LoRA r=64, scaling 2; batch 16; `duration_sec: 60` (one episode = one window)
- AdamW weight decay 0.1; OneCycle `pct_start` 0.05
- first codebook ×100; `text_padding_weight` 0.5
- **`mask_main_speech_audio: true`** (main-speech mask pre 0.08 s / post 0.24 s)
- all three patches applied
- seed 0, one seed per run
- every checkpoint kept

| Phase | Run | Data | Peak LR | Epochs | Steps | Checkpoints |
|---|---|---|---|---|---|---|
| 0 | R0_devloss | none (base model) | 0 | — | 2 no-op | dev loss only |
| 1 | R1–R5 | train_1h (113 eps) | 2e-6, 1e-5, 5e-5, 2e-4, 5e-4 | 20 | 142 each | 21 each (every 7 steps ≈ 0.99 epoch, plus the last) |
| 2 | R6–R7 | train_10h (1,101 eps) | top 2 from phase 1 | E (from phase 1) | ⌈E·1101/16⌉: 345 at E=5, 689 at E=10 | every 69 steps (≈ 1 epoch), plus the last |

- `max_steps = ceil(E · n / 16)`, with n counted from the jsonl. `configs/<run>.epochs.json`
  maps each checkpoint to its exact epoch.
- **Phase 1 chooses E and the top 2 LRs** from dev. 2e-6 is expected to be a no-op at 142
  steps. It doubles as a pipeline check: if it doesn't match R0, something in the
  LoRA / eval path changes behavior by itself.
- **Report caveat:** the 1 h point at E epochs is the epoch-E checkpoint of a 20-epoch
  OneCycle run, not a run trained to exactly E.
- **Dropped or on hold** (user's call): a separate 1 h rerun at exactly E; the 100 h build
  and run; the stretched-holds ablation (×3 would break the 60 s cap); the 1,000 h stretch; a
  behavior-loss validation script; a second seed.

**Compute.** Upstream's README measures ≈ 12k tokens/s on 1× H100, counting batch × seconds ×
12.5 × 9 tokens per step: 16 × 60 × 12.5 × 9 = 108k → **≈ 9 s/step** (the earlier planning
figure of 15 s/step was pessimistic). Measure in the smoke run.
- Phase 1 training: 710 steps ≈ 1.8 h.
- Per-epoch dev loss on 200 episodes (13 forward batches every 7 steps; forward only ≈ ⅓ of
  a training step per batch) adds roughly +60%, so ≈ 2.9 h, ~35 min per run.
- Phase 2: E=5 ≈ 0.9 h, E=10 ≈ 1.7 h per run (eval adds ~10% there).
- Total ≈ 6–7 H100-hours of training. With setup, smoke and generation evals, ~10–12
  GPU-hours ≈ $30–36 at RunPod's ~$3/hr.
- **Wall-clock on 4× H100** (one run per GPU via `run_parallel.sh`; same GPU-hours):
  phase 1 ≈ 1.2 h (two run-lengths) instead of ~2.9 h on 1×. Phase 2 at E=10 ≈ 1.7 h
  (R6 and R7 side by side, while the other 2 GPUs run generation evals).
  Don't split one run across GPUs: moshi-finetune's `batch_size` is per GPU, so that would
  change the recipe, and 142-step runs gain little.
- Disk: phase 1 checkpoints ≈ 105 × 0.39 GB ≈ 41 GB.

**Exposure accounting:** each episode is one window, so exposure = steps × batch × GPUs.
`python -m silence.make_splits exposure --episodes N --mean-dur S --steps K` gives epochs
and audio-hours seen. Report unique episodes, packs and TTS clips, and total / speech /
hold-silence hours, separately (`ledger.json`).

## Eval harness (details agreed 2026-10-14; `eval_score.py` written)

**Agreed decisions (user, 2026-10-14):**
1. Timing from Silero VAD on Moshi's decoded audio; words from Moshi's own text stream for
   dev sweeps. Whisper ASR only on final test evals and the listening log.
2. 1 sample per episode for dev sweeps, 3 for final test evals. Same seed per episode across
   models.
3. LLM judge for resume appropriateness: test evals only, 1 sample (~2.5k calls ≈ $10; ask
   before running).
4. Generation on dev_80 every epoch in phase 1 if R0 shows ≤ 2 min per checkpoint, else every
   2; dev_200 for the top candidates and in phase 2.
5. Re-force the script between scored windows (per-frame forcing mask), so every hold is
   scored, not just the first. Ordinary episodes: free-run after the first user turn.
6. Thresholds as below (2.5 s ack grace; ≥ 1 s or > 3 words; 3 s windows; B1 tolerance
   max(1.5 s, 0.2N)); report their sensitivity.

**`silence/eval_score.py`** (done): free windows per episode, VAD segments, per-hold
violation / outcome (on_time | late | silent) / latency / timer error / talking-through /
distractor reactions, ordinary responses (episodes without holds only), barge-in into
mid-turn pauses. B1 N = 3 s holds have no violation region (grace 2.5 s + tolerance 1.5 s
> N), so `violated` is None there and only their timing counts. **Oracle test on dev_80**
(ground-truth Moshi channel): 0% violations (n=58), 100% on time (n=63), cue latency median
0.92 s, timer error median −0.11 s, 100% ordinary responses, 0 distractor reactions. Negative
control (user channel scored as Moshi): 76% violations, 94% silent at release.
`silero-vad` must be added to the GPU env for eval.

Original proposal:
- An offline driver around `moshi.models.lm.LMGen`, with **prefix forcing**: force Moshi's
  text + audio tokens to ground truth until `instr_end` (ordinary: until the first user turn
  ends), then free-run. This needs a small override of the sampled tokens for the first k
  frames. **Must batch** episodes: at batch 1 generation is only ~2–3× realtime.
- Score **decoded audio** (Mimi decode → Silero VAD + ASR with word timestamps); log the text
  stream as a diagnostic.
- Metrics from each episode's `events`:
  - **hold violation:** ≥ 1 s or > 3 words of Moshi speech inside [instr_end + 2.5 s ack
    grace, release]
  - **distractor violation:** a violation starting within 3 s after a distractor, per code
  - **resume success / latency:** onset within [cue_end, cue_end + 3 s]. B1: success if
    |onset − (instr_end + N)| ≤ max(1.5 s, 0.2N), plus the signed error.
  - **resume appropriateness:** LLM judge on the post-release transcript
  - **ordinary responsiveness:** a response within 3 s of the user's turn end; compare to R0
  - **over-suppression:** no response when one was due
  - optional voice-drift speaker similarity. The audio mask means Moshi's speech audio gets
    no training signal, so drift at high LR shows up only here or by ear.
- One sample per episode at default temperatures; 95% bootstrap CIs. Expect ±13 pts on
  pooled hold behavior with dev_80, ±8 with dev_200, ±10 per 90-episode test suite.
- First GPU job of the harness: R0 on dev, which doubles as the throughput benchmark. That
  decides whether generation runs every epoch or every 2–4.
- Plots: rate vs. data size (log x), one line per LR, per suite. A listening log of
  successes and failures per suite.

## Known gaps / TODO

- Whether anyone listened to the demo isn't recorded; ask.
- Turn gaps are a placeholder lognormal. Noise is synthetic.
- The B1 fillers (D2c) come from each pack's `mutter`.
- Pack reuse is silent when a family runs out (test_distract / test_cond, ~3 B2 scripts).
- `build.py train` must learn to skip dev packs before any further train build.
- `requirements.txt` misses `whisper_timestamped` (calibrate_offset.py) and
  `huggingface_hub[cli]`; `sentencepiece` and `torch` come in only through moshi /
  pocket-tts. The GPU box doesn't use it (uv.lock there).
- The demo wavs are gitignored, so `demo/index.html` plays only in the main checkout.
  `configs/` is gitignored, since it's generated on the box with absolute paths.
- The KyutaiTTS (1.6B, GPU) backend is optional and untested; calibrate its word offset first.
- 100 h, if it comes back: ~78 h of speech ≈ 6.5 h CPU here, and it needs more packs (B2
  first) and the dev-pack exclusion.

## Working with the user (Jaewoo)

- Casual and direct. He pushes back when estimates look inflated; give real numbers with
  the arithmetic.
- Render math in LaTeX in explanations.
- Ask before generating at scale or spending money (API calls for thousands of packs, GPU
  rentals).
