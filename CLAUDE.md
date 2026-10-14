# CLAUDE.md: moshi-silence

Handoff from a planning session (claude.ai, 2026-10-14). Read this fully before touching code.

## The task (one-day research sprint)

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

## Current status (as of 2026-10-14, ~5 pm ET)

**Environment on this Mac: set up, nothing to install.**
- `.venv/` (uv, Python 3.12) with torch 2.9.1, moshi 0.2.13, pocket-tts-timestamped 1.2.0,
  anthropic 1.14.0, sphn, sentencepiece, matplotlib. Activate with `source .venv/bin/activate`.
- `../moshi-finetune/`: a clone of kyutai-labs/moshi-finetune. check.py imports its
  Interleaver; training happens on a rented GPU, not here.
- `models/pocket-tts/languages/english_2026-09/` (372 MB): Pocket TTS weights
  (model.safetensors, tokenizer.json/.model, embeddings/ with all 21 English voices + 6
  non-English we don't use). Downloaded from the ungated
  `kyutai/pocket-tts-without-voice-cloning`; PocketTTS loads it from disk, so no HF access
  is needed at run time.
- `.env`: `ANTHROPIC_API_KEY` (gitignored). Never print or log it.
- `cache/tts/pocket/`: rendered clips (wav + word-timestamp json), keyed by
  sha1(backend|voice|text); ~5.2k at 5 pm and growing with the 10 h build. Rebuilds reuse them.
- **Not on this Mac:** Moshi model weights (`kyutai/moshiko-pytorch-bf16`, ~15 GB). The
  trainer and eval download them into the HF cache on the GPU box. Training a 7B bf16 model
  isn't feasible here.
- **Git:** local repo on `master`, no remote. Two commits (initial generator, demo refresh).
  As of 5 pm a lot of work is uncommitted (most of `silence/`, `silence/transcript.py`,
  `patches/`, the rebuilt `demo/`). Commit it before branching off for new work.

**Generator: working.**
- The Pocket TTS demo is now 14 episodes (one per type, plus B1 double hold, COND phone call,
  and minute-scale B1), and **14/14 pass check.py** through moshi-finetune's own Interleaver.
  - B1 re-ask error vs instr_end + N: −0.09 to +0.14 s (jitter is ±0.25 s by design)
  - Moshi word starts vs its audio onset: within ±0.04 s
  - holds: all PAD, with EPAD on the frame right before the release
- `python -m silence.transcript <episode.json>` prints an episode as a timed transcript
  (who says what when, holds, distractor tags), which is handy for reading episodes
  without listening.

**Content packs: done.** `packs/packs.jsonl` has 2,500 packs (500 per family: B1, B2, B3,
ORD, DECOY; `--test-frac 0.15` reserves test-only packs). The smoke test (`packs/smoke.jsonl`,
40 packs) came first. The full run took 322 API calls (~0.69M input / ~0.81M output tokens)
and rejected 620 candidates (written to `packs/rejects.jsonl` with the reason). The biggest
reasons were schema checks (no D3a: 206; D3a on the last line: 40; decoy word missing: 26;
unanswered question: 16), too long for a 60 s episode (B3: 106, B2: 99), and near-duplicates
(93). Only a handful failed the speakability filter. We used 500 per family, not 1500; that
covers 10 h (1,000 episodes) with no pack reuse. 100 h would reuse packs ~5×, so top up first
(rerunning with the same `--out` tops each family up to `--per-family`).

**Train set: built and checked.** `data/train` has 1,150 episodes (10.49 h), and **all 1,150
pass check.py**.
- `--n 1000` only reached 9.07 h, because episodes average 32.6 s, not ~36 s. It was extended
  to `--n 1150` with the same seed. The first 1,000 episodes are byte-identical, since types and
  packs come out in a fixed order and clips come from the cache.
- `make_splits --hours 1 10` gives `train_1h.jsonl` = 113 episodes / 1.01 h and
  `train_10h.jsonl` = 1,101 episodes / 10.01 h, the first-K prefixes, so 1 h ⊂ 10 h.
- Logs: `data/build_train.log` (first 1,000: 7,661 clips, 37.6 min) and
  `data/build_train_1150.log`.
- Pocket TTS runs at **~12× realtime with 4 workers** on this Mac (up to 17× on the
  extension), about twice the earlier 6× estimate.
- No pack is used twice. Unused train packs are left over (B1 ~200, B2 ~190, B3 ~220,
  ORD ~90, DECOY ~310), because packs were generated equally per family while the mix isn't.
  These are the source for the dev set.
- (2026-10-14) Two sessions extended `data/train` at the same moment (`--n 1110` and
  `--n 1150`). The manifest was restored to 1,150 afterwards. **Only one session at a time
  should write to `data/`.**

**Dev set: built and checked.** `data/dev` has 200 episodes (1.97 h), and all 200 pass
check.py. Built by `silence/build_dev.py`; the log is `data/build_dev.log`.
- `dev_80.jsonl` (0.77 h) is the first 80 of `dev_200.jsonl`. Both are exactly balanced:
  20 / 50 per family (B1, B2, B3, ORD+DECOY), with each family split by its TRAIN_MIX
  sub-mix, and families round-robin so every prefix of 4k has k per family.
- Train distribution (train voices, phrasings, conditions), so it measures "learned the
  behavior on new scripts", not generalization. Episodes are ≤ 60 s.
- **No overlap with train** (the user's requirement). Dev packs exclude every pack used
  anywhere in `data/train` (all 1,150 episodes, incl. B1 `pack2`). An overlap screen also
  drops any candidate pack that shares a spoken line verbatim with a training pack, or whose
  key-text content-word Jaccard against a same-family training pack is ≥ 0.5. That screen
  dropped 200 of 928 candidates, mostly B2/B3 packs sharing a short line.
- `python -m silence.check_overlap data/train data/dev` reports:
  - 0 shared packs, 0 shared episodes
  - closest script Jaccard 0.47 (median 0.22); same-topic pairs remain, e.g. two different
    "car squeals on cold mornings" ORD packs
  - 1 pack line shared (the one-word ack "Okay.")
  - 407 shared phrase-bank line instances (137 distinct): instructions, cues, acks. These
    are shared by design; held-out wording is what `test_word` is for.
- **Use:** validation loss on all 200 every epoch. Generation eval on `dev_80` in phase 1 and
  on all 200 in phase 2 and the final checkpoint choice. Never trained on, never reported as
  the result.
- **TODO:** `build.py train` doesn't know about dev yet. Any future train build (e.g. 100 h)
  must skip dev packs (`meta.pack` / `pack2` in `data/dev`), or dev stops being held out.

**moshi-finetune patch: `patches/moshi-finetune-audio-mask.patch`.** It adds
`mask_main_speech_audio` (plus `main_speech_mask_pre_sec` 0.08 / `main_speech_mask_post_sec`
0.24). The audio loss is skipped on frames where Moshi speaks, padded by those margins. Text
loss is unchanged, and the audio loss still applies wherever Moshi is silent. That way the
synthetic `michael` TTS voice can't pull Moshi's voice. It touches `args.py`,
`data/interleaver.py`, `eval.py`, `train.py`, so validation loss is masked the same way.
Apply it on the GPU box's moshi-finetune checkout. **It's on for every run.**

**Fixed during the demo; keep both fixes if you touch tts.py / assemble.py:**
- Pocket TTS's attention timestamps put the first word at ~0 s whenever a clip opens with
  silence (median 0.3 s early, up to 1 s). Before the fix, Moshi's text tokens led its audio
  by up to a second, and every gap was measured from the wrong point; a cue latency that
  printed as 0.54 s was really ~1.3 s. `tts.retime()` moves any word stamped before the
  energy onset (10 ms frames within −30 dB of the clip peak) onto the onset.
- Pocket clips carry a ~−42 dBFS noise floor that leaked into Moshi's channel during holds.
  `assemble.gate()` zeroes every clip outside [onset − 50 ms, offset + 100 ms] with 10 ms
  fades. check.py skips the same margins when testing hold silence, and its `align=` column
  checks word times against audio onset (fails beyond ±0.15 s).
- (2026-10-14) ~5% of Pocket clips open with up to 1 s of breath/noise poking above −30 dB,
  which `speech_span()` took for the onset (Moshi text up to 0.9 s ahead of its audio).
  The onset now has to be followed by real speech (a frame > −15 dB within 150 ms, or a
  dense 100 ms reaching −23 dB). Checked on all 281 cached clips: 15 move, 14 clearly noise.

**Done since the 2 pm handoff:** git repo, demo rebuilt (14/14), pack smoke test, full
pack run (2,500), training plan revised (see "Run matrix"). The demo listening pass isn't
recorded anywhere; ask the user whether it happened.

**Not started:**
- the 7 test suites (`--n 90` each); build after the train set finishes
- `make_splits.py` on the 10 h build (1 h / 10 h jsonl files + ledger)
- the training-config generator (yamls from `ledger.json`) and the GPU-box bootstrap
- the eval harness (`LMGen` driver)
- the human-recorded test set
- training runs
- the report and demo

## Next steps, in order

1. ~~Check the train build, then `make_splits`~~ (done, 1,150/1,150 pass). Still worth doing:
   read 3–5 episodes with `silence.transcript` and listen to a couple. On the GPU box, rerun
   `make_splits.py data/train --hours 1 10 --abs` so the jsonl files hold absolute paths.
2. **Build the 7 test suites** (in progress; `--n 90` each, from
   `packs/packs_test_screened.jsonl`, log at `data/build_test.log`). `silence/screen_packs.py`
   dropped 39 of 387 test packs that contain a line spoken in train/dev or are near-duplicates
   (Jaccard ≥ 0.5) of a train/dev pack, leaving B1 81, B2 60, B3 51, ORD 79, DECOY 77.
   Unlike the dev screen, this one compares against lines actually spoken in train/dev
   episodes. Raw pack strings over-count: e.g. `friend_away` "Okay, I'm back" lines are only
   spoken in test-only D4 episodes. `test_distract` / `test_cond` need ~63 B2 packs, so
   about 3 B2 scripts repeat within those suites. Run `silence.check_overlap` per suite
   afterwards.
3. ~~Build the dev set~~ (done: `data/dev`, 200 episodes, see above).
4. ~~Training configs + GPU bootstrap~~ (written, untested on a GPU):
   - `silence/make_configs.py` writes one yaml per run plus a checkpoint → epoch map. Phase 1
     is 113 episodes × 20 epochs = 142 steps, a checkpoint and dev loss every 7 steps
     (≈ 0.99 epoch), 21 checkpoints per run.
   - `scripts/gpu_bootstrap.sh` clones moshi-finetune at `2acc879f`, applies both patches,
     downloads the pinned weights, makes jsonl paths absolute, writes the configs, and runs
     a 6-step smoke train that must show 2 checkpoints and no NaN.
   - Upload ~4.3 GB of data (train 3.6 GB, dev 0.65 GB) plus the test suites.
5. **Eval harness**, then R0 → R1–R5 → choose E and the top 2 LRs → R6–R7, then the listening
   log and the report. Eval details are still to be discussed with the user.

The 100 h build is on hold along with the 100 h run.

## Pronunciation (Pocket TTS garbles unusual words; the user heard "latte" break)

- `gen_packs.py`'s prompt asks for plain, common, easy-to-pronounce English: no loan words,
  unusual brands, acronyms, rare place names, or uncommon first names.
- `gen_packs.is_speakable()` drops any pack with digits, non-ASCII characters, all-caps
  acronyms (except I / OK / TV), or a word in `HARD_WORDS`. Add to `HARD_WORDS` whenever
  listening turns up a new bad word. The rejection counts print at the end of a run.
- Phrase banks and demo packs pass the same filter.
- Stronger catch-all, not built yet: transcribe every rendered clip with an ASR (e.g.
  whisper on this Mac) and drop clips above ~10% WER. Worth adding if listening keeps
  turning up garbled words.

## Layout

```
packs/packs.jsonl          LLM content packs (2,500); rejects.jsonl, smoke.jsonl, gen_packs.log
patches/                   moshi-finetune-audio-mask.patch (apply on the GPU box)
silence/phrasebank.py      instruction / cue / distractor phrasings, each split train vs test
silence/splits.py          which axis each test suite holds out (one axis per suite)
silence/voices.py          voice pools: pocket (default), kyutai, espeak
silence/specs.py           episode builders + TRAIN_MIX
silence/tts.py             TTS backends (pocket | kyutai | espeak) + ClipCache
silence/assemble.py        timeline -> stereo wav + {alignments, events, meta} json
silence/build.py           CLI: specs -> TTS -> episodes + manifest.jsonl + episodes.jsonl
silence/check.py           checks via moshi-finetune's own Interleaver; plots; index.html
silence/build_dev.py       dev set: unused + overlap-screened train packs, balanced, nested dev_80 ⊂ dev_200
silence/check_overlap.py   train vs held-out overlap audit (packs, near-dup scripts, shared lines, voices)
silence/screen_packs.py    held-out pack file minus packs overlapping train/dev (for the test suites)
silence/make_configs.py    moshi-finetune yamls per run (fixed epochs) + checkpoint→epoch maps
scripts/gpu_bootstrap.sh   GPU box: moshi-finetune + patches, pinned weights, configs, smoke train
patches/moshi-finetune-eval-reload.patch   fresh eval loader per evaluation (upstream NaN bug)
silence/make_splits.py     nested 1h ⊂ 10h ⊂ 100h jsonl subsets + ledger.json + exposure math
silence/gen_packs.py       LLM content packs via Anthropic API (reads .env); schema + speakability filters
silence/transcript.py      print built episodes as timed transcripts
silence/list_kyutai_voices.py, calibrate_offset.py   only for the optional 1.6B Kyutai TTS
```

## Design decisions (agreed with the user; don't relitigate without asking)

**Behavior families and training mix (by episode count): 15 / 25 / 20 / 40.**

| Family | Share | Variants |
|---|---|---|
| B1 timed hold ("give me N seconds") | 15% | 11 std / 2 early answer / 2 rescind |
| B2 side conversation ("stay quiet until I'm back") | 25% | all std, 1–3 distractors each |
| B3 hear me out ("let me describe all three options") | 20% | 16 std / 2 finishes early / 2 rescind |
| Ordinary | 40% | 30 normal Q&A (⅓ of user turns have a 0.8–2 s mid-turn pause) / 10 lexical decoys |

- B2 rescind was deliberately cut (a cue release is already the release).
- Contrast episodes (early answer, rescind, finishes early) exist so the model learns to
  follow the instruction rather than "go deaf for a while".
- Lexical decoys use hold words with no hold intent ("I just got back from Boston",
  "took ten seconds") so the model can't key on single words.
- B1 N ∈ {3,5,10,15,20,30} s for train; {8,12,25} s and "half a minute" / "a minute" test-only.

**Distractor codes** (in both train and test, with separate train/test lexicons):
D1a friend→user question, D1b user→friend question, D1c friend question that could be for
Moshi, D2a 2–8 s stall in a side conversation, D2b 3–5 s pause between options, D2c fillers
during a B1 hold, D3a return-cue lookalike, D3b completion lookalike. **D4 (test only):** the
*friend* says "Okay, I'm back!"; Moshi must stay quiet because the cue must come from the user.

**Test suites** (each changes one axis vs. train; see `splits.py`): test_script,
test_word, test_voice, test_distract (dense distractors + D4), test_cond (held-out waiting
conditions: keyword release "don't say anything until I say pineapple", phone call with
only the user audible, minute-scale B1), test_ordinary (regression), test_all (everything
new), plus ~40 human-recorded episodes (user + 1–2 friends on one mic; same event schema).

**Voices (Pocket TTS):** Moshi always `michael` (one fixed voice so its output voice doesn't
change between conditions). Test-only: eve, george, jane, peter_yearsley. Train: the other 16
English voices. Only 4 unseen voices is a known limitation; say so in the report.

**Episodes are ≤ 60 s** so each is exactly one training window at `duration_sec: 60`.

## How moshi-finetune actually consumes the data (verified by reading the source)

- Stereo wav, **left = Moshi, right = user**. JSON next to each wav with
  `{"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...]}`. Extra keys are ignored, which
  is why `events` and `meta` live in the same file.
- `train.py` builds the Interleaver with `keep_main_only=True`: **only Moshi's words become
  text tokens**. User speech is audio-only input.
- Text stream: word tokens at their start frame, PAD (id 3) elsewhere, EPAD (id 0) on the
  frame before a word that follows padding. zero token = -1. 12.5 Hz frames.
- Loss: text + 8 audio codebooks of Moshi's channel; user codebooks are input only.
  `text_padding_weight` (default 0.5) down-weights **both** PAD and EPAD. A separate (higher)
  EPAD weight is a worthwhile ablation: EPAD is the "start talking now" decision.
- Files are chunked into consecutive `duration_sec` windows from t=0, each an independent
  context. Hence the ≤60 s episode cap.
- Moshi's context is 3000 frames = 240 s (sliding window).
- **`train.py` has no resume.** It refuses an existing `run_dir` unless `overwrite_run_dir`.
  Don't use preemptible GPUs for training.
- `InterleavedTokenizer` passes a frame count where `prepare_item` expects seconds; it works
  only because the result is then cropped. Don't "fix" it.
- Recommended LoRA config: r=128 (we plan r=64), scaling 2, lr 2e-6 (Kyutai's example pairs
  it with 2000 steps; the code default in `args.py` is 1e-4). 1×H100: ~12k tok/s, peak ~40 GB
  at batch 16 × 100 s.
- Optimizer: AdamW, betas (0.9, 0.95), eps 1e-8 (`train.py:202`). Adam normalizes gradient
  scale, so loss weights (`first_codebook_weight_multiplier`, `text_padding_weight`) change
  emphasis, not step size.
- LR schedule: `OneCycleLR(max_lr=lr, total_steps=max_steps, pct_start)` (`train.py:210`):
  warmup from lr/25, cosine decay to ~0, average ≈ lr/2. **A mid-run checkpoint is not the
  same as a shorter run.** It was taken at high LR, before annealing.
- LoRA: `W' = W + scaling · B·A` on every linear layer, B initialized to zero (so step 0 is
  exactly the base model), A Kaiming. `scaling: 2` is absolute, equivalent to the usual
  alpha = 2r.
- Built-in eval (`do_eval`, `eval_data`, `eval_freq`): forward passes on up to 40 batches,
  logging text loss and audio loss separately (`finetune/eval.py`). The audio-mask patch
  covers it.

- **Training epochs are real epochs.** `get_dataset_iterator` reshuffles each pass without
  replacement (`sphn` `shuffle(with_replacement=False)`), so epoch = steps × batch / n holds.
- **Eval-loader bug (upstream main):** `train.py` builds the eval loader once, but for eval
  data it's a finite generator. The first evaluation exhausts it, and every later evaluation
  iterates zero batches and logs NaN. It also drops the last partial batch (200 dev episodes
  at batch 16 would score only 192). Fixed by `patches/moshi-finetune-eval-reload.patch`:
  a fresh loader per evaluation, plus the partial batch kept. The partial batch counts as one
  batch in the average, a slight overweight. Upstream has a `fix_eval` branch that may
  address the same bug; not checked.
- **`num_ckpt_keep` defaults to 3** and deletes older checkpoints. Our configs set
  `num_ckpt_keep: null` to keep every per-epoch checkpoint.
- **No revision argument:** `train.py` calls `CheckpointInfo.from_hf_repo` without a
  revision. To pin the model, download a snapshot at a fixed revision
  (`2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`, current on 2026-10-14) and point
  `moshi_path` / `mimi_path` / `tokenizer_path` at the files (`model.safetensors`,
  `tokenizer-e351c8d8-checkpoint125.safetensors`, `tokenizer_spm_32k_3.model`).
- **Patches:** both apply in order (audio-mask, then eval-reload) to clean upstream main
  `2acc879f`, and the result matches the local `../moshi-finetune` (branch
  `silence-audio-mask`) file for file.

## Run matrix (agreed 2026-10-14, ~5 pm; supersedes the fixed-steps plan)

**Epochs are fixed per phase, not steps.** 1 h is a proof of concept: does the model start
picking up the behavior at all?

Fixed for every run:
- same pinned `kyutai/moshiko-pytorch-bf16` revision
- LoRA r=64, scaling 2; batch 16; `duration_sec: 60` (one episode = one window)
- AdamW, weight decay 0.1, OneCycle `pct_start` 0.05
- `first_codebook_weight_multiplier` 100, `text_padding_weight` 0.5
- **`mask_main_speech_audio: true`** (patch above)
- seed 0, one seed per run

`max_steps = ceil(E · n_episodes / 16)`, with `n_episodes` read from `ledger.json`, not
hardcoded (1 h ≈ 100 episodes, so 1 epoch ≈ 6.25 steps).

| Phase | Run | Data | LR | Epochs | Steps |
|---|---|---|---|---|---|
| 0 | R0 | none (base model) | — | — | 0 |
| 1 | R1–R5 | 1 h | 2e-6, 1e-5, 5e-5, 2e-4, 5e-4 | 20, checkpoint every epoch | ~125 each |
| 2 | R6–R7 | 10 h | top 2 LRs from phase 1 | E (chosen from phase 1) | E · 1000 / 16 each |

- **Phase 1 chooses E and the top 2 LRs** from per-epoch evaluation on the dev set. 2e-6 is
  expected to be a no-op at ~125 steps (~0.06× Kyutai's LR × steps). It doubles as a pipeline
  check: if it doesn't match R0, the LoRA merge or the eval is changing behavior by itself.
- **Report caveat:** the 1 h point at E epochs is the epoch-E checkpoint of a 20-epoch
  OneCycle run, not a run trained to exactly E. It's slightly handicapped next to 10 h; label
  it that way.
- **Per-epoch evaluation on a small dev set** (agreed in principle; details to be discussed):
  ~100 train-distribution episodes (training mix, train voices and phrasings, a different
  seed, packs not in the 10 h build), used only for model selection, never for training or
  reporting. The 7 test suites stay untouched until the final evaluation of the chosen
  checkpoints. Cheap layer: built-in eval loss every epoch (`eval_freq` = steps per epoch).
  Expensive layer: generation eval (harness below) on saved checkpoints, every epoch or every
  2 depending on measured harness throughput.
- **Dropped or on hold** (user's call): a separate 1 h rerun at exactly E; the 100 h run and
  100 h build; the stretched-holds ablation (as designed, ×3 holds would break the 60 s cap);
  the 1,000 h stretch (one epoch alone is 62,500 steps); the behavior-loss validation script;
  a second seed.

Compute (assumes ~15 s/step, i.e. ~1 H100-hour per 240 steps; measure on the box):

| | E = 5 | E = 10 |
|---|---|---|
| R1–R5 (5 × 125 steps) | 2.6 h | 2.6 h |
| R6–R7 | 2 × 313 steps ≈ 2.6 h | 2 × 625 steps ≈ 5.2 h |
| **Training total** | **~5.2 H100-h ≈ $7–18** | **~7.8 H100-h ≈ $10–27** |

Plus eval GPU time, which can't be estimated until the harness exists. Saving a checkpoint
every epoch for R1–R5 is 100 adapter checkpoints (rough guess 0.3–0.5 GB each), so provision
~50 GB of disk.

Each episode is one window, so exposure = episode presentations = steps × batch × GPUs;
`make_splits.py exposure` computes epochs and audio-hours seen. Report unique episodes, unique
packs, unique TTS clips, total / speech / hold-silence hours separately.

The user plans to rent an H100 (Vast.ai / RunPod / Hyperbolic, ~$1.3–3.5/hr), on-demand rather
than preemptible because `train.py` can't resume. Data generation and eval can run elsewhere.

## Eval harness (not written yet)

- Offline driver around `moshi.models.lm.LMGen`. **Prefix forcing:** force Moshi's text +
  audio tokens to ground truth until `instr_end` (ordinary: until the first user turn ends),
  then free-run. Otherwise Moshi won't ask the scripted question and the user's instruction
  makes no sense. Needs a small patch to override sampled tokens for the first k frames.
- Score **decoded audio** (Mimi decode → Silero VAD + ASR with word timestamps); log the
  text stream only as a diagnostic.
- Metrics, using each episode's `events`:
  - **hold violation:** ≥1 s or >3 words of Moshi speech inside
    [instr_end + 2.5 s ack grace, release]
  - **distractor violation:** a violation starting within 3 s after a distractor, per code
  - **resume success / latency:** onset within [cue_end, cue_end + 3 s]. B1: success if
    |onset − (instr_end + N)| ≤ max(1.5 s, 0.2N), plus the signed error.
  - **resume appropriateness:** LLM judge on the post-release transcript
  - **ordinary responsiveness:** responds within 3 s of the user's turn end; compare to R0
  - **over-suppression:** no response when one was due
  - optional voice-drift speaker similarity
- One sample per episode at default temperatures; 95% bootstrap CIs.
- Plots: rate vs. data size (log x), one line per LR, per suite.
- A listening log of successes and failures per suite.

## Known gaps / TODO

- Whether anyone has listened to the demo isn't recorded; ask.
- `gen_packs.py` default model is `claude-sonnet-5-5`. It forces
  `base_url="https://api.anthropic.com"` and reads the key from `.env`.
- `packs/demo_packs.jsonl` was deleted and `build.py` no longer has a `demo` split, so the old
  demo commands don't work. How the current 14-episode `demo/` was built isn't recorded.
- Turn-gap distribution is a placeholder lognormal; ideally fit it on DailyTalkContiguous.
- Noise is synthetic (pink / brown / filtered white).
- Distractor lines are inserted from a lexicon, so side conversations can be slightly
  incoherent around them. Possible fix: have the LLM write side conversations with tagged
  distractor slots.
- B1 fillers come from each pack's `mutter` field; they're tagged D2c.
- `demo/` is a plumbing and listening check only.
- Pocket TTS on this Mac: measured ~11.9× realtime with 4 workers on the 10 h build
  (1,000 episodes ≈ 7,661 clips ≈ 7.8 h of speech ≈ 40 min). If 100 h comes back, that's
  ~78 h of speech ≈ 6.5 h of CPU here, and it needs more packs first.
- The KyutaiTTS (1.6B, GPU) backend is optional and untested; calibrate its word offset first.

## Working with the user (Jaewoo)

- Casual and direct. He pushes back when estimates look inflated; give real numbers with
  the arithmetic.
- Render math in LaTeX in explanations.
- Ask before generating at scale or spending money (API calls for thousands of packs, GPU
  rentals).
