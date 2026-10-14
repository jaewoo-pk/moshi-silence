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

## Current status (as of 2026-10-14, ~2 pm ET)

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
- `cache/tts/pocket/`: 108 rendered clips (wav + word-timestamp json), keyed by
  sha1(backend|voice|text). Rebuilds reuse them.
- **Not on this Mac:** Moshi model weights (`kyutai/moshiko-pytorch-bf16`, ~15 GB). The
  trainer and eval download them into the HF cache on the GPU box. Training a 7B bf16 model
  isn't feasible here.
- **Not a git repo yet.** `git init` and commit before making changes (`.gitignore` already
  excludes .env, .venv, models, cache, data, and demo wavs).

**Generator: working.**
- The Pocket TTS demo (12 episodes, one per type) builds, and **12/12 pass check.py**
  through moshi-finetune's own Interleaver. That was verified on the cached clips after the
  fix below.
- `demo/` on disk is still from the run *before* the fix (`checks.json` shows 5/12).
  Rerun the two demo commands below to refresh it. It takes seconds and doesn't re-run TTS.
- Demo numbers after the fix:
  - B1 re-ask error vs instr_end + N: −0.16 / +0.23 s (jitter is ±0.25 s by design)
  - cue → Moshi resume: 0.38–0.87 s
  - Moshi word starts vs its audio onset: within ±0.05 s
  - holds: all PAD, with EPAD on the frame right before the release

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

**Not started:**
- the user listening to the demo (do this before scaling)
- LLM content packs (`gen_packs.py` is written but has never been run)
- full train / test datasets
- a moshi-finetune training yaml for the run matrix
- the eval harness (`LMGen` driver)
- the human-recorded test set
- training runs
- the report and demo

## Next steps, in order

The next real step is **generating the full dataset**. Two cheap gates come first, because
a problem found after rendering 100 h of audio is expensive.

1. **Listen (≈5 min).** The user reran the demo on this Mac after the fix (12/12 pass). Have
   him listen to 2–3 episodes in `demo/index.html` with headphones (left = Moshi,
   right = user + friend + noise). Listen for what checks can't catch: friend loudness,
   pacing, noise level, mispronounced words. Note: `packs/demo_packs.jsonl` was just edited
   ("latte" → "iced coffee", "Alfama" → "the old town"), so the next demo build re-renders
   those two clips.
2. **Pack smoke test.** `python -m silence.gen_packs --out packs/smoke.jsonl --per-family 10`.
   Read the output against `packs/demo_packs.jsonl` and check the schema, variety, and
   pronounceability (see below). **Ask before the full run** (`--per-family 1500`, ~7.5k packs).
3. **Generate:**
   - all 7 test suites (`--n 90` each)
   - the train set at 10 h (`--n 1000`)
   - `make_splits.py` for the 1 h / 10 h jsonl files and the ledger

   That's roughly an hour of Pocket TTS on this Mac's CPU. Time the first ~50 episodes to
   get the real throughput.
4. **Extend to 100 h** (`--n 10000`, same seed) in the background or on the GPU box (~60 h of
   speech ≈ 10 h of CPU here). Episode types, packs, and specs come out in the same order for
   the same seed, so the first 1,000 episodes are identical to the 10 h build and reuse its
   cached clips. 1 h / 10 h training can start while 100 h renders.
5. Training configs (from `../moshi-finetune/example/moshi_7B.yaml`: LoRA r=64,
   `duration_sec: 60`, batch 16, `max_steps: 250`, `ckpt_freq: 50`, one yaml per run) and
   the eval harness, then R0 → R1–R6 → R7, the listening log, and the report.

Optional hygiene: `git init` and commit (`.gitignore` is ready).

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
packs/demo_packs.jsonl     hand-written content packs; the schema reference for gen_packs
silence/phrasebank.py      instruction / cue / distractor phrasings, each split train vs test
silence/splits.py          which axis each test suite holds out (one axis per suite)
silence/voices.py          voice pools: pocket (default), kyutai, espeak
silence/specs.py           episode builders + TRAIN_MIX
silence/tts.py             TTS backends (pocket | kyutai | espeak) + ClipCache
silence/assemble.py        timeline -> stereo wav + {alignments, events, meta} json
silence/build.py           CLI: specs -> TTS -> episodes + manifest.jsonl + episodes.jsonl
silence/check.py           checks via moshi-finetune's own Interleaver; plots; index.html
silence/make_splits.py     nested 1h ⊂ 10h ⊂ 100h jsonl subsets + ledger.json + exposure math
silence/gen_packs.py       LLM content packs via Anthropic API (reads .env)
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
- Recommended LoRA config: r=128 (we plan r=64), scaling 2, lr 2e-6 (low for LoRA; we sweep).
  1×H100: ~12k tok/s, peak ~40 GB at batch 16 × 100 s.

## Run matrix (planned)

Same pinned `kyutai/moshiko-pytorch-bf16` revision for every run; LoRA r=64; fixed steps
S=250, batch 16, `duration_sec: 60`; checkpoint every 50 steps (free exposure curve).

| Run | Data | LR |
|---|---|---|
| R0 | none (base model) | — |
| R1–R3 | 1 h | 2e-6, 1e-5, 5e-5 |
| R4–R6 | 10 h | same three |
| R7 | 100 h | best 1–2 LRs |
| stretch | 100 h for 1 epoch; 1,000 h | best LR |
| ablation | 1 h with every hold stretched ×3 (more hours, same unique content) | best LR |

Each episode is one window, so exposure = episode presentations = steps × batch × GPUs;
`make_splits.py exposure` computes epochs and audio-hours seen. Report unique episodes, unique
packs, unique TTS clips, total / speech / hold-silence hours separately.

Compute estimate: ~1 H100-hour per run, ~20 H100-hours for the whole study. The user plans
to rent an H100 (Vast.ai / RunPod / Hyperbolic, ~$1.3–3.5/hr); data generation and eval
can run elsewhere.

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

- Nobody has listened to the demo yet.
- `gen_packs.py` untested. Default model `claude-sonnet-5-5`. It forces
  `base_url="https://api.anthropic.com"` and reads the key from `.env`.
- Turn-gap distribution is a placeholder lognormal; ideally fit it on DailyTalkContiguous.
- Noise is synthetic (pink / brown / filtered white).
- Distractor lines are inserted from a lexicon, so side conversations can be slightly
  incoherent around them. Possible fix: have the LLM write side conversations with tagged
  distractor slots.
- B1 fillers come from each pack's `mutter` field; they're tagged D2c.
- `demo/` uses the hand-written packs in `packs/demo_packs.jsonl`; it's a plumbing and
  listening check only.
- Pocket TTS on this Mac runs on CPU (~6× real-time on Apple Silicon). 100 h of episodes is
  ~60 h of speech to synthesize, so it may be worth running that build on the GPU box.
- The KyutaiTTS (1.6B, GPU) backend is optional and untested; calibrate its word offset first.

## Working with the user (Jaewoo)

- Casual and direct. He pushes back when estimates look inflated; give real numbers with
  the arithmetic.
- Render math in LaTeX in explanations.
- Ask before generating at scale or spending money (API calls for thousands of packs, GPU
  rentals).
