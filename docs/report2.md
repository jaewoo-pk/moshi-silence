# Teaching a Full-Duplex Speech Model When to Stay Silent: Data Requirements for Instruction-Conditioned Silence in Moshi

*Draft (paper format), 2026-10-15. Only the Methods section is written; the other sections are stubs.
Results to fill in come from `docs/report.md`, `docs/training_record.md` and `results/`.*

---

## Abstract

**TODO.** (~200 words: question, approach, headline numbers for 1 h / 10 h / 50 h, generalization, data-requirement
estimate.)

## 1. Introduction

**TODO.** Full-duplex models (Moshi) listen and speak at once, so knowing when *not* to speak is a learned
behavior, not a turn-taking rule. Question: how much synthetic fine-tuning data teaches instruction-conditioned
silence ("wait N seconds", "stay quiet until I'm back", "let me finish"), and does it generalize to new wording,
voices and conditions? Contributions: (i) a synthetic-conversation generator with held-out axes, (ii) an
evaluation harness that scores the model's own generated audio, (iii) data-size and learning-rate sweeps with
exposure tracked separately from unique data, (iv) an analysis of the silence bias of the fine-tuning objective.

## 2. Related work

**TODO.** Moshi / Mimi; full-duplex dialogue models; turn-taking and backchannel prediction; synthetic speech data
for dialogue fine-tuning; LoRA.

---

## 3. Methods

### 3.1 Study design

We fine-tune Moshi on synthetic two-party and three-party conversations. Moshi must learn how to (a) stay silent and (b) resume at the right moment. Three controlled factors:

1. **Amount of unique training data:** 1 h, 10 h and 50 h of conversation audio. The 1 h set is a prefix of the
   10 h set; the 50 h set uses a revised recipe (§3.4.8).
2. **Peak learning rate:** 2e-6 to 5e-4.
3. **Text-loss weight on the "start speaking" token (EPAD):** ×1 to ×100.

Every run starts from the same checkpoint and uses the same seed and evaluation. Unique data, hours and training
exposure are reported separately (§3.5.3).

### 3.2 Base model and fine-tuning code

| Component | Version |
|---|---|
| Base checkpoint | `kyutai/moshiko-pytorch-bf16`, Hugging Face revision `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`: 7B Moshi LM, Mimi codec, SentencePiece 32k text tokenizer |
| Fine-tuning code | `kyutai-labs/moshi-finetune` at commit `2acc879fe7c48f885a18f6cc9548bccb2674d87b`, plus 4 patches (below) |
| Environment | moshi-finetune's `uv.lock` (`uv sync --frozen`): moshi `061cc4c` (0.2.4a1), torch 2.6.0+cu124, sphn 0.1.12, huggingface-hub 0.28.1; CPython 3.12.15 via uv 0.13.1. Full freeze: `results/phase1/setup/pip-freeze.txt` |
| Eval-only extras | silero-vad 6.2.3 |
| Data generation | Python 3.12, torch 2.9.1 (CPU), `pocket-tts-timestamped` 1.2.0, anthropic SDK 1.14.0 |

The repository has no `config.json`, so moshi uses its built-in 7B configuration. All jobs run with
`HF_HUB_OFFLINE=1` against a local snapshot of the pinned revision.

**Patches to moshi-finetune** (`patches/`, applied in this order by `scripts/gpu_bootstrap.sh`):

1. `moshi-finetune-audio-mask.patch`, new option `mask_main_speech_audio`.
   - It removes the audio loss on Moshi's own speech frames, from 0.08 s before each word to 0.24 s after it.
   - The audio loss remains on Moshi's silent frames.
   - Reason: Moshi's channel in our data is a TTS voice, which we do not want the model to imitate.
2. `moshi-finetune-eval-reload.patch`.
   - Builds a fresh dev-loss loader for each evaluation; upstream's loader is a one-shot generator, so every
     evaluation after the first logs NaN.
   - Keeps the last partial dev batch.
   - Logs the text and audio dev loss separately.
3. `moshi-finetune-text-tail.patch`. In a file shorter than the training window, text frames after the end
   were PAD targets and counted in the text loss (~46% of all text frames for the 1 h set). This patch ignores
   them, like the audio.
4. `moshi-finetune-loss-weights.patch`, used from §4.3 on. It adds `text_epad_weight` (a separate text-loss
   weight on EPAD frames) and `audio_loss_weight` (total = text + w · audio). Both affect training only; the
   dev loss keeps the original weighting.

Upstream behaviors we rely on (verified in the source):

- Training iterates real epochs: shuffle without replacement, reseeded each epoch.
- `num_ckpt_keep` defaults to 3; we set it to `null` to keep every checkpoint.
- `train.py` cannot resume (no optimizer state is saved), so runs are never interrupted on purpose.

### 3.3 Target behaviors

Five episode families. Each episode is one conversation of ≤ 60 s.

| Family | Instruction (example) | Release Moshi must respond to | Variants (training weights) |
|---|---|---|---|
| **B1 timed hold** | "Give me ten seconds to think, then ask me again." | N seconds after the instruction ends, with no external cue: Moshi re-asks | `std` 9, `double` 2 (two holds in one episode), `early` 2 (user answers before N), `rescind` 2 (user cancels the hold) |
| **B2 side conversation** | "I'm going to talk to my friend. Stay quiet until I tell you I'm back." | The user's return cue ("Okay, I'm back.") | `std` 25 |
| **B3 hear me out** | "Let me describe all three options before you comment." | The user's completion signal ("That's all three.") | `std` 16, `early` 2 (finishes after fewer options), `rescind` 2 |
| **ORD ordinary** | Normal question answering | Each user question | 30; ⅓ of user turns contain a 0.8–2 s mid-turn pause |
| **DECOY** | Hold words without hold intent ("I just got back from Boston", "it took ten seconds") | Each user question | 10 |

Training mix v1 (1 h and 10 h sets, `specs.TRAIN_MIX`), by episode count: B1 15%, B2 25%, B3 20%, ORD 30%, DECOY 10%.

**Contrast variants** (`early`, `rescind`, `double`) require the model to track the instruction rather than going
quiet for a fixed time.

**Hold durations.**
- Training N ∈ {3, 5, 10, 15, 20, 30} s, restricted per episode to values that fit the 57 s length budget.
- `early` and `rescind` use N ≥ 10; `double` uses N ≤ 12.
- Held out for testing: N ∈ {8, 12, 25} s, plus "half a minute" and "a minute" (the latter only in `test_cond`).

**Distractors.** They are written by the script LLM inside each conversation (§3.4.1), so they are held out together
with the scripts. They are not inserted from a lexicon.

| Code | Description | Where |
|---|---|---|
| D1a | Friend asks the user a question | B2 side conversation |
| D1b | User asks the friend a question | B2 |
| D1c | Friend question that could be addressed to Moshi | B2 |
| D2a | 2–8 s stall inside the side conversation | 50% of B2 (every B2 in dense suites) |
| D2b | 3–5 s pause between options | 60% of B3 (every B3 in dense suites) |
| D2c | User mutters/fillers during a timed hold | 50% of B1 `std` with N ≥ 5 |
| D3a | Phrase resembling the return cue, said to the friend | exactly one per B2 |
| D3b | Phrase resembling the completion signal | 40% of B3 (every B3 in dense suites) |
| D4 | The *friend* says "Okay, I'm back!" (Moshi must stay silent) | test only |

### 3.4 Synthetic data generation

The pipeline: content packs (§3.4.1) → episode specification (§3.4.3) → TTS (§3.4.4) → mixing (§3.4.5) →
labels (§3.4.6) → automatic checks (§3.4.7). Commands: §3.10.

#### 3.4.1 Content packs (LLM-written scripts)

`silence/gen_packs.py` asks `claude-sonnet-5-5` for **packs**: the content of one conversation (topic lines, the
side conversation with tagged distractors, B3 options, the B1 question / answer / re-ask, ORD turns).

**Requests.**
- 10 packs of one family per request, low effort, no extended thinking.
- Structured outputs: a JSON schema per family.
- Each request gets 2–3 themes from a list of 29 and one of 10 user speaking styles (terse, chatty, hesitant,
  formal, older adult, teenager, non-native, upbeat, tired, …).
- Half of the requests allow the user to address Moshi by name in about half the examples.
- Each prompt includes a random 40 of the family's already-accepted scenarios, to avoid repeats.
- Up to 32 requests run concurrently; reruns top each family up to its target.

**Constraints in the prompt.**
- The user always speaks first.
- Every side-conversation question is answered.
- Each B2 has exactly one D3a, on line 2–4 of a 5–6-line side conversation.
- Per-field word limits.
- Plain, common English: no brand names, acronyms, loan words or rare names, because the TTS mispronounces them.

**Filters** (`validate`, `is_speakable`, `near_dup`). A pack is rejected for:
- schema violations;
- a Moshi-first opening;
- an unanswered question;
- a missing or misplaced D3a (a D3a on the last line is repaired by dropping that line);
- a D3a spoken by the friend;
- a missing decoy word;
- a worst-case episode length above 57 s, estimated at 0.1 + 0.28 s per word over 12 sampled timing setups;
- unspeakable text: digits, non-ASCII, all-caps acronyms other than I / OK / TV, camelCase, or a word in `HARD_WORDS`;
- an exact or near duplicate within the family: content-word Jaccard ≥ 0.6 on the family's key text.

**Train/test split.** By hash: `sha1(key_text) % 1000 < 150` → test (15%).

**Output (v1).**
- 2,500 packs (500 per family), from 322 requests: 0.69M input and 0.81M output tokens, ≈ $9.50.
- 620 candidates rejected, logged in `packs/rejects.jsonl`. The main reasons: no D3a 206, B3 too long 106,
  B2 too long 99, near-duplicate 93.
- 387 packs fell in the test split.

#### 3.4.2 Phrase banks (instruction wording)

The instruction, cue and response wording comes from hand-written phrase banks (`silence/phrasebank.py`), not from
the packs. That lets one test suite change only the wording. Every list has disjoint train and test entries:

| Slot | Train | Test |
|---|---|---|
| B1 instruction | 20 | 6 |
| B1 durations (spoken forms) | 6 | 3 |
| B1 "ask me another" | 6 | 2 |
| B1 early-answer prefix / rescind | 4 / 4 | 2 / 2 |
| B2 instruction | 20 | 6 |
| B2 return cue | 15 | 5 |
| B3 instruction | 15 | 5 |
| B3 completion | 12 | 4 |
| B3 early / rescind | 2 / 2 | 1 / 1 |
| Friend arrival lines / friend names | 4 / 6 | 2 / 3 |

Moshi's acknowledgments ("Sure.", "Okay, I'll wait.") come from one shared list.

#### 3.4.3 Episode specification and timing

`silence/specs.py` turns a pack into a timeline. Each episode draws from its own RNG, `Random(f"{seed}:{episode_id}")`:

**Voices and room.**
- The user and friend voices are two distinct voices from the split's pool.
- Friend gain U(−9, −4) dB.
- Friend reverb RT60 U(0.2, 0.6) s.
- Noise type ∈ {pink, brown, low-passed white}, at an SNR of U(15, 35) dB.

**Turn timing.**

| Gap | Distribution |
|---|---|
| Human → next human turn | 1.5 × clip(lognormal(μ = −0.9, σ = 0.45), 0.12, 1.2) s |
| Human → Moshi response | 1.5 × U(0.3, 0.9) s |
| User ↔ friend | 15% of the time an overlap of U(0, 0.25) s; otherwise 1.5 × clip(lognormal(−1.1, 0.5), 0.1, 1.0) s |
| First speech | the user, at 0.6 s |

The 1.5 scale (`GAP_SCALE`) was set by listening. Distractor stalls, mid-turn pauses and hold timing are not scaled.

**Hold logic.**
- An acknowledgment is spoken by Moshi in 70% of holds.
- **B1:** Moshi's re-ask comes N + U(−0.25, 0.25) s after the instruction ends.
  - `early`: the user answers at U(0.3, 0.7)·N.
  - `rescind`: the user cancels at U(1.5, max(2, 0.5N)) s.
  - D2c mutters are spaced through the hold and end ≥ 1 s before the re-ask.
- **B2:** the friend arrives, the side conversation runs with its tagged distractors (+ D2a stall), the user gives
  the return cue, and Moshi resumes.
- **B3:** options are spoken with D2b pauses and an optional D3b; the user gives the completion signal, and Moshi
  comments.

**Length.**
- The episode budget is 57 s.
- Training episodes must be ≤ 60 s, so each is exactly one 60 s training window.
- Over-length B2 episodes first get a shorter side conversation (two lines fewer per try, up to 4 tries) and are
  otherwise redrawn with the next pack.
- No episode opens with Moshi.

**Episode-type sequence.** Drawn from the mix with `Random(f"types:{seed}:{split}")`. Packs are dealt without
replacement per family, so no script is reused within a training set.

#### 3.4.4 Speech synthesis

**Engine.** Kyutai **Pocket TTS** (`pocket-tts-timestamped` 1.2.0, model `english_2026-09`), CPU, 24 kHz, with
word timestamps from attention.
- Moshi's lines always use voice `michael`.
- Users and friends use 16 training voices: alba, anna, azelma, bill_boerst, caro_davy, charles, cosette,
  eponine, fantine, jean, javert, marius, mary, paul, stuart_bell, vera.
- 4 voices are held out: eve, george, jane, peter_yearsley.

**Caching.** Clips are cached by `sha1(backend|voice|text)[:16]` and stored raw.

**Three timing corrections**, applied when episodes are assembled:
1. `tts.retime`: words stamped before the clip's energy onset are moved to the onset. Pocket stamps the first word
   at ~0 s when a clip opens with silence.
2. `speech_span` onset rule: a frame above −30 dB (relative to the clip) counts as the onset only if real speech
   follows. That means a frame above −15 dB within 150 ms, or a 100 ms window with ≥ 40% of frames above
   threshold that reaches −23 dB. This ignores breath or noise at the start of ~5% of clips.
3. `assemble.gate`: each clip is zeroed outside [onset − 50 ms, offset + 100 ms], with 10 ms fades. This removes
   Pocket's ~−42 dBFS noise floor from Moshi's channel during holds.

**Throughput.** ~12× realtime with 4 processes on an Apple M5. The 10 h build rendered 7,661 clips (7.8 h of
speech) in 37.6 min.

#### 3.4.5 Mixing

`silence/assemble.py`:

1. Each clip is normalized to −20 dBFS RMS. Fillers are at −6 dB.
2. Friend clips are convolved with a synthetic room impulse response (exponentially decaying Gaussian noise with
   the drawn RT60) and set to the drawn gain.
3. Noise is added **to the user channel only**, at the drawn SNR relative to −20 dBFS.
4. Each channel is peak-limited to 0.99, and a 0.8 s tail is added.
5. Output: a stereo 24 kHz PCM_16 WAV.
   - **Left = Moshi:** scripted speech, exact digital silence elsewhere.
   - **Right = user + friend + noise.**

#### 3.4.6 Labels

Each episode is written as three files:
- `<id>.wav`;
- `<id>.json` with `alignments` (Moshi's words only, as `[word, [start, end], "SPEAKER_MAIN"]`), `events` and `meta`;
- `<id>.spec.json` (the full timeline).

`events` records:
- every hold: instruction end, release type (timer or cue), release reference time (N after the instruction for
  timers, the end of the cue for cues), and Moshi's scripted onset;
- every human turn, with speaker and times;
- every distractor, with code and time;
- mid-turn pauses.

Only Moshi's words become text tokens in training (moshi-finetune `keep_main_only`). The user and friend are
audio-only inputs.

#### 3.4.7 Automatic checks

`silence/check.py` tokenizes every episode with moshi-finetune's own Interleaver. An episode fails on any of:
- hold-silence amplitude > 1e-3 on Moshi's channel outside the gate margins;
- a word vs audio onset misalignment > 0.15 s;
- a B1 timer error > 0.3 s;
- a cue → resume latency outside [0.25, 1.5] s;
- a human turn < 6 dB above the floor;
- a human/Moshi overlap > 0.05 s;
- a training episode > 60 s;
- Moshi speaking first;
- a missing EPAD on the frame before each release, or non-PAD text inside a hold.

All 1,150 train, 200 dev and 630 test episodes pass.

#### 3.4.8 The 50 h set (recipe v2)

Built later to test scaling. It differs from v1 in more than size, so we report it separately (§3.5):

**Scripts.**
- 5,470 new packs, from `gen_packs --seed 1` (crashed at request ~578) then `--seed 2`, with `--test-frac 0`.
  Per-family targets: B1 2,120, B2 1,580, B3 1,240, ORD 2,250, DECOY 780.
- A **reverse screen** (`silence/screen_train_packs.py`) drops new packs that contain a line spoken anywhere in
  dev or the test suites, or that have key-text Jaccard ≥ 0.5 with a test or dev pack. 306 dropped → 5,164 new
  packs, 7,664 in total.
- Dev packs are excluded with `build.py --exclude data/dev`.

**Mix** `TRAIN_MIX_V2`: B1 25% (std 17, double 4, early 2, rescind 2), B2 20%, B3 15% (std 11, early 2,
rescind 2), ORD 30%, DECOY 10%. D2c mutters in 70% of eligible B1 holds (was 50%).

**Voices (user and friend).**
- An engine is drawn per speaker: Pocket 40%, ElevenLabs Flash v2.5 15%, Fish Audio s2-pro 45%.
- Training voices: 16 Pocket, 17 ElevenLabs premade, and 30 Fish public-library voices vetted as generic. Voices
  named after or cloned from real people or characters were excluded.
- 4 held-out voices per engine.
- Moshi stays Pocket `michael`.
- Hosted-engine clips:
  - are validated: format check, silence, speaking rate within 0.12–1.0 s/word, up to 3 resamples;
  - get 0.15 s of padding;
  - get synthetic word times, spread over the speech span. Only human turn onsets and offsets are used downstream.

**Build.** 5,600 planned. 5,597 were assembled (1 failed Fish line, 2 over 60 s); 12 failed `check.py` and were
removed by `filter_checked`.
- **Result: 5,585 episodes / 51.24 h**; first-K prefixes of 1 / 10 / 25 / 50 h.
- ASR spot check (faster-whisper base.en, 366 clips): mean WER 0.021 (ElevenLabs), 0.013 (Fish), 0.036 (Pocket).

**v3 variant** (`--mix v3`, for the follow-up run R24). It reuses v2's plan and audio and changes only timing:
- with `LONG_SILENCES`, B2 stalls are U(8, 20) s in 50% of eligible episodes, and B3 pauses reach U(3, 12) s;
- a separate RNG stream leaves every v2 draw unchanged;
- after checking, `silence/rebalance.py` caps B1 at 15% of episodes.

### 3.5 Datasets, splits and exposure accounting

#### 3.5.1 Training sets

| Set | Episodes | Unique packs | Unique TTS clips | Total h | Speech h | Moshi speech h | Hold-silence h | Mean episode |
|---|---|---|---|---|---|---|---|---|
| `train_1h` (v1) | 113 | 116 | 885 | 1.01 | 0.70 | 0.25 | 0.39 | 32.2 s |
| `train_10h` (v1) | 1,101 | 1,135 | 8,463 | 10.01 | 7.05 | 2.48 | 3.73 | 32.7 s |
| `train50/train_50h` (v2) | 5,457 | 5,664 | 39,968 | 50.00 | 34.98 | 12.89 | 17.12 | 33.0 s |

- `train_1h` and `train_10h` are first-K prefixes of one 1,150-episode build, cut at 1 h and 10 h of total audio
  (`silence/make_splits.py`). So 1 h ⊂ 10 h.
- `train_1h` contains 71 holds (48 cue-released, 23 timed). Its type counts:

  | Type | B1 std / double / early / rescind | B2 | B3 std / early / rescind | ORD | DECOY |
  |---|---|---|---|---|---|
  | Episodes | 17 / 3 / 1 / 1 | 24 | 18 / 2 / 2 | 31 | 14 |

  These are random draws from the mix, so there are only 1–2 contrast episodes per variant.
- `train_10h` contains 699 holds (519 cue, 180 timed).

#### 3.5.2 Development and test sets

**Dev** (`silence/build_dev.py`, model selection only).
- 200 episodes / 1.97 h, built from train-split packs not used in `data/train`. It uses training voices and
  wording.
- Balanced: 50 per family group (B1, B2, B3, ORD + DECOY), each split by its training sub-mix.
- `dev_80` (0.77 h; 58 silence-scored holds, 63 releases, 42 ordinary questions) is the first 80 episodes of `dev_200`.
- Candidates sharing any spoken line with a training pack, or with key-text Jaccard ≥ 0.5 to one, were dropped
  (200 of 928).
- Audit: 0 shared packs or episodes with train; closest script Jaccard 0.47.

**Test suites** (`silence/build.py <suite>`, from the 348 test packs that survive a screen against lines spoken in
train or dev). 7 suites × 90 episodes, 6.1 h, each holding out one axis:

| Suite | Scripts | Wording | Voices | Mix |
|---|---|---|---|---|
| `test_script` | test | train | train | training mix |
| `test_word` | test | **test** (incl. N ∈ {8, 12, 25} s) | train | training mix |
| `test_voice` | test | train | **test** (4 unseen) | training mix |
| `test_distract` | test | train | train | B2 50 / B2-with-D4 20 / B3 30; every B2 gets a D2a stall, every B3 a D2b pause + D3b |
| `test_cond` | test | train | train | keyword release ("pineapple") 35 / phone call 35 / "half a minute"–"a minute" B1 30 |
| `test_ordinary` | test | train | train | ORD 70 / DECOY 30 |
| `test_all` | test | **test** | **test** | training mix |

- All 630 test episodes pass `check.py`.
- 0 shared packs or episodes with train; closest train script Jaccard ≤ 0.44.
- Test episodes have no 60 s cap ("a minute" holds run past it).
- The suites are **not paired**: each draws its own episode types and pack order. Suites that use the training
  mix share only 42–68 of their 90 packs pairwise. Comparisons between models on one suite are fully paired.

**Human-recorded set** (`silence/record.py`, `silence/human.py`; **TODO:** fill in counts once recorded and
scored). One speaker reads the user side of held-out `test_all` episodes into a microphone in three blocks:
- **R:** 10 s of room tone.
- **B, own words:** for each hold, the instruction and the return cue or rescind, said naturally from a card that
  gives only the intent. Recorded before A to avoid priming.
- **A, verbatim:** every user line of ~24 episodes, shown with the two preceding turns for context.

The clips replace the TTS user lines in the original episodes (same ids, so the same noise and room draws). The
friend and Moshi stay TTS. This yields three paired suites:
- `test_human_tts`: the unchanged TTS twins;
- `test_human`: the human voice, verbatim;
- `test_human_free`: hold episodes with the own-words instruction and cue.

#### 3.5.3 Exposure accounting

- Each training episode is exactly one 60 s window. Shorter episodes are padded, and the padding is masked in both
  text and audio loss (patch 3).
- **Exposure** = episode presentations = optimizer steps × batch size (16).
- **Epochs** = presentations / unique episodes.
- **Audio seen** = epochs × unique hours.
- We report unique episodes, unique packs, unique TTS clips, total / speech / Moshi-speech / hold-silence hours,
  and exposure separately (`make_splits` writes `ledger.json`; `make_splits exposure` computes epochs and audio
  seen). Repeating examples or lengthening silences changes exposure or hours, not unique episodes.

### 3.6 Training objective (how an episode becomes targets)

moshi-finetune (`InterleavedTokenizer`) converts each 60 s window into 750 frames at 12.5 Hz:

**Inputs.** Mimi codes (8 codebooks) of both channels, plus Moshi's text stream. Moshi's text and audio are
teacher-forced; the user's codes are input only.

**Text targets.**
- Each of Moshi's words is tokenized (SentencePiece) and placed from its start frame.
- All other frames are PAD (id 3).
- The frame just before a word that follows padding is EPAD (id 0): the model's "start speaking now" decision.
- In our training sets, ~90% of text frames are PAD and ~0.6% are EPAD.

**Loss.** L = L_text + w_a · L_audio, with w_a = 1 except where stated.
- L_text is cross-entropy over text frames. Word tokens have weight 1; PAD and EPAD have `text_padding_weight` 0.5.
  With patch 4, EPAD gets its own weight `text_epad_weight`.
- L_audio is cross-entropy over Moshi's 8 audio codebooks, with weights (100, 1, …, 1) normalized: the first
  (semantic) codebook carries ≈ 93%. With patch 1, it is masked on Moshi's speech frames.
- Weights only change emphasis, not step size, because AdamW normalizes gradient scale.
- Under the default weights, EPAD carries ≈ 0.6% of the text-loss weight. Because of the speech mask, the audio
  loss is computed only on Moshi's silent frames (~72% of frames). §4.3 analyzes this silence bias.

### 3.7 Training procedure

**Hyperparameters common to all runs** (`silence/make_configs.py`; exact YAMLs in `results/*/configs/`):

| Setting | Value |
|---|---|
| Adaptation | LoRA on every `nn.Linear` (main transformer, depformer, heads); base weights frozen; rank 64, scaling 2 (≈ 194M trainable parameters, 0.39 GB per bf16 adapter) |
| Window / batch | `duration_sec` 60; `batch_size` 16 per step; 1 GPU per run (no data parallelism) |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.1 |
| Schedule | OneCycle: warm-up from peak/25 over the first 5% of steps, cosine decay to ≈ 0 at the last step |
| Loss | `first_codebook_weight_multiplier` 100; `text_padding_weight` 0.5; `mask_main_speech_audio` true |
| Precision | bf16, gradient checkpointing |
| Seed | 0 (identical data order across runs; step-1 loss is the same in every run) |
| Dev loss | all of `dev_200` at every checkpoint |
| Checkpoints | `save_adapters: true`, `num_ckpt_keep: null` (all kept); checkpoint and dev loss on the same steps, plus the last step |

**Run length.** Epochs are fixed per phase, not steps: `max_steps = ceil(E · n / 16)`. Each config has a
`<run>.epochs.json` mapping checkpoints to epochs.
- A checkpoint taken mid-run is a snapshot of the full OneCycle schedule (taken at a higher LR), not a shorter run.
- The R22 runs use a fixed step count instead, to hold compute equal across data sizes.

**Run matrix:**

| Phase | Runs | Train data | Peak LR | EPAD × | Steps (epochs) | Checkpoint every |
|---|---|---|---|---|---|---|
| 0 | R0 | none (base model, 2 steps at lr 0) | 0 | — | 2 | dev loss only |
| 1 | R1, R6, R2, R7, R3, R4 | `train_1h` | 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 2e-4 | 1 | 142 (20) | 7 steps (≈ 1 epoch) → 21 |
| 2 | R8–R11 | `train_10h` | 2e-5, 5e-5, 2e-4, 5e-4 | 1 | 345 (5) | 17 steps (≈ ¼ epoch) → 21 |
| 3a | R16–R19 | `train_1h` | 5e-5 | 3, 10, 30, 100 | 142 planned; 133 reached (17.8)* | 7 steps → 18 |
| 3b | R20, R21 | `train_10h` | 5e-5 | 3, 10 | 345 (5) | 17 steps → 21 |
| 4 | R22 | `train50/train_50h` (v2) | 5e-5 | 10 | 345 fixed (1.01) | 17 steps → 21 |
| 5 | R24 | v3, B1 capped at 15% | 5e-5 | 10 | 2 epochs | 31 steps |

\*R16–R19 crashed when the network volume filled while saving step 133. Each kept 18 complete checkpoints
(epochs 1–17.8); they are analyzed as-is.

- EPAD × k means `text_epad_weight` = 0.5 · k.
- R5 (1 h, 5e-4) was planned and dropped before it started.
- R23 (the v2 recipe at 10 h, 345 steps) is configured (`scripts/run50h.sh 23`) but was not run.

**Hardware.**
- RunPod Secure Cloud, on-demand NVIDIA H100 80 GB SXM, data center US-NE-1, $3.99 per GPU-hour.
- A network volume holds the env, weights, data and checkpoints (150 GB, grown to 250 GB).
- Concurrent runs each get their own GPU (`scripts/run_parallel.sh`).

**Measured speed.**
- ≈ 10 s/step (≈ 11.3k tokens/s), with 29.1 GB peak memory.
- Model load plus the first step ≈ 2.6 min; a dev-loss evaluation on 200 episodes ≈ 47 s.
- One 1 h run took ≈ 35 min; one 10 h run ≈ 69 min.

### 3.8 Evaluation

#### 3.8.1 Generation protocol

We score the model's own generated audio (`silence/eval_gen.py`).

1. **Encode.** Both channels of an evaluation episode are Mimi-encoded, 4 episodes at a time (Mimi is fp32, ~3 GB
   per 60 s episode). The script's text stream is built with moshi-finetune's own Interleaver.
2. **Generate, with the script forced outside free windows.** `ForcedLMGen`, a copy of moshi's `LMGen.step`, steps
   through the episode.
   - Outside the *free windows*, Moshi's text token and 8 audio codebooks are forced to the script. Forced text is
     substituted before the depformer, as in training.
   - Inside free windows, the model runs freely.
   - Free windows, one per hold, run from the instruction end to the next scripted human turn after the release.
     Each is at least the release reference + 4 s, and is capped 1 s before the next hold's instruction.
   - Ordinary episodes run free from the end of the first user turn.
   - Forcing between windows means every hold is scored independently, even after the model went off script
     earlier.
3. **Alignment.** In training, the model reads the user frame t−1 at step t. A naive file replay through `LMGen`
   delivers each user frame 80 ms early, so `eval_gen` feeds frame s−1 at step s (`--user-shift 1`). The selftest
   verifies this on the 7B in bf16:
   - 100% of the input columns the generator feeds are integer-identical to the training forward's inputs
     (0.3% with no shift);
   - text-logit argmax agreement is 99.7%;
   - script text cross-entropy is 0.491 streaming vs 0.490 full forward.
4. **Sampling.**
   - moshi defaults: temperature 0.8 (audio) / 0.7 (text), top-k 250 / 25.
   - 1 sample per episode for dev, batch 40.
   - The KV cache is sized to the batch's frame count.
   - Seed `seed + 1000·sample + batch_offset`, with batches fixed by the length-sorted episode list. Every model
     gets the same random numbers on the same episode.
   - When checkpoints of one run are evaluated together, the base weights load once and only LoRA tensors are
     swapped. A base model with zero LoRA reproduces the LoRA-free model token for token.
5. **Decode and segment.** Moshi's generated codes are Mimi-decoded. Silero VAD 6.2.3 (min silence 300 ms) gives
   speech segments. Moshi's text stream gives its words.

**Speed.** 28 ms per step at batch 40 on one H100. dev_80 takes ≈ 57–70 s per checkpoint.

#### 3.8.2 Metrics

Per-event metrics come from `silence/eval_score.py`, with these thresholds:

| Threshold | Value |
|---|---|
| Acknowledgment grace after the instruction | 2.5 s |
| Violation | ≥ 1.0 s of speech or > 3 words in the hold region |
| Timed-release tolerance | max(1.5 s, 0.2·N) |
| Cue anticipation allowed | 0.25 s |
| Resume / response / reaction windows | 3 s |
| Minimum reaction segment | 0.5 s |
| Barge-in overlap | 0.5 s |

- **Hold violated.** Speech ≥ 1.0 s, or > 3 words, inside [instruction end + 2.5 s, release ref − tolerance].
  The tolerance is max(1.5 s, 0.2·N) for timed holds and 0.25 s for cues. When this region is empty (e.g. N = 3 s),
  the hold is not scored for silence.
- **Resume outcome.** *On time* if a new speech onset falls in [ref − tolerance, ref + 3 s]; for timed holds it
  must also be within ±tolerance of ref. Otherwise *late* or *no onset*. We also record cue latency and the timer
  error (onset − ref).
- **Ordinary answered.** An onset within 3 s after a user question. Scored only in hold-free episodes.
- **Distractor reaction.** A speech segment ≥ 0.5 s that starts within 3 s after the distractor. Reported per code.
- **Barge-in.** ≥ 0.5 s of speech overlapping a user's mid-turn pause.

**Headline two-axis score** (`results/*/analysis/rescore_speak_silence.py`, computed from saved segments with no
regeneration):
- *Silent when due* = 1 − hold-violation rate, over holds scored for silence.
- *Speaks when due* = the share of release events where Moshi spoke. Release events are cue releases, timed
  releases and ordinary questions, pooled. "Spoke" is lenient: any new onset from ref − tolerance to the end of
  the free window (≈ ref + 4 s).
- **Score = mean(silent when due, speaks when due).**
- Both axes are reported because either one alone is maximized by a degenerate model (always silent or always
  talking).
- *Timed re-ask* is "spoke" restricted to timed releases.

#### 3.8.3 Aggregation, model selection and test protocol

- Rates are pooled over events within a set. 95% percentile bootstrap CIs resample episodes (1,000 resamples,
  `silence/eval_report.py`).
- Per-checkpoint noise on dev_80 is ≈ ±0.06 in score (±10–12 pp per rate).
- **Plateau** averages: epochs 10–20 for 1 h runs (10–17.8 for R16–R19 and their baseline), epochs 2–5 for 10 h runs.
- **Selection:** every checkpoint of every run is scored on dev_80. The selected checkpoint is the best dev_80
  score, with ties going to the earlier epoch. dev_200 is used to confirm candidates (`scripts/dev200_confirm.sh`).
  **Dev loss is logged but not used for selection.**
- **Test protocol.** The 7 test suites are generated once per selected model (`scripts/test_suites.sh`), with the
  same sampling and seeds as dev.
  - Reported here with 1 sample per episode; the planned final protocol is 3 samples (`--samples 3`, used for the
    human set).
  - Test-suite results are never used for selection.

#### 3.8.4 Validation of the harness

- **Oracle.** Scoring the ground-truth Moshi channel on dev_80 gives 0% holds violated, 100% resumed on time,
  100% ordinary answered and 0 distractor reactions. Median cue latency is 0.92 s; median timer error is −0.11 s.
- **Negative control.** Scoring the user channel as if it were Moshi gives 76% violated and 94% no onset.
- **Forced generation.** With `--force-all`, the generator's output scores identically to the oracle on the same
  episodes, so the Mimi round trip preserves timing.

#### 3.8.5 Dev loss and listening

- Teacher-forced text and audio dev loss on `dev_200` is logged at every checkpoint (patch 2). It serves as a
  health check.
- Listening material is generated from saved codes on a CPU, with no GPU needed (`silence/demo_phase1.py`,
  `silence/transcript_gen.py`):
  - the listening demo: 9 dev conversations × representative models, chosen where models disagree most;
  - transcripts with generated words;
  - a live-chat panel serving two checkpoints with `moshi.server --lora-weight` (`silence/chat_gui.py`, 13 test
    scripts in `silence/chat_scripts.py`).

### 3.9 Compute and cost

| Item | Amount |
|---|---|
| Training + evaluation GPU time | ≈ 24 H100-hours (≈ $95 at $3.99/h) |
| Script generation (Claude Sonnet 5.5) | ≈ $10 (v1) + ≈ $20 (v2) |
| Hosted TTS for the 50 h set | Fish $9.39; ElevenLabs within a Starter plan (42k credits) |
| One 1 h run (20 epochs) + its dev_80 evaluation | ≈ 1 H100-hour |
| One 10 h run (5 epochs) + its dev_80 evaluation | ≈ 1.5 H100-hours |

### 3.10 Reproduction

Prerequisites:
- an Anthropic API key, needed only to regenerate packs; the generated `packs/*.jsonl` are in the repository;
- for the 50 h set only, ElevenLabs and Fish Audio keys;
- a machine with ≥ 1 H100 80 GB and a ≥ 250 GB volume.

**Data (CPU):**

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt silero-vad==6.2.3
# Pocket TTS weights: kyutai/pocket-tts-without-voice-cloning → models/pocket-tts/

# 1. Scripts (or reuse packs/packs.jsonl)
python -m silence.gen_packs --out packs/packs.jsonl --per-family 500 --workers 32 --rejects packs/rejects.jsonl

# 2. Training set, nested subsets, checks
python -m silence.build train --packs packs/packs.jsonl --n 1150 --out data/train
python -m silence.make_splits data/train --hours 1 10
python -m silence.check data/train --finetune ../moshi-finetune

# 3. Dev set (unused train packs, overlap-screened) + audit
python -m silence.build_dev --exclude data/train --out data/dev
python -m silence.check_overlap data/train data/dev

# 4. Test suites from screened test packs
python -m silence.screen_packs --split test --against data/train data/dev --out packs/packs_test_screened.jsonl
for s in test_script test_word test_voice test_distract test_cond test_ordinary test_all; do
  python -m silence.build $s --packs packs/packs_test_screened.jsonl --n 90 --out data/$s
  python -m silence.check data/$s --finetune ../moshi-finetune
done

# 5. (optional) 50 h v2 set
python -m silence.gen_packs --out packs/packs_v2.jsonl --seed 1 --test-frac 0 --workers 48 \
    --per-family-json '{"B1":2120,"B2":1580,"B3":1240,"ORD":2250,"DECOY":780}'   # then --seed 2 to top up
python -m silence.screen_train_packs --packs packs/packs_v2.jsonl --base packs/packs.jsonl \
    --heldout data/dev data/test_* --out packs/packs_v2_screened.jsonl
python -m silence.build train --tts multi --mix v2 --packs packs/packs_v2_screened.jsonl \
    --exclude data/dev --n 5600 --out data50/train50
python -m silence.check data50/train50 --finetune ../moshi-finetune --workers 16
python -m silence.filter_checked data50/train50
python -m silence.make_splits data50/train50 --hours 1 10 25 50 --abs
```

**GPU** (see the header of `scripts/gpu_bootstrap.sh` for the upload commands; use `rsync -rlpt`):

```bash
# env (uv.lock, two-step whisper fix), pinned weights, 4 patches, absolute-path jsonls,
# phase-1 configs, 6-step smoke train, eval selftest, 40-episode benchmark
bash scripts/gpu_bootstrap.sh

# Phase 1 (1 h LR sweep) + dev_80 generation evals
bash scripts/run_parallel.sh configs/R0_devloss.yaml configs/R?_1h_*.yaml
python -m silence.make_configs --phase 1 --first 6 --no-base --lrs 5e-6 2e-5 --train ... --eval ... --moshi-dir ... --runs ...
bash scripts/eval_dev.sh dev_80 R0 R1_1h_lr2e-06 R2_1h_lr1e-05 R3_1h_lr5e-05 R4_1h_lr0.0002 R6_1h_lr5e-06 R7_1h_lr2e-05

# Phase 2 (10 h × 4 LRs), EPAD sweeps, 50 h, v3: each script writes its configs and runs its evals
bash scripts/phase2.sh
bash scripts/ablation.sh            # R16–R19
bash scripts/epad10h.sh both        # R20, R21
bash scripts/run50h.sh 22           # R22 (23 = recipe control)
bash scripts/run_v3.sh              # R24

# Held-out tests and dev_200 confirmation of selected checkpoints
OUT=/workspace/eval/test bash scripts/test_suites.sh base /workspace/runs/<run>/checkpoints/checkpoint_XXXXXX ...
bash scripts/dev200_confirm.sh <run>/checkpoint_XXXXXX ...

# Reports and plots (CPU)
python -m silence.eval_report <eval root> --configs configs --runs <runs> --out <dir>
python -m silence.plot_training <runs> --configs <configs> --out <dir>
```

### 3.11 Deviations from the original plan

| Planned | Done | Reason |
|---|---|---|
| 1 h, 10 h, 100 h, with 1,000 h as a stretch | 1 h, 10 h, 50 h | Time; 50 h was built in ~70 min |
| 50 h at the same recipe | Recipe v2 | More voices and more timed holds. This confounds size with recipe |
| R5 (1 h, 5e-4) | Dropped | 5e-5 and 2e-4 had already memorized the scripts |
| 20 epochs for R16–R19 | 17.8 epochs | Disk-full crash |
| dev_200 for every candidate | dev_80 per checkpoint; dev_200 only for finalists | Cost |
| 3 test samples per episode | 1 for synthetic suites; 3 for the human set | Cost |
| LLM judge and Whisper scoring of response content | Not built | Timing metrics only |

---

## 4. Results

**TODO.** Port from `docs/report.md` §5:
- 4.1 Baseline.
- 4.2 Learning rate (1 h and 10 h tables; two-phase learning; dev loss vs behavior).
- 4.3 Data size and exposure (plateau at ~70 steps; **figure: score vs unique hours**).
- 4.4 Silence bias and EPAD weighting.
- 4.5 50 h.
- 4.6 Generalization (test-suite table with CIs; distractors; human set).
- 4.7 Failure modes and listening.

## 5. Discussion

**TODO.**
- Data-requirement estimate.
- Why loss is a poor selector.
- Silence bias of the duplex objective.
- What generalized and what failed.
- Limitations: synthetic Moshi voice, single seed, 1 sample, unpaired suites, Pocket-only held-out voices.
- Next experiments.

## 6. Conclusion

**TODO.**

## References

**TODO.** Moshi (Défossez et al., 2024); moshi-finetune; Pocket TTS; Silero VAD; LoRA (Hu et al., 2021);
OneCycle (Smith & Topin, 2019).

## Appendix

**TODO.**
- Full per-run tables (`docs/training_record.md` §10).
- Episode transcript examples (`results/transcripts.md`).
- Audio examples.
