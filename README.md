# moshi-silence

Synthetic data for teaching Moshi conditional silence: when to stay quiet on
instruction, and when to speak again. Builds episodes in the format
[kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) trains on.

## Layout

```
packs/demo_packs.jsonl     hand-written content packs (schema reference)
silence/phrasebank.py      instruction / cue / distractor phrasings, split train vs test
silence/splits.py          what each test suite holds out (one axis at a time)
silence/voices.py          voice pools (espeak for the demo, Kyutai VCTK for real)
silence/specs.py           episode builders: B1 timed hold, B2 side conversation,
                           B3 hear-me-out, ordinary, decoys, test-only conditions
silence/tts.py             TTS backends (espeak | kyutai) with word timestamps + clip cache
silence/assemble.py        timeline -> stereo wav + {alignments, events, meta} json
silence/build.py           CLI: specs -> TTS -> episodes + manifest
silence/check.py           checks via moshi-finetune's own Interleaver, plots, index.html
silence/make_splits.py     nested 1h/10h/100h subsets + data ledger + exposure math
silence/gen_packs.py       LLM content packs (Anthropic API)
silence/list_kyutai_voices.py, calibrate_offset.py   GPU-box setup helpers
```

## Episode format

`episodes/<id>.wav` is stereo 24 kHz: **left = Moshi, right = user + friend + noise**.
`episodes/<id>.json` holds:

- `alignments`: Moshi's words only, `[word, [start, end], "SPEAKER_MAIN"]`. This is the
  only key the trainer reads.
- `events`: eval labels: `instr_end`, `hold`, `release` (kind, reference time, Moshi
  onset), `distractors` (code + time span), `response_points`, all turns with times.
- `meta`: pack, voices, templates, mixing params, TTS clip ids (for the ledger).

## Training mix (by episode count)

| B1 timed hold | B2 side conversation | B3 hear me out | Ordinary |
|---|---|---|---|
| 15% (11 std / 2 early answer / 2 rescind) | 25% | 20% (16 std / 2 early / 2 rescind) | 40% (30 normal, ⅓ with mid-turn pauses / 10 decoys) |

## Setup (CPU is enough)

```
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
# Pocket TTS weights (ungated repo; needs Hugging Face access once):
hf download kyutai/pocket-tts-without-voice-cloning --include "languages/english_2026-09/*" --local-dir models/pocket-tts
cp .env.example .env    # put your ANTHROPIC_API_KEY here (gitignored)
# for check.py only: a moshi-finetune checkout
git clone https://github.com/kyutai-labs/moshi-finetune ../moshi-finetune
```

TTS is Kyutai Pocket TTS (~100M params, runs on CPU) through the
`pocket-tts-timestamped` fork, which gives word timestamps from the model's own
attention, so no separate aligner is needed. Voices: 16 predefined voices for
training, 4 held out for test, `michael` fixed for Moshi (see `silence/voices.py`).

## Demo: one episode of each type

```
python -m silence.build demo --out demo
python -m silence.check demo --finetune ../moshi-finetune --plots   # writes demo/index.html
```

## Full data

```
python -m silence.gen_packs --out packs/packs.jsonl --per-family 1500
python -m silence.build train --packs packs/packs.jsonl --n 10000 --out data/train
for s in test_script test_word test_voice test_distract test_cond test_ordinary test_all; do
  python -m silence.build $s --packs packs/packs.jsonl --n 90 --out data/$s; done
python -m silence.make_splits data/train --hours 1 10 100 --abs
python -m silence.check data/train --finetune ../moshi-finetune
```

Then point moshi-finetune's `data.train_data` at `data/train/train_1h.jsonl` (etc.).
`--tts kyutai` switches to the 1.6B Kyutai TTS on a GPU (calibrate its word offset first
with `python -m silence.calibrate_offset`). `--tts espeak` is a robotic offline fallback.

## Known gaps

- `KyutaiTTS` (optional GPU backend) mirrors `TTSModel.simple_generate` but is untested.
- Pocket TTS has 20 English voices, so the voice-generalization test has only 4 unseen
  speakers.
- Turn-gap distribution is a placeholder; fit it on DailyTalkContiguous timestamps.
- Background noise is synthetic (pink / brown / filtered white); swap in real noise
  clips split by file for the final runs.
- Distractor lines are inserted from a lexicon, so side conversations can be a little
  incoherent around them.
