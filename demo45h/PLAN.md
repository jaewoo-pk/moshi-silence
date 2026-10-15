# Demo of the 45h model: build plan

Decisions (user, 2026-10-15): compare **base vs the 45h model** only (the 45h model = R24,
`R24_v3_b1-15_lr5e-05_epad10/checkpoint_000527`); include failure cards; include the user's own voice;
live video tentatively yes; the website is **minimal** and **runs on any computer**.

## Deliverable: `demo45h/site/`, one self-contained folder

- `index.html` with all data inlined: no `fetch`, no server, no CDN, no fonts to download. It works by
  double-clicking it (`file://`), from a USB stick, a zip, or any static host. It can also be published as a
  private shareable link.
- `audio/*.mp3`: about 20 clips (10 conversations × 2 models), ~0.5 MB each.
- `video.mp4` (optional): ~2 min, 720p, ≤ 10 MB.
- Minimal design: one column, system font, light/dark following the OS, no framework. Page order:
  1. headline + 3 numbers;
  2. video;
  3. cards;
  4. 2 figures;
  5. "how clips were chosen" + links.

Each card has:
- a one-line setup;
- a **base / 45h** toggle;
- an audio player (Moshi left, user + friend right);
- a thin timeline: requested silence, the moment Moshi should speak, Moshi's speech in red where it talked in the hold;
- the transcript with Moshi's generated words;
- the scored verdict.

## Cards

Candidates come from `pick.py` → `candidates.json`. All are held out from training.

| Card | Shows |
|---|---|
| 1 side conversation | stays quiet while the user talks to a friend, returns on "I'm back" |
| 2 cue lookalike | ignores "I'm done…" said to the friend |
| 3 hear me out, with pauses | holds through long pauses between options |
| 4 hear me out, cancelled | speaks again when the user takes the instruction back |
| 5 timed hold | re-asks after N seconds with no cue |
| 6 ordinary question with a pause | still answers, doesn't barge into the pause |
| 7a my voice, scripted / 7b my voice, own words | works on a real human voice |
| 8a FAIL: the friend says "I'm back" / 8b FAIL: a minute-long hold | honest limits |

## Steps

1. **Pick** (done): `python demo45h/pick.py`.
2. **Pod session**, 1× H100, US-NE-1, ~45 min ≈ $3:
   - copy the `codes.npz` files for base + the 45h model, for the test suites and the human set, from
     `/workspace/eval/` (a few MB);
   - if doing the video, load base + the 45h model in `silence/chat_gui.py` and record with OBS (QuickTime
     can't capture system audio);
   - terminate the pod.
3. **Listen + choose:** decode the top-3 candidates per card on the Mac (Mimi, CPU), listen, choose one per card.
4. **Build the site** (`demo45h/build.py`): decode the chosen clips to mp3, inline the transcripts, timelines and
   verdicts, copy 2 figures.
5. **Check** in Safari and Chrome, at phone width, offline (Wi-Fi off), from a different folder.
   Optionally publish as a private link.
