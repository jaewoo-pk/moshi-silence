"""Turn an episode spec + cached TTS clips into Moshi fine-tuning files.

Output per episode (layout expected by kyutai-labs/moshi-finetune):
  <id>.wav   stereo int16 24 kHz. LEFT = Moshi, RIGHT = user (+ friend + noise)
  <id>.json  {"alignments": Moshi's words only, as [word, [start, end], "SPEAKER_MAIN"],
              "events": labels for evaluation, "meta": provenance}
The trainer only reads "alignments" (finetune/data/interleaver.py), so the extra
keys ride along for free.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from .tts import SR, _speech_bounds, retime

TARGET_DBFS = -20.0
TAIL_S = 0.8


def _db(x):
    return 10 ** (x / 20)


def _norm_clip(audio, target_dbfs=TARGET_DBFS):
    s0, s1 = _speech_bounds(audio)
    seg = audio[int(s0 * SR): int(s1 * SR)]
    rms = np.sqrt((seg ** 2).mean()) if len(seg) else 0.0
    return audio * (_db(target_dbfs) / rms) if rms > 1e-6 else audio


def _rir(rt60, rng):
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60) * 0.3
    h[: int(0.003 * SR)] = 0
    h[0] = 1.0
    return (h / np.sqrt((h ** 2).sum())).astype(np.float32)


def _noise(kind, n, rng):
    w = rng.standard_normal(n)
    if kind == "white_lp":
        k = np.ones(8) / 8
        x = np.convolve(w, k, mode="same")
    else:
        f = np.fft.rfftfreq(n, 1 / SR)
        f[0] = f[1]
        shape = 1 / np.sqrt(f) if kind == "pink" else 1 / f
        shape[f < 40] = 0  # no DC rumble
        x = np.fft.irfft(np.fft.rfft(w) * shape, n)
    return (x / (np.sqrt((x ** 2).mean()) + 1e-12)).astype(np.float32)


def _union(intervals):
    tot, cur = 0.0, None
    for a, b in sorted(intervals):
        if cur is None or a > cur[1]:
            if cur:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    return tot + (cur[1] - cur[0] if cur else 0.0)


PRE_S, POST_S, FADE_S = 0.05, 0.10, 0.01


def gate(audio, on, off):
    """Zero the clip outside [on - PRE_S, off + POST_S] with short fades, so the noise
    floor between words of different turns never reaches Moshi's (silent) channel."""
    a = np.zeros_like(audio)
    i0, i1 = max(0, int((on - PRE_S) * SR)), min(len(audio), int((off + POST_S) * SR))
    a[i0:i1] = audio[i0:i1]
    f = int(FADE_S * SR)
    if i1 - i0 > 2 * f:
        a[i0:i0 + f] *= np.linspace(0, 1, f)
        a[i1 - f:i1] *= np.linspace(1, 0, f)
    return a


def place(spec, cache):
    """Resolve timing. Returns placed turns with absolute word times + anchors."""
    anchors, placed, t_last = {"start": 0.0}, [], 0.0
    stalls = []
    for turn in spec["turns"]:
        voice = spec["voices"][turn["spk"]]
        key, audio, words = cache.get(turn["text"], voice)
        words, on, off = retime(audio, words)
        audio = gate(audio, on, off)
        first = words[0][1]
        if "at" in turn:
            a, off = turn["at"]
            onset = anchors[a] + off
        else:
            gap = turn["gap"]
            if "stall" in turn:
                code, gap = turn["stall"]
                if code.startswith("D") and not code.endswith("_gap"):
                    stalls.append({"code": code, "t": [round(t_last, 3), round(t_last + gap, 3)], "spk": None})
            onset = t_last + gap
        clip_start = max(0.0, onset - first)
        w_abs = [[w, round(s + clip_start, 3), round(e + clip_start, 3)] for w, s, e in words]
        if turn.get("anchor"):
            anchors[f"{turn['anchor']}_start"] = w_abs[0][1]
            anchors[f"{turn['anchor']}_end"] = w_abs[-1][2]
        t_last = max(t_last, w_abs[-1][2])
        placed.append({**turn, "clip": key, "clip_start": clip_start, "audio": audio, "words": w_abs})
    return placed, anchors, stalls, t_last


def assemble(spec, cache, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(hashlib.sha1(spec["id"].encode()).hexdigest()[:8], 16))
    p = spec["params"]
    placed, anchors, stalls, t_last = place(spec, cache)
    dur = t_last + TAIL_S
    n = int(np.ceil(dur * SR))

    moshi, user = np.zeros(n, np.float32), np.zeros(n, np.float32)
    rir = _rir(p["friend_rt60"], rng)
    for t in placed:
        a = _norm_clip(t["audio"]) * _db(t.get("gain_db", 0.0))
        if t["spk"] == "friend":
            a = _norm_clip(fftconvolve(a, rir)[: len(a) + len(rir) // 2]) * _db(p["friend_gain_db"])
        i = int(round(t["clip_start"] * SR))
        a = a[: n - i]
        (moshi if t["spk"] == "moshi" else user)[i: i + len(a)] += a
    user += _noise(p["noise"], n, rng) * _db(TARGET_DBFS - p["snr_db"])
    for ch in (moshi, user):
        pk = np.abs(ch).max()
        if pk > 0.99:
            ch *= 0.99 / pk
    sf.write(out_dir / f"{spec['id']}.wav", np.stack([moshi, user], 1), SR, subtype="PCM_16")

    # ---------------------------------------------------------------- labels
    alignments = [[w, [s, e], "SPEAKER_MAIN"] for t in placed if t["spk"] == "moshi" for w, s, e in t["words"]]
    alignments.sort(key=lambda a: a[1][0])

    def span(t):
        return [t["words"][0][1], t["words"][-1][2]]

    moshi_turns = [{"role": t.get("role"), "t": span(t), "text": t["text"]} for t in placed if t["spk"] == "moshi"]
    human_turns = [{"spk": t["spk"], "t": span(t), "text": t["text"], **({"midpause": True} if t.get("midpause") else {})}
                   for t in placed if t["spk"] != "moshi"]
    distractors = [{"code": t["event"], "t": span(t), "spk": t["spk"], "text": t["text"]} for t in placed if t.get("event")]
    distractors += stalls
    distractors.sort(key=lambda d: d["t"][0])

    rel = next((t for t in placed if t.get("release")), None)
    release = dict(p.get("release", {"kind": "none"}))
    events = {"type": spec["type"], "instr_end": anchors.get("instr_end")}
    if rel is not None and "instr_end" in anchors:
        onset = rel["words"][0][1]
        if release["kind"] == "timer":
            release["ref"] = round(anchors["instr_end"] + release["N"], 3)
        else:
            release["ref"] = anchors.get("cue_end")
        release["moshi_onset"] = onset
        events["hold"] = [anchors["instr_end"], onset]
    events["release"] = release
    events["distractors"] = distractors
    # Every point where Moshi responds to a human: used for responsiveness metrics.
    resp, last_human = [], None
    for t in placed:
        if t["spk"] != "moshi":
            last_human = span(t)[1]
        elif t.get("role") in ("response", "release") and last_human is not None:
            resp.append({"human_end": last_human, "moshi_onset": span(t)[0], "role": t["role"]})
    events["response_points"] = resp
    events["moshi_turns"] = moshi_turns
    events["human_turns"] = human_turns

    meta = {k: spec[k] for k in ("id", "type", "split", "pack", "voices", "params", "templates")}
    meta["clips"] = [t["clip"] for t in placed]
    (out_dir / f"{spec['id']}.json").write_text(json.dumps(
        {"alignments": alignments, "events": events, "meta": meta}, ensure_ascii=False))

    speech = [span(t) for t in placed]
    stats = {
        "duration": round(dur, 3),
        "speech_s": round(_union(speech), 3),
        "moshi_speech_s": round(_union([span(t) for t in placed if t["spk"] == "moshi"]), 3),
        "hold_s": round(events["hold"][1] - events["hold"][0], 3) if "hold" in events else 0.0,
    }
    return {"path": f"{spec['id']}.wav", "duration": round(n / SR, 6)}, stats
