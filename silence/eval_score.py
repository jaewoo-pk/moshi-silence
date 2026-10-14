"""Score Moshi's behavior on generated episodes (or the ground truth, as an oracle test).

  python -m silence.eval_score /workspace/eval/R3/ckpt70/dev_80   # eval_gen output -> scores.jsonl + summary
  python -m silence.eval_score --oracle data/dev --jsonl data/dev/dev_80.jsonl     # ground-truth Moshi channel
  python -m silence.eval_score --oracle data/dev --jsonl data/dev/dev_80.jsonl --channel right   # negative control

Inputs per episode: the episode's `events`, speech segments from Silero VAD on Moshi's audio
(computed by eval_gen on the decoded audio, or here on the wav's left channel for the oracle)
and a word list [[word, t_start], ...] (Moshi's text stream, or the alignments for the
oracle). Timing comes from the VAD; the words only feed the word-count rules.

Only the free-run windows are scored (`free_windows`). Outside them eval_gen forced Moshi's
channel to the script, so speech there is the script's, not the model's. Thresholds are in
`Thresholds`; the report shows their sensitivity.

Per hold:
  violated        >= viol_speech s of speech, or > viol_words words, inside the hold region
                  [instr_end + grace, release ref - tolerance]; None if that region is empty
                  (B1 N=3 s), so only the timing is scored
  outcome         on_time | late | no_onset, from the first speech onset in the resume window
                  [ref - tol_pre, ref + resume_win]; timer holds must also be within +-tol of ref
  latency         onset - ref (None if silent); timer_err is the same thing for B1
  talking_through speech that started before the resume window is still going at ref
  distractors     per distractor: `reacted` = a speech segment (>= react_min s) STARTED between its
                  start and react_win s after its end (inside the hold region): attributable to
                  it. `overlapped` = >= react_min s of speech in that span, however it started
                  (also counts a model that talks straight through)
Ordinary responses: responded (any speech within respond_win s after the user's turn ends),
latency of the first onset; barge-in into mid-turn pauses; onsets inside human turns.
"""
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FR = 12.5


@dataclass
class Thresholds:
    grace: float = 2.5          # s after instr_end in which an ack is allowed
    viol_speech: float = 1.0    # s of speech inside the hold region = violation
    viol_words: int = 3         # more words than this inside the hold region = violation
    cue_pre: float = 0.25       # a cue release may be anticipated by this much
    timer_tol_abs: float = 1.5  # B1: |onset - (instr_end + N)| <= max(abs, rel * N)
    timer_tol_rel: float = 0.2
    resume_win: float = 3.0     # s after the release reference to start speaking
    respond_win: float = 3.0    # s after a user turn to respond (ordinary)
    react_win: float = 3.0      # s after a distractor in which speech counts as a reaction
    react_min: float = 0.5      # minimum segment length for a distractor reaction
    pause_overlap: float = 0.5  # speech overlapping a mid-turn pause by this much = barge-in

    def timer_tol(self, n):
        return max(self.timer_tol_abs, self.timer_tol_rel * n)


# ---------------------------------------------------------------- windows
def free_windows(events, duration, th=Thresholds()):
    """[[t0, t1], ...] where Moshi runs free; forced to the script elsewhere. One window per
    hold, from the instruction end to the next scripted human turn after the release (at
    least resume_win + 1 s past the release reference, so the resume is observable).
    Ordinary episodes: free from the end of the first human turn."""
    holds = events["holds"]
    human = sorted(events["human_turns"], key=lambda t: t["t"][0])
    if not holds:
        return [[round(human[0]["t"][1], 3), round(duration, 3)]]
    out = []
    for i, h in enumerate(holds):
        t0 = h["instr_end"]
        ref, onset = h["release"]["ref"], h["release"]["moshi_onset"]
        nxt = [t["t"][0] for t in human if t["t"][0] > onset]
        t1 = max(nxt[0] if nxt else duration, ref + th.resume_win + 1.0)
        if i + 1 < len(holds):
            t1 = min(t1, holds[i + 1]["instr_end"] - 1.0)
        out.append([round(t0, 3), round(min(t1, duration), 3)])
    return out


def clip_segments(segs, windows):
    out = []
    for s, e in segs:
        for w0, w1 in windows:
            a, b = max(s, w0), min(e, w1)
            if b > a:
                out.append((a, b))
    return sorted(out)


def in_windows(t, windows):
    return any(w0 <= t < w1 for w0, w1 in windows)


# ---------------------------------------------------------------- VAD
_vad = None


def vad_segments(audio, sr, min_silence_ms=300):
    """Speech segments [(start, end)] in seconds from Silero VAD (expects mono float32)."""
    global _vad
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    from scipy.signal import resample_poly

    if _vad is None:
        _vad = load_silero_vad()
    if sr != 16000:
        from math import gcd
        g = gcd(16000, sr)
        audio = resample_poly(audio, 16000 // g, sr // g)
    ts = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)), _vad,
                               sampling_rate=16000, threshold=0.5, min_speech_duration_ms=100,
                               min_silence_duration_ms=min_silence_ms, speech_pad_ms=30, return_seconds=True)
    return [(float(t["start"]), float(t["end"])) for t in ts]


def _overlap(segs, a, b):
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in segs)


# ---------------------------------------------------------------- scoring
def score_hold(h, distractors, segs, words, th):
    rel = h["release"]
    timer = rel["kind"] == "timer"
    instr_end, ref = h["instr_end"], rel["ref"]
    tol = th.timer_tol(rel["N"]) if timer else None
    tol_pre = tol if timer else th.cue_pre
    region = (instr_end + th.grace, ref - tol_pre)
    speech_s = _overlap(segs, *region)
    n_words = sum(region[0] <= t < region[1] for _, t in words)
    n_words_hold = sum(instr_end <= t < region[1] for _, t in words)
    viol_starts = [max(s, region[0]) for s, e in segs if e > region[0] and s < region[1]]
    # Short timer holds (e.g. N=3 s: grace 2.5 s + tolerance 1.5 s > N) leave no region in
    # which talking is unambiguously a violation; only their timing is scored.
    violated = (speech_s >= th.viol_speech or n_words > th.viol_words) if region[1] > region[0] else None

    win = (ref - tol_pre, ref + th.resume_win)
    onsets = [s for s, _ in segs if win[0] <= s <= win[1]]
    onset = onsets[0] if onsets else None
    talking_through = any(s < win[0] < e and e > ref for s, e in segs)
    if onset is None:
        outcome = "no_onset"  # silent, or still talking from before the window (see talking_through)
    elif timer and abs(onset - ref) > tol:
        outcome = "late"
    else:
        outcome = "on_time"
    first_post_grace = next((s for s, _ in segs if s >= region[0]), None)

    reacts = []
    for d in distractors:
        if not (instr_end <= d["t"][0] < ref):
            continue
        lo, hi = max(d["t"][0], region[0]), min(d["t"][1] + th.react_win, region[1])
        hit = any(lo <= s <= hi and e - s >= th.react_min for s, e in segs)
        over = _overlap(segs, lo, hi) >= th.react_min if hi > lo else False
        reacts.append({"code": d["code"], "t": d["t"], "reacted": hit, "overlapped": over})

    return {
        "kind": rel["kind"], "N": rel.get("N"), "instr_end": instr_end, "ref": ref,
        "violated": violated, "speech_in_region_s": round(speech_s, 3), "words_in_region": n_words,
        "words_in_hold": n_words_hold, "first_violation_t": round(viol_starts[0] - instr_end, 3) if viol_starts else None,
        "outcome": outcome, "latency": round(onset - ref, 3) if onset is not None else None,
        "timer_err": round(first_post_grace - ref, 3) if timer and first_post_grace is not None else None,
        "talking_through": talking_through,
        "distractors": reacts,
    }


def score_episode(events, duration, segs, words, th=Thresholds()):
    windows = free_windows(events, duration, th)
    segs = clip_segments(segs, windows)
    words = [(w, t) for w, t in words if in_windows(t, windows)]
    human = sorted(events["human_turns"], key=lambda t: t["t"][0])
    out = {"type": events["type"], "duration": duration, "free": windows,
           "free_s": round(sum(b - a for a, b in windows), 3),
           "speech_s": round(sum(e - s for s, e in segs), 3), "n_words": len(words)}

    out["holds"] = [score_hold(h, events["distractors"], segs, words, th) for h in events["holds"]]

    # Ordinary responsiveness only on episodes without holds; in hold episodes the response
    # after a release falls outside the scored window and the release itself is scored above.
    resp = []
    for rp in events["response_points"] if not events["holds"] else []:
        if rp["role"] != "response" or not in_windows(rp["human_end"], windows):
            continue
        he = rp["human_end"]
        responded = _overlap(segs, he, he + th.respond_win) > 0
        on = [s for s, _ in segs if he - 0.3 <= s <= he + th.respond_win]
        resp.append({"human_end": he, "responded": responded,
                     "latency": round(on[0] - he, 3) if on else None})
    out["responses"] = resp

    barge = []
    for i, t in enumerate(human):
        if not t.get("midpause") or i == 0:
            continue
        p0, p1 = human[i - 1]["t"][1], t["t"][0]
        if not in_windows(p0, windows):
            continue
        hit = any(p0 <= s < p1 for s, _ in segs) or _overlap(segs, p0, p1) >= th.pause_overlap
        barge.append({"pause": [p0, p1], "barged": hit})
    out["barge_ins"] = barge

    inter = 0
    for t in human:
        a, b = t["t"]
        if in_windows(a, windows):
            inter += sum(a + 0.3 <= s <= b - 0.3 for s, _ in segs)
    out["interrupt_onsets"] = inter
    return out


# ---------------------------------------------------------------- text stream -> words
def words_from_text_tokens(tokens, piece, pad=3, epad=0):
    """[[word, t_sec], ...] from Moshi's text stream (one token per 80 ms frame). A word starts
    at a sentencepiece piece with the '▁' prefix; following non-prefixed pieces extend it."""
    words = []
    for f, tok in enumerate(tokens):
        if tok in (pad, epad) or tok < 0:
            continue
        p = piece(int(tok))
        if p.startswith("▁") or not words:
            words.append([p.lstrip("▁"), round(f / FR, 3)])
        else:
            words[-1][0] += p
    return [w for w in words if w[0]]


# ---------------------------------------------------------------- summary
def summarize(rows):
    """Plain rates over a list of score rows (no CIs; see eval_report)."""
    holds = [h for r in rows for h in r["holds"]]
    resp = [x for r in rows for x in r["responses"]]
    first = [r["responses"][0] for r in rows if r["responses"]]
    barge = [b for r in rows for b in r["barge_ins"]]
    timers = [h for h in holds if h["kind"] == "timer"]
    cues = [h for h in holds if h["kind"] == "cue"]
    reacts = [d for h in holds for d in h["distractors"]]

    def rate(xs, key):
        xs = [x for x in xs if x[key] is not None]
        return f"{100 * np.mean([x[key] for x in xs]):5.1f}% (n={len(xs)})" if xs else "   n/a"

    def med(xs, key):
        v = [x[key] for x in xs if x[key] is not None]
        return f"median {np.median(v):+.2f} s, p90 {np.percentile(np.abs(v), 90):.2f} s (n={len(v)})" if v else "n/a"

    lines = [
        f"episodes {len(rows)}; holds {len(holds)} (timer {len(timers)}, cue {len(cues)})",
        f"hold violated        {rate(holds, 'violated')}",
        f"talking through      {rate(holds, 'talking_through')}",
        f"resume on time       {rate([{'x': h['outcome'] == 'on_time'} for h in holds], 'x')}",
        f"  no onset in window {rate([{'x': h['outcome'] == 'no_onset'} for h in holds], 'x')}",
        f"  cue latency        {med(cues, 'latency')}",
        f"  timer error        {med(timers, 'timer_err')}",
        f"words in hold (incl. ack): mean {np.mean([h['words_in_hold'] for h in holds]) if holds else 0:.1f}",
        f"ordinary responded   {rate(resp, 'responded')}; first turn {rate(first, 'responded')}; {med(first, 'latency')}",
        f"barge-in             {rate(barge, 'barged')}",
        f"interrupt onsets     {sum(r['interrupt_onsets'] for r in rows)} in {len(rows)} episodes",
    ]
    for key, label in (("reacted", "distractor triggered"), ("overlapped", "speech after distr.")):
        by_code = {}
        for d in reacts:
            by_code.setdefault(d["code"], []).append(d[key])
        if by_code:
            lines.append(f"{label:21}" + "  ".join(
                f"{c} {100 * np.mean(v):.0f}% ({len(v)})" for c, v in sorted(by_code.items())))
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI
def read_wav(path):
    """float32 [N] or [N, C] and sample rate; scipy only (the GPU env has no soundfile)."""
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    return x.astype(np.float32), sr


def score_all(items, th):
    """items: (row_id, episode_json_path, duration, segments, words). Returns score rows."""
    rows = []
    for i, (rid, ep_json, duration, segs, words) in enumerate(items):
        d = json.loads(Path(ep_json).read_text())
        row = {"id": rid, **score_episode(d["events"], duration, segs, words, th)}
        row["segments"] = [[round(s, 3), round(e, 3)] for s, e in segs]
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  scored {i + 1}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gen_dir", nargs="?", help="eval_gen output dir (<id>.json with segments + words, meta.json)")
    ap.add_argument("--oracle", help="episode set dir: score the ground-truth Moshi channel")
    ap.add_argument("--jsonl", help="oracle: subset jsonl (default: episodes.jsonl of the set)")
    ap.add_argument("--channel", default="left", choices=["left", "right"],
                    help="oracle: which channel to treat as Moshi (right = negative control)")
    ap.add_argument("--out", help="scores jsonl (default: <gen_dir>/scores.jsonl or <set>/oracle_scores_<channel>.jsonl)")
    for f, v in asdict(Thresholds()).items():
        ap.add_argument(f"--{f}", type=type(v), default=v)
    a = ap.parse_args()
    th = Thresholds(**{f: getattr(a, f) for f in asdict(Thresholds())})

    if a.oracle:
        set_dir = Path(a.oracle)
        ep_dir = set_dir / "episodes"
        jl = Path(a.jsonl) if a.jsonl else set_dir / "episodes.jsonl"
        rows_in = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
        ids = [Path(r["path"]).stem if "path" in r else r["id"] for r in rows_in]
        ch = 0 if a.channel == "left" else 1

        def items():
            for eid in ids:
                x, sr = read_wav(ep_dir / f"{eid}.wav")
                d = json.loads((ep_dir / f"{eid}.json").read_text())
                words = [(w, s) for w, (s, e), spk in d["alignments"]]
                yield eid, ep_dir / f"{eid}.json", len(x) / sr, vad_segments(x[:, ch], sr), words

        out = Path(a.out) if a.out else set_dir / f"oracle_scores_{a.channel}.jsonl"
    else:
        gen = Path(a.gen_dir)
        meta = json.loads((gen / "meta.json").read_text())
        ep_dir = Path(meta["episodes_dir"])

        def items():
            for rid in meta["ids"]:
                rec = json.loads((gen / f"{rid}.json").read_text())
                segs = [tuple(x) for x in rec["segments"]]
                yield rid, ep_dir / f"{rec['id']}.json", rec["duration"], segs, [tuple(w) for w in rec["words"]]

        out = Path(a.out) if a.out else gen / "scores.jsonl"

    rows = score_all(items(), th)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {out}")
    print(summarize(rows))


if __name__ == "__main__":
    main()
