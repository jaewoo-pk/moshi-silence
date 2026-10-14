"""Print built episodes as timed transcripts: who says what when, where Moshi holds,
and which lines are distractors.

  python -m silence.transcript data/train/episodes/tr_000123.json [more.json ...]
"""
import json
import sys
from pathlib import Path


def transcript(d):
    ev, meta = d["events"], d["meta"]
    turns = [{"spk": "moshi", **t} for t in ev["moshi_turns"]] + list(ev["human_turns"])
    turns.sort(key=lambda t: t["t"][0])
    tags = {(x["t"][0], x["spk"]): x["code"] for x in ev["distractors"] if x["spk"]}
    stalls = {round(x["t"][1], 2): x for x in ev["distractors"] if x["spk"] is None}
    holds = ev.get("holds") or []
    ends = {round(h["instr_end"], 3): h for h in holds}
    onsets = {round(h["release"]["moshi_onset"], 3): h for h in holds}

    dur = max(t["t"][1] for t in turns) + 0.8
    out = [f"── {meta['id']}  {ev['type']}  ({meta['split']}, {dur:.1f} s, user={meta['voices']['user']}, "
           f"friend={meta['voices']['friend']})"]
    for t in turns:
        s, e = t["t"]
        h = onsets.get(round(s, 3))
        if h:
            r = h["release"]
            how = f"timer N={r['N']} s, re-ask {r['moshi_onset'] - r['ref']:+.2f} s" if r["kind"] == "timer" \
                else f"{r['moshi_onset'] - r['ref']:.2f} s after the cue"
            out.append(f"        └─ hold ends: Moshi silent {s - h['instr_end']:.1f} s ({how})")
        st = next((x for k, x in stalls.items() if abs(k - s) < 0.05), None)
        if st:
            out.append(f"        ··· {st['t'][1] - st['t'][0]:.1f} s silence [{st['code']}]")
        tag = tags.get((s, t["spk"]))
        notes = [f"[{tag}]" if tag else "", "(after a mid-turn pause)" if t.get("midpause") else "",
                 f"({t['role']})" if t["spk"] == "moshi" and t.get("role") in ("ack", "release") else ""]
        out.append(f"  {s:5.1f}  {t['spk'].upper():6s} {t['text']}  {' '.join(n for n in notes if n)}".rstrip())
        if round(e, 3) in ends:
            out.append("        ┌─ hold starts: Moshi must stay silent")
    return "\n".join(out)


def main():
    for p in sys.argv[1:]:
        print(transcript(json.loads(Path(p).read_text())))
        print()


if __name__ == "__main__":
    main()
