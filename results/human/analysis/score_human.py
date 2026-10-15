# Human-recorded test sets (2026-10-15): the user's own voice (test_human: scripted; test_human_free: own wording on
# instruction / cue turns) vs the same scripts with the TTS user voice (test_human_tts, paired control). 3 samples per
# episode, common seeds across models. Same metrics as results/test/analysis/score_tests.py.
#   python results/human/analysis/score_human.py   -> results/human/summary.md
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

H = Path("results/human"); DATA = Path.home() / "repo/moshi-silence/data"
SETS = ["test_human_tts", "test_human", "test_human_free"]
MODELS = [("base", "base"), ("checkpoint_000042", "1 h ×10 (R17)"), ("checkpoint_000153", "10 h ×10 (R21)"),
          ("checkpoint_000345", "50 h v2 ×10 (R22)"), ("checkpoint_000527", "45 h v3 ×10 (R24)")]

def score(sc, st):
    silent, cue_any, tim_any, ordr, react, by_type = [], [], [], [], defaultdict(list), defaultdict(list)
    for l in open(sc):
        r = json.loads(l)
        eid = re.sub(r"_s\d+$", "", r["id"])
        ev = json.load(open(DATA / st / "episodes" / f"{eid}.json"))["events"]
        gen = json.load(open(sc.parent / f"{r['id']}.json"))
        segs, free = gen["segments"], gen["free"]
        for h, eh in zip(r["holds"], ev["holds"]):
            if h["violated"] is not None:
                silent.append(not h["violated"]); by_type[r["type"]].append(not h["violated"])
            rel = eh["release"]; ref = rel["ref"]; timer = rel["kind"] == "timer"
            tol = max(1.5, 0.2 * rel["N"]) if timer else 0.25
            wins = [b for a, b in free if a <= ref <= b + 0.01]
            win_end = wins[0] if wins else ref + 4
            spoke = any(ref - tol <= a < win_end for a, b in segs) and not h["talking_through"]
            (tim_any if timer else cue_any).append(spoke)
            for d in h.get("distractors", []):
                react[d["code"]].append(d["reacted"])
        ordr += [x["responded"] for x in r["responses"]]
    m = lambda v: (float(np.mean(v)), len(v)) if v else (float("nan"), 0)
    s, sp = m(silent), m(cue_any + tim_any + ordr)
    return dict(silent=s, speaks=sp, score=(s[0] + sp[0]) / 2, cue=m(cue_any), timed=m(tim_any), ordinary=m(ordr),
                react={k: m(v) for k, v in react.items()}, by_type={k: m(v) for k, v in by_type.items()})

res = {}
for st in SETS:
    for ck, lab in MODELS:
        sc = H / "raw" / st / ck / "scores.jsonl"
        if sc.exists():
            res.setdefault(lab, {})[st] = score(sc, st)
labels = [lab for _, lab in MODELS if lab in res]
pct = lambda v: "–" if not v[1] else f"{100 * v[0]:.0f}%"
out = ["# Human-recorded test sets: behavior by model", "",
       "Speaker s01 (the user), one mic; friend lines TTS. 3 samples per episode, common seeds. test_human_tts = the same "
       "24 scripts with the TTS user voice (paired control). Cells: **score** (silent when due / speaks when due).", "",
       "| Set | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
names = {"test_human_tts": "TTS voice, same scripts (control)", "test_human": "your voice, scripted",
         "test_human_free": "your voice, own wording"}
for st in SETS:
    cells = []
    for lab in labels:
        x = res[lab].get(st)
        cells.append("–" if not x else f"**{x['score']:.2f}** ({pct(x['silent'])} / {pct(x['speaks'])})")
    out.append(f"| {st}: {names[st]} | " + " | ".join(cells) + " |")
out += ["", "## Details", ""]
for st in SETS:
    out += [f"### {st}", "", "| Metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for name, k in [("silent when due", "silent"), ("speaks when due", "speaks"), ("cue release: spoke", "cue"),
                    ("timed re-ask", "timed"), ("ordinary answered", "ordinary")]:
        vals = [res[l].get(st, {}).get(k, (float("nan"), 0)) for l in labels]
        if any(v[1] for v in vals):
            out.append(f"| {name} (n={max(v[1] for v in vals)}) | " + " | ".join(pct(v) for v in vals) + " |")
    for t in sorted({t for l in labels for t in res[l].get(st, {}).get("by_type", {})}):
        vals = [res[l].get(st, {}).get("by_type", {}).get(t, (float("nan"), 0)) for l in labels]
        out.append(f"| silent in {t} holds (n={max(v[1] for v in vals)}) | " + " | ".join(pct(v) for v in vals) + " |")
    for c in sorted({c for l in labels for c in res[l].get(st, {}).get("react", {})}):
        vals = [res[l].get(st, {}).get("react", {}).get(c, (float("nan"), 0)) for l in labels]
        out.append(f"| reacted to {c} (n={max(v[1] for v in vals)}) | " + " | ".join(pct(v) for v in vals) + " |")
    out.append("")
(H / "summary.md").write_text("\n".join(out)); (H / "summary.json").write_text(json.dumps(res, indent=1))
print("\n".join(out[:9]))
