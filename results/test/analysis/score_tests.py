# Held-out test suites (2026-10-15): two-axis behavior per suite and model, same definitions as the dev
# analyses (results/epad10h/analysis/rescore_speak_silence.py): silent when due = 1 - hold violated;
# speaks when due = lenient "spoke" over cue releases + timed releases + ordinary questions; timed re-ask;
# per-distractor reaction rates. Output: results/test/summary.md + summary.json. No regeneration.
#   python results/test/analysis/score_tests.py
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

T = Path("results/test"); DATA = Path.home() / "repo/moshi-silence/data"
SUITES = ["test_script", "test_word", "test_voice", "test_distract", "test_cond", "test_ordinary", "test_all"]
# raw dir -> {checkpoint dir: model label}
MODELS = {"raw": {"base": "base", "checkpoint_000042": "1 h ×10 (R17 ep 5.9)", "checkpoint_000153": "10 h ×10 (R21 ep 2.2)"}}
LABELS = {"raw_R22": "50 h v2 ×10 (R22", "raw_R24": "45 h v3 ×10 (R24"}
for d in sorted(T.glob("raw_R2*")):
    for ck in {p.parent.name for p in d.glob("*/*/scores.jsonl")}:
        MODELS[d.name] = {ck: f"{LABELS[d.name]} step {int(ck.split('_')[1])})"}
ORDER = ["base", "1 h ×10", "10 h ×10", "50 h v2", "45 h v3"]

def score(sc, suite):
    ep = {}
    silent, cue_any, cue_ok, tim_any, tim_ok, ordr, react = [], [], [], [], [], [], defaultdict(list)
    sil_by_type = defaultdict(list)
    for l in open(sc):
        r = json.loads(l)
        ev = json.load(open(DATA / suite / "episodes" / f"{r['id']}.json"))["events"]
        gen = json.load(open(sc.parent / f"{r['id']}.json"))
        segs, free = gen["segments"], gen["free"]
        for h, eh in zip(r["holds"], ev["holds"]):
            if h["violated"] is not None:
                silent.append(not h["violated"]); sil_by_type[r["type"]].append(not h["violated"])
            rel = eh["release"]; ref = rel["ref"]; timer = rel["kind"] == "timer"
            tol = max(1.5, 0.2 * rel["N"]) if timer else 0.25
            wins = [b for a, b in free if a <= ref <= b + 0.01]
            win_end = wins[0] if wins else ref + 4
            spoke = any(ref - tol <= a < win_end for a, b in segs) and not h["talking_through"]
            (tim_any if timer else cue_any).append(spoke)
            (tim_ok if timer else cue_ok).append(h["outcome"] == "on_time")
            for d in h.get("distractors", []):
                react[d["code"]].append(d["reacted"])
        ordr += [x["responded"] for x in r["responses"]]
    m = lambda v: (float(np.mean(v)), len(v)) if v else (float("nan"), 0)
    s, sp = m(silent), m(cue_any + tim_any + ordr)
    return dict(silent=s, speaks=sp, score=(s[0] + sp[0]) / 2 if s[1] and sp[1] else (sp[0] if sp[1] else s[0]),
                cue_spoke=m(cue_any), cue_on_time=m(cue_ok), timed_reask=m(tim_any), ordinary=m(ordr),
                react={k: m(v) for k, v in sorted(react.items())}, silent_by_type={k: m(v) for k, v in sorted(sil_by_type.items())})

res = {}
for raw, cks in MODELS.items():
    for ck, label in cks.items():
        for suite in SUITES:
            sc = T / raw / suite / ck / "scores.jsonl"
            if sc.exists():
                res.setdefault(label, {})[suite] = score(sc, suite)
labels = sorted(res, key=lambda x: next(i for i, o in enumerate(ORDER) if x.startswith(o)))
pct = lambda v: "–" if v[1] == 0 else f"{100 * v[0]:.0f}%"
out = ["# Held-out test suites: behavior by model", "",
       "7 suites × 90 episodes, 1 sample per episode, VAD-based scoring (same definitions as dev). "
       "Cells: **score** (silent / speaks). Score = mean of silent-when-due and speaks-when-due "
       "(test_ordinary has no holds: score = speaks).", "",
       "| Suite | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
for suite in SUITES:
    cells = []
    for lab in labels:
        x = res[lab].get(suite)
        cells.append("–" if not x else f"**{x['score']:.2f}** ({pct(x['silent'])} / {pct(x['speaks'])})")
    out.append(f"| {suite} | " + " | ".join(cells) + " |")
out += ["", "## Details per suite", ""]
for suite in SUITES:
    out += [f"### {suite}", "", "| Metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    rows = [("silent when due", "silent"), ("speaks when due", "speaks"), ("cue release: spoke", "cue_spoke"),
            ("cue release: on time", "cue_on_time"), ("timed re-ask", "timed_reask"), ("ordinary answered", "ordinary")]
    for name, k in rows:
        vals = [res[l].get(suite, {}).get(k, (float('nan'), 0)) for l in labels]
        if any(v[1] for v in vals):
            out.append(f"| {name} (n={max(v[1] for v in vals)}) | " + " | ".join(pct(v) for v in vals) + " |")
    codes = sorted({c for l in labels for c in res[l].get(suite, {}).get("react", {})})
    for c in codes:
        vals = [res[l].get(suite, {}).get("react", {}).get(c, (float('nan'), 0)) for l in labels]
        out.append(f"| reacted to {c} (n={max(v[1] for v in vals)}) | " + " | ".join(pct(v) for v in vals) + " |")
    types = sorted({t for l in labels for t in res[l].get(suite, {}).get("silent_by_type", {})})
    for t in types:
        vals = [res[l].get(suite, {}).get("silent_by_type", {}).get(t, (float('nan'), 0)) for l in labels]
        out.append(f"| silent in {t} holds (n={max(v[1] for v in vals)}) | " + " | ".join(pct(v) for v in vals) + " |")
    out.append("")
(T / "summary.md").write_text("\n".join(out))
(T / "summary.json").write_text(json.dumps(res, indent=1))
print("\n".join(out[:16]))
