# Human test set (2026-10-15): behavior per model on the three paired suites built by silence/human.py,
# same definitions as results/test/analysis/score_tests.py (silent when due = 1 - hold violated; speaks when
# due = lenient "spoke" over cue releases + timed releases + ordinary questions; score = their mean).
# 3 samples per episode, pooled. Paired differences (human - TTS, own words - verbatim) are bootstrapped over
# episodes (the samples of an episode stay together). Output: results/human/summary.md + summary.json.
#   python results/human/analysis/score_human.py
import json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

H = Path("results/human"); DATA = Path.home() / "repo/moshi-silence/data"
SUITES = ["test_human_tts", "test_human", "test_human_free"]
LABELS = {"base": "base", "checkpoint_000153": "10 h ×10 (R21 ep 2.2)", "checkpoint_000345": "50 h ×10 (R22 step 345)"}


def per_episode(sc, suite):
    """{episode id: {metric: [bools over samples and holds]}}"""
    ep = defaultdict(lambda: defaultdict(list))
    for l in open(sc):
        r = json.loads(l)
        eid = re.sub(r"_s\d+$", "", r["id"])
        ev = json.load(open(DATA / suite / "episodes" / f"{eid}.json"))["events"]
        gen = json.load(open(sc.parent / f"{r['id']}.json"))
        segs, free = gen["segments"], gen["free"]
        e = ep[eid]
        for h, eh in zip(r["holds"], ev["holds"]):
            if h["violated"] is not None:
                e["silent"].append(not h["violated"])
            rel = eh["release"]; ref = rel["ref"]; timer = rel["kind"] == "timer"
            tol = max(1.5, 0.2 * rel["N"]) if timer else 0.25
            wins = [b for a, b in free if a <= ref <= b + 0.01]
            win_end = wins[0] if wins else ref + 4
            spoke = any(ref - tol <= a < win_end for a, b in segs) and not h["talking_through"]
            e["timed" if timer else "cue"].append(spoke)
            e["speaks"].append(spoke)
            if not timer:
                e["cue_on_time"].append(h["outcome"] == "on_time")
            for d in h.get("distractors", []):
                e[f"react_{d['code']}"].append(d["reacted"])
        for x in r["responses"]:
            e["ordinary"].append(x["responded"]); e["speaks"].append(x["responded"])
    return ep


def pooled(ep, ids, k):
    v = [x for i in ids for x in ep.get(i, {}).get(k, [])]
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)


def score(ep, ids):
    s, sp = pooled(ep, ids, "silent"), pooled(ep, ids, "speaks")
    return (s[0] + sp[0]) / 2 if s[1] and sp[1] else sp[0]


def boot_diff(epa, epb, ids, f, n=2000, seed=0):
    """Paired bootstrap over episodes of f(b) - f(a); returns (diff, lo, hi)."""
    rng = np.random.default_rng(seed)
    ids = list(ids)
    d = f(epb, ids) - f(epa, ids)
    bs = []
    for _ in range(n):
        smp = list(rng.choice(ids, len(ids)))
        # duplicate ids count twice: expand per-episode lists
        A = {f"{i}#{j}": epa.get(i, {}) for j, i in enumerate(smp)}
        B = {f"{i}#{j}": epb.get(i, {}) for j, i in enumerate(smp)}
        bs.append(f(B, list(A)) - f(A, list(A)))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d, float(lo), float(hi)


res = {}
for suite in SUITES:
    for sc in sorted((H / "raw" / suite).glob("*/scores.jsonl")):
        res.setdefault(LABELS.get(sc.parent.name, sc.parent.name), {})[suite] = per_episode(sc, suite)
labels = [l for l in LABELS.values() if l in res]
ids_all = [json.loads(l)["id"] for l in open(DATA / "test_human_tts" / "episodes.jsonl")]
ids_hold = [json.loads(l)["id"] for l in open(DATA / "test_human_free" / "episodes.jsonl")]
pct = lambda v: "–" if v[1] == 0 else f"{100 * v[0]:.0f}%"
sil = lambda ep, ids: pooled(ep, ids, "silent")[0]
spk = lambda ep, ids: pooled(ep, ids, "speaks")[0]

out = ["# Human test set: behavior by model", "",
       "One speaker (s01, ~20 min of recording) voices the user side of 24 held-out `test_all` episodes; the friend "
       "and Moshi's scripted lines stay TTS. Three paired suites (same episode ids, so the same noise and friend RIR): "
       "**TTS** = the original episodes, **human** = every user line recorded verbatim, **own words** = the 16 hold "
       "episodes with the instruction and the release cue / rescind said in the speaker's own words. 3 samples per "
       "episode, VAD scoring, same definitions as `results/test/summary.md`. Cells: **score** (silent / speaks).", ""]
for title, ids, suites in [("All 24 episodes", ids_all, SUITES[:2]), ("The 16 hold episodes", ids_hold, SUITES)]:
    out += [f"## {title}", "", "| Model | " + " | ".join({"test_human_tts": "TTS", "test_human": "human (verbatim)",
            "test_human_free": "human (own words)"}[s] for s in suites) + " |", "|---|" + "---|" * len(suites)]
    for lab in labels:
        cells = []
        for s in suites:
            ep = res[lab].get(s)
            cells.append("–" if ep is None else
                         f"**{score(ep, ids):.2f}** ({pct(pooled(ep, ids, 'silent'))} / {pct(pooled(ep, ids, 'speaks'))})")
        out.append(f"| {lab} | " + " | ".join(cells) + " |")
    out.append("")

out += ["## Paired differences (95% bootstrap CI over episodes)", "",
        "| Model | Comparison | episodes | score | silent when due | speaks when due |", "|---|---|---|---|---|---|"]
fmt = lambda t: f"{100 * t[0]:+.0f} pp [{100 * t[1]:+.0f}, {100 * t[2]:+.0f}]"
fmts = lambda t: f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"
diffs = {}
for lab in labels:
    r = res[lab]
    for name, a, b, ids in [("human − TTS", "test_human_tts", "test_human", ids_all),
                            ("own words − verbatim", "test_human", "test_human_free", ids_hold),
                            ("own words − TTS", "test_human_tts", "test_human_free", ids_hold)]:
        if a in r and b in r:
            t = (boot_diff(r[a], r[b], ids, score), boot_diff(r[a], r[b], ids, sil), boot_diff(r[a], r[b], ids, spk))
            diffs[f"{lab} | {name}"] = t
            out.append(f"| {lab} | {name} | {len(ids)} | {fmts(t[0])} | {fmt(t[1])} | {fmt(t[2])} |")
out.append("")

out += ["## Details", ""]
rows = [("silent when due", "silent"), ("speaks when due", "speaks"), ("cue release: spoke", "cue"),
        ("cue release: on time", "cue_on_time"), ("timed re-ask", "timed"), ("ordinary answered", "ordinary")]
for s in SUITES:
    ids = ids_hold if s == "test_human_free" else ids_all
    out += [f"### {s}", "", "| Metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    keys = rows + [(f"reacted to {k[6:]}", k) for k in sorted({k for l in labels for e in res[l].get(s, {}).values()
                                                                 for k in e if k.startswith("react_")})]
    for name, k in keys:
        vals = [pooled(res[l].get(s, {}), ids, k) for l in labels]
        if any(v[1] for v in vals):
            out.append(f"| {name} (n={max(v[1] for v in vals)}) | " + " | ".join(pct(v) for v in vals) + " |")
    out.append("")

(H / "summary.md").write_text("\n".join(out))
(H / "summary.json").write_text(json.dumps({"diffs": diffs, "scores": {
    l: {s: {"score": score(ep, ids_hold if s == "test_human_free" else ids_all),
            "silent": pooled(ep, ids_hold if s == "test_human_free" else ids_all, "silent"),
            "speaks": pooled(ep, ids_hold if s == "test_human_free" else ids_all, "speaks")}
        for s, ep in r.items()} for l, r in res.items()}}, indent=1))
print("\n".join(out))
