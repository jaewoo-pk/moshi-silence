"""Figures for the v3 / R24 section of docs/decisions.md (separate from make_decision_figs.py, whose
figures stay as they were).

    PYTHONPATH=. python docs/figs/make_v3_figs.py

Reads results/run_v3/analysis/speak_silence.json (dev_80 per checkpoint, R21 / R22 / R24),
results/run_v3/eval_raw (per-hold scores), results/test/summary.json, results/human/summary.json.
Colors (validated categorical slots): 10 h R21 orange, 50 h v2 R22 aqua, 45 h v3 R24 yellow, 1 h R17 blue,
base dashed gray; aqua and yellow are < 3:1 on the surface, so every mark carries a value or direct label.
"""
import json
from pathlib import Path
from statistics import mean

import numpy as np

from silence import plotstyle as ps

R = Path("results")
OUT = Path("docs/figs")
plt = ps.setup()

C17, C21, C22, C24 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RUNS = [("R21_10h_lr5e-05_epad10", C21, "10 h, recipe v1 (R21)"),
        ("R22_50h_lr5e-05_epad10", C22, "50 h, recipe v2 (R22)"),
        ("R24_v3_b1-15_lr5e-05_epad10", C24, "45 h, recipe v3 (R24)")]

ss = json.loads((R / "run_v3/analysis/speak_silence.json").read_text())


def step(r):
    return int(r["ck"].split("_")[-1])


# ---------------------------------------------------------------- 1. dev_80 score vs training step
fig, ax = plt.subplots(figsize=(7, 3.4))
for run, c, lab in RUNS:
    rr = sorted([r for r in ss if r["run"] == run], key=step)
    xs, ys = [step(r) for r in rr], [(r["silent"] + r["speak"]) / 2 for r in rr]
    ax.plot(xs, ys, color=c, marker="o", ms=3.5, label=lab)
    ax.annotate(f"{ys[-1]:.2f}", (xs[-1], ys[-1]), xytext=(6, -12 if run.startswith("R21") else 0),
                textcoords="offset points", va="center",
                fontsize=8, color=ps.INK_2)
ax.axhline(0.29, color=ps.BASE, ls="--", lw=1.2, label="base model")
ax.set_xlabel("optimizer steps (batch 16)")
ax.set_ylabel("score (dev_80)")
ax.set_ylim(0.2, 1.0)
ax.set_xlim(0, 590)
ax.set_title("v3 data: the 45 h run ends above both earlier models")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "v3_dev_curve.png", dpi=200)
plt.close(fig)


# ---------------------------------------------------------------- 2. silence by hold type (last 8 checkpoints)
def by_type(run, last=8):
    d = R / "run_v3/eval_raw" / run / "dev_80"
    cks = sorted(p for p in d.iterdir() if (p / "scores.jsonl").exists())[-last:]
    acc = {}
    for ck in cks:
        for line in (ck / "scores.jsonl").read_text().splitlines():
            ep = json.loads(line)
            for h in ep["holds"]:
                if h.get("violated") is not None:
                    acc.setdefault(ep["type"], []).append(0 if h["violated"] else 1)
    return {k: mean(v) for k, v in acc.items()}, {k: len(v) // last for k, v in acc.items()}


tt = {run: by_type(run) for run, _, _ in RUNS}
order = ["B2_std", "B3_std", "B3_early", "B3_rescind", "B1_std", "B1_early", "B1_rescind", "B1_double"]
order = [t for t in order if all(t in tt[r][0] for r, _, _ in RUNS)]
fig, ax = plt.subplots(figsize=(7.5, 4.3))
y = np.arange(len(order))
h = 0.26
for j, (run, c, lab) in enumerate(RUNS):
    vals = [100 * tt[run][0][t] for t in order]
    ax.barh(y + (j - 1) * h, vals, h, color=c, edgecolor=ps.SURFACE, linewidth=1, label=lab)
    for i, v in enumerate(vals):
        ax.text(v + 1, i + (j - 1) * h, f"{v:.0f}%", va="center", fontsize=6.5, color=ps.INK_2)
n = tt[RUNS[0][0]][1]
ax.set_yticks(y)
ax.set_yticklabels([f"{t}  ({n[t]}/ckpt)" for t in order])
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_xlabel("% of holds kept silent (dev_80, last 8 checkpoints)")
ax.grid(True, axis="x")
ax.grid(False, axis="y")
ax.set_title("v3 recovers the holds 50 h broke", pad=30)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "v3_by_type.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3. held-out suites + human recordings
ts = json.loads((R / "test/summary.json").read_text())
hs = json.loads((R / "human/summary.json").read_text())
tkey = {k.split("(")[-1].split()[0]: k for k in ts}            # base -> 'base', R21 -> '10 h ×10 (R21 ep 2.2)', ...
tkey["base"] = "base"
hkey = {k.split("(")[-1].rstrip(")"): k for k in hs}
hkey["base"] = "base"
suites = ["test_script", "test_word", "test_voice", "test_distract", "test_cond", "test_ordinary", "test_all"]
slab = {"test_script": "new scripts", "test_word": "new wording", "test_voice": "unseen voices",
        "test_distract": "dense distractors", "test_cond": "new conditions", "test_ordinary": "ordinary only",
        "test_all": "all held out"}
M = [("R21", C21, "10 h v1 (R21)"), ("R22", C22, "50 h v2 (R22)"), ("R24", C24, "45 h v3 (R24)")]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
y = np.arange(len(suites))
for j, (m, c, lab) in enumerate(M):
    vals = [ts[tkey[m]][s]["score"] for s in suites]
    a1.barh(y + (j - 1) * h, vals, h, color=c, edgecolor=ps.SURFACE, linewidth=1, label=lab)
    for i, v in enumerate(vals):
        a1.text(v + 0.01, i + (j - 1) * h, f"{v:.2f}", va="center", fontsize=6.5, color=ps.INK_2)
base_vals = [ts["base"][s]["score"] for s in suites]
for i, v in enumerate(base_vals):
    a1.plot([v, v], [i - 1.5 * h, i + 1.5 * h], color=ps.BASE, ls="--", lw=1.2, label="base model" if i == 0 else None)
a1.set_yticks(y)
a1.set_yticklabels([slab[s] for s in suites])
a1.invert_yaxis()
a1.set_xlim(0, 1.12)
a1.set_xlabel("score (held-out test suite, 90 episodes, 1 sample)")
a1.grid(True, axis="x")
a1.grid(False, axis="y")
a1.set_title("Held-out synthetic suites")
a1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=7)

conds = ["test_human_tts", "test_human", "test_human_free"]
clab = ["TTS voice\n(same scripts)", "my voice,\nscripted", "my voice,\nown wording"]
HM = [("base", ps.BASE, "base model", "--"), ("R17", C17, "1 h (R17)", "-"), ("R21", C21, "10 h v1 (R21)", "-"),
      ("R22", C22, "50 h v2 (R22)", "-"), ("R24", C24, "45 h v3 (R24)", "-")]
x = np.arange(len(conds))
for m, c, lab, ls_ in HM:
    vals = [hs[hkey[m]][k]["score"] for k in conds]
    a2.plot(x, vals, color=c, ls=ls_, marker="o", ms=5, label=lab, lw=1.6 if m == "base" else 2)
a2.set_xticks(x)
a2.set_xticklabels(clab)
a2.set_xlim(-0.3, 2.3)
a2.set_ylim(0.2, 0.95)
a2.set_ylabel("score (24 episodes, 3 samples)")
a2.set_title("Human-recorded set (speaker: me)")
a2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "v3_tests.png", dpi=200)
plt.close(fig)

print("wrote v3_dev_curve.png v3_by_type.png v3_tests.png")
for run, _, _ in RUNS:
    print(run, {t: round(tt[run][0][t], 2) for t in order})
print({m: [round(hs[hkey[m]][k]["score"], 2) for k in conds] for m, *_ in HM})
