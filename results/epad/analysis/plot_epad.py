"""EPAD-weight sweep plot: behavior vs epoch per EPAD multiple (1 h, lr 5e-5), from speak_silence.json.
Run from the repo root: PYTHONPATH=. python results/epad/analysis/plot_epad.py"""
import json
import numpy as np
from silence import plotstyle as ps

RUNS = [("×1 (baseline)", "R3_1h_lr5e-05"), ("×3", "R16_1h_lr5e-05_epad3"), ("×10", "R17_1h_lr5e-05_epad10"),
        ("×30", "R18_1h_lr5e-05_epad30"), ("×100", "R19_1h_lr5e-05_epad100")]
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal blue ramp (validated), low -> high EPAD
PANELS = [("silent", "silent when it should be"), ("speak", "speaks when it should"),
          ("tim_any", "timed hold: re-asked on its own"), ("score", "score = mean(silent, speaks)")]
out = json.load(open("results/epad/analysis/speak_silence.json"))
base = next(o for o in out if o["run"] == "R0")
plt = ps.setup()
fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), sharex=True)
for ax, (k, title) in zip(axes, PANELS):
    for (lab, run), c in zip(RUNS, RAMP):
        rs = sorted([o for o in out if o["run"] == run and o["ep"] <= 17.9], key=lambda o: o["ep"])
        x = np.array([0.0] + [o["ep"] for o in rs])
        val = lambda o: (o["silent"] + o["speak"]) / 2 if k == "score" else o[k]
        y = np.array([val(base)] + [val(o) for o in rs])
        ax.plot(x, y, color=c, lw=2, marker="o", ms=3.5, mew=0, label=f"EPAD {lab}")
    ax.axhline(val(base), color=ps.BASE, lw=1.5, ls=(0, (4, 3)), label="base model")
    ax.axvspan(9.5, 17.9, color=ps.GRID, alpha=0.5, lw=0)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}" if k != "score" else f"{v:.1f}"))
    ax.set_title(title)
    ax.set_xlabel("epoch")
seen = {}
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        seen.setdefault(l, h)
fig.legend(list(seen.values()), list(seen), loc="upper right", ncol=len(seen), bbox_to_anchor=(0.995, 1.0))
fig.suptitle("EPAD-weight sweep (1 h, lr 5e-5, dev_80): shaded = plateau window used for comparison",
             x=0.01, ha="left", fontsize=11, fontweight="bold", color=ps.INK)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("results/epad/analysis/epad_sweep.png", dpi=150)
print("wrote results/epad/analysis/epad_sweep.png")
