"""Figures for docs/decisions.md, computed from the saved results (no GPU).

    PYTHONPATH=. python docs/figs/make_decision_figs.py

Reads results/*/analysis/speak_silence.json (per-checkpoint two-axis rescoring of dev_80),
results/phase1/runs/*/metrics.eval.jsonl (dev loss), results/run50h/eval_raw (per-hold scores),
results/test/summary.json (held-out suites). Writes docs/figs/*.png.
Colors: data size is identity (1 h blue, 10 h orange, 50 h aqua, base dashed gray), in every figure;
peak-LR colors (silence.plotstyle) only where runs at one data size are compared by LR.
"""
import json
from pathlib import Path
from statistics import mean

from silence import plotstyle as ps

R = Path("results")
OUT = Path("docs/figs")
OUT.mkdir(parents=True, exist_ok=True)
plt = ps.setup()

C1, C10, C50 = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3 (aqua < 3:1: always labeled)


def load(phase):
    return json.loads((R / phase / "analysis" / "speak_silence.json").read_text())


def step(row):
    return 0 if row["ck"] == "base" else int(row["ck"].split("_")[-1])


def score(row):
    return (row["silent"] + row["speak"]) / 2


def plateau(rows, run, lo, hi, key=None):
    sel = [r for r in rows if r["run"] == run and lo <= r["ep"] <= hi]
    f = key or score
    return mean(f(r) for r in sel)


p1, p2, ep1, ep10, p50 = load("phase1"), load("phase2"), load("epad"), load("epad10h"), load("run50h")
base = next(r for r in p1 if r["run"] == "R0")
RUNS_1H = {2e-6: "R1_1h_lr2e-06", 5e-6: "R6_1h_lr5e-06", 1e-5: "R2_1h_lr1e-05", 2e-5: "R7_1h_lr2e-05",
           5e-5: "R3_1h_lr5e-05", 2e-4: "R4_1h_lr0.0002"}
RUNS_10H = {2e-5: "R8_10h_lr2e-05", 5e-5: "R9_10h_lr5e-05", 2e-4: "R10_10h_lr0.0002", 5e-4: "R11_10h_lr0.0005"}


def lr_tick(lr):
    return f"{lr:.0e}".replace("e-0", "e-")


# ---------------------------------------------------------------- 1. LR sweep plateau, 1 h vs 10 h
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharex=True)
for ax, (key, title) in zip(axes, [(score, "Score (mean of the two)"), (lambda r: r["silent"], "Silent when due"),
                                   (lambda r: r["speak"], "Speaks when due")]):
    for runs, rows, lo, hi, c, lab in [(RUNS_1H, p1, 10, 21, C1, "1 h (plateau ep 10–20)"),
                                        (RUNS_10H, p2, 2, 5.1, C10, "10 h (plateau ep 2–5)")]:
        xs = sorted(runs)
        ys = [plateau(rows, runs[x], lo, hi, key) for x in xs]
        ax.plot(xs, ys, color=c, marker="o", ms=5, label=lab)
    ax.axhline(key(base), color=ps.BASE, ls="--", lw=1.2, label="base model")
    ax.set_xscale("log")
    ax.set_xticks(sorted(set(RUNS_1H) | set(RUNS_10H)))
    ax.set_xticklabels([lr_tick(x) for x in sorted(set(RUNS_1H) | set(RUNS_10H))], rotation=45)
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.set_xlabel("peak learning rate")
axes[0].legend(loc="lower right")
axes[0].annotate("too low:\nquiet model", (2e-6, plateau(p1, RUNS_1H[2e-6], 10, 21)), xytext=(2.2e-6, 0.62),
                 fontsize=8, color=ps.INK_2)
axes[0].annotate("too high:\nquiet model", (5e-4, plateau(p2, RUNS_10H[5e-4], 2, 5.1)), xytext=(1.3e-4, 0.42), arrowprops=dict(arrowstyle="-", color=ps.MUTED, lw=0.8),
                 fontsize=8, color=ps.INK_2)
fig.tight_layout()
fig.savefig(OUT / "lr_plateau.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. dev loss vs behavior (1 h)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 5.2), sharex=True)
for lr in (1e-5, 5e-5, 2e-4):
    run = RUNS_1H[lr]
    ev = [json.loads(l) for l in (R / "phase1/runs" / run / "metrics.eval.jsonl").read_text().splitlines()]
    xs = [0] + [e["step"] * 16 / 113 for e in ev]
    ys = [0.5369] + [e["text_eval_loss"] for e in ev]
    a1.plot(xs, ys, color=ps.run_color(run), label=f"lr {lr_tick(lr)}")
    i = min(range(1, len(ys)), key=lambda k: ys[k])
    a1.plot(xs[i], ys[i], "o", color=ps.run_color(run), ms=7, mec=ps.SURFACE, mew=1.5)
    rows = sorted([r for r in p1 if r["run"] == run], key=lambda r: r["ep"])
    a2.plot([0] + [r["ep"] for r in rows], [score(base)] + [score(r) for r in rows], color=ps.run_color(run),
            label=f"lr {lr_tick(lr)}")
    a2.axvline(xs[i], color=ps.run_color(run), lw=0.8, ls=":")
a1.axhline(0.5369, color=ps.BASE, ls="--", lw=1.2, label="base model")
a2.axhline(score(base), color=ps.BASE, ls="--", lw=1.2, label="base model")
a1.set_title("Dev text loss (lower = better fit): bottoms at epoch 3–10, then rises (memorization)")
a1.set_ylabel("text cross-entropy")
a2.set_title("Behavior score on dev_80: keeps improving long after the loss minimum (dotted)")
a2.set_ylabel("score")
a2.set_ylim(0, 1)
a2.set_xlabel("epoch (1 h data, 113 episodes)")
a1.legend(loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(OUT / "loss_vs_behavior.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3. plateau is set by optimizer steps
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for rows, run, c, lab in [(p1, RUNS_1H[2e-4], C1, "1 h, lr 2e-4 (best 1 h)"),
                          (p2, RUNS_10H[5e-5], C10, "10 h, lr 5e-5 (best 10 h)")]:
    rr = sorted([r for r in rows if r["run"] == run], key=step)
    ax.plot([0] + [step(r) for r in rr], [score(base)] + [score(r) for r in rr], color=c, marker="o", ms=3.5, label=lab)
ax.axhline(score(base), color=ps.BASE, ls="--", lw=1.2, label="base model")
ax.axvline(70, color=ps.MUTED, lw=1, ls=":")
ax.text(74, 0.33, "~70 steps ≈ 1,100 conversations seen\n= 10 epochs of 1 h, ≈ 1 epoch of 10 h", fontsize=8, color=ps.INK_2)
ax.set_xlabel("optimizer steps (batch 16)")
ax.set_ylabel("score")
ax.set_ylim(0, 1)
ax.set_title("Behavior plateaus after ~70 steps at both data sizes")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "steps_plateau.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 4. silence bias of the text targets
pad, epad = 89.7, 0.64
words = 100 - pad - epad
w = {"PAD (stay silent)": pad * 0.5, "word tokens": words * 1.0, "EPAD (start talking)": epad * 0.5}
tot = sum(w.values())
bars = {"share of text frames": {"PAD (stay silent)": pad, "word tokens": words, "EPAD (start talking)": epad},
        "share of text-loss weight": {k: 100 * v / tot for k, v in w.items()}}
cols = {"PAD (stay silent)": "#c3c2b7", "word tokens": C1, "EPAD (start talking)": C10}
fig, ax = plt.subplots(figsize=(7.5, 2.2))
for y, (lab, parts) in enumerate(bars.items()):
    x0 = 0
    for k, v in parts.items():
        ax.barh(y, v, left=x0, color=cols[k], edgecolor=ps.SURFACE, linewidth=2, height=0.55, label=k if y == 0 else None)
        if v > 5:
            ax.text(x0 + v / 2, y, f"{v:.1f}%", ha="center", va="center", fontsize=8, color=ps.INK)
        x0 += v
    e = parts["EPAD (start talking)"]
    ax.annotate(f"EPAD {e:.2f}%", (100, y), xytext=(101.5, y), va="center", fontsize=8, color=ps.INK)
ax.set_yticks(range(len(bars)))
ax.set_yticklabels(list(bars))
ax.set_xlim(0, 115)
ax.set_xticks([0, 25, 50, 75, 100])
ax.grid(False)
ax.set_title("train_1h text targets: the 'start talking' decision is ~0.6% of the text loss")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.18), ncol=3)
fig.tight_layout()
fig.savefig(OUT / "silence_bias.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 5. EPAD weight: trade-off + timed re-ask
E1 = [(1, "R3_1h_lr5e-05"), (3, "R16_1h_lr5e-05_epad3"), (10, "R17_1h_lr5e-05_epad10"),
      (30, "R18_1h_lr5e-05_epad30"), (100, "R19_1h_lr5e-05_epad100")]
E10 = [(1, "R9_10h_lr5e-05"), (3, "R20_10h_lr5e-05_epad3"), (10, "R21_10h_lr5e-05_epad10")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
import numpy as np
g = np.linspace(0.5, 1.0, 50)
for s in (0.8, 0.85, 0.9):
    a1.plot(g, 2 * s - g, color=ps.AXIS, lw=1, ls=":", zorder=0,
            label="equal score 0.80 / 0.85 / 0.90 (up-right is better)" if s == 0.8 else None)
for E, rows, lo, hi, c, lab in [(E1, ep1, 10, 17.9, C1, "1 h (plateau ep 10–17.8)"),
                                (E10, ep10, 2, 5.1, C10, "10 h (plateau ep 2–5)")]:
    xs = [plateau(rows, run, lo, hi, lambda r: r["silent"]) for _, run in E]
    ys = [plateau(rows, run, lo, hi, lambda r: r["speak"]) for _, run in E]
    a1.plot(xs, ys, color=c, marker="o", ms=6, label=lab, mec=ps.SURFACE, mew=1.5)
    for (k, _), x, y in zip(E, xs, ys):
        off = {(C1, 3): (-6, -13), (C1, 10): (6, 3), (C1, 30): (-6, 7), (C1, 100): (6, 4)}.get((c, k), (6, 4))
        a1.annotate(f"×{k}", (x, y), xytext=off, textcoords="offset points", fontsize=8, color=ps.INK_2,
                    ha="right" if off[0] < 0 else "left")
a1.set_xlim(0.65, 1.02)
a1.set_ylim(0.65, 1.0)
a1.set_xlabel("silent when due")
a1.set_ylabel("speaks when due")
a1.set_title("EPAD weight moves models along the trade-off")
a1.grid(True, axis="both")
a1.legend(loc="lower left")

ks = [1, 3, 10, 30, 100]
tim = lambda r: r["tim_any"]
mid1 = [plateau(ep1, run, 4, 9.5, tim) for _, run in E1]
pl1 = [plateau(ep1, run, 10, 17.9, tim) for _, run in E1]
pl10 = {k: plateau(ep10, run, 2, 5.1, tim) for k, run in E10}
x = np.arange(len(ks))
bw = 0.26
a2.bar(x - bw, [100 * v for v in mid1], bw, color="none", edgecolor=C1, hatch="////", linewidth=1.2, label="1 h, mid-training (ep 4–9)")
a2.bar(x, [100 * v for v in pl1], bw, color=C1, edgecolor=ps.SURFACE, label="1 h, plateau")
a2.bar([i + bw for i, k in enumerate(ks) if k in pl10], [100 * pl10[k] for k in ks if k in pl10], bw, color=C10,
       edgecolor=ps.SURFACE, label="10 h, plateau")
for i, k in enumerate(ks):
    for dx, v in [(-bw, mid1[i]), (0, pl1[i])] + ([(bw, pl10[k])] if k in pl10 else []):
        a2.text(i + dx, 100 * v + 1.5, f"{100 * v:.0f}", ha="center", fontsize=7, color=ps.INK_2)
a2.axhline(100 * tim(base), color=ps.BASE, ls="--", lw=1.2, label="base model")
a2.set_xticks(x)
a2.set_xticklabels([f"×{k}" for k in ks])
a2.set_xlabel("EPAD weight (× PAD's 0.5)")
a2.set_ylabel("% of timed holds re-asked")
a2.set_ylim(0, 100)
a2.set_title("Timed holds: re-asks after 'give me N seconds'")
a2.legend(loc="upper left", fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "epad.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 6. 50 h diagnosis: silence by hold type
def by_type(run):
    d = R / "run50h/eval_raw" / run / "dev_80"
    cks = sorted(p for p in d.iterdir() if (p / "scores.jsonl").exists())[-10:]
    acc = {}
    for ck in cks:
        for line in (ck / "scores.jsonl").read_text().splitlines():
            ep = json.loads(line)
            for h in ep["holds"]:
                if h.get("violated") is None:
                    continue
                acc.setdefault(ep["type"], []).append(0 if h["violated"] else 1)
    return {k: mean(v) for k, v in acc.items()}, {k: len(v) for k, v in acc.items()}


t21, n21 = by_type("R21_10h_lr5e-05_epad10")
t22, _ = by_type("R22_50h_lr5e-05_epad10")
order = ["B2_std", "B3_std", "B3_early", "B1_std", "B1_early", "B1_double", "B1_rescind", "B3_rescind"]
order = [t for t in order if t in t21 and t in t22]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
y = np.arange(len(order))
ax.barh(y - 0.2, [100 * t21[t] for t in order], 0.38, color=C10, edgecolor=ps.SURFACE, label="10 h (R21, recipe v1)")
ax.barh(y + 0.2, [100 * t22[t] for t in order], 0.38, color=C50, edgecolor=ps.SURFACE, label="50 h (R22, recipe v2)")
for i, t in enumerate(order):
    ax.text(100 * t21[t] + 1, i - 0.2, f"{100 * t21[t]:.0f}%", va="center", fontsize=7, color=ps.INK_2)
    ax.text(100 * t22[t] + 1, i + 0.2, f"{100 * t22[t]:.0f}%", va="center", fontsize=7, color=ps.INK_2)
ax.set_yticks(y)
ax.set_yticklabels([f"{t}  (n={n21[t]})" for t in order])
ax.invert_yaxis()
ax.set_xlim(0, 112)
ax.set_xlabel("% of holds kept silent (dev_80, last 10 checkpoints)")
ax.grid(True, axis="x")
ax.grid(False, axis="y")
ax.set_title("50 h lost silence in holds with long gaps, not in side conversations", pad=30)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "r22_by_type.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 7. held-out test suites
s = json.loads((R / "test/summary.json").read_text())
models = list(s)
mcol = [ps.BASE, C1, C10, C50]
mlab = ["base model", "1 h, EPAD ×10 (R17)", "10 h, EPAD ×10 (R21)", "50 h, EPAD ×10 (R22)"]
suites = ["test_script", "test_word", "test_voice", "test_distract", "test_cond", "test_ordinary", "test_all"]
slab = {"test_script": "new scripts", "test_word": "new wording", "test_voice": "unseen voices",
        "test_distract": "dense distractors", "test_cond": "new conditions", "test_ordinary": "ordinary only",
        "test_all": "all held out"}
fig, ax = plt.subplots(figsize=(7.5, 5.2))
y = np.arange(len(suites))
h = 0.2
for j, (m, c, lab) in enumerate(zip(models, mcol, mlab)):
    vals = [s[m][su]["score"] for su in suites]
    ax.barh(y + (j - 1.5) * h, vals, h, color=c, edgecolor=ps.SURFACE, linewidth=1, label=lab)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i + (j - 1.5) * h, f"{v:.2f}", va="center", fontsize=6.5, color=ps.INK_2)
ax.set_yticks(y)
ax.set_yticklabels([f"{slab[su]}\n({su})" for su in suites])
ax.invert_yaxis()
ax.set_xlim(0, 1.1)
ax.set_xlabel("score (held-out test suite, 90 episodes, 1 sample)")
ax.grid(True, axis="x")
ax.grid(False, axis="y")
ax.set_title("Held-out generalization: 10 h best on 6 of 7 suites", pad=34)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.01), ncol=2, fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "test_suites.png", dpi=200)
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
print("R21/R22 silent by type:", {t: (round(t21[t], 2), round(t22[t], 2)) for t in order})
print("EPAD timed re-ask mid1/pl1/pl10:", [round(v, 2) for v in mid1], [round(v, 2) for v in pl1], pl10)
