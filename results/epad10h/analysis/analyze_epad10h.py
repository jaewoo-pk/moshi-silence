"""EPAD weight at 10 h (lr 5e-5): R20 (x3), R21 (x10) vs R9 (x1) and R8 (2e-5, x1). Reads
analysis/speak_silence.json (rescore_speak_silence.py) + eval/summary.csv (eval_report: text dev loss,
on-time, violated). Writes analysis/epad10h.png, analysis/epad10h_table.json and prints markdown tables.
Run from the repo root:
  python results/epad10h/analysis/rescore_speak_silence.py results/epad10h
  PYTHONPATH=. python results/epad10h/analysis/analyze_epad10h.py"""
import csv, json
import numpy as np
from silence import plotstyle as ps

D = "results/epad10h"
RUNS = [("EPAD ×1 (R9)", "R9_10h_lr5e-05"), ("EPAD ×3 (R20)", "R20_10h_lr5e-05_epad3"),
        ("EPAD ×10 (R21)", "R21_10h_lr5e-05_epad10")]
RAMP = ["#86b6ef", "#2a78d6", "#0d366b"]  # ordinal blue ramp (same steps as epad_sweep.png), low -> high EPAD
REF = ("lr 2e-5, EPAD ×1 (R8)", "R8_10h_lr2e-05")
PLATEAU = (1.95, 5.02)  # 10 h plateau window used in phase 2 (epochs 2-5)
PANELS = [("silent", "silent when it should be"), ("speak", "speaks when it should"),
          ("tim_any", "timed hold: re-asked on its own"), ("score", "score = mean(silent, speaks)")]

out = json.load(open(f"{D}/analysis/speak_silence.json"))
for o in out:
    o["score"] = (o["silent"] + o["speak"]) / 2
summ = {(r["run"], r["ckpt"]): r for r in csv.DictReader(open(f"{D}/eval/summary.csv"))}
for o in out:
    s = summ.get((o["run"], o["ck"]), {})
    f = lambda k: float(s[k]) if s.get(k) not in (None, "", "nan") else float("nan")
    o["text_dev_loss"], o["on_time"], o["violated"] = f("text_dev_loss"), f("on_time"), f("hold_violated")
base = next(o for o in out if o["run"] == "R0")
rows = lambda run: sorted([o for o in out if o["run"] == run], key=lambda o: o["ep"])

# ---- plot
plt = ps.setup()
fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), sharex=True)
for ax, (k, title) in zip(axes, PANELS):
    lines = [(lab, run, c, "-", 2) for (lab, run), c in zip(RUNS, RAMP)] + [(REF[0], REF[1], ps.run_color(REF[1]), ":", 1.6)]
    for lab, run, c, ls, lw in lines:
        rs = rows(run)
        if not rs:
            continue
        x = np.array([0.0] + [o["ep"] for o in rs]); y = np.array([base[k]] + [o[k] for o in rs])
        ax.plot(x, y, color=c, lw=lw, ls=ls, marker="o", ms=3.5 if ls == "-" else 2.5, mew=0, label=lab)
    ax.axhline(base[k], color=ps.BASE, lw=1.5, ls=(0, (4, 3)), label="base model")
    ax.axvspan(*PLATEAU, color=ps.GRID, alpha=0.5, lw=0)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, k=k: f"{v:.0%}" if k != "score" else f"{v:.1f}"))
    ax.set_title(title); ax.set_xlabel("epoch")
seen = {}
for ax in axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        seen.setdefault(l, h)
fig.legend(list(seen.values()), list(seen), loc="upper right", ncol=len(seen), bbox_to_anchor=(0.995, 1.0))
fig.suptitle("EPAD weight at 10 h (lr 5e-5, dev_80): shaded = plateau window (epochs 2–5)",
             x=0.01, ha="left", fontsize=11, fontweight="bold", color=ps.INK)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(f"{D}/analysis/epad10h.png", dpi=150)
print(f"wrote {D}/analysis/epad10h.png\n")

# ---- tables
allruns = RUNS + [REF]
pct = lambda v: "—" if v != v else f"{v:.0%}"
print("### Per checkpoint (score / silent / speaks / timed re-ask / ordinary / text dev loss)\n")
print("| epoch | " + " | ".join(lab for lab, _ in allruns) + " |")
print("|---|" + "---|" * len(allruns))
eps = sorted({round(o["ep"], 2) for r in allruns for o in rows(r[1])})
print(f"| 0 (base) | " + " | ".join(f"{base['score']:.2f} ({pct(base['silent'])}/{pct(base['speak'])}/{pct(base['tim_any'])}/{pct(base['ordr'])}, {base['text_dev_loss']:.3f})" for _ in allruns) + " |")
for e in eps:
    cells = []
    for _, run in allruns:
        o = next((o for o in rows(run) if abs(o["ep"] - e) < 0.02), None)
        cells.append("" if o is None else f"**{o['score']:.2f}** ({pct(o['silent'])}/{pct(o['speak'])}/{pct(o['tim_any'])}/{pct(o['ordr'])}, {o['text_dev_loss']:.3f})")
    print(f"| {e:.2f} | " + " | ".join(cells) + " |")

print("\n### Plateau (epochs 2–5) and best checkpoint\n")
print("| Run | Plateau score | Silent | Speaks | Cue spoke / on time | Timed re-ask | Ordinary | Best ckpt (epoch): score, silent / speaks, timed, ordinary | Min text dev loss (epoch) |")
print("|---|---|---|---|---|---|---|---|---|")
table = {}
for lab, run in allruns:
    rs = rows(run)
    if not rs:
        continue
    pl = [o for o in rs if PLATEAU[0] <= o["ep"] <= PLATEAU[1]]
    a = lambda k: float(np.nanmean([o[k] for o in pl]))
    best = max(rs, key=lambda o: (round(o["score"], 6), -o["ep"]))  # ties -> earlier epoch
    ml = min(rs, key=lambda o: o["text_dev_loss"] if o["text_dev_loss"] == o["text_dev_loss"] else 9)
    table[run] = dict(plateau={k: a(k) for k in ["score", "silent", "speak", "cue_any", "cue_ok", "tim_any", "ordr"]},
                      best=best, min_loss=dict(ck=ml["ck"], ep=ml["ep"], loss=ml["text_dev_loss"]))
    print(f"| {lab} | {a('score'):.2f} | {pct(a('silent'))} | {pct(a('speak'))} | {pct(a('cue_any'))} / {pct(a('cue_ok'))} | {pct(a('tim_any'))} | {pct(a('ordr'))} | "
          f"`{best['ck']}` ({best['ep']:.2f}): {best['score']:.2f}, {pct(best['silent'])} / {pct(best['speak'])}, {pct(best['tim_any'])}, {pct(best['ordr'])} | {ml['text_dev_loss']:.3f} ({ml['ep']:.2f}) |")
print(f"\nBase model: score {base['score']:.2f}, silent {pct(base['silent'])}, speaks {pct(base['speak'])}, timed {pct(base['tim_any'])}, ordinary {pct(base['ordr'])}, text dev loss {base['text_dev_loss']:.3f}")
print("n per checkpoint: silence-scored holds %d, cue releases %d, timer releases %d, ordinary %d" % (base["n_sil"], base["n_cue"], base["n_tim"], base["n_ord"]))
json.dump(table, open(f"{D}/analysis/epad10h_table.json", "w"), indent=1)
