"""Training / dev-loss curves and a per-run summary from moshi-finetune's metric logs.

  python -m silence.plot_training results/phase1/runs --configs results/phase1/configs --out results/phase1/plots

<runs>/<run>/metrics.train.jsonl (every step: loss, text_loss, audio_loss, lr, memory, time) and
<runs>/<run>/metrics.eval.jsonl (every checkpoint: eval_loss, text_eval_loss, audio_eval_loss) are
written by train.py. Pull them back from the box without the checkpoints:
  rsync -rlpt --include '*/' --include 'metrics.*.jsonl' --include 'args.yaml' --exclude '*' \
      box:/workspace/runs/ results/phase1/runs/
  rsync -rlpt box:/workspace/moshi-silence/configs/ results/phase1/configs/

Step -> epoch uses configs/<run>.epochs.json (episodes, batch); without it the x axis is steps.
R0 (base model, no training) contributes its dev loss as a dashed reference line.

Outputs in --out:
  loss_curves.png     train loss (thin: per step, thick: 1-epoch trailing mean) and dev loss per
                      checkpoint; total / text / audio; one line per LR
  lr_schedule.png     the OneCycle schedule each run followed (log scale)
  dev_loss_vs_lr.png  dev loss at the best and at the final checkpoint vs peak LR
  training_summary.csv / .md   per run: final train loss, best / final dev loss (+ epoch),
                      median s/step, peak memory
"""
import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from . import plotstyle as ps

LOSSES = [("loss", "eval_loss", "total"), ("text_loss", "text_eval_loss", "text"),
          ("audio_loss", "audio_eval_loss", "audio")]


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_run(run_dir, configs):
    name = run_dir.name
    train = read_jsonl(run_dir / "metrics.train.jsonl")
    evals = read_jsonl(run_dir / "metrics.eval.jsonl")
    per_epoch = None  # epochs per step
    ep = Path(configs) / f"{name}.epochs.json" if configs else None
    if ep and ep.exists():
        d = json.loads(ep.read_text())
        per_epoch = d["batch"] / d["episodes"]
    return {"name": name, "train": train, "eval": evals, "per_epoch": per_epoch,
            "is_base": name.startswith("R0")}


def x_of(run, steps):
    steps = np.asarray(steps, float)
    return steps * run["per_epoch"] if run["per_epoch"] else steps


def trailing_mean(y, k):
    y = np.asarray(y, float)
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i + 1 - k)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def steps_per_epoch(run):
    return max(1, round(1 / run["per_epoch"])) if run["per_epoch"] else 7


def sec_per_step(run):
    """Median wall time per training step, excluding step 1 (model load) and steps that ran a
    dev eval / checkpoint (their interval includes it)."""
    tr = run["train"]
    eval_steps = {r["step"] for r in run["eval"]}
    dts = []
    for a, b in zip(tr, tr[1:]):
        if b["step"] - a["step"] != 1 or b["step"] in eval_steps or a["step"] < 1:
            continue
        dts.append((datetime.fromisoformat(b["at"]) - datetime.fromisoformat(a["at"])).total_seconds())
    return float(np.median(dts)) if dts else float("nan")


def summarize(runs):
    rows = []
    for r in runs:
        tr, ev = r["train"], r["eval"]
        row = {"run": r["name"], "peak_lr": ps.run_lr(r["name"]) or 0.0,
               "steps": tr[-1]["step"] if tr else 0,
               "epochs": round(tr[-1]["step"] * r["per_epoch"], 2) if tr and r["per_epoch"] else "",
               "s_per_step": round(sec_per_step(r), 2) if tr else "",
               "peak_mem_gb": round(max(t.get("peak_allocated_mem", 0) for t in tr), 1) if tr else ""}
        if tr and not r["is_base"]:
            k = steps_per_epoch(r)
            for key, _, short in LOSSES:
                row[f"train_{short}_last_epoch"] = round(float(np.mean([t[key] for t in tr[-k:]])), 4)
        for _, key, short in LOSSES:
            vals = [(e[key], e["step"]) for e in ev if key in e and np.isfinite(e[key])]
            if not vals:
                continue
            best, bstep = min(vals)
            row[f"dev_{short}_best"] = round(best, 4)
            row[f"dev_{short}_best_epoch"] = round(bstep * r["per_epoch"], 2) if r["per_epoch"] else bstep
            row[f"dev_{short}_final"] = round(vals[-1][0], 4)
        rows.append(row)
    return rows


def write_summary(rows, out):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "training_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    cols = ["run", "peak_lr", "steps", "epochs", "train_total_last_epoch", "dev_total_best",
            "dev_total_best_epoch", "dev_total_final", "dev_text_best", "dev_text_best_epoch",
            "dev_audio_best", "s_per_step", "peak_mem_gb"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(
            (f"{r[c]:.0e}" if c == "peak_lr" and r.get(c) else str(r.get(c, ""))) for c in cols) + " |")
    md = "\n".join(lines) + "\n"
    (out / "training_summary.md").write_text(md)
    return md


def robust_top(series, ref=None):
    """Upper y limit that keeps a diverging run from flattening the others: the 95th percentile
    of all plotted values (+ headroom), at least 10% above the reference line."""
    vals = np.concatenate([np.asarray(s, float) for s in series if len(s)]) if series else np.array([])
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return None
    top = np.percentile(vals, 95) * 1.15
    if ref is not None:
        top = max(top, ref * 1.1)
    # always show where the runs start (the first ~epoch), so only later divergence is cut off
    starts = [np.nanmax(np.asarray(s, float)[:8]) for s in series if len(s)]
    if starts:
        top = max(top, max(starts) * 1.05)
    return top


def plot_losses(plt, runs, out, title="Phase 1"):
    trained = [r for r in runs if not r["is_base"] and (r["train"] or r["eval"])]
    base = next((r for r in runs if r["is_base"] and r["eval"]), None)
    xlabel = "epoch" if all(r["per_epoch"] for r in trained) else "step"
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharex=True)
    for col, (tkey, ekey, short) in enumerate(LOSSES):
        ref = base["eval"][-1].get(ekey) if base else None
        ax_t, ax_d = axes[0, col], axes[1, col]
        plotted_t, plotted_d, off = [], [], []
        for r in sorted(trained, key=lambda r: ps.run_lr(r["name"]) or 0):
            c = ps.run_color(r["name"])
            tr = [t for t in r["train"] if tkey in t]
            if tr:
                x = x_of(r, [t["step"] for t in tr])
                y = np.array([t[tkey] for t in tr], float)
                sm = trailing_mean(y, steps_per_epoch(r))
                ax_t.plot(x, y, color=c, lw=1, alpha=0.25)
                ax_t.plot(x, sm, color=c, lw=2, label=ps.run_label(r["name"]))
                plotted_t.append(sm)
            ev = [e for e in r["eval"] if ekey in e]
            if ev:
                x = x_of(r, [e["step"] for e in ev])
                y = np.array([e[ekey] for e in ev], float)
                if ref is not None:  # step 0 is exactly the base model (LoRA B = 0)
                    x, y = np.insert(x, 0, 0.0), np.insert(y, 0, ref)
                ax_d.plot(x, y, color=c, lw=2, marker="o", ms=3.5, mew=0, label=ps.run_label(r["name"]))
                plotted_d.append(y)
        for ax, plotted, title in ((ax_t, plotted_t, f"train loss: {short}"),
                                   (ax_d, plotted_d, f"dev loss: {short}")):
            if ref is not None and ax is ax_d:
                ax.axhline(ref, color=ps.BASE, lw=1.5, ls=(0, (4, 3)), label="base model (R0)")
            top = robust_top(plotted, ref if ax is ax_d else None)
            if top is not None:
                over = [ps.run_label(r["name"]) for r, s in zip(
                    sorted(trained, key=lambda r: ps.run_lr(r["name"]) or 0), plotted)
                    if np.nanmax(s) > top] if len(plotted) == len(trained) else []
                lo = min(np.nanmin(s) for s in plotted)
                ax.set_ylim(max(0.0, lo - 0.08 * (top - lo)), top)
                if over:
                    ax.text(1.0, 1.015, ", ".join(over) + " runs off the top", transform=ax.transAxes,
                            ha="right", va="bottom", fontsize=7.5, color=ps.INK_2)
            ax.set_title(title)
        axes[1, col].set_xlabel(xlabel)
    axes[0, 0].set_ylabel("cross-entropy")
    axes[1, 0].set_ylabel("cross-entropy")
    seen = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    if seen:
        fig.legend(list(seen.values()), list(seen), loc="upper right", ncol=len(seen),
                   bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(f"{title}: loss vs epoch by peak LR (train: thin = per step, thick = 1-epoch mean)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=ps.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out / "loss_curves.png", dpi=150)
    plt.close(fig)


def plot_lr(plt, runs, out):
    trained = [r for r in runs if not r["is_base"] and r["train"]]
    if not trained:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for r in sorted(trained, key=lambda r: ps.run_lr(r["name"]) or 0):
        tr = r["train"]
        ax.plot(x_of(r, [t["step"] for t in tr]), [t["lr"] for t in tr], color=ps.run_color(r["name"]),
                label=ps.run_label(r["name"]))
    ax.set_yscale("log")
    ax.set_ylim(bottom=min(ps.run_lr(r["name"]) or 1 for r in trained) / 100)
    ax.set_title("LR schedule (OneCycle: 5% warmup, cosine decay)")
    ax.set_xlabel("epoch" if all(r["per_epoch"] for r in trained) else "step")
    ax.set_ylabel("learning rate")
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "lr_schedule.png", dpi=150)
    plt.close(fig)


def plot_vs_lr(plt, runs, rows, out):
    by = {r["run"]: r for r in rows}
    trained = sorted([r for r in runs if not r["is_base"] and r["eval"] and ps.run_lr(r["name"])],
                     key=lambda r: ps.run_lr(r["name"]))
    if len(trained) < 2:
        return
    base = next((r for r in runs if r["is_base"] and r["eval"]), None)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))
    for ax, (_, ekey, short) in zip(axes, LOSSES):
        xs = [ps.run_lr(r["name"]) for r in trained]
        best = [by[r["name"]].get(f"dev_{short}_best", np.nan) for r in trained]
        final = [by[r["name"]].get(f"dev_{short}_final", np.nan) for r in trained]
        ax.plot(xs, best, color=ps.INK_2, lw=2, marker="o", ms=7, mew=0, label="best checkpoint")
        ax.plot(xs, final, color=ps.MUTED, lw=1.5, marker="o", ms=7, mfc=ps.SURFACE, mew=1.5,
                label="final checkpoint")
        for r, x, y in zip(trained, xs, best):
            e = by[r["name"]].get(f"dev_{short}_best_epoch")
            if e != "" and e is not None and np.isfinite(y):
                ax.annotate(f"ep {e:.0f}", (x, y), textcoords="offset points", xytext=(7, -10),
                            ha="left", fontsize=7, color=ps.INK_2)
        if base:
            ax.axhline(base["eval"][-1][ekey], color=ps.BASE, lw=1.5, ls=(0, (4, 3)),
                       label="base model (R0)")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:.0e}".replace("e-0", "e-") for x in xs])
        ax.minorticks_off()
        ax.set_xlabel("peak LR")
        ax.set_title(f"dev loss: {short}")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "dev_loss_vs_lr.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", help="dir with <run>/metrics.{train,eval}.jsonl")
    ap.add_argument("--configs", help="dir with <run>.epochs.json (step -> epoch)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=r"^R\d", help="regex on run names (default: R*, skips _smoke)")
    ap.add_argument("--title", default="Phase 1")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(d, a.configs) for d in sorted(Path(a.runs).iterdir())
            if d.is_dir() and re.search(a.only, d.name)]
    runs = [r for r in runs if r["train"] or r["eval"]]
    if not runs:
        raise SystemExit(f"no metrics under {a.runs} matching {a.only!r}")
    rows = summarize(runs)
    print(write_summary(rows, out))
    plt = ps.setup()
    if plt is None:
        print("matplotlib not available: wrote the summary only")
        return
    plot_losses(plt, runs, out, a.title)
    plot_lr(plt, runs, out)
    plot_vs_lr(plt, runs, rows, out)
    print(f"plots -> {out}/loss_curves.png, lr_schedule.png, dev_loss_vs_lr.png")


if __name__ == "__main__":
    main()
