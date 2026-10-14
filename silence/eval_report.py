"""Aggregate eval_score outputs into rates with bootstrap CIs, tables and per-epoch curves.

  python -m silence.eval_report /workspace/eval --configs configs --runs /workspace/runs --out report/phase1
  python -m silence.eval_report /workspace/eval --configs configs --sensitivity --out report/phase1

Expects eval_gen / eval_score output laid out as
  <eval root>/<run>/<set>/<checkpoint_XXXXXX | base>/scores.jsonl   (several checkpoints)
  <eval root>/<run>/<set>/scores.jsonl                             (one model; ckpt = "-")
where <run> is a config name (R3_1h_lr5e-05, or R0 for the base model). Checkpoint -> epoch
comes from configs/<run>.epochs.json; "base" is epoch 0. The dev loss per checkpoint comes from
<runs>/<run>/metrics.eval.jsonl (written by train.py at the same steps as the checkpoints).

Rates are pooled over holds (or responses) of the episodes, with 95% percentile bootstrap CIs
from resampling EPISODES (holds within an episode are not independent). Outputs: summary.csv,
summary.md and, if matplotlib is importable, one plot per episode set (rate vs epoch, one line
per run, base model dashed). --sensitivity rescores every set at alternative thresholds from
the saved VAD segments and words (no regeneration) and writes sensitivity.csv.
"""
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .eval_score import Thresholds, score_episode

CODES = ["D1a", "D1b", "D1c", "D2a", "D2b", "D2c", "D3a", "D3b", "D4"]


# ---------------------------------------------------------------- metrics
def metrics(rows):
    """Point estimates over a list of eval_score rows."""
    holds = [h for r in rows for h in r["holds"]]
    viol = [h["violated"] for h in holds if h["violated"] is not None]
    cue = [h["latency"] for h in holds if h["kind"] == "cue" and h["latency"] is not None]
    tim = [abs(h["timer_err"]) for h in holds if h["kind"] == "timer" and h["timer_err"] is not None]
    first = [r["responses"][0]["responded"] for r in rows if r["responses"]]
    resp = [x["responded"] for r in rows for x in r["responses"]]
    lat = [r["responses"][0]["latency"] for r in rows if r["responses"] and r["responses"][0]["latency"] is not None]
    barge = [b["barged"] for r in rows for b in r["barge_ins"]]
    m = {
        "hold_violated": np.mean(viol) if viol else np.nan,
        "on_time": np.mean([h["outcome"] == "on_time" for h in holds]) if holds else np.nan,
        "no_onset": np.mean([h["outcome"] == "no_onset" for h in holds]) if holds else np.nan,
        "late": np.mean([h["outcome"] == "late" for h in holds]) if holds else np.nan,
        "talking_through": np.mean([h["talking_through"] for h in holds]) if holds else np.nan,
        "words_in_hold": np.mean([h["words_in_hold"] for h in holds]) if holds else np.nan,
        "cue_latency_med": np.median(cue) if cue else np.nan,
        "timer_abs_err_med": np.median(tim) if tim else np.nan,
        "ord_first_responded": np.mean(first) if first else np.nan,
        "ord_responded": np.mean(resp) if resp else np.nan,
        "ord_latency_med": np.median(lat) if lat else np.nan,
        "barge_in": np.mean(barge) if barge else np.nan,
    }
    for c in CODES:
        v = [d for h in holds for d in h["distractors"] if d["code"] == c]
        m[f"trig_{c}"] = np.mean([d["reacted"] for d in v]) if v else np.nan
        m[f"over_{c}"] = np.mean([d.get("overlapped", d["reacted"]) for d in v]) if v else np.nan
    m["n_episodes"] = len(rows)
    m["n_holds"] = len(holds)
    m["n_viol_holds"] = len(viol)
    return m


CI_KEYS = ["hold_violated", "on_time", "no_onset", "talking_through", "words_in_hold", "cue_latency_med",
           "timer_abs_err_med", "ord_first_responded", "ord_latency_med", "barge_in"]


def bootstrap(rows, n=1000, seed=0):
    """95% percentile CIs over episodes for CI_KEYS."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    draws = {k: [] for k in CI_KEYS}
    for _ in range(n):
        m = metrics([rows[i] for i in rng.choice(idx, len(idx))])
        for k in CI_KEYS:
            draws[k].append(m[k])
    out = {}
    for k, v in draws.items():
        v = np.array(v, float)
        v = v[np.isfinite(v)]
        out[k] = (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)
    return out


# ---------------------------------------------------------------- discovery
def find_scores(root):
    """[(run, set, ckpt, path)] for every scores.jsonl under root."""
    out = []
    for p in sorted(Path(root).rglob("scores.jsonl")):
        rel = p.relative_to(root).parts[:-1]
        if len(rel) == 3:
            out.append((rel[0], rel[1], rel[2], p))
        elif len(rel) == 2:
            out.append((rel[0], rel[1], "-", p))
    return out


def epoch_of(configs, run, ckpt):
    if ckpt == "base" or run.startswith("R0"):
        return 0.0
    f = Path(configs) / f"{run}.epochs.json" if configs else None
    if f and f.exists():
        return json.loads(f.read_text())["checkpoint_epoch"].get(ckpt, np.nan)
    return np.nan


def dev_loss(runs, run, ckpt):
    """text / audio dev loss at the checkpoint's step from <runs>/<run>/metrics.eval.jsonl."""
    if not runs or not ckpt.startswith("checkpoint_"):
        return np.nan, np.nan
    f = Path(runs) / run / "metrics.eval.jsonl"
    if not f.exists():
        return np.nan, np.nan
    step = int(ckpt.split("_")[1])
    for line in f.read_text().splitlines():
        d = json.loads(line)
        if d.get("step") == step:
            return d.get("text_eval_loss", np.nan), d.get("audio_eval_loss", np.nan)
    return np.nan, np.nan


# ---------------------------------------------------------------- sensitivity
def rescore(gen_dir, th):
    meta = json.loads((gen_dir / "meta.json").read_text())
    ep_dir = Path(meta["episodes_dir"])
    rows = []
    for rid in meta["ids"]:
        rec = json.loads((gen_dir / f"{rid}.json").read_text())
        ev = json.loads((ep_dir / f"{rec['id']}.json").read_text())["events"]
        rows.append(score_episode(ev, rec["duration"], [tuple(s) for s in rec["segments"]],
                                  [tuple(w) for w in rec["words"]], th))
    return rows


SENS = [("viol_speech", [0.5, 1.0, 2.0]), ("viol_words", [1, 3, 6]), ("resume_win", [2.0, 3.0, 5.0]),
        ("grace", [1.5, 2.5, 4.0])]


# ---------------------------------------------------------------- output
def fmt(v, lo=None, hi=None, pct=True):
    if v is None or not np.isfinite(v):
        return "–"
    s = f"{100 * v:.0f}%" if pct else f"{v:.2f}"
    if lo is not None and np.isfinite(lo):
        s += f" [{100 * lo:.0f}, {100 * hi:.0f}]" if pct else f" [{lo:.2f}, {hi:.2f}]"
    return s


def plot(table, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available: skipping plots")
        return
    panels = [("hold_violated", "hold violated"), ("on_time", "resumed on time"),
              ("ord_first_responded", "ordinary: answered"), ("text_dev_loss", "text dev loss")]
    for s in sorted({r["set"] for r in table}):
        rs = [r for r in table if r["set"] == s]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
        for ax, (k, title) in zip(axes, panels):
            base = [r for r in rs if r["ckpt"] == "base" or r["run"].startswith("R0")]
            for run in sorted({r["run"] for r in rs if not r["run"].startswith("R0")}):
                pts = sorted((r["epoch"], r[k], r.get(f"{k}_lo", np.nan), r.get(f"{k}_hi", np.nan))
                             for r in rs if r["run"] == run and r["ckpt"] != "base" and np.isfinite(r["epoch"]))
                if not pts:
                    continue
                x, y, lo, hi = map(np.array, zip(*pts))
                ax.plot(x, y, marker=".", label=run)
                if np.isfinite(lo).any():
                    ax.fill_between(x, lo, hi, alpha=0.15)
            if base and np.isfinite(base[0][k]):
                ax.axhline(base[0][k], ls="--", c="k", lw=1, label="base")
            ax.set_title(title)
            ax.set_xlabel("epoch")
        axes[0].legend(fontsize=7)
        fig.suptitle(s)
        fig.tight_layout()
        fig.savefig(out / f"curves_{s}.png", dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="eval root (<run>/<set>/<ckpt>/scores.jsonl)")
    ap.add_argument("--configs", help="dir with <run>.epochs.json")
    ap.add_argument("--runs", help="training runs dir (<run>/metrics.eval.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--sensitivity", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    table = []
    for run, s, ckpt, p in find_scores(a.root):
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        m = metrics(rows)
        ci = bootstrap(rows, a.boot)
        tl, al = dev_loss(a.runs, run, ckpt)
        r = {"run": run, "set": s, "ckpt": ckpt, "epoch": epoch_of(a.configs, run, ckpt),
             "text_dev_loss": tl, "audio_dev_loss": al, **m}
        for k, (lo, hi) in ci.items():
            r[f"{k}_lo"], r[f"{k}_hi"] = lo, hi
        table.append(r)
        print(f"{run:22} {s:14} {ckpt:18} ep {r['epoch']:5.1f}  viol {fmt(m['hold_violated'], *ci['hold_violated'])}  "
              f"on time {fmt(m['on_time'], *ci['on_time'])}  ord {fmt(m['ord_first_responded'], *ci['ord_first_responded'])}")
    if not table:
        raise SystemExit(f"no scores.jsonl under {a.root}")
    keys = list(table[0])
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)

    md = ["| run | set | ckpt | epoch | hold violated | resumed on time | no onset | talking through | "
          "cue latency (s) | B1 abs timer err (s) | ordinary answered | barge-in | text dev loss |",
          "|" + "---|" * 13]
    for r in sorted(table, key=lambda r: (r["set"], r["run"], r["epoch"] if np.isfinite(r["epoch"]) else -1)):
        md.append(f"| {r['run']} | {r['set']} | {r['ckpt']} | {r['epoch']:.1f} | "
                  f"{fmt(r['hold_violated'], r['hold_violated_lo'], r['hold_violated_hi'])} | "
                  f"{fmt(r['on_time'], r['on_time_lo'], r['on_time_hi'])} | {fmt(r['no_onset'])} | "
                  f"{fmt(r['talking_through'])} | {fmt(r['cue_latency_med'], pct=False)} | "
                  f"{fmt(r['timer_abs_err_med'], pct=False)} | "
                  f"{fmt(r['ord_first_responded'], r['ord_first_responded_lo'], r['ord_first_responded_hi'])} | "
                  f"{fmt(r['barge_in'])} | {fmt(r['text_dev_loss'], pct=False)} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    plot(table, out)

    if a.sensitivity:
        sens = []
        for run, s, ckpt, p in find_scores(a.root):
            for field, values in SENS:
                for v in values:
                    m = metrics(rescore(p.parent, replace(Thresholds(), **{field: v})))
                    sens.append({"run": run, "set": s, "ckpt": ckpt, "field": field, "value": v,
                                 **{k: m[k] for k in ("hold_violated", "on_time", "ord_first_responded")}})
        with (out / "sensitivity.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sens[0]))
            w.writeheader()
            w.writerows(sens)
    print(f"wrote {out}/summary.csv, summary.md" + (", sensitivity.csv" if a.sensitivity else ""))


if __name__ == "__main__":
    main()
