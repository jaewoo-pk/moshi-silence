"""Fix 1 (2026-10-15, after R22): cap the share of one episode family in a built train set, keeping order.

  python -m silence.rebalance /workspace/data50/train_v3 --family B1 --share 0.15 --out train_v3_b1-15.jsonl

Reads <dir>/manifest.jsonl + episodes.jsonl (after filter_checked), keeps every episode of the other families and
the first B1 episodes (in build order) until B1 is --share of the episodes. Writes an absolute-path jsonl for
training plus a ledger (silence.make_splits.ledger) next to it.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from .make_splits import ledger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--family", default="B1")
    ap.add_argument("--share", type=float, default=0.15)
    ap.add_argument("--out", required=True, help="file name inside data_dir")
    a = ap.parse_args()
    d = Path(a.data_dir)
    man = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines()]
    rows = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines()]
    fam = lambda r: r["type"].split("_")[0]
    n_other = sum(fam(r) != a.family for r in rows)
    cap = round(n_other * a.share / (1 - a.share))
    keep, used = [], 0
    for m, r in zip(man, rows):
        if fam(r) == a.family:
            if used >= cap:
                continue
            used += 1
        keep.append((m, r))
    (d / a.out).write_text("".join(json.dumps({"path": str((d / m["path"]).resolve()), "duration": m["duration"]}) + "\n"
                                   for m, _ in keep))
    metas = [json.loads((d / m["path"]).with_suffix(".json").read_text())["meta"] for m, _ in keep]
    led = ledger([r for _, r in keep], metas)
    (d / (Path(a.out).stem + "_ledger.json")).write_text(json.dumps(led, indent=1))
    c = Counter(fam(r) for _, r in keep)
    print(f"{a.out}: {len(keep)} episodes, {sum(r['duration'] for _, r in keep) / 3600:.2f} h; "
          f"families {dict(c)} ({a.family} {100 * c[a.family] / len(keep):.1f}%)")


if __name__ == "__main__":
    main()
