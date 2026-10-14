"""Write a pack file for held-out builds with overlapping packs removed.

  python -m silence.screen_packs --split test --against data/train data/dev \\
      --out packs/packs_test_screened.jsonl

Keeps only packs of --split, minus any that contain a line spoken verbatim in the --against
builds (from their episode specs, so pack fields never used in training, like friend_away
outside D4, don't count), or whose key text has content-word Jaccard >= --sim against a pack
used there. Test suites draw only test packs, so building them from
this file changes nothing else.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from silence.build import load_packs
from silence.build_dev import screen, spoken_lines, used_packs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", default="packs/packs.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--against", nargs="+", default=["data/train", "data/dev"])
    ap.add_argument("--sim", type=float, default=0.5)
    ap.add_argument("--out", default="packs/packs_test_screened.jsonl")
    a = ap.parse_args()

    packs = load_packs(a.packs)
    ref_ids = set().union(*(used_packs(d) for d in a.against))
    cand = {k: p for k, p in packs.items() if p.get("split", "train") == a.split}
    # screen() passes non-train packs through untouched, so present candidates as train
    as_train = {k: {**p, "split": "train"} for k, p in cand.items()}
    heard = set().union(*(spoken_lines(d) for d in a.against))
    keep, dropped = screen(as_train, [packs[k] for k in ref_ids if k in packs], a.sim, ref_lines=heard)
    fam = lambda p: "DECOY" if p.get("decoy") else p["family"]
    print(f"{a.split} packs: {dict(sorted(Counter(fam(p) for p in cand.values()).items()))}")
    print(f"dropped (vs {', '.join(a.against)}): {dict(sorted(dropped.items()))}")
    print(f"kept: {dict(sorted(Counter(fam(cand[k]) for k in keep).items()))}")
    Path(a.out).write_text("".join(json.dumps(cand[k]) + "\n" for k in keep))
    print(f"wrote {len(keep)} packs to {a.out}")


if __name__ == "__main__":
    main()
