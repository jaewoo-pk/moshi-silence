"""Reverse screen for a train top-up (50 h build, 2026-10-15): drop NEW train packs that
overlap any held-out material, so no test / dev script leaks into training.

  python -m silence.screen_train_packs --packs packs/packs_v2.jsonl --base packs/packs.jsonl \\
      --heldout data/dev data/test_* --out packs/packs_v2_screened.jsonl

New = train packs in --packs whose id is not in --base. A new pack is dropped if it contains a
line actually spoken in any --heldout build (normalized; from the episode specs), or if its key
text has content-word Jaccard >= --sim with any test pack or any pack used in a --heldout build
(same family). Same rules as screen_packs, in the other direction. Base packs pass through.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from silence.build import load_packs
from silence.build_dev import screen, spoken_lines, used_packs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--heldout", nargs="+", required=True)
    ap.add_argument("--sim", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    packs = load_packs(a.packs)
    base = set(load_packs(a.base))
    new = {k: p for k, p in packs.items() if k not in base and p.get("split", "train") == "train"}
    ref_ids = set().union(*(used_packs(d) for d in a.heldout)) | {k for k, p in packs.items() if p.get("split") == "test"}
    heard = set().union(*(spoken_lines(d) for d in a.heldout))
    keep, dropped = screen(new, [packs[k] for k in ref_ids if k in packs], a.sim, ref_lines=heard)
    fam = lambda p: "DECOY" if p.get("decoy") else p["family"]
    print(f"new train packs: {dict(sorted(Counter(fam(p) for p in new.values()).items()))}")
    print(f"dropped (vs {len(ref_ids)} held-out packs, {len(heard)} held-out spoken lines): {dict(sorted(dropped.items()))}")
    print(f"kept new: {dict(sorted(Counter(fam(p) for p in keep.values()).items()))}")
    out = [p for k, p in packs.items() if k in base or k in keep]
    Path(a.out).write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in out))
    print(f"wrote {len(out)} packs ({len(keep)} new) to {a.out}")


if __name__ == "__main__":
    main()
