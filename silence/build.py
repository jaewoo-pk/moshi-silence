"""Build episodes: specs -> TTS (cached) -> wav/json -> manifest + ledger.

  # Demo: one episode of each type (Pocket TTS on CPU):
  python -m silence.build demo --out demo

  # Real data:
  python -m silence.build train --packs packs/packs.jsonl --n 10000 --out data/train
  python -m silence.build test_word --packs packs/packs.jsonl --n 90 --out data/test_word

Episodes are written in a fixed shuffled order, so the first K episodes of the
train manifest form the nested 1 h / 10 h / 100 h subsets (see make_splits.py).
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import specs
from .assemble import assemble
from .tts import ClipCache, get_backend

MAX_TRAIN_DUR = 60.0  # one training window at duration_sec: 60

DEMO = [  # (type, pack, split, extra kwargs)
    ("B1_std", "b1_0001", "train", {}),
    ("B1_early", "b1_0002", "train", {}),
    ("B1_rescind", "b1_0001", "train", {}),
    ("B2_std", "b2_0001", "train", {"codes": ["D1a", "D2a", "D3a"]}),
    ("B3_std", "b3_0001", "train", {}),
    ("B3_early", "b3_0002", "train", {}),
    ("B3_rescind", "b3_0002", "train", {}),
    ("ORD", "ord_0001", "train", {"pause_prob": 1.0}),
    ("DECOY", "dec_0002", "train", {}),
    ("B1_std", "b1_0002", "test_voice", {}),
    ("B2_D4", "b2_0002", "test_distract", {"codes": ["D1c"]}),
    ("COND_keyword", "b2_0002", "test_cond", {"codes": ["D1a"]}),
]


def load_packs(path):
    packs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return {p["id"]: p for p in packs}


def family_of(typ):
    return specs.BUILDERS[typ][1]


class PackSampler:
    """Hands out packs per family without replacement; reuse is counted, never hidden."""

    def __init__(self, packs, which, seed):
        self.r = random.Random(f"packs:{seed}:{which}")
        self.by_fam = defaultdict(list)
        for p in packs.values():
            if p.get("split", "train") != which:
                continue
            fam = "DECOY" if p.get("decoy") else p["family"]
            self.by_fam[fam].append(p)
        self.queues = {f: [] for f in self.by_fam}
        self.uses = Counter()

    def next(self, fam):
        if not self.by_fam.get(fam):
            raise ValueError(f"no {fam} packs for this split")
        if not self.queues[fam]:
            self.queues[fam] = self.r.sample(self.by_fam[fam], len(self.by_fam[fam]))
        p = self.queues[fam].pop()
        self.uses[p["id"]] += 1
        return p


def build_one(ep_id, typ, pack, split, backend_name, seed, cache, out_dir, max_dur=None, **kw):
    """Build spec, render missing clips, assemble. Shrinks side conversations if too long."""
    max_side = None
    for attempt in range(4):
        spec = specs.build(ep_id, typ, pack, split, backend_name, seed, **kw,
                           **({"max_side": max_side} if max_side else {}))
        cache.render_missing([(t["text"], spec["voices"][t["spk"]]) for t in spec["turns"]])
        row, stats = assemble(spec, cache, out_dir)
        if max_dur is None or stats["duration"] <= max_dur or not typ.startswith("B2"):
            return spec, row, stats
        max_side = max(2, len(pack["side"]) - 2 * (attempt + 1))
    return spec, row, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", help="demo | train | test_script | test_word | test_voice | test_distract | test_cond | test_ordinary | test_all")
    ap.add_argument("--packs", default="packs/demo_packs.jsonl")
    ap.add_argument("--n", type=int, default=0, help="number of episodes (ignored for demo)")
    ap.add_argument("--tts", default="pocket", choices=["pocket", "kyutai", "espeak"])
    ap.add_argument("--model-dir", default="models/pocket-tts", help="local Pocket TTS download")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="cache/tts")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    packs = load_packs(a.packs)
    backend = get_backend(a.tts, **({"model_dir": a.model_dir} if a.tts == "pocket" else {}))
    cache = ClipCache(Path(a.cache) / a.tts, backend)

    if a.split == "demo":
        plan = [(f"demo_{i:02d}_{t}", t, packs[p], s, kw) for i, (t, p, s, kw) in enumerate(DEMO)]
    else:
        from .splits import SPLITS
        cfg = SPLITS[a.split]
        sampler = PackSampler(packs, cfg["packs"], a.seed)
        r = random.Random(f"types:{a.seed}:{a.split}")
        if a.split == "train":
            mix = specs.TRAIN_MIX
        elif a.split == "test_ordinary":
            mix = {"ORD": 70, "DECOY": 30}
        elif a.split == "test_cond":
            mix = {"COND_keyword": 35, "COND_phone": 35, "B1_minutes": 30}
        elif a.split == "test_distract":
            mix = {"B2_std": 50, "B2_D4": 20, "B3_std": 30}
        else:
            mix = {k: v for k, v in specs.TRAIN_MIX.items()}
        types = r.choices(list(mix), weights=list(mix.values()), k=a.n)
        prefix = "tr" if a.split == "train" else a.split.replace("test_", "te_")
        plan = [(f"{prefix}_{i:06d}", t, sampler.next(family_of(t)), a.split, {}) for i, t in enumerate(types)]

    manifest, ledger_rows = [], []
    for ep_id, typ, pack, split, kw in plan:
        spec, row, stats = build_one(ep_id, typ, pack, split, a.tts, a.seed, cache, out / "episodes",
                                     max_dur=MAX_TRAIN_DUR if split == "train" else None, **kw)
        (out / "episodes" / f"{ep_id}.spec.json").write_text(json.dumps(spec, indent=1))
        row = {**row, "path": f"episodes/{row['path']}"}
        manifest.append(row)
        ledger_rows.append({"id": ep_id, "type": typ, "split": split, "pack": pack["id"], **stats,
                            "clips": len(spec["turns"])})
        print(f"{ep_id:28s} {typ:14s} {stats['duration']:6.1f}s  hold {stats['hold_s']:5.1f}s")

    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger_rows))
    print(f"\nwrote {len(manifest)} episodes to {out}")


if __name__ == "__main__":
    main()
