"""Build episodes: specs -> TTS (cached) -> wav/json -> manifest + ledger.

  python -m silence.build train --packs packs/packs.jsonl --n 10000 --out data/train
  python -m silence.build test_word --packs packs/packs.jsonl --n 90 --out data/test_word

Episodes are written in a fixed shuffled order, so the first K episodes of the
train manifest form the nested 1 h / 10 h / 100 h subsets (see make_splits.py).

Three phases: (1) plan every episode's spec, (2) render all missing TTS clips across
--workers single-thread processes (Pocket TTS is latency-bound, so processes scale and
threads don't), (3) assemble episodes in parallel. Over-length train episodes then get
a shorter side conversation or a different pack.
"""
import argparse
import json
import os
import platform
import random
import subprocess
import time
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

from . import specs
from .assemble import assemble
from .tts import ClipCache, get_backend

# One training window at duration_sec: 60. moshi-finetune cuts every file into windows
# at 0, 60, 120 s, ...; a 60.3 s file would add a 0.3 s window that starts mid-episode
# with no instruction in context. So a train episode must not exceed this, strictly.
MAX_TRAIN_DUR = 60.0


def _moshi_opens(p):
    first = (p.get("topic") or p.get("turns") or [{}])[0]
    return first.get("spk") == "moshi"


def load_packs(path):
    """Packs by id. Drops packs written before the user-speaks-first rule (B2 / ORD
    packs whose first line is Moshi's), so no episode ever opens with Moshi."""
    packs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    kept = [p for p in packs if not _moshi_opens(p)]
    if len(kept) < len(packs):
        print(f"dropped {len(packs) - len(kept)} packs that open with Moshi")
    return {p["id"]: p for p in kept}


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
    """Build spec, render missing clips, assemble. Shrinks B2 side conversations if too
    long. Returns None if the episode still exceeds max_dur (caller picks another pack)."""
    max_side = None
    for attempt in range(4):
        spec = specs.build(ep_id, typ, pack, split, backend_name, seed, **kw,
                           **({"max_side": max_side} if max_side else {}))
        cache.render_missing([(t["text"], spec["voices"][t["spk"]]) for t in spec["turns"]])
        row, stats = assemble(spec, cache, out_dir)
        if max_dur is None or row["duration"] <= max_dur:
            return spec, row, stats
        if not typ.startswith("B2") or typ == "COND_phone":
            break
        max_side = max(2, len(pack["side"]) - 2 * (attempt + 1))
    return None


class _NameOnly:
    """Stands in for a TTS backend where only the cache key (backend name) is needed."""

    def __init__(self, name):
        self.name = name


def default_workers():
    if platform.system() == "Darwin":  # efficiency cores don't help; use performance cores
        try:
            return int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"]))
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass
    return os.cpu_count() or 1


_W = {}


def _init_render(tts, model_dir, cache_root):
    import torch
    torch.set_num_threads(1)
    backend = get_backend(tts, **({"model_dir": model_dir} if tts == "pocket" else {}))
    _W["cache"] = ClipCache(cache_root, backend)


def _render(items):
    import soundfile as sf
    c = _W["cache"]
    c.render_missing(items)
    return len(items), sum(sf.info(c.root / f"{c.key(t, v)}.wav").duration for t, v in items)


def _init_assemble(tts, cache_root):
    _W["cache"] = ClipCache(cache_root, _NameOnly(tts))


def _assemble(args):
    spec, out_dir = args
    return spec["id"], assemble(spec, _W["cache"], out_dir)


def render_parallel(items, a, cache_root, workers):
    """Render (text, voice) pairs not yet in the cache; prints throughput and ETA."""
    todo = sorted(items)
    if not todo:
        return
    chunks = [todo[i: i + 8] for i in range(0, len(todo), 8)]
    t0, n_done, audio = time.time(), 0, 0.0
    print(f"rendering {len(todo)} clips on {workers} workers ...", flush=True)
    with get_context("spawn").Pool(workers, _init_render, (a.tts, a.model_dir, str(cache_root))) as pool:
        for n, sec in pool.imap_unordered(_render, chunks):
            n_done += n
            audio += sec
            el = time.time() - t0
            if n_done == len(todo) or n_done // 8 % max(1, len(chunks) // 20) == 0:
                eta = el / n_done * (len(todo) - n_done)
                print(f"  {n_done}/{len(todo)} clips  {audio / 60:.1f} min audio  "
                      f"{audio / el:.1f}x realtime  elapsed {el / 60:.1f} min  ETA {eta / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", help="train | test_script | test_word | test_voice | test_distract | test_cond | test_ordinary | test_all")
    ap.add_argument("--packs", default="packs/packs.jsonl")
    ap.add_argument("--n", type=int, required=True, help="number of episodes")
    ap.add_argument("--tts", default="pocket", choices=["pocket", "kyutai", "espeak"])
    ap.add_argument("--model-dir", default="models/pocket-tts", help="local Pocket TTS download")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="cache/tts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=default_workers(), help="TTS / assembly processes")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    packs = load_packs(a.packs)
    cache_root = Path(a.cache) / a.tts
    name_cache = ClipCache(cache_root, _NameOnly(a.tts))

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
    plan = [(f"{prefix}_{i:06d}", t) for i, t in enumerate(types)]
    max_dur = MAX_TRAIN_DUR if a.split == "train" else None

    t_start = time.time()
    # 1. plan
    planned = []
    for ep_id, typ in plan:
        pack = sampler.next(family_of(typ))
        kw = {"pack2": sampler.next("B1")} if typ == "B1_double" else {}
        planned.append((ep_id, typ, pack, kw, specs.build(ep_id, typ, pack, a.split, a.tts, a.seed, **kw)))
    # 2. render
    need = {(t["text"], sp["voices"][t["spk"]]) for *_, sp in planned for t in sp["turns"]}
    render_parallel([x for x in need if not name_cache.has(*x)], a, cache_root, a.workers)
    # 3. assemble
    print(f"assembling {len(planned)} episodes ...", flush=True)
    with get_context("spawn").Pool(a.workers, _init_assemble, (a.tts, str(cache_root))) as pool:
        built = dict(pool.imap_unordered(_assemble, [(sp, str(out / "episodes")) for *_, sp in planned], chunksize=4))

    # 4. over-length train episodes: shorter side conversation, else the next pack
    dropped, backend = Counter(), None
    for i, (ep_id, typ, pack, kw, sp) in enumerate(planned):
        row, _ = built[ep_id]
        if max_dur is None or row["duration"] <= max_dur:
            continue
        if backend is None:
            backend = get_backend(a.tts, **({"model_dir": a.model_dir} if a.tts == "pocket" else {}))
            cache = ClipCache(cache_root, backend)
        res = build_one(ep_id, typ, pack, a.split, a.tts, a.seed, cache, out / "episodes", max_dur=max_dur, **kw)
        for _ in range(5):
            if res:
                break
            dropped[typ] += 1
            pack = sampler.next(family_of(typ))
            res = build_one(ep_id, typ, pack, a.split, a.tts, a.seed, cache, out / "episodes", max_dur=max_dur, **kw)
        else:
            raise RuntimeError(f"{ep_id} ({typ}): 5 packs in a row exceed {max_dur} s")
        sp, row, stats = res
        planned[i] = (ep_id, typ, pack, kw, sp)
        built[ep_id] = (row, stats)

    manifest, ledger_rows = [], []
    for ep_id, typ, pack, kw, sp in planned:
        row, stats = built[ep_id]
        (out / "episodes" / f"{ep_id}.spec.json").write_text(json.dumps(sp, indent=1))
        manifest.append({**row, "path": f"episodes/{row['path']}"})
        ledger_rows.append({"id": ep_id, "type": typ, "split": a.split, "pack": pack["id"],
                            **({"pack2": kw["pack2"]["id"]} if "pack2" in kw else {}), **stats,
                            "clips": len(sp["turns"])})
    hours = sum(r["duration"] for r in ledger_rows) / 3600
    print(f"built {len(planned)} episodes ({hours:.2f} h) in {(time.time() - t_start) / 60:.1f} min")

    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger_rows))
    print(f"\nwrote {len(manifest)} episodes to {out}")
    if dropped:
        print(f"redrawn for exceeding {max_dur} s: {dict(dropped)}")


if __name__ == "__main__":
    main()
