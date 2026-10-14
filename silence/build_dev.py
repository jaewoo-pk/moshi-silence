"""Dev (validation) set: train distribution, unseen scripts. Used only to choose epochs,
LRs and checkpoints; never trained on, never reported as a result.

  python -m silence.build_dev --exclude data/train --out data/dev

- packs: train-split packs NOT used by any episode under --exclude (all of them, not
  just the 10 h prefix), so every dev script is new to every training subset
- phrasings, voices, conditions: the train ones (SPLITS["train"])
- exactly --per-family episodes per family (B1 / B2 / B3 / ORD+DECOY), each family split
  by its TRAIN_MIX sub-mix, variants spread evenly through the family's list
- families interleaved round-robin, so any prefix of 4k episodes has k per family:
  dev_80.jsonl (20 per family) is the first 80 of dev_200.jsonl
- every episode <= 60 s (one training window, so the trainer's eval loss sees it whole)

Future train builds must skip these packs too (see meta.pack in data/dev).
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path

from silence import specs
from silence.build import (MAX_TRAIN_DUR, ClipCache, PackSampler, _NameOnly, _assemble,
                           _init_assemble, build_one, default_workers, family_of, get_backend,
                           load_packs, render_parallel)
from silence.make_splits import ledger

FAMILIES = ["B1", "B2", "B3", "ORD"]


def fam_of_type(typ):
    return "ORD" if typ in ("ORD", "DECOY") else family_of(typ)


def type_plan(per_family, seed):
    """Exact per-family counts from TRAIN_MIX, variants evenly spread, families round-robin."""
    by_fam = defaultdict(dict)
    for typ, w in specs.TRAIN_MIX.items():
        by_fam[fam_of_type(typ)][typ] = w
    r = random.Random(f"dev-types:{seed}")
    lists = {}
    for f in FAMILIES:
        mix = by_fam[f]
        tot = sum(mix.values())
        # largest-remainder rounding so counts sum to per_family exactly
        raw = {t: per_family * w / tot for t, w in mix.items()}
        cnt = {t: int(v) for t, v in raw.items()}
        for t in sorted(raw, key=lambda t: raw[t] - cnt[t], reverse=True)[: per_family - sum(cnt.values())]:
            cnt[t] += 1
        # evenly spaced: item i of variant t sits at (i + jitter) / cnt[t]
        keyed = [((i + r.random()) / c, t) for t, c in cnt.items() for i in range(c)]
        lists[f] = [t for _, t in sorted(keyed)]
    return [lists[f][i] for i in range(per_family) for f in FAMILIES]


_META_KEYS = {"id", "family", "split", "style", "decoy", "spk", "tag", "est_max_s"}


def pack_strings(x):
    """Every spoken string in a pack (skips metadata fields)."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            if k not in _META_KEYS:
                yield from pack_strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from pack_strings(v)


def _norm(s):
    return " ".join("".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).split())


def screen(candidates, train_packs, thresh):
    """Drop candidate packs that overlap the training packs: any spoken line shared verbatim
    (after case / punctuation normalization), or key-text content-word Jaccard >= thresh
    against any training pack of the same family (gen_packs' own near-dup measure)."""
    from silence.gen_packs import content_words, key_text
    fam = lambda p: "DECOY" if p.get("decoy") else p["family"]
    tr_lines, tr_words = set(), defaultdict(list)
    for p in train_packs:
        tr_lines |= {_norm(s) for s in pack_strings(p)}
        tr_words[fam(p)].append(content_words(key_text(fam(p), p)))
    keep, dropped = {}, Counter()
    for k, p in candidates.items():
        if p.get("split", "train") != "train":
            keep[k] = p
            continue
        if {_norm(s) for s in pack_strings(p)} & tr_lines:
            dropped[(fam(p), "shared line")] += 1
            continue
        w = content_words(key_text(fam(p), p))
        if any(len(w & v) / max(1, len(w | v)) >= thresh for v in tr_words[fam(p)]):
            dropped[(fam(p), f"similar >= {thresh}")] += 1
            continue
        keep[k] = p
    return keep, dropped


def used_packs(data_dir):
    used = set()
    for l in (Path(data_dir) / "episodes.jsonl").read_text().splitlines():
        row = json.loads(l)
        used.add(row["pack"])
        if "pack2" in row:
            used.add(row["pack2"])
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", default="packs/packs.jsonl")
    ap.add_argument("--exclude", nargs="+", default=["data/train"], help="built dirs whose packs to skip")
    ap.add_argument("--per-family", type=int, default=50)
    ap.add_argument("--subsets", type=int, nargs="+", default=[80, 200], help="nested prefix sizes")
    ap.add_argument("--out", default="data/dev")
    ap.add_argument("--tts", default="pocket", choices=["pocket", "kyutai", "espeak"])
    ap.add_argument("--model-dir", default="models/pocket-tts")
    ap.add_argument("--cache", default="cache/tts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--abs", action="store_true", help="absolute wav paths in the subset jsonls")
    ap.add_argument("--sim", type=float, default=0.5,
                    help="drop candidate packs this similar to any training pack (content-word Jaccard)")
    ap.add_argument("--screen-only", action="store_true", help="print the overlap screen and exit")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    packs = load_packs(a.packs)
    excluded = set().union(*(used_packs(d) for d in a.exclude))
    pool = {k: p for k, p in packs.items() if k not in excluded}
    print(f"{len(excluded)} packs excluded (used in {', '.join(a.exclude)}); "
          f"{sum(p.get('split', 'train') == 'train' for p in pool.values())} train packs left")
    pool, dropped_overlap = screen(pool, [packs[k] for k in excluded if k in packs], a.sim)
    left = Counter("DECOY" if p.get("decoy") else p["family"] for p in pool.values()
                   if p.get("split", "train") == "train")
    print(f"overlap screen dropped {sum(dropped_overlap.values())}: {dict(sorted(dropped_overlap.items()))}")
    print(f"clean train packs left per family: {dict(sorted(left.items()))}")
    if a.screen_only:
        return
    sampler = PackSampler(pool, "train", f"dev{a.seed}")
    cache_root = Path(a.cache) / a.tts

    types = type_plan(a.per_family, a.seed)
    plan = [(f"dv_{i:06d}", t) for i, t in enumerate(types)]

    # 1. plan (split="train": train phrasings, voices, conditions)
    planned = []
    for ep_id, typ in plan:
        pack = sampler.next(family_of(typ))
        kw = {"pack2": sampler.next("B1")} if typ == "B1_double" else {}
        planned.append((ep_id, typ, pack, kw, specs.build(ep_id, typ, pack, "train", a.tts, a.seed, **kw)))
    # 2. render
    name_cache = ClipCache(cache_root, _NameOnly(a.tts))
    need = {(t["text"], sp["voices"][t["spk"]]) for *_, sp in planned for t in sp["turns"]}
    render_parallel([x for x in need if not name_cache.has(*x)], a, cache_root, a.workers)
    # 3. assemble
    print(f"assembling {len(planned)} episodes ...", flush=True)
    with get_context("spawn").Pool(a.workers, _init_assemble, (a.tts, str(cache_root))) as p:
        built = dict(p.imap_unordered(_assemble, [(sp, str(out / "episodes")) for *_, sp in planned], chunksize=4))
    # 4. over-length: shorter side conversation, else the next unused pack
    dropped, cache = Counter(), None
    for i, (ep_id, typ, pack, kw, sp) in enumerate(planned):
        if built[ep_id][0]["duration"] <= MAX_TRAIN_DUR:
            continue
        if cache is None:
            backend = get_backend(a.tts, **({"model_dir": a.model_dir} if a.tts == "pocket" else {}))
            cache = ClipCache(cache_root, backend)
        res = build_one(ep_id, typ, pack, "train", a.tts, a.seed, cache, out / "episodes", max_dur=MAX_TRAIN_DUR, **kw)
        for _ in range(5):
            if res:
                break
            dropped[typ] += 1
            pack = sampler.next(family_of(typ))
            res = build_one(ep_id, typ, pack, "train", a.tts, a.seed, cache, out / "episodes", max_dur=MAX_TRAIN_DUR, **kw)
        else:
            raise RuntimeError(f"{ep_id} ({typ}): 5 packs in a row exceed {MAX_TRAIN_DUR} s")
        sp, row, stats = res
        planned[i] = (ep_id, typ, pack, kw, sp)
        built[ep_id] = (row, stats)

    manifest, rows = [], []
    for ep_id, typ, pack, kw, sp in planned:
        row, stats = built[ep_id]
        (out / "episodes" / f"{ep_id}.spec.json").write_text(json.dumps(sp, indent=1))
        manifest.append({**row, "path": f"episodes/{row['path']}"})
        rows.append({"id": ep_id, "type": typ, "split": "dev", "pack": pack["id"],
                     **({"pack2": kw["pack2"]["id"]} if "pack2" in kw else {}), **stats,
                     "clips": len(sp["turns"])})
    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    overlap = {r["pack"] for r in rows} | {r["pack2"] for r in rows if "pack2" in r}
    assert not overlap & excluded, "dev reused a training pack"

    metas = [json.loads((out / m["path"]).with_suffix(".json").read_text())["meta"] for m in manifest]
    led = {}
    for n in sorted(a.subsets):
        k = min(n, len(manifest))
        lines = [json.dumps({"path": str((out / m["path"]).resolve() if a.abs else Path(m["path"])),
                             "duration": m["duration"]}) + "\n" for m in manifest[:k]]
        (out / f"dev_{k}.jsonl").write_text("".join(lines))
        led[f"dev_{k}.jsonl"] = {**ledger(rows[:k], metas[:k]),
                                 "by_family": dict(Counter(fam_of_type(r["type"]) for r in rows[:k]))}
        print(f"dev_{k}.jsonl: {k} episodes, {led[f'dev_{k}.jsonl']['hours_total']:.2f} h, "
              f"{led[f'dev_{k}.jsonl']['by_family']}")
    led["excluded_from"] = a.exclude
    (out / "ledger.json").write_text(json.dumps(led, indent=1))
    if dropped:
        print(f"redrawn for exceeding {MAX_TRAIN_DUR} s: {dict(dropped)}")


if __name__ == "__main__":
    main()
