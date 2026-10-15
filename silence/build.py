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
from .tts import ClipCache, MultiCache, get_backend

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
        clips = [(t["text"], spec["voices"][t["spk"]]) for t in spec["turns"]]
        cache.render_missing(clips)
        if not all(cache.has(*x) for x in clips):  # a hosted engine failed on a line: next pack
            return None
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
    def path(k):
        f = c.root / f"{k}.wav"
        return f if f.exists() else c.root / f"{k}.flac"
    return len(items), sum(sf.info(path(c.key(t, v))).duration for t, v in items)


def _init_assemble(tts, cache_root):
    _W["cache"] = MultiCache(cache_root) if tts == "multi" else ClipCache(cache_root, _NameOnly(tts))


def _assemble(args):
    spec, out_dir = args
    return spec["id"], assemble(spec, _W["cache"], out_dir)


def render_parallel(items, a, cache_root, workers):
    """Render (text, voice) pairs not yet in the cache; prints throughput and ETA."""
    todo = sorted(items, reverse=getattr(a, "reverse", False))  # --reverse: help another shard from its far end
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
    ap.add_argument("--tts", default="pocket", choices=["pocket", "kyutai", "espeak", "multi"],
                    help="multi = user/friend voices from Pocket / ElevenLabs / Fish (voices.MULTI), cache under --cache/<engine>")
    ap.add_argument("--mix", default="v1", choices=["v1", "v2", "v3"],
                    help="train mix: v1 = specs.TRAIN_MIX, v2 = TRAIN_MIX_V2 (+ more mutters), v3 = v2's plan + LONG_SILENCES")
    ap.add_argument("--exclude", nargs="*", default=[], help="built sets whose packs (pack / pack2) must not be used, e.g. data/dev")
    ap.add_argument("--render-only", default=None, help="comma-separated engines: render this shard's missing clips, then exit")
    ap.add_argument("--reverse", action="store_true", help="render clips in reverse key order (take over another shard's tail)")
    ap.add_argument("--max-drop", type=float, default=0.03, help="max share of episodes dropped for failed hosted clips")
    ap.add_argument("--shard", default="0/1", help="i/n: this process renders clips whose key hash %% n == i")
    ap.add_argument("--model-dir", default="models/pocket-tts", help="local Pocket TTS download")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="cache/tts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=default_workers(), help="TTS / assembly processes")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    packs = load_packs(a.packs)
    excl = set()
    for d in a.exclude:
        for l in (Path(d) / "episodes.jsonl").read_text().splitlines():
            e = json.loads(l)
            excl.update(x for x in (e.get("pack"), e.get("pack2")) if x)
    if excl:
        n0 = len(packs)
        packs = {k: v for k, v in packs.items() if k not in excl}
        print(f"excluded {n0 - len(packs)} packs used in {a.exclude}")
    if a.mix in ("v2", "v3"):
        specs.MUTTER_P = 0.7
    if a.mix == "v3":
        specs.LONG_SILENCES = True
    multi = a.tts == "multi"
    cache_root = Path(a.cache) if multi else Path(a.cache) / a.tts
    name_cache = MultiCache(cache_root) if multi else ClipCache(cache_root, _NameOnly(a.tts))

    from .splits import SPLITS
    cfg = SPLITS[a.split]
    sampler = PackSampler(packs, cfg["packs"], a.seed)
    r = random.Random(f"types:{a.seed}:{a.split}")
    if a.split == "train":
        mix = specs.TRAIN_MIX_V2 if a.mix in ("v2", "v3") else specs.TRAIN_MIX
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
    missing = [x for x in need if not name_cache.has(*x)]
    if multi:
        from .voices import engine_of
        si, sn = map(int, a.shard.split("/"))
        # The final (assembly) run renders nothing: every clip must come from the render-only
        # shards; failed hosted clips drop their episodes (bounded by --max-drop).
        engines = a.render_only.split(",") if a.render_only else []
        by = defaultdict(list)
        for t, v in missing:
            if int(name_cache.key(t, v), 16) % sn == si:
                by[engine_of(v)].append((t, v))
        import hashlib
        plan_hash = hashlib.sha1(json.dumps(sorted(need)).encode()).hexdigest()[:12]  # the clip set (floats differ by 1 ulp across libm)
        print(f"plan hash {plan_hash} ({len(planned)} episodes; every shard must print the same)", flush=True)
        print("missing clips by engine:", {e: len(x) for e, x in by.items()},
              "| est. speech h:", {e: round(sum(specs.speak_dur(t) for t, _ in x) / 3600, 2) for e, x in by.items()}, flush=True)
        # Hosted engines: account-wide request limits, so render them from ONE process only
        # (--render-only el,fish --shard 0/1); Pocket shards use --render-only pocket.
        import threading
        ths, errors, failed = [], [], []
        for eng in ("el", "fish"):
            if eng in engines and by[eng]:
                def run(eng=eng):
                    try:
                        c, items, t0 = MultiCache(cache_root), sorted(by[eng]), time.time()
                        for i in range(0, len(items), 200):
                            c.render_missing(items[i: i + 200])
                            print(f"  [{eng}] {min(i + 200, len(items))}/{len(items)} clips  {len(c.failed)} failed  "
                                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
                        failed.extend(c.failed)
                    except BaseException as e:  # noqa: BLE001 - surfaced below, the process exits non-zero
                        errors.append((eng, repr(e)))
                        raise
                th = threading.Thread(target=run)
                th.start()
                ths.append(th)
        if "pocket" in engines and by["pocket"]:
            a.tts = "pocket"
            try:
                render_parallel(by["pocket"], a, cache_root / "pocket", a.workers)
            finally:
                a.tts = "multi"
        for th in ths:
            th.join()
        if failed:
            with (out / "tts_failed.jsonl").open("a") as f:
                f.writelines(json.dumps({"text": t, "voice": v, "error": e}) + "\n" for t, v, e in failed)
            print(f"{len(failed)} hosted-engine clips failed (see {out / 'tts_failed.jsonl'}); their episodes are dropped")
        if errors:
            raise SystemExit(f"engine thread(s) crashed: {errors}")
        if a.render_only:
            print("render-only shard done")
            return
        missing = {x for x in need if not name_cache.has(*x)}
        n_pocket = sum(1 for x in missing if engine_of(x[1]) == "pocket")
        if n_pocket:
            raise SystemExit(f"{n_pocket} Pocket clips still missing (other shards not done?)")
        if missing:  # failed hosted-engine lines: drop those episodes rather than fake the voice
            keep = [x for x in planned if not any((t["text"], x[4]["voices"][t["spk"]]) in missing for t in x[4]["turns"])]
            n_drop = len(planned) - len(keep)
            print(f"dropping {n_drop} episodes with a failed hosted-engine clip ({len(missing)} clips)")
            if n_drop > a.max_drop * len(planned):
                raise SystemExit(f"{n_drop} dropped > --max-drop {a.max_drop:.0%}: fix the engine failures first")
            planned = keep
    else:
        render_parallel(missing, a, cache_root, a.workers)
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
            if multi:
                backend, cache = True, MultiCache(cache_root, model_dir=a.model_dir)
            else:
                backend = get_backend(a.tts, **({"model_dir": a.model_dir} if a.tts == "pocket" else {}))
                cache = ClipCache(cache_root, backend)
        def attempt(pack):
            try:
                return build_one(ep_id, typ, pack, a.split, a.tts, a.seed, cache, out / "episodes", max_dur=max_dur, **kw)
            except (KeyError, RuntimeError) as e:  # multi: a new clip needs a hosted engine (no key here) or failed
                if not multi:
                    raise
                dropped[f"redraw error {type(e).__name__}"] += 1
                return None
        res = attempt(pack)
        # multi: no new packs here (their clips would need slow single-process Pocket or a hosted
        # engine on the assembly box): only build_one's B2 shrink, which reuses rendered lines.
        for _ in range(0 if multi else 5):
            if res:
                break
            dropped[typ] += 1
            pack = sampler.next(family_of(typ))
            res = attempt(pack)
        if not res:  # still too long (or failed): drop the episode, and its over-length files
            dropped[f"{typ} (episode dropped)"] += 1
            planned[i] = None
            for ext in (".wav", ".json", ".spec.json"):
                (out / "episodes" / f"{ep_id}{ext}").unlink(missing_ok=True)
            continue
        sp, row, stats = res
        planned[i] = (ep_id, typ, pack, kw, sp)
        built[ep_id] = (row, stats)

    planned = [x for x in planned if x is not None]
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
