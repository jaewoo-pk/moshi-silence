"""Overlap audit between a training build and a held-out build (dev or test).

  python -m silence.check_overlap data/train data/dev

Reports shared pack ids, shared episode ids, near-duplicate scripts (gen_packs' content-word
Jaccard on key texts), spoken lines shared verbatim (split into phrase-bank lines, shared by
design, vs lines from the held-out pack's own script, which would be a leak) and voices.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from silence.gen_packs import content_words, key_text

train_dir, dev_dir = Path(sys.argv[1]), Path(sys.argv[2])
packs = {json.loads(l)["id"]: json.loads(l) for l in open("packs/packs.jsonl")}


def load(d):
    rows = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines()]
    specs = {r["id"]: json.loads((d / "episodes" / f"{r['id']}.spec.json").read_text()) for r in rows}
    return rows, specs


def pack_ids(rows):
    return {r["pack"] for r in rows} | {r["pack2"] for r in rows if "pack2" in r}


tr_rows, tr_specs = load(train_dir)
dv_rows, dv_specs = load(dev_dir)
tr_p, dv_p = pack_ids(tr_rows), pack_ids(dv_rows)
print(f"1. pack ids: train {len(tr_p)}, dev {len(dv_p)}, shared {len(tr_p & dv_p)}")
print(f"2. episode ids shared: {len({r['id'] for r in tr_rows} & {r['id'] for r in dv_rows})}")


def fam(p):
    return "DECOY" if p.get("decoy") else p["family"]


# 3. near-duplicate scripts (gen_packs' own measure: Jaccard of content words of the key text)
tr_by_fam = {}
for pid in tr_p:
    p = packs[pid]
    tr_by_fam.setdefault(fam(p), []).append((pid, content_words(key_text(fam(p), p))))
sims = []
for pid in dv_p:
    p = packs[pid]
    w = content_words(key_text(fam(p), p))
    best = max(((len(w & v) / max(1, len(w | v)), tid) for tid, v in tr_by_fam.get(fam(p), [])), default=(0, None))
    sims.append((best[0], pid, best[1]))
sims.sort(reverse=True)
s = [x[0] for x in sims]
print(f"3. dev->closest train script (Jaccard): max {s[0]:.2f}, median {sorted(s)[len(s) // 2]:.2f}, "
      f">=0.6 (gen_packs dup threshold): {sum(x >= 0.6 for x in s)}, >=0.4: {sum(x >= 0.4 for x in s)}")
for sim, d, t in sims[:3]:
    print(f"   {sim:.2f}  dev {d}: {key_text(fam(packs[d]), packs[d])[:90]!r}")
    print(f"         train {t}: {key_text(fam(packs[t]), packs[t])[:90]!r}")

# 4. spoken lines: shared texts, split into pack-derived (would be a leak) vs phrase bank (by design)
tr_lines = Counter(t["text"] for sp in tr_specs.values() for t in sp["turns"])
pack_derived, template = Counter(), Counter()
for sp in dv_specs.values():
    blob = json.dumps(packs[sp["pack"]]) + json.dumps(packs.get(sp["params"].get("pack2"), {}))
    for t in sp["turns"]:
        if t["text"] in tr_lines:
            (pack_derived if json.dumps(t["text"])[1:-1] in blob else template)[t["text"]] += 1
n_dev_lines = sum(len(sp["turns"]) for sp in dv_specs.values())
print(f"4. dev spoken lines: {n_dev_lines}; also in train: {sum(pack_derived.values()) + sum(template.values())}")
print(f"   from the phrase bank (shared by design): {sum(template.values())} "
      f"({len(template)} distinct, e.g. {list(template)[:3]})")
print(f"   from the dev pack's own script: {sum(pack_derived.values())} ({len(pack_derived)} distinct)")
for txt, n in pack_derived.most_common(10):
    print(f"     {n}x {txt!r}")

# 5. voices
tr_v = {v for sp in tr_specs.values() for k, v in sp["voices"].items() if k != "moshi"}
dv_v = {v for sp in dv_specs.values() for k, v in sp["voices"].items() if k != "moshi"}
print(f"5. user/friend voices: dev uses {len(dv_v)}, all in the train pool: {dv_v <= tr_v}")
