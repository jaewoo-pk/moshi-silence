"""Overlap audit between a training build and a held-out build (dev or a test suite).

  python -m silence.check_overlap data/train data/dev
  python -m silence.check_overlap data/train data/test_script data/test_word ...

Reports shared pack ids, shared episode ids, near-duplicate scripts (gen_packs' content-word
Jaccard on key texts), spoken lines shared verbatim (split into phrase-bank lines, shared by
design unless the suite holds wording out, vs lines from the held-out pack's own script,
which would be a leak) and user/friend voices.
"""
import argparse
import json
from pathlib import Path

from silence.gen_packs import content_words, key_text


def load(d):
    rows = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines()]
    specs = {r["id"]: json.loads((d / "episodes" / f"{r['id']}.spec.json").read_text()) for r in rows}
    return rows, specs


def pack_ids(rows):
    return {r["pack"] for r in rows} | {r["pack2"] for r in rows if "pack2" in r}


def fam(p):
    return "DECOY" if p.get("decoy") else p["family"]


def audit(packs, train_dir, held_dir):
    tr_rows, tr_specs = load(train_dir)
    ho_rows, ho_specs = load(held_dir)
    tr_p, ho_p = pack_ids(tr_rows), pack_ids(ho_rows)
    print(f"== {train_dir} vs {held_dir}")
    print(f"1. pack ids: train {len(tr_p)}, held-out {len(ho_p)}, shared {len(tr_p & ho_p)}")
    print(f"2. episode ids shared: {len({r['id'] for r in tr_rows} & {r['id'] for r in ho_rows})}")

    # 3. near-duplicate scripts (gen_packs' own measure: Jaccard of content words of the key text)
    tr_by_fam = {}
    for pid in tr_p:
        p = packs[pid]
        tr_by_fam.setdefault(fam(p), []).append((pid, content_words(key_text(fam(p), p))))
    sims = []
    for pid in ho_p:
        p = packs[pid]
        w = content_words(key_text(fam(p), p))
        best = max(((len(w & v) / max(1, len(w | v)), tid) for tid, v in tr_by_fam.get(fam(p), [])),
                   default=(0.0, None))
        sims.append((best[0], pid, best[1]))
    sims.sort(reverse=True)
    s = sorted(x[0] for x in sims)
    print(f"3. held-out -> closest train script (Jaccard): max {s[-1]:.2f}, median {s[len(s) // 2]:.2f}, "
          f">=0.6 (gen_packs dup threshold): {sum(x >= 0.6 for x in s)}, >=0.5: {sum(x >= 0.5 for x in s)}")
    for sim, h, t in sims[:2]:
        if t:
            print(f"   {sim:.2f}  held-out {h}: {key_text(fam(packs[h]), packs[h])[:80]!r}")
            print(f"         train    {t}: {key_text(fam(packs[t]), packs[t])[:80]!r}")

    # 4. spoken lines shared verbatim: from the held-out pack (a leak) vs the phrase bank
    tr_lines = {t["text"] for sp in tr_specs.values() for t in sp["turns"]}
    from_pack, from_bank = {}, {}
    for sp in ho_specs.values():
        blob = json.dumps(packs[sp["pack"]]) + json.dumps(packs.get(sp["params"].get("pack2"), {}))
        for t in sp["turns"]:
            if t["text"] in tr_lines:
                d = from_pack if json.dumps(t["text"])[1:-1] in blob else from_bank
                d[t["text"]] = d.get(t["text"], 0) + 1
    n = sum(len(sp["turns"]) for sp in ho_specs.values())
    print(f"4. held-out spoken lines: {n}; also spoken in train: {sum(from_pack.values()) + sum(from_bank.values())}")
    print(f"   phrase bank: {sum(from_bank.values())} ({len(from_bank)} distinct)")
    print(f"   held-out pack's own script: {sum(from_pack.values())} ({len(from_pack)} distinct)"
          + (f": {sorted(from_pack, key=from_pack.get, reverse=True)[:5]}" if from_pack else ""))

    tr_v = {v for sp in tr_specs.values() for k, v in sp["voices"].items() if k != "moshi"}
    ho_v = {v for sp in ho_specs.values() for k, v in sp["voices"].items() if k != "moshi"}
    print(f"5. user/friend voices: held-out uses {len(ho_v)}, {len(ho_v - tr_v)} not in train"
          + (f" {sorted(ho_v - tr_v)}" if ho_v - tr_v else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("held_out", nargs="+")
    ap.add_argument("--packs", default="packs/packs.jsonl", help="must contain every pack id used (packs.jsonl does)")
    a = ap.parse_args()
    packs = {json.loads(l)["id"]: json.loads(l) for l in open(a.packs)}
    for h in a.held_out:
        audit(packs, Path(a.train), Path(h))


if __name__ == "__main__":
    main()
