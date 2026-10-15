"""Drop episodes that fail check.py from a built set (50 h build, 2026-10-15).

  python -m silence.filter_checked data50/train50

Reads <dir>/checks.json. Failing episodes' files move to <dir>/rejected/; manifest.jsonl and
episodes.jsonl are rewritten without them (originals kept as *.all.jsonl), order unchanged, so
make_splits' first-K prefixes stay nested. Writes rejected.json (id, type, reasons) and prints
the failure reasons by type. Also writes train_all.jsonl (absolute paths).
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def main():
    d = Path(sys.argv[1])
    checks = json.loads((d / "checks.json").read_text())
    bad = {c["id"]: c for c in checks if not c["ok"]}
    ids_checked = {c["id"] for c in checks}
    man = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines()]
    rows = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines()]
    assert len(man) == len(rows) and all(Path(m["path"]).stem == r["id"] for m, r in zip(man, rows))
    missing = [r["id"] for r in rows if r["id"] not in ids_checked]
    if missing:
        raise SystemExit(f"{len(missing)} manifest episodes have no check result, e.g. {missing[:3]}")
    for name in ("manifest.jsonl", "episodes.jsonl"):
        if not (d / name.replace(".jsonl", ".all.jsonl")).exists():
            shutil.copy(d / name, d / name.replace(".jsonl", ".all.jsonl"))
    (d / "rejected").mkdir(exist_ok=True)
    for i in bad:
        for ext in (".wav", ".json", ".spec.json"):
            f = d / "episodes" / f"{i}{ext}"
            if f.exists():
                f.rename(d / "rejected" / f.name)
    keep = [(m, r) for m, r in zip(man, rows) if r["id"] not in bad]
    (d / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m, _ in keep))
    (d / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for _, r in keep))
    (d / "train_all.jsonl").write_text("".join(
        json.dumps({"path": str((d / m["path"]).resolve()), "duration": m["duration"]}) + "\n" for m, _ in keep))
    (d / "rejected.json").write_text(json.dumps([{"id": i, "type": c["type"], "fails": c["fails"]} for i, c in bad.items()], indent=1))
    reasons = Counter((c["type"], f.split(" ")[0]) for c in bad.values() for f in c["fails"])
    hours = sum(r["duration"] for _, r in keep) / 3600
    print(f"kept {len(keep)}/{len(rows)} episodes ({hours:.2f} h); rejected {len(bad)}")
    for (t, why), n in reasons.most_common(30):
        print(f"  {t:14s} {why:30s} {n}")


if __name__ == "__main__":
    main()
