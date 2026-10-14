"""Nested training subsets + the data ledger.

  python -m silence.make_splits data/train --hours 1 10 100 --abs

Writes data/train/train_{H}h.jsonl (first K episodes of the fixed build order, so
1h ⊂ 10h ⊂ 100h) and data/train/ledger.json with, per subset:
  unique episodes / packs / TTS clips, clip reuse, total hours, speech hours,
  Moshi speech hours, hold-silence hours, per-type counts.

Training exposure is a property of a run, not of a dataset:
  python -m silence.make_splits exposure --episodes 1000 --mean-dur 35 --steps 250 --batch 16 --gpus 1
Each episode fits in one training window, so every sample the trainer draws is
one episode presentation (the window is zero-padded to duration_sec).
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def ledger(rows, metas):
    clips = Counter(c for m in metas for c in m["clips"])
    packs = {m["pack"] for m in metas}
    h = lambda k: round(sum(r[k] for r in rows) / 3600, 3)
    by_type = Counter(r["type"] for r in rows)
    return {
        "episodes": len(rows),
        "unique_packs": len(packs),
        "unique_tts_clips": len(clips),
        "clip_uses": sum(clips.values()),
        "hours_total": h("duration"),
        "hours_speech_any_channel": h("speech_s"),
        "hours_moshi_speech": h("moshi_speech_s"),
        "hours_hold_silence": h("hold_s"),
        "mean_episode_s": round(sum(r["duration"] for r in rows) / max(1, len(rows)), 2),
        "by_type": dict(sorted(by_type.items())),
    }


def exposure(n_episodes, mean_dur, steps, batch, gpus=1):
    pres = steps * batch * gpus
    return {"episode_presentations": pres, "epochs": round(pres / n_episodes, 2),
            "audio_hours_seen": round(pres * mean_dur / 3600, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--hours", type=float, nargs="+", default=[1, 10, 100])
    ap.add_argument("--abs", action="store_true", help="write absolute wav paths (do this on the training box)")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--mean-dur", type=float)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gpus", type=int, default=1)
    a = ap.parse_args()

    if a.data_dir == "exposure":
        print(json.dumps(exposure(a.episodes, a.mean_dur, a.steps, a.batch, a.gpus), indent=1))
        return

    d = Path(a.data_dir)
    manifest = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines()]
    rows = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines()]
    metas = [json.loads((d / m["path"]).with_suffix(".json").read_text())["meta"] for m in manifest]

    out, cum, k = {"full": ledger(rows, metas)}, 0.0, 0
    for H in sorted(a.hours):
        while k < len(rows) and cum < H * 3600:
            cum += rows[k]["duration"]
            k += 1
        if cum < H * 3600:
            print(f"only {cum / 3600:.2f} h available, skipping {H} h")
            break
        name = f"train_{H:g}h.jsonl"
        lines = []
        for m in manifest[:k]:
            p = (d / m["path"]).resolve() if a.abs else Path(m["path"])
            lines.append(json.dumps({"path": str(p), "duration": m["duration"]}) + "\n")
        (d / name).write_text("".join(lines))
        out[name] = ledger(rows[:k], metas[:k])
        print(f"{name}: {k} episodes, {cum / 3600:.2f} h")
    (d / "ledger.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
