"""Human test set: swap the recorded user clips (silence/record.py) into the held-out
test episodes they were read from and re-assemble them.

    python -m silence.human [--rec <data>/human/s01] [--data <main checkout>/data]

Writes three suites next to the others, all with the ORIGINAL episode ids, so each
episode gets the same noise / friend-RIR draws as its TTS twin (assemble() seeds on the id)
and every comparison is paired:
  test_human_tts   the TTS episodes the session was planned from, copied unchanged
  test_human       the same episodes, every user line in the human voice (block A, verbatim)
  test_human_free  hold episodes only: as test_human, but the instruction and the return cue /
                   rescind are the own-words takes (block B)
The friend and Moshi stay TTS (Pocket cache). Human word times are spread over each clip's
speech span (as for the hosted engines): only onset / offset of human turns are used
downstream, and only Moshi's words become text tokens.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import soundfile as sf

from .assemble import assemble
from .record import main_checkout
from .tts import SR, ClipCache, _api_words, _Name

FREE = "free:"


class HumanCache:
    """ClipCache.get() for spec voices "human:<speaker>"; every other voice goes to `base`.
    Text "free:<key>" fetches a block-B clip by key, any other text the verbatim clip."""

    def __init__(self, base, clips_dir, speaker):
        self.base, self.clips, self.speaker = base, Path(clips_dir), speaker
        self.voice = f"human:{speaker}"

    def key(self, text, voice):
        if voice != self.voice:
            return self.base.key(text, voice)
        if text.startswith(FREE):
            return text[len(FREE):]
        return hashlib.sha1(f"human|{self.speaker}|{text}".encode()).hexdigest()[:16]

    def has(self, text, voice):
        if voice != self.voice:
            return self.base.has(text, voice)
        return (self.clips / f"{self.key(text, voice)}.wav").exists()

    def get(self, text, voice):
        if voice != self.voice:
            return self.base.get(text, voice)
        k = self.key(text, voice)
        audio, sr = sf.read(self.clips / f"{k}.wav", dtype="float32")
        assert sr == SR, (k, sr)
        words = _api_words("[own words]" if text.startswith(FREE) else text, audio)
        return f"human_{k}", audio, words

    def render_missing(self, items):
        missing = [x for x in items if not self.has(*x)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} clips not recorded / not cached, e.g. {missing[:3]}")
        return 0


def write_suite(out, rows, ledger):
    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in rows))
    (out / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ledger))
    print(f"wrote {len(rows)} episodes to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--rec", type=Path, default=None, help="recording dir (default <data>/human/s01)")
    ap.add_argument("--pocket-cache", type=Path, default=None, help="default <main checkout>/cache/tts/pocket")
    a = ap.parse_args()
    root = main_checkout()
    data = a.data or root / "data"
    rec = a.rec or data / "human" / "s01"
    plan = json.loads((rec / "session.json").read_text())
    speaker, src = plan["speaker"], data / plan["suite"] / "episodes"
    cache = HumanCache(ClipCache(a.pocket_cache or root / "cache" / "tts" / "pocket", _Name("pocket")),
                       rec / "clips", speaker)
    free = {}       # episode -> {turn index: key}
    for it in plan["items"]:
        if it["block"] == "B":
            free.setdefault(it["episode"], {})[it["turn"]] = it["key"]

    suites = {"test_human_tts": [], "test_human": [], "test_human_free": []}
    ledgers = {k: [] for k in suites}
    for k in suites:
        d = data / k
        if d.exists():
            shutil.rmtree(d)
        (d / "episodes").mkdir(parents=True)
    for ep in plan["episodes"]:
        orig = json.loads((src / f"{ep['id']}.spec.json").read_text())
        # TTS twin: copied as built
        for ext in (".wav", ".json", ".spec.json"):
            shutil.copy2(src / f"{ep['id']}{ext}", data / "test_human_tts" / "episodes" / f"{ep['id']}{ext}")
        dur = sf.info(src / f"{ep['id']}.wav").duration
        suites["test_human_tts"].append({"path": f"episodes/{ep['id']}.wav", "duration": round(dur, 6)})
        ledgers["test_human_tts"].append({"id": ep["id"], "type": ep["type"], "split": "test_human_tts",
                                          "pack": orig["pack"], "duration": round(dur, 3)})
        variants = [("test_human", {})]
        if ep["id"] in free:
            variants.append(("test_human_free", free[ep["id"]]))
        for suite, swap in variants:
            spec = json.loads(json.dumps(orig))
            spec["split"] = suite
            spec["voices"]["user"] = cache.voice
            for i, key in swap.items():
                assert spec["turns"][i]["spk"] == "user" and spec["turns"][i].get("anchor"), (ep["id"], i)
                spec["turns"][i]["text"] = FREE + key
            out = data / suite / "episodes"
            (out / f"{spec['id']}.spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1))
            row, stats = assemble(spec, cache, out)
            suites[suite].append({**row, "path": f"episodes/{row['path']}"})
            ledgers[suite].append({"id": spec["id"], "type": spec["type"], "split": suite, "pack": spec["pack"],
                                   "speaker": speaker, "swapped_turns": sorted(swap), **stats})
    for k in suites:
        write_suite(data / k, suites[k], ledgers[k])


if __name__ == "__main__":
    main()
