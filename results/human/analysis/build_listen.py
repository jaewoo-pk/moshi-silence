# Listening page for the human test set (2026-10-15): every episode x {TTS, human verbatim, human own words}
# x {R21 ep 2.2, R22 step 345}, sample 0 decoded from the saved Mimi codes (CPU), mixed with the input channel
# (L = generated Moshi + 0.3 input, R = input + 0.3 Moshi, as in results/demo), plus the scorer's verdicts for all 3
# samples and timelines of the scored windows. Output: results/human/listen/{index.html, audio/*.mp3}.
#   python results/human/analysis/build_listen.py [--no-audio]
import argparse, json, re, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

H = Path("results/human"); DATA = Path.home() / "repo/moshi-silence/data"
MIMI = Path.home() / "repo/moshi-silence/models/moshiko/tokenizer-e351c8d8-checkpoint125.safetensors"
SUITES = {"test_human_tts": "TTS", "test_human": "Your voice", "test_human_free": "Own words"}
MODELS = {"checkpoint_000153": "R21", "checkpoint_000345": "R22"}
GRACE, CUE_PRE = 2.5, 0.25
_mimi = None


def _decode(job):
    """(codes npz, key, user wav, out mp3) -> writes the mp3."""
    global _mimi
    import torch, soundfile as sf
    torch.set_num_threads(2)
    if _mimi is None:
        from moshi.models import loaders
        _mimi = loaders.get_mimi(str(MIMI), device="cpu")
        _mimi.set_num_codebooks(8)
    npz, key, user_wav, out = job
    codes = np.load(npz)[key]
    with torch.no_grad():
        m = _mimi.decode(torch.from_numpy(codes[1:].astype(np.int64))[None]).float().numpy()[0, 0]
    u, sr = sf.read(user_wav, dtype="float32")
    u = u[:, 1]
    n = min(len(m), len(u))
    st = np.stack([m[:n] + 0.3 * u[:n], u[:n] + 0.3 * m[:n]], 1)
    pk = np.abs(st).max()
    if pk > 0.95:
        st *= 0.95 / pk
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "x.wav"
        sf.write(tmp, st, sr, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:a", "libmp3lame", "-b:a", "64k", str(out)], check=True)
    return out.name


def timer_tol(n):
    return max(1.5, 0.2 * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    out = H / "listen"; (out / "audio").mkdir(parents=True, exist_ok=True)
    plan = json.loads((DATA / "human/s01/session.json").read_text())
    intents = {(it["episode"], it["anchor"]): it["intent"] for it in plan["items"] if it["block"] == "B"}
    eps, jobs = [], []
    for ep in plan["episodes"]:
        eid = ep["id"]
        E = {"id": eid, "type": ep["type"], "rows": []}
        for suite, slab in SUITES.items():
            jf = DATA / suite / "episodes" / f"{eid}.json"
            if not jf.exists():
                continue
            ev = json.loads(jf.read_text())["events"]
            spec = json.loads((DATA / suite / "episodes" / f"{eid}.spec.json").read_text())
            anchors = {i: t.get("anchor") for i, t in enumerate(spec["turns"]) if t["spk"] != "moshi"}
            human = []
            hi = [i for i, t in enumerate(spec["turns"]) if t["spk"] != "moshi"]
            for k, t in zip(hi, ev["human_turns"]):
                txt = t["text"]
                own = txt.startswith("free:")
                if own:
                    txt = intents.get((eid, anchors[k]), "")
                human.append({"spk": t["spk"], "t": t["t"], "text": txt, "own": own})
            moshi = [{"role": t["role"], "t": t["t"], "text": t["text"]} for t in ev["moshi_turns"]]
            holds = []
            for h in ev["holds"]:
                rel = h["release"]
                timer = rel["kind"] == "timer"
                tol = timer_tol(rel["N"]) if timer else CUE_PRE
                holds.append({"instr_end": h["instr_end"], "ref": rel["ref"], "kind": rel["kind"], "N": rel.get("N"),
                              "region": [h["instr_end"] + GRACE, rel["ref"] - tol]})
            dis = [{"code": d["code"], "t": d["t"], "spk": d["spk"], "text": d.get("text")} for d in ev["distractors"]]
            for ck, mlab in MODELS.items():
                d = H / "raw" / suite / ck
                rows = {json.loads(l)["id"]: json.loads(l) for l in open(d / "scores.jsonl")}
                samples = [rows[f"{eid}_s{j}"] for j in range(3)]
                rec = json.loads((d / f"{eid}_s0.json").read_text())
                free = rec["free"]
                infree = lambda t: any(x <= t <= y for x, y in free)
                words = [[w, round(t, 2)] for w, t in rec["words"] if infree(t) and w not in ("<s>", "</s>")]
                fn = f"{suite}_{mlab}_{eid}.mp3"
                if not a.no_audio and not (out / "audio" / fn).exists():
                    jobs.append((str(d / "codes.npz"), f"{eid}_s0", str(DATA / suite / "episodes" / f"{eid}.wav"), out / "audio" / fn))
                verdict = []
                for s in samples:
                    verdict.append({
                        "holds": [{"violated": h["violated"], "outcome": h["outcome"], "latency": h["latency"],
                                   "speech": h["speech_in_region_s"], "words": h["words_in_region"],
                                   "through": h["talking_through"],
                                   "reacted": [x["code"] for x in h.get("distractors", []) if x["reacted"]]}
                                  for h in s["holds"]],
                        "responses": [x["responded"] for x in s["responses"]]})
                E["rows"].append({"suite": suite, "cond": slab, "model": mlab, "dur": rec["duration"], "free": free,
                                  "segs": rec["segments"], "words": words, "human": human, "moshi": moshi,
                                  "holds": holds, "dis": dis, "v": verdict, "audio": f"audio/{fn}"})
        eps.append(E)
    if jobs:
        print(f"decoding {len(jobs)} clips with {a.workers} workers ...", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            for i, name in enumerate(ex.map(_decode, jobs), 1):
                if i % 16 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} {name}", flush=True)
    summary = json.loads((H / "summary.json").read_text())
    tpl = (H / "analysis" / "listen_template.html").read_text()
    html = tpl.replace("/*__DATA__*/null", json.dumps({"episodes": eps, "summary": summary}, separators=(",", ":")))
    (out / "index.html").write_text(html)
    print(f"wrote {out / 'index.html'} ({len(html) / 1e6:.2f} MB), {len(list((out / 'audio').glob('*.mp3')))} clips")


if __name__ == "__main__":
    main()
