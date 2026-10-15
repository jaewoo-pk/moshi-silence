# Re-score phase-1 dev_80 outputs on two axes: silent when due (holds kept) vs speaks when due
# (cue releases + timed releases + ordinary questions), splitting cue vs timed releases. Lenient
# "spoke": any new onset from ref - tol until the free window ends (~ref + 4 s). No regeneration.
# Run from the repo root: python results/phase1/analysis/rescore_speak_silence.py [results/phaseN]

import json, glob, sys, numpy as np
from pathlib import Path
BASE = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/phase1')
from pathlib import Path
R = BASE / "eval_raw"
ep = {}
for p in glob.glob("/Users/parkjay/repo/moshi-silence/data/dev/episodes/dv_*.json"):
    if p.endswith(".spec.json"): continue
    ep[Path(p).stem] = json.load(open(p))["events"]
def epoch(run, ck):
    return 0.0 if run == "R0" else json.load(open(BASE / "configs" / f"{run}.epochs.json"))["checkpoint_epoch"][ck]
out = []
for sc in sorted(R.glob("*/dev_80/*/scores.jsonl")):
    run, ck = sc.parts[-4], sc.parts[-2]
    rows = [json.loads(l) for l in open(sc)]
    silent, cue_any, cue_ok, tim_any, tim_ok, tim_err, ordr = [], [], [], [], [], [], []
    for r in rows:
        gen = json.load(open(sc.parent / f"{r['id']}.json"))
        segs, free = gen["segments"], gen["free"]
        for h, eh in zip(r["holds"], ep[r["id"]]["holds"]):
            if h["violated"] is not None: silent.append(not h["violated"])
            rel = eh["release"]; ref = rel["ref"]; timer = rel["kind"] == "timer"
            tol = max(1.5, 0.2 * rel["N"]) if timer else 0.25
            wins = [b for a, b in free if a <= ref <= b + 0.01]
            win_end = wins[0] if wins else ref + 4
            onsets = [a for a, b in segs if a >= ref - tol and a < win_end]
            spoke = bool(onsets) and not h["talking_through"]
            (tim_any if timer else cue_any).append(spoke)
            (tim_ok if timer else cue_ok).append(h["outcome"] == "on_time")
            if timer and onsets: tim_err.append((onsets[0] - ref) / rel["N"])
        ordr += [x["responded"] for x in r["responses"]]
    m = lambda v: float(np.mean(v)) if v else float("nan")
    out.append(dict(run=run, ck=ck, ep=epoch(run, ck), silent=m(silent), speak=m(cue_any + tim_any + ordr),
                    cue_any=m(cue_any), cue_ok=m(cue_ok), tim_any=m(tim_any), tim_ok=m(tim_ok),
                    tim_relerr=float(np.median(tim_err)) if tim_err else float("nan"),
                    n_sil=len(silent), n_tim=len(tim_any), n_cue=len(cue_any), n_ord=len(ordr), ordr=m(ordr)))
json.dump(out, open(BASE / "analysis" / "speak_silence.json", "w"))
o0 = out[0]; print(f"n: silence-scored holds {o0['n_sil']}, cue releases {o0['n_cue']}, timer releases {o0['n_tim']}, ordinary {o0['n_ord']}")
order = sorted({o["run"] for o in out}, key=lambda r: (r != "R0", float(r.split("lr")[1]) if "lr" in r else 0))
print(f"{'run':16} | silent-when-due  speaks-when-due | cue: spoke / on time | timer: spoke / on time  med err | ordinary   (plateau ep 10-20)")
for run in order:
    rs = [o for o in out if o["run"] == run and (run == "R0" or o["ep"] >= 9.5)]
    a = lambda k: np.nanmean([o[k] for o in rs])
    print(f"{run:16} | {a('silent'):6.0%} {a('speak'):15.0%}  | {a('cue_any'):5.0%} / {a('cue_ok'):4.0%}      | {a('tim_any'):5.0%} / {a('tim_ok'):4.0%}   {a('tim_relerr'):+.2f}N | {a('ordr'):4.0%}")
