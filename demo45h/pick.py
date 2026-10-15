"""Propose conversations for the 45h-model demo, from saved verdicts only (no audio needed).

    python demo45h/pick.py            # prints the top candidates per card, writes demo45h/candidates.json

Models: base vs the 45h model (R24 = checkpoint_000527). Sources, all held out from training:
  test suites  results/test/raw/<suite>/base, results/test/raw_R24/<suite>/checkpoint_000527   (1 sample)
  human set    results/human/raw/<suite>/{base,checkpoint_000527}                            (3 samples: _s0.._s2)
Success cards need: the 45h model right and base wrong. Failure cards need: the 45h model wrong.
"Right" = every hold kept silent (or not scorable) and resumed on time, no distractor reaction, every
ordinary question answered, no barge-in. Ranking: base's error is obvious (more words in the hold),
then shorter episodes first (easier listening).
"""
import json
from collections import defaultdict
from pathlib import Path

R = Path("results")
OUT = Path("demo45h/candidates.json")
K45 = "checkpoint_000527"


def load(path):
    rows = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        rows[r["id"]] = r
    return rows


def dists(r):
    return [d for h in r["holds"] for d in h.get("distractors", [])]


def ok(r):
    for h in r["holds"]:
        if h["violated"] or h["outcome"] != "on_time":
            return False
    if any(d["reacted"] for d in dists(r)):
        return False
    if any(not x["responded"] for x in r["responses"]):
        return False
    return not any(b["barged"] for b in r["barge_ins"])


def base_words(r):
    return sum(h.get("words_in_hold") or 0 for h in r["holds"]) + sum(not x["responded"] for x in r["responses"]) * 5


def has(r, code):
    return any(d["code"] == code for d in dists(r))


def reacted(r, code):
    return any(d["code"] == code and d["reacted"] for d in dists(r))


# card -> (sources, type filter, extra filter on the 45h row, want_45h_ok)
CARDS = {
    "1_side_conversation": (["test_script", "test_voice", "test_all"], {"B2_std"}, lambda m, b: has(m, "D1a"), True),
    "2_cue_lookalike": (["test_script", "test_voice", "test_all", "test_distract"], {"B2_std"}, lambda m, b: has(m, "D3a") and reacted(b, "D3a"), True),
    "3_hear_me_out_pauses": (["test_distract", "test_script", "test_all"], {"B3_std"}, lambda m, b: has(m, "D2b"), True),
    "4_hear_me_out_cancelled": (["test_script", "test_word", "test_voice", "test_all"], {"B3_rescind"}, lambda m, b: True, True),
    "5_timed_hold": (["test_script", "test_voice", "test_word", "test_all"], {"B1_std"}, lambda m, b: (m["holds"][0].get("N") or 0) >= 8, True),
    "6_ordinary_with_pause": (["test_ordinary", "test_script"], {"ORD"}, lambda m, b: bool(m["barge_ins"]), True),
    "7a_my_voice_scripted": (["test_human"], None, lambda m, b: bool(m["holds"]), True),
    "7b_my_voice_own_words": (["test_human_free"], None, lambda m, b: bool(m["holds"]), True),
    "8a_FAIL_friend_says_im_back": (["test_distract"], {"B2_D4"}, lambda m, b: reacted(m, "D4") or not ok(m), False),
    "8b_FAIL_minute_hold": (["test_cond"], {"B1_minutes"}, lambda m, b: True, False),
}


def sources(suite):
    if suite.startswith("test_human"):
        d = R / "human/raw" / suite
        return load(d / "base/scores.jsonl"), load(d / K45 / "scores.jsonl")
    return load(R / "test/raw" / suite / "base/scores.jsonl"), load(R / "test/raw_R24" / suite / K45 / "scores.jsonl")


def main():
    cache = {}
    out = {}
    used = set()  # a conversation goes on one card only
    for card, (suites, types, extra, want_ok) in CARDS.items():
        cands = []
        for s in suites:
            if s not in cache:
                cache[s] = sources(s)
            base, m45 = cache[s]
            human = s.startswith("test_human")
            ids = sorted({i.rsplit("_s", 1)[0] for i in m45}) if human else sorted(m45)
            for ep in ids:
                rows45 = [m45[f"{ep}_s{k}"] for k in range(3) if f"{ep}_s{k}" in m45] if human else [m45[ep]]
                b = base.get(f"{ep}_s0" if human else ep)
                m = rows45[0]
                if b is None or (types and m["type"] not in types) or not extra(m, b):
                    continue
                n_ok = sum(ok(r) for r in rows45)
                if want_ok:
                    if not ok(m) or n_ok < (2 if human else 1) or ok(b):
                        continue
                elif ok(m):
                    continue
                cands.append({"suite": s, "id": ep, "type": m["type"], "duration": round(m["duration"], 1),
                              "45h_ok_samples": f"{n_ok}/{len(rows45)}", "base_words_in_hold": base_words(b),
                              "45h_holds": [(h["outcome"], h["violated"], h.get("latency")) for h in m["holds"]]})
        key = (lambda c: (-c["base_words_in_hold"], c["duration"])) if want_ok else (lambda c: c["duration"])
        cands.sort(key=key)
        cands = [c for c in cands if (c["suite"], c["id"]) not in used]
        out[card] = cands[:3]
        used |= {(c["suite"], c["id"]) for c in cands[:3]}
        print(f"\n{card}: {len(cands)} candidates")
        for c in cands[:3]:
            print(f"  {c['suite']}/{c['id']} {c['type']} {c['duration']}s  base words-in-hold {c['base_words_in_hold']}"
                  f"  45h {c['45h_ok_samples']} {c['45h_holds']}")
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
