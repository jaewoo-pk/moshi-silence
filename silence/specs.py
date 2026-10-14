"""Episode specs: what is said, by whom, in which voice, and WHEN.

A spec is plain JSON. No audio here. Timing is expressed per turn as either
  "gap": seconds after the previous speech ended (any speaker), or
  "at":  [anchor, offset] where anchor is e.g. "instr_end" or "start".
The assembler resolves these against the real TTS word times.

Turn fields:
  spk        "moshi" | "user" | "friend"
  text       what is said
  gap / at   timing (see above)
  anchor     register this turn's times as <anchor>_start / <anchor>_end
  role       Moshi turns: topic | ack | release | response
  release    True on the Moshi turn that ends the hold
  event      distractor code (D1a, D3a, ...) attached to this turn
  stall      [code, seconds] -> a silent gap of that length BEFORE this turn,
             recorded as a distractor event (D2a / D2b)
  gain_db    extra gain (fillers are quieter)
"""
import random

from . import phrasebank as pb
from .splits import SPLITS
from .voices import load_pool, pool_for

# ------------------------------------------------------------------ timing samplers
# Placeholder distributions. Replace turn_gap with the one fitted on DailyTalk
# (scripts/fit_gaps.py) once HF is reachable.

def turn_gap(r):        # human -> next human turn
    return min(max(r.lognormvariate(-0.9, 0.45), 0.12), 1.2)

def moshi_gap(r):       # human -> Moshi response
    return r.uniform(0.3, 0.9)

def side_gap(r):        # user <-> friend; sometimes slight overlap
    return r.uniform(-0.25, 0.0) if r.random() < 0.15 else min(max(r.lognormvariate(-1.1, 0.5), 0.1), 1.0)

def start():
    return ["start", 0.6]


def _lower_first(s):
    w = s.split(" ", 1)[0]
    return s if w in ("I", "I'm", "I've", "I'll") else s[0].lower() + s[1:]


class Ctx:
    """Per-episode sampling context: rng + which phrasing/voice/distractor pools to use."""

    def __init__(self, ep_id, split, backend, seed):
        self.id, self.split, self.backend = ep_id, split, backend
        self.cfg = SPLITS[split]
        self.r = random.Random(f"{seed}:{ep_id}")
        pool = pool_for(backend, self.cfg["voices"])
        user, friend = self.r.sample(pool, 2)
        self.voices = {"moshi": load_pool(backend)["moshi"], "user": user, "friend": friend}
        self.params = {
            "friend_gain_db": round(self.r.uniform(-9, -4), 1),
            "friend_rt60": round(self.r.uniform(0.2, 0.6), 2),
            "noise": self.r.choice(["pink", "brown", "white_lp"]),
            "snr_db": round(self.r.uniform(15, 35), 1),
            "ack": self.r.random() < 0.7,
        }
        self.templates = {}

    def pick(self, name, d, axis="phr"):
        entries = pb.bank(d, self.cfg[axis])
        i = self.r.randrange(len(entries))
        self.templates[name] = f"{self.cfg[axis]}:{i}"
        return entries[i]

    def ack(self, fam, turns):
        if self.params["ack"]:
            turns.append({"spk": "moshi", "text": self.r.choice(pb.ACKS[fam]), "gap": self.r.uniform(0.3, 0.7), "role": "ack"})

    def spec(self, typ, pack, turns, **extra):
        return {"id": self.id, "type": typ, "split": self.split, "pack": pack["id"],
                "voices": self.voices, "params": {**self.params, **extra}, "templates": self.templates,
                "turns": turns}


# ------------------------------------------------------------------ B1: timed hold
def build_b1(c: Ctx, pack, variant="std", cond=False):
    r = c.r
    if cond:
        durs = pb.B1_DURATIONS["test_cond"]
    else:
        durs = pb.B1_DURATIONS[c.cfg["phr"]]
    choices = [n for n in durs if variant != "early" or n >= 10]
    N = r.choice(choices)
    instr = c.pick("instr", pb.B1_INSTR).format(N=durs[N])
    T = [
        {"spk": "moshi", "text": pack["question"], "at": start(), "role": "topic"},
        {"spk": "user", "text": instr, "gap": turn_gap(r), "anchor": "instr"},
    ]
    c.ack("B1", T)
    if variant == "std":
        if r.random() < 0.5 and N >= 5:  # thinking fillers inside the hold (D2c)
            ts = sorted(r.uniform(2.0, N - 1.5) for _ in range(r.randint(1, 2)))
            for t in ts:
                T.append({"spk": "user", "text": r.choice(pack["mutter"]), "at": ["instr_end", t],
                          "gain_db": -6, "event": "D2c"})
        T.append({"spk": "moshi", "text": pack["reask"], "at": ["instr_end", N + r.uniform(-0.25, 0.25)],
                  "role": "release", "release": True})
        T.append({"spk": "user", "text": pack["answer"], "gap": turn_gap(r)})
        T.append({"spk": "moshi", "text": pack["reaction"], "gap": moshi_gap(r), "role": "response"})
        release = {"kind": "timer", "N": N}
    elif variant == "early":
        pre = c.pick("early", pb.B1_EARLY_PREFIX)
        T.append({"spk": "user", "text": f"{pre} {pack['answer']}", "at": ["instr_end", r.uniform(0.3, 0.7) * N],
                  "anchor": "cue"})
        T.append({"spk": "moshi", "text": pack["reaction"], "gap": moshi_gap(r), "role": "release", "release": True})
        release = {"kind": "cue"}
    elif variant == "rescind":
        T.append({"spk": "user", "text": c.pick("rescind", pb.B1_RESCIND),
                  "at": ["instr_end", r.uniform(1.5, max(2.0, 0.5 * N))], "anchor": "cue"})
        T.append({"spk": "moshi", "text": pack["reask"], "gap": moshi_gap(r), "role": "release", "release": True})
        T.append({"spk": "user", "text": pack["answer"], "gap": turn_gap(r)})
        T.append({"spk": "moshi", "text": pack["reaction"], "gap": moshi_gap(r), "role": "response"})
        release = {"kind": "cue"}
    else:
        raise ValueError(variant)
    typ = "B1_minutes" if cond else f"B1_{variant}"
    return c.spec(typ, pack, T, N=N, N_text=durs[N], release=release)


# ------------------------------------------------------------------ B2: side conversation
D_SPEAKER = {"D1a": "friend", "D1b": "user", "D1c": "friend", "D3a": "user", "D4": "friend"}


def _side_with_distractors(c: Ctx, pack, codes, max_lines=None):
    """Side conversation lines + distractors inserted at random positions (never first)."""
    r = c.r
    side = [dict(t) for t in pack["side"]][: max_lines or len(pack["side"])]
    T = []
    for t in side:
        T.append({"spk": t["spk"], "text": t["text"], "gap": side_gap(r)})
    for code in codes:
        pos = r.randint(1, len(T))  # insert before T[pos] (or append)
        if code in ("D2a",):
            if pos < len(T):
                T[pos]["stall"] = ["D2a", round(r.uniform(2.0, 8.0), 2)]
            else:
                T[-1]["stall"] = ["D2a", round(r.uniform(2.0, 8.0), 2)]
            continue
        lex = pb.bank(pb.DISTRACT[code], c.cfg["distract"])
        line = {"spk": D_SPEAKER[code], "text": r.choice(lex), "gap": side_gap(r), "event": code}
        if code == "D4":  # friend steps away, then says the return cue themselves
            T.insert(pos, {"spk": "friend", "text": "Hold on, let me grab something.", "gap": side_gap(r)})
            line["stall"] = ["D4_gap", round(r.uniform(2.5, 5.0), 2)]
            T.insert(pos + 1, line)
        else:
            T.insert(pos, line)
    return T


def build_b2(c: Ctx, pack, variant="std", n_distract=None, max_side=None, codes=None):
    r = c.r
    T = []
    for i, t in enumerate(pack["topic"]):
        T.append({"spk": t["spk"], "text": t["text"], "role": "topic" if t["spk"] == "moshi" else None,
                  **({"at": start()} if i == 0 else {"gap": moshi_gap(r) if t["spk"] == "moshi" else turn_gap(r)})})
    friend = r.choice(pb.FRIEND_NAMES[c.cfg["phr"]])
    if variant == "phone":
        instr = r.choice(pb.PHONE_INSTR)
    elif variant == "keyword":
        kw = r.choice(pb.KEYWORDS)
        instr = r.choice(pb.KW_INSTR).format(kw=kw)
    else:
        if r.random() < 0.6:
            T.append({"spk": "friend", "text": c.pick("arrival", pb.FRIEND_ARRIVAL), "gap": turn_gap(r)})
        instr = c.pick("instr", pb.B2_INSTR).format(friend=friend)
    T.append({"spk": "user", "text": instr, "gap": turn_gap(r), "anchor": "instr"})
    c.ack("KW" if variant == "keyword" else "B2", T)

    pool = ["D1a", "D1b", "D1c", "D2a", "D3a"]
    if n_distract is None:
        n_distract = r.randint(2, 4) if c.cfg.get("dense") else r.randint(1, 3)
    codes = list(codes) if codes is not None else r.sample(pool, min(n_distract, len(pool)))
    if variant == "D4":
        codes.append("D4")
    if variant == "phone":
        # Only the user is audible; the other side of the call is silence.
        side = [{"spk": "user", "text": t["text"], "gap": r.uniform(1.5, 4.0)} for t in pack["side"] if t["spk"] == "user"]
        codes = []
    else:
        side = _side_with_distractors(c, pack, codes, max_side)
    side[0]["gap"] = turn_gap(r) + 0.3
    T += side

    if variant == "phone":
        cue = r.choice(pb.PHONE_RELEASE)
    elif variant == "keyword":
        cue = r.choice(pb.KW_RELEASE).format(kw=kw, kw_cap=kw.capitalize())
    else:
        cue = c.pick("cue", pb.B2_RETURN_CUE)
    T.append({"spk": "user", "text": cue, "gap": turn_gap(r), "anchor": "cue"})
    T.append({"spk": "moshi", "text": pack["resume"], "gap": moshi_gap(r), "role": "release", "release": True})
    T.append({"spk": "user", "text": pack["followup"], "gap": turn_gap(r)})
    typ = {"std": "B2_std", "D4": "B2_D4", "phone": "COND_phone", "keyword": "COND_keyword"}[variant]
    return c.spec(typ, pack, T, friend=friend, distractors=codes, release={"kind": "cue"})


# ------------------------------------------------------------------ B3: hear me out
def build_b3(c: Ctx, pack, variant="std", dense=None):
    r = c.r
    opts = pack["options"]
    k = len(opts)
    T = [
        {"spk": "moshi", "text": pack["opener"], "at": start(), "role": "topic"},
        {"spk": "user", "text": pack["context"], "gap": turn_gap(r)},
        {"spk": "user", "text": c.pick("instr", pb.B3_INSTR).format(k=pb.NUM_WORD[k]), "gap": r.uniform(0.2, 0.5),
         "anchor": "instr"},
    ]
    c.ack("B3", T)
    n_present = {"std": k, "early": 2, "rescind": 1}[variant]
    dense = c.cfg.get("dense") if dense is None else dense
    # Distractors between options (only meaningful if more options are coming).
    d2b_after = r.randrange(n_present - 1) if n_present > 1 and (dense or r.random() < 0.6) else None
    d3b_after = r.randrange(n_present - 1) if n_present > 1 and (dense or r.random() < 0.4) else None
    codes = []
    for i in range(n_present):
        sents = opts[i]
        between = r.uniform(1.0, 2.5)
        first = {"spk": "user", "text": f"{pb.B3_ORDINALS[k][i]} {_lower_first(sents[0])}",
                 "gap": (turn_gap(r) + 0.3) if i == 0 else between}
        if i > 0 and d2b_after == i - 1:
            first["stall"] = ["D2b", round(r.uniform(3.0, 5.0), 2)]
            codes.append("D2b")
        T.append(first)
        for s in sents[1:]:
            T.append({"spk": "user", "text": s, "gap": r.uniform(0.2, 0.6)})
        if d3b_after == i:
            T.append({"spk": "user", "text": r.choice(pb.bank(pb.DISTRACT["D3b"], c.cfg["distract"])),
                      "gap": r.uniform(0.3, 0.8), "event": "D3b"})
            codes.append("D3b")
    if variant == "std":
        cue = c.pick("cue", pb.B3_COMPLETION).format(k=pb.NUM_WORD[k])
        comment = pack["comment"]
    elif variant == "early":
        cue = c.pick("cue", pb.B3_EARLY)
        comment = pack["comment_first_two"]
    else:
        cue = c.pick("cue", pb.B3_RESCIND)
        comment = pack["comment_first"]
    T.append({"spk": "user", "text": cue, "gap": turn_gap(r), "anchor": "cue"})
    T.append({"spk": "moshi", "text": comment, "gap": moshi_gap(r), "role": "release", "release": True})
    T.append({"spk": "user", "text": pack["followup"], "gap": turn_gap(r)})
    return c.spec(f"B3_{variant}", pack, T, k=k, n_presented=n_present, distractors=codes, release={"kind": "cue"})


# ------------------------------------------------------------------ Ordinary (+ decoys)
def _split_clause(text, r):
    """Split a user turn at a clause boundary for a mid-turn pause, or return None."""
    for sep in [", but ", ", and ", ", ", " and ", " but "]:
        if sep in text:
            a, b = text.split(sep, 1)
            if len(a.split()) >= 3 and len(b.split()) >= 3:
                keep = sep.strip(" ,")
                return a + ("," if sep.startswith(",") else ""), (keep + " " + b) if keep else b
    return None


def build_ord(c: Ctx, pack, pause_prob=1 / 3):
    r = c.r
    T = []
    n_pause = 0
    for i, t in enumerate(pack["turns"]):
        if i == 0:
            T.append({"spk": t["spk"], "text": t["text"], "at": start(), "role": "topic" if t["spk"] == "moshi" else None})
            continue
        if t["spk"] == "moshi":
            T.append({"spk": "moshi", "text": t["text"], "gap": moshi_gap(r), "role": "response"})
            continue
        parts = _split_clause(t["text"], r) if r.random() < pause_prob else None
        if parts:
            T.append({"spk": "user", "text": parts[0], "gap": turn_gap(r)})
            T.append({"spk": "user", "text": parts[1], "gap": r.uniform(0.8, 2.0), "midpause": True})
            n_pause += 1
        else:
            T.append({"spk": "user", "text": t["text"], "gap": turn_gap(r)})
    typ = "DECOY" if pack.get("decoy") else "ORD"
    return c.spec(typ, pack, T, midpauses=n_pause, decoy=pack.get("decoy"), release={"kind": "none"})


# ------------------------------------------------------------------ dispatch
BUILDERS = {
    "B1_std": (build_b1, "B1", {"variant": "std"}),
    "B1_early": (build_b1, "B1", {"variant": "early"}),
    "B1_rescind": (build_b1, "B1", {"variant": "rescind"}),
    "B2_std": (build_b2, "B2", {"variant": "std"}),
    "B3_std": (build_b3, "B3", {"variant": "std"}),
    "B3_early": (build_b3, "B3", {"variant": "early"}),
    "B3_rescind": (build_b3, "B3", {"variant": "rescind"}),
    "ORD": (build_ord, "ORD", {}),
    "DECOY": (build_ord, "DECOY", {}),
    # test-only
    "B2_D4": (build_b2, "B2", {"variant": "D4"}),
    "COND_keyword": (build_b2, "B2", {"variant": "keyword"}),
    "COND_phone": (build_b2, "B2", {"variant": "phone"}),
    "B1_minutes": (build_b1, "B1", {"variant": "std", "cond": True}),
}

# Training mix (by episode count), agreed 15 / 25 / 20 / 40.
TRAIN_MIX = {
    "B1_std": 11, "B1_early": 2, "B1_rescind": 2,
    "B2_std": 25,
    "B3_std": 16, "B3_early": 2, "B3_rescind": 2,
    "ORD": 30, "DECOY": 10,
}


def build(ep_id, typ, pack, split, backend, seed, **kw):
    fn, _, fixed = BUILDERS[typ]
    c = Ctx(ep_id, split, backend, seed)
    return fn(c, pack, **fixed, **kw)
