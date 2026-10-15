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
  release    on the Moshi turn that ends a hold: True (the hold described by
             params["release"]) or that hold's own release dict (B1_double)
  hold_anchor  which instruction anchor this release closes (default "instr")
  event      distractor code (D1a, D3a, ...) attached to this turn
  stall      [code, seconds] -> a silent gap of that length BEFORE this turn,
             recorded as a distractor event (D2a / D2b)
  gain_db    extra gain (fillers are quieter)
"""
import random

from . import phrasebank as pb
from .assemble import TAIL_S
from .splits import SPLITS
from .voices import load_pool, pool_for

# Train episodes must fit one 60 s window (build.MAX_TRAIN_DUR). Length budgets aim a few
# seconds lower because est_duration() is an estimate.
BUDGET_DUR = 57.0

# ------------------------------------------------------------------ timing samplers
# Placeholder distributions. Replace turn_gap with the one fitted on DailyTalk
# (scripts/fit_gaps.py) once HF is reachable.

# Ordinary between-turn silences are scaled up 1.5x (listening: the originals felt rushed).
# Distractor pauses (D2a / D2b stalls, mid-turn pauses) and hold timing are not scaled.
GAP_SCALE = 1.5

def gap(r, lo, hi):     # an ordinary pause, uniform in [lo, hi] before scaling
    return GAP_SCALE * r.uniform(lo, hi)

def turn_gap(r):        # human -> next human turn
    return GAP_SCALE * min(max(r.lognormvariate(-0.9, 0.45), 0.12), 1.2)

def moshi_gap(r):       # human -> Moshi response
    return gap(r, 0.3, 0.9)

def side_gap(r):        # user <-> friend; sometimes slight overlap
    return r.uniform(-0.25, 0.0) if r.random() < 0.15 else GAP_SCALE * min(max(r.lognormvariate(-1.1, 0.5), 0.1), 1.0)

def start():
    return ["start", 0.6]

# Every episode opens with the user; Moshi never speaks first. B2 / ORD packs carry their
# own opening (gen_packs enforces a user first line; build.load_packs drops older packs).

def est_dur(text):      # generous upper bound on a clip's spoken length (Pocket TTS ~2.5-3 words/s)
    return 0.5 + 0.45 * len(text.split())


def speak_dur(text):    # spoken length: 0.275 s/word fitted on 156 Pocket TTS clips (p90 +0.6 s); the
    return 0.25 + 0.28 * len(text.split())  # constant makes whole rendered episodes come out right


def est_duration(spec):
    """Episode length from the spec alone, with speak_dur() standing in for TTS clips.
    Resolves gap / at / stall the way assemble.place() does; used for length budgets
    before anything is rendered."""
    anchors, t_last = {"start": 0.0}, 0.0
    for t in spec["turns"]:
        if "at" in t:
            a, off = t["at"]
            on = anchors[a] + off
        else:
            on = t_last + (t["stall"][1] if "stall" in t else t["gap"])
        end = on + speak_dur(t["text"])
        if t.get("anchor"):
            anchors[f"{t['anchor']}_start"], anchors[f"{t['anchor']}_end"] = on, end
        t_last = max(t_last, end)
    return t_last + TAIL_S


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
            turns.append({"spk": "moshi", "text": self.r.choice(pb.ACKS[fam]), "gap": gap(self.r, 0.3, 0.7), "role": "ack"})

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
    choices = [n for n in durs if variant == "std" or n >= 10]  # early / rescind need room to cut the hold short
    if c.split == "train":  # only hold lengths that keep the episode inside one 60 s window
        # The ack falls inside the hold (which runs from instr_end), so only these lines and
        # the 4 gaps between them add to N.
        lines = [pack["opener"], pack["question"], "Give me ten seconds to think, then ask me again.",
                 pack["reask"], pack["answer"], pack["reaction"]]
        base = start()[1] + sum(speak_dur(x) for x in lines) + 4 * GAP_SCALE * 0.7 + TAIL_S
        frac = 1.0 if variant == "std" else 0.7  # early / rescind cut in by 0.7 N at the latest
        choices = [n for n in choices if base + frac * n <= BUDGET_DUR] or [min(choices)]
    N = r.choice(choices)
    instr = c.pick("instr", pb.B1_INSTR).format(N=durs[N])
    T = [{"spk": "user", "text": pack["opener"], "at": start()},  # "Quiz me on something." -> Moshi asks
         {"spk": "moshi", "text": pack["question"], "gap": moshi_gap(r), "role": "topic"}]
    T.append({"spk": "user", "text": instr, "gap": turn_gap(r), "anchor": "instr"})
    c.ack("B1", T)
    if variant == "std":
        if r.random() < 0.5 and N >= 5:  # thinking fillers inside the hold (D2c)
            # Placed one after another; each must end >= 1 s before the re-ask, judged by an
            # upper-bound duration estimate (spec time has no audio yet; check.py verifies).
            t0 = 2.0
            for text in r.sample(pack["mutter"], min(r.randint(1, 2), len(pack["mutter"]))):
                est = est_dur(text)
                if t0 + est > N - 1.0:
                    break
                t = r.uniform(t0, N - 1.0 - est)
                T.append({"spk": "user", "text": text, "at": ["instr_end", t], "gain_db": -6, "event": "D2c"})
                t0 = t + est + 1.0
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


def build_b1_double(c: Ctx, pack, pack2):
    """Two timed holds in one episode: the first runs to its re-ask, answer and reaction,
    then the user asks for another question (pack2) and puts Moshi on hold again. Teaches
    that a hold ends at its release and that a new instruction starts a new one.
    Both N are short and drawn so the episode fits the length budget."""
    r = c.r
    durs = pb.B1_DURATIONS[c.cfg["phr"]]
    T = [{"spk": "user", "text": pack["opener"], "at": start()},
         {"spk": "moshi", "text": pack["question"], "gap": moshi_gap(r), "role": "topic"}]
    holds = []  # (anchor, re-ask turn) per hold; N filled in below
    for p, anchor in [(pack, "instr"), (pack2, "instr2")]:
        if anchor == "instr2":
            T.append({"spk": "user", "text": c.pick("another", pb.B1_ANOTHER), "gap": turn_gap(r)})
            T.append({"spk": "moshi", "text": p["question"], "gap": moshi_gap(r), "role": "topic"})
        T.append({"spk": "user", "text": c.pick(anchor, pb.B1_INSTR), "gap": turn_gap(r), "anchor": anchor})
        c.ack("B1", T)
        reask = {"spk": "moshi", "text": p["reask"], "role": "release", "hold_anchor": anchor}
        T.append(reask)
        T.append({"spk": "user", "text": p["answer"], "gap": turn_gap(r)})
        T.append({"spk": "moshi", "text": p["reaction"], "gap": moshi_gap(r), "role": "response"})
        holds.append((anchor, reask))
    # Episode length without the holds, then the (N1, N2) pairs that still fit.
    for anchor, reask in holds:
        reask["at"] = [f"{anchor}_end", 0.0]
    base = est_duration({"turns": [{**t, "text": t["text"].format(N="ten seconds")} for t in T]})
    short = [n for n in durs if n <= 12]
    pairs = [(a, b) for a in short for b in short if base + a + b <= BUDGET_DUR] or [(min(short), min(short))]
    Ns = r.choice(pairs)
    for (anchor, reask), N in zip(holds, Ns):
        reask["at"] = [f"{anchor}_end", N + r.uniform(-0.25, 0.25)]
        reask["release"] = {"kind": "timer", "N": N}
    for t in T:
        if t.get("anchor") in ("instr", "instr2"):
            t["text"] = t["text"].format(N=durs[Ns[0] if t["anchor"] == "instr" else Ns[1]])
    return c.spec("B1_double", pack, T, N=Ns[0], N2=Ns[1], pack2=pack2["id"],
                  release={"kind": "timer", "N": Ns[0]})


# ------------------------------------------------------------------ B2: side conversation
def _is_q(t):
    return "?" in t["text"]


def _side_with_distractors(c: Ctx, pack, n_stall, d4=False, max_lines=None):
    """The pack's side conversation, which already carries its distractors as line tags
    (D1a/D1b/D1c questions to the other person, D3a return-cue lookalikes; see gen_packs).
    Adds D2a stalls (pauses before the side conversation is finished) and, for D4, the
    friend stepping away and announcing they're back. Nothing is inserted between a
    question and its answer."""
    r = c.r
    side = [dict(t) for t in pack["side"]]
    if max_lines:
        side = side[:max_lines]
        while len(side) > 2 and _is_q(side[-1]):  # never end on an unanswered question
            side.pop()
    T = [{"spk": t["spk"], "text": t["text"], "gap": side_gap(r), **({"event": t["tag"]} if t.get("tag") else {})}
         for t in side]

    def slots(end):  # insert-before positions that don't split an exchange
        return [p for p in range(1, len(T) + (1 if end else 0)) if not _is_q(T[p - 1]) and not T[p - 1].get("glue")]

    for _ in range(n_stall):
        ok = [p for p in slots(end=False) if "stall" not in T[p]]
        if ok:
            T[r.choice(ok)]["stall"] = ["D2a", round(r.uniform(2.0, 8.0), 2)]
    if d4:  # test only: the cue comes from the wrong person
        away, back = pack["friend_away"]
        pos = r.choice(slots(end=True) or [len(T)])
        T[pos:pos] = [{"spk": "friend", "text": away, "gap": side_gap(r), "glue": True},
                      {"spk": "friend", "text": back, "gap": side_gap(r), "event": "D4",
                       "stall": ["D4_gap", round(r.uniform(2.5, 5.0), 2)]}]
    for prev, t in zip(T, T[1:]):  # side_gap may overlap the other person, never oneself
        if t["spk"] == prev["spk"]:
            t["gap"] = max(t["gap"], 0.15)
    for t in T:
        t.pop("glue", None)
    return T


def build_b2(c: Ctx, pack, variant="std", max_side=None, stalls=None):
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

    if stalls is None:
        stalls = 1 if c.cfg.get("dense") or r.random() < 0.5 else 0
    if variant == "phone":
        # Only the user is audible; the other side of the call is silence.
        side = [{"spk": "user", "text": t["text"], "gap": r.uniform(1.5, 4.0)} for t in pack["side"] if t["spk"] == "user"]
    else:
        side = _side_with_distractors(c, pack, stalls, d4=variant == "D4", max_lines=max_side)
    codes = sorted({t["event"] for t in side if t.get("event")} | {t["stall"][0] for t in side if t.get("stall", [""])[0] == "D2a"})
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
    if c.split == "train":  # a D2a pause that would overflow the 60 s window is shortened (to >= 2 s) or dropped
        over = est_duration({"turns": T}) - BUDGET_DUR
        st = next((t for t in side if t.get("stall", [""])[0] == "D2a"), None)
        if over > 0 and st:
            if st["stall"][1] - over >= 2.0:
                st["stall"][1] = round(st["stall"][1] - over, 2)
            else:
                del st["stall"]
                codes = [x for x in codes if x != "D2a"]
    typ = {"std": "B2_std", "D4": "B2_D4", "phone": "COND_phone", "keyword": "COND_keyword"}[variant]
    return c.spec(typ, pack, T, friend=friend, distractors=codes, release={"kind": "cue"})


# ------------------------------------------------------------------ B3: hear me out
def build_b3(c: Ctx, pack, variant="std", dense=None):
    r = c.r
    opts = pack["options"]
    k = len(opts)
    T = [{"spk": "user", "text": pack["context"], "at": start()}]  # the user opens with what they're deciding
    T.append({"spk": "user", "text": c.pick("instr", pb.B3_INSTR).format(k=pb.NUM_WORD[k]), "gap": gap(r, 0.2, 0.5),
              "anchor": "instr"})
    c.ack("B3", T)
    n_present = {"std": k, "early": 2, "rescind": 1}[variant]
    dense = c.cfg.get("dense") if dense is None else dense
    # Distractors between options (only meaningful if more options are coming).
    d2b_after = r.randrange(n_present - 1) if n_present > 1 and (dense or r.random() < 0.6) else None
    d3b_after = r.randrange(n_present - 1) if n_present > 1 and (dense or r.random() < 0.4) else None
    codes = []
    for i in range(n_present):
        sents = opts[i]
        between = gap(r, 1.0, 2.5)
        first = {"spk": "user", "text": f"{pb.B3_ORDINALS[k][i]} {_lower_first(sents[0])}",
                 "gap": (turn_gap(r) + 0.3) if i == 0 else between}
        if i > 0 and d2b_after == i - 1:
            first["stall"] = ["D2b", round(r.uniform(3.0, 5.0), 2)]
            codes.append("D2b")
        T.append(first)
        for s in sents[1:]:
            T.append({"spk": "user", "text": s, "gap": gap(r, 0.2, 0.6)})
        if d3b_after == i:
            T.append({"spk": "user", "text": pack["aside"], "gap": gap(r, 0.3, 0.8), "event": "D3b"})
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
    "B1_double": (build_b1_double, "B1", {}),  # needs pack2=<second B1 pack>
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

# Training mix (by episode count), agreed 15 / 25 / 20 / 40. B1_double takes 2 of B1_std's 11.
TRAIN_MIX = {
    "B1_std": 9, "B1_double": 2, "B1_early": 2, "B1_rescind": 2,
    "B2_std": 25,
    "B3_std": 16, "B3_early": 2, "B3_rescind": 2,
    "ORD": 30, "DECOY": 10,
}


def build(ep_id, typ, pack, split, backend, seed, **kw):
    fn, _, fixed = BUILDERS[typ]
    c = Ctx(ep_id, split, backend, seed)
    return fn(c, pack, **fixed, **kw)


# Longest configuration of each family's standard episode: B2 always gets a stall, B3
# always gets the pause and the aside between options. Arrival / ack / N still vary by seed.
_BUDGET = {"B1": ("B1_std", {}), "B2": ("B2_std", {"stalls": 1}), "B3": ("B3_std", {"dense": True}),
           "ORD": ("ORD", {"pause_prob": 1.0}), "DECOY": ("DECOY", {"pause_prob": 1.0})}


def longest_episode(fam, pack, n=12):
    """Estimated length of the longest standard episode this pack produces, over n sampled
    configurations. Packs above BUDGET_DUR would only fit after cutting their content
    (or not at all), so gen_packs rejects them up front."""
    typ, kw = _BUDGET[fam]
    pack = {"id": "budget", **pack}
    return max(est_duration(build(f"budget_{i}", typ, pack, "train", "pocket", 0, **kw)) for i in range(n))
