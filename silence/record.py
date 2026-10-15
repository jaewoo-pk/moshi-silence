"""Prompter for the human-recorded test set: one person reads the USER side of held-out
test episodes into the mic; the friend stays TTS (its lines only show as context).

Three blocks, in this order:
  R  10 s of room tone (noise floor of the room / mic).
  B  own words: for each hold episode, the instruction and the return cue / rescind,
     said however you'd naturally say it (the card gives the intent, never the scripted
     wording). Recorded before A so the scripted wording can't prime it.
  A  verbatim: every user line of ~24 test_all episodes (held-out wording, test voices),
     shown with the two preceding turns so the prosody fits the conversation.

A later `human` TTS backend swaps these clips in for the TTS user lines and re-assembles
the episodes, so timing, labels, check.py and scoring stay the same, paired with the TTS
version of each episode.

    python -m silence.record              # plan on first use, then record (resumable)
    python -m silence.record --plan-only  # write session.json and print the plan
    python -m silence.record --status
    python -m silence.record --list-devices

Output (default <main checkout>/data/human/<speaker>/, gitignored):
  session.json            the plan (items in recording order)
  clips/<key>.wav         kept take, 24 kHz mono PCM_16, trimmed to speech + 0.2 / 0.3 s
  clips/<key>.json        text or intent, item id, speech span, levels
  raw/<key>.<n>.wav       every take, untrimmed (with 0.5 s pre-roll)
  takes.jsonl             one row per take
Block A keys are sha1("human|<speaker>|<text>")[:16], i.e. ClipCache.key for a backend
named "human"; a line that recurs across episodes is recorded once. Block B keys are
free_<episode>_<anchor>.
"""
import argparse
import hashlib
import json
import os
import queue
import random
import select
import shutil
import subprocess
import sys
import termios
import textwrap
import threading
import time
import tty
from collections import deque
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from silence.tts import speech_span

SR = 24_000
PREROLL_S = 0.5          # audio kept from before the start key, so a first syllable isn't cut
PAD_PRE, PAD_POST = 0.2, 0.3
ROOMTONE_S = 10.0
WPS, LINE_OVERHEAD_S, FREE_ITEM_S = 2.5, 5.0, 12.0   # for the time-left estimate

# Episodes per type from test_all (90 episodes; it has no B1_early).
WANT = [("B1_std", 4), ("B1_double", 1), ("B1_rescind", 1), ("B2_std", 6),
        ("B3_std", 2), ("B3_early", 1), ("B3_rescind", 1), ("ORD", 6), ("DECOY", 2)]
SCENES = {
    "B1": "Moshi quizzes you; you ask for time to think.",
    "B2": "A friend drops by (their lines are pre-recorded). You tell Moshi to stay quiet, "
          "talk with the friend, then come back to Moshi.",
    "B3": "You lay out options to Moshi and ask it to hold off until you're done.",
    "ORD": "An ordinary chat with Moshi.",
}
NUM = {3: "three", 5: "five", 8: "eight", 10: "ten", 12: "twelve", 15: "fifteen",
       20: "twenty", 25: "twenty-five", 30: "thirty"}


def family(typ):
    f = typ.split("_")[0]
    return "ORD" if f == "DECOY" else f


def main_checkout():
    """The main checkout (data/ lives there, not in worktrees)."""
    try:
        d = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                           cwd=Path(__file__).parent, capture_output=True, text=True, check=True)
        return Path(d.stdout.strip()).parent
    except (OSError, subprocess.CalledProcessError):
        return Path.cwd()


# ------------------------------------------------------------------------- plan

def intent(typ, anchor, n_secs):
    """Block B card: what to say, never how."""
    if anchor in ("instr", "instr2"):
        if typ.startswith("B1"):
            n = NUM.get(n_secs, str(n_secs))
            return (f"Ask Moshi to give you {n} seconds to think, then ask you again.",
                    f"mention {n} seconds")
        if typ.startswith("B2"):
            return ("Someone just came over and you want to talk to them. Tell Moshi to stay "
                    "quiet until you say you're back.", "")
        return ("Tell Moshi you're going to go through three options, and it should wait "
                "until you've covered all of them.", "")
    return {
        "B1_rescind": ("Take it back: Moshi doesn't need to wait, it can ask you now.", ""),
        "B2_std": ("You're done talking to them. Tell Moshi you're back.", ""),
        "B3_std": ("You've finished all three. Tell Moshi it can comment now.", ""),
        "B3_early": ("Turns out there are only two. Tell Moshi it can go ahead.", ""),
        "B3_rescind": ("Stop listing early, and ask Moshi what it thinks of the first option.", ""),
    }[typ]


def addressee(spec, i):
    """Who user line i is said to: moshi, friend, or self (asides / mutters)."""
    t = spec["turns"][i]
    if t.get("event") in ("D2c",):
        return "self"
    if t.get("event") == "D3b":
        return "aside"
    if spec["type"].startswith("B2"):
        anchors = [j for j, u in enumerate(spec["turns"]) if u.get("anchor")]
        a = {spec["turns"][j]["anchor"]: j for j in anchors}
        if a.get("instr", 1e9) < i < a.get("cue", -1):
            return "friend"
    return "moshi"


def context(spec, i, k=2):
    lab = {"user": "YOU", "moshi": "MOSHI", "friend": "FRIEND"}
    return [[lab[t["spk"]], t["text"]] for t in spec["turns"][max(0, i - k): i]]


def hold_n(spec, anchor):
    for t in spec["turns"]:
        if t.get("hold_anchor", "instr") == anchor and isinstance(t.get("release"), dict):
            return t["release"].get("N")
    return spec["params"].get("N")      # rescind / early: the release is a cue, N sits in params


def pick_episodes(specs, seed=0):
    rng = random.Random(seed)
    by = {}
    for s in specs:
        by.setdefault(s["type"], []).append(s)
    out = []
    for typ, k in WANT:
        pool = sorted(by.get(typ, []), key=lambda s: s["id"])
        rng.shuffle(pool)
        if typ == "B1_std":      # different N first
            seen, first, rest = set(), [], []
            for s in pool:
                n = s["params"]["release"].get("N")
                (rest if n in seen else first).append(s)
                seen.add(n)
            pool = first + rest
        elif typ in ("ORD", "DECOY"):   # mid-turn pauses first (barge-in check)
            pool.sort(key=lambda s: not any(t.get("midpause") for t in s["turns"]))
        if len(pool) < k:
            sys.exit(f"test set has only {len(pool)} {typ} episodes, need {k}")
        out += pool[:k]
    # round-robin over families so block A alternates conversation kinds
    fams = {}
    for s in out:
        fams.setdefault(family(s["type"]), []).append(s)
    order, queues = [], [fams[f] for f in ("B1", "B2", "B3", "ORD") if f in fams]
    while any(queues):
        for q in queues:
            if q:
                order.append(q.pop(0))
    return order


def build_plan(data_dir, speaker, suite="test_all", seed=0):
    specs = [json.loads(p.read_text()) for p in sorted((data_dir / suite / "episodes").glob("*.spec.json"))]
    if not specs:
        sys.exit(f"no specs under {data_dir / suite / 'episodes'}")
    eps = pick_episodes(specs, seed)
    items = [{"id": "R00", "block": "R", "kind": "roomtone", "key": "roomtone", "seconds": ROOMTONE_S}]
    for s in eps:      # block B
        for i, t in enumerate(s["turns"]):
            a = t.get("anchor")
            if t["spk"] != "user" or a not in ("instr", "instr2", "cue"):
                continue
            want, must = intent(s["type"], a, hold_n(s, a))
            items.append({"id": f"B{sum(it['block'] == 'B' for it in items) + 1:02d}", "block": "B",
                          "kind": "free", "episode": s["id"], "type": s["type"], "turn": i, "anchor": a,
                          "intent": want, "must": must, "scripted": t["text"], "context": context(s, i),
                          "key": f"free_{s['id']}_{a}"})
    seen = set()
    for e, s in enumerate(eps):     # block A
        for i, t in enumerate(s["turns"]):
            if t["spk"] != "user":
                continue
            key = hashlib.sha1(f"human|{speaker}|{t['text']}".encode()).hexdigest()[:16]
            if key in seen:
                continue
            seen.add(key)
            nxt = s["turns"][i + 1] if i + 1 < len(s["turns"]) else {}
            items.append({"id": f"A{sum(it['block'] == 'A' for it in items) + 1:03d}", "block": "A",
                          "kind": "line", "episode": s["id"], "type": s["type"], "ep_index": e,
                          "turn": i, "text": t["text"], "to": addressee(s, i),
                          "pause_after": bool(nxt.get("midpause")), "continues": bool(t.get("midpause")),
                          "quiet": t.get("gain_db", 0) < 0, "context": context(s, i), "key": key})
    return {"speaker": speaker, "suite": suite, "seed": seed, "data": str(data_dir),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "episodes": [{"id": s["id"], "type": s["type"]} for s in eps], "items": items}


def item_seconds(it):
    if it["kind"] == "roomtone":
        return it["seconds"] + 5
    if it["kind"] == "free":
        return FREE_ITEM_S
    return len(it["text"].split()) / WPS + LINE_OVERHEAD_S


def describe(plan):
    its = plan["items"]
    a = [it for it in its if it["block"] == "A"]
    b = [it for it in its if it["block"] == "B"]
    words = sum(len(it["text"].split()) for it in a)
    from collections import Counter
    c = Counter(e["type"] for e in plan["episodes"])
    print(f"{len(plan['episodes'])} episodes from {plan['suite']}: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    print(f"block B (own words): {len(b)} clips; block A (verbatim): {len(a)} lines, {words} words")
    print(f"estimated recording time: {sum(map(item_seconds, its)) / 60:.0f} min")


# ------------------------------------------------------------------------- audio

class Mic:
    """Input stream -> queue of mono float32 blocks at the device rate. `fake` replays a
    wav in real time instead (tests)."""

    def __init__(self, device=None, fake=None):
        self.q, self.overflows = queue.Queue(), 0
        if fake:
            x, self.sr = sf.read(fake, dtype="float32", always_2d=True)
            x = x[:, -1]
            self.name, self._stop = f"fake:{Path(fake).name}", threading.Event()

            def run():
                n, i = int(self.sr * 0.02), 0
                while not self._stop.is_set():
                    self.q.put(x[i: i + n] if i + n <= len(x) else np.zeros(n, np.float32))
                    i = (i + n) % max(len(x), 1)
                    time.sleep(0.02)
            self._t = threading.Thread(target=run, daemon=True)
            self._t.start()
            self.stream = None
            return
        import sounddevice as sd
        info = sd.query_devices(device, "input")
        self.name, self.sr = info["name"], int(info["default_samplerate"])
        def cb(d, frames, t, status):
            if status.input_overflow:
                self.overflows += 1
            self.q.put(d[:, 0].copy())
        # high latency: a GIL-busy UI thread must never make CoreAudio drop input blocks
        self.stream = sd.InputStream(device=device, channels=1, samplerate=self.sr, dtype="float32",
                                     blocksize=int(self.sr * 0.02), latency="high", callback=cb)
        self.stream.start()

    def pull(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        else:
            self._stop.set()


def to_sr(x, sr):
    if sr == SR:
        return x.astype(np.float32)
    g = gcd(SR, sr)
    return resample_poly(x, SR // g, sr // g).astype(np.float32)


def db(v):
    return float(20 * np.log10(max(float(v), 1e-9)))


def analyze(x, it):
    """Levels, speech span and warnings for a 24 kHz take."""
    peak = float(np.abs(x).max()) if len(x) else 0.0
    r = {"dur": round(len(x) / SR, 3), "peak_db": round(db(peak), 1), "warn": []}
    if it["kind"] == "roomtone":
        r["rms_db"] = round(db(np.sqrt(np.mean(x ** 2))), 1)
        if r["dur"] < it["seconds"] - 1:
            r["warn"].append(f"only {r['dur']:.1f} s: let it run the full {it['seconds']:.0f} s")
        if r["peak_db"] > -30:
            r["warn"].append(f"room is loud (peak {r['peak_db']} dBFS)")
        return r
    if peak < 10 ** (-40 / 20):
        r["warn"].append("nothing heard (peak below -40 dBFS)")
        return r
    on, off = map(float, speech_span(x))
    r["speech"] = [round(on, 3), round(off, 3)]
    if (np.abs(x) > 0.99).sum() > 2:
        r["warn"].append("clipped: move back from the mic")
    elif r["peak_db"] < -30:
        r["warn"].append(f"quiet (peak {r['peak_db']} dBFS): move closer")
    n = int(0.02 * SR)
    m = len(x) // n
    e = (x[: m * n].reshape(m, n) ** 2).mean(1) + 1e-12
    t = np.arange(m) * 0.02
    inside = (t >= on) & (t < off)
    if inside.any() and (~inside).sum() >= 5:
        r["snr_db"] = round(float(10 * np.log10(e[inside].mean() / np.percentile(e[~inside], 50))), 1)
        if r["snr_db"] < 20:
            r["warn"].append(f"noisy (SNR {r['snr_db']} dB)")
    if on < 0.05:
        r["warn"].append("speech starts at the very beginning: start may be cut")
    if off > len(x) / SR - 0.05:
        r["warn"].append("speech runs to the very end: stopped too early?")
    if it["kind"] == "line":
        w = len(it["text"].split())
        if off - on < 0.12 * w or off - on < 0.25:
            r["warn"].append(f"short for {w} words ({off - on:.1f} s): cut off?")
    return r


def trim(x, r):
    if "speech" not in r:
        return x
    a = max(0, int((r["speech"][0] - PAD_PRE) * SR))
    b = min(len(x), int((r["speech"][1] + PAD_POST) * SR))
    y = x[a:b].copy()
    f = min(int(0.01 * SR), len(y) // 2)
    if f:
        ramp = np.linspace(0, 1, f, dtype=np.float32)
        y[:f] *= ramp
        y[-f:] *= ramp[::-1]
    return y


# ------------------------------------------------------------------------- session

class Session:
    def __init__(self, out):
        self.out = out
        self.plan = json.loads((out / "session.json").read_text())
        self.items = self.plan["items"]
        (out / "clips").mkdir(exist_ok=True)
        (out / "raw").mkdir(exist_ok=True)
        self._done = {p.stem for p in (out / "clips").glob("*.wav")}

    def done(self, it):
        return it["key"] in self._done

    def n_takes(self, it):
        return len(list((self.out / "raw").glob(f"{it['key']}.*.wav")))

    def save_raw(self, it, x, r):
        n = self.n_takes(it) + 1
        sf.write(self.out / "raw" / f"{it['key']}.{n}.wav", x, SR, subtype="PCM_16")
        return n

    def keep(self, it, x, r, take):
        y = trim(x, r)
        meta = {"item": it["id"], "block": it["block"], "episode": it.get("episode"), "take": take,
                "text": it.get("text"), "intent": it.get("intent"), "speaker": self.plan["speaker"],
                "sr": SR, **r}
        if "speech" in r:
            a = max(0.0, r["speech"][0] - PAD_PRE)
            meta["speech"] = [round(r["speech"][0] - a, 3), round(r["speech"][1] - a, 3)]
        tmp = self.out / "clips" / f".{it['key']}.tmp.wav"
        sf.write(tmp, y, SR, subtype="PCM_16")
        os.replace(tmp, self.out / "clips" / f"{it['key']}.wav")
        self._done.add(it["key"])
        (self.out / "clips" / f"{it['key']}.json").write_text(json.dumps(meta, ensure_ascii=False))

    def log(self, it, r, take, kept):
        row = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "item": it["id"],
               "key": it["key"], "take": take, "kept": kept, **r}
        with open(self.out / "takes.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def left_s(self, i):
        return sum(item_seconds(it) for it in self.items[i:] if not self.done(it))

    def status(self):
        for blk in "RBA":
            its = [it for it in self.items if it["block"] == blk]
            print(f"block {blk}: {sum(map(self.done, its))}/{len(its)} recorded")
        print(f"time left ≈ {self.left_s(0) / 60:.0f} min")


# ------------------------------------------------------------------------- terminal UI

B, DIM, RED, GRN, YEL, CYN, RST = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"
TO = {"moshi": "to Moshi", "friend": "to your friend", "self": "muttering to yourself, quieter",
      "aside": "an aside, half to yourself"}
INTRO = {
    "R": "Room tone: stay silent for 10 s. Press SPACE, then don't move or talk.",
    "B": "OWN WORDS. Say it the way you'd naturally say it. Don't read anything off the "
         "screen; the card says what to get across, not how. Keep any number it asks for.",
    "A": "VERBATIM. Read each line as if you were in the conversation. Small slips are fine; "
         "redo with [b] if a word comes out wrong.",
}


class UI:
    def __init__(self, sess, mic):
        self.s, self.mic = sess, mic
        self.pre = deque(maxlen=max(1, int(PREROLL_S / 0.02) + 2))
        self.rec, self.take, self.rec_t0 = False, [], 0.0
        self.level, self.msg, self.review = -90.0, "", None
        self.i = next((k for k, it in enumerate(sess.items) if not sess.done(it)), len(sess.items))
        cols = shutil.get_terminal_size((100, 30)).columns
        self.width = min(100, cols) if cols >= 40 else 100
        self.silent_since, self.any_signal = time.time(), False
        self.player = None

    # -- drawing
    def wrap(self, text, indent):
        return textwrap.fill(text, self.width - 2, initial_indent=indent, subsequent_indent=" " * len(indent))

    def draw(self):
        o = ["\033[H\033[2J"]
        s = self.s
        o.append(f"{DIM}human test set · speaker {s.plan['speaker']} · mic: {self.mic.name}{RST}")
        if self.i >= len(s.items):
            o.append(f"\n{GRN}{B}All items recorded.{RST} [b] back to redo the last one, [q] quit.")
            o.append(self.meter())
            print("\n".join(o), end="", flush=True)
            return
        it = s.items[self.i]
        blk = [x for x in s.items if x["block"] == it["block"]]
        pos = blk.index(it) + 1
        head = f"block {it['block']} · {pos}/{len(blk)}"
        if it["block"] == "A":
            head += f" · episode {it['ep_index'] + 1}/{len(s.plan['episodes'])} ({it['type']})"
        n_done = sum(map(s.done, s.items))
        head += f" · {n_done}/{len(s.items)} done · ~{s.left_s(self.i) / 60:.0f} min left"
        o.append(f"{DIM}{head}{RST}")
        o.append(f"{DIM}{self.wrap(INTRO[it['block']], '')}{RST}\n")
        if it["block"] == "A" and (self.i == 0 or s.items[self.i - 1].get("episode") != it["episode"]):
            o.append(f"{CYN}{B}New conversation.{RST} {CYN}{SCENES[family(it['type'])]}{RST}\n")
        for who, txt in it.get("context", []):
            o.append(f"{DIM}{self.wrap(txt, f'  {who:<7}')}{RST}")
        if it.get("context"):
            o.append("")
        if it["kind"] == "roomtone":
            o.append(f"  {B}(silence, {it['seconds']:.0f} s){RST}")
        elif it["kind"] == "free":
            o.append(f"  {YEL}▶ to Moshi, in your own words{RST}")
            o.append(B + self.wrap(it["intent"], "  ") + RST)
            if it["must"]:
                o.append(f"  {YEL}(must {it['must']}){RST}")
        else:
            tag = TO[it["to"]]
            if it["quiet"] and it["to"] != "self":
                tag += " (quieter)"
            if it["continues"]:
                tag += "; continue the sentence from before the pause"
            if it["pause_after"]:
                tag += "; you pause mid-sentence here, so don't end it like a finished sentence"
            o.append(f"  {YEL}▶ {tag}{RST}")
            o.append(B + self.wrap(f"“{it['text']}”", "  ") + RST)
        if s.done(it) and not self.review:
            o.append(f"\n  {GRN}✓ already recorded ({s.n_takes(it)} take(s)); SPACE re-records{RST}")
        o.append("")
        if self.rec:
            o.append(f"{RED}{B}● REC {time.time() - self.rec_t0:4.1f} s{RST}   [space] stop   [x] cancel")
        elif self.review:
            o.append(f"{YEL}{B}Check this take:{RST}")
            for w in self.review[1]["warn"]:
                o.append(f"  {YEL}- {w}{RST}")
            o.append("[space] redo   [k] keep anyway   [p] play it   [b] back   [q] quit")
        else:
            o.append("[space] record   [p] play saved   [b] back   [s] skip   [q] quit")
        if self.msg:
            o.append(self.msg)
        o.append(self.meter())
        print("\n".join(o), end="", flush=True)

    def meter(self):
        lv = max(-60.0, min(0.0, self.level))
        n = int((lv + 60) / 60 * 30)
        col = RED if lv > -3 else GRN if lv > -35 else DIM
        warn = ""
        if not self.any_signal and time.time() - self.silent_since > 2:
            warn = (f"  {RED}mic sends digital silence: allow your terminal app under System Settings → "
                    f"Privacy & Security → Microphone, then restart{RST}")
        return f"\n{DIM}level{RST} {col}{'█' * n}{DIM}{'·' * (30 - n)}{RST} {lv:5.0f} dB{warn}"

    # -- actions
    def start(self):
        self.review, self.msg = None, ""
        self.take, self.rec, self.rec_t0 = list(self.pre), True, time.time()
        self.mic.overflows = 0

    def stop(self, cancel=False):
        self.rec = False
        if cancel:
            self.msg = f"{DIM}take discarded{RST}"
            return
        it = self.s.items[self.i]
        x = to_sr(np.concatenate(self.take) if self.take else np.zeros(1, np.float32), self.mic.sr)
        r = analyze(x, it)
        if self.mic.overflows:
            r["warn"].append(f"{self.mic.overflows} audio block(s) dropped by the mic stream: redo")
            r["overflows"] = self.mic.overflows
        take = self.s.save_raw(it, x, r)
        if r["warn"]:
            self.review = (x, r, take)
            self.s.log(it, r, take, kept=False)
            return
        self.commit(x, r, take)

    def commit(self, x, r, take):
        it = self.s.items[self.i]
        self.s.keep(it, x, r, take)
        self.s.log(it, r, take, kept=True)
        info = f"{r['dur']:.1f} s, peak {r['peak_db']} dBFS"
        if "snr_db" in r:
            info += f", SNR {r['snr_db']} dB"
        if "rms_db" in r:
            info += f", room {r['rms_db']} dBFS rms"
        self.msg = f"{GRN}✓ {it['id']} saved ({info}){RST}"
        self.review = None
        self.i += 1

    def play(self):
        """Play through macOS's afplay in its own process: playing from this busy Python loop
        (sounddevice) starves the output stream and crackles, though the files are clean."""
        if self.player and self.player.poll() is None:
            self.player.terminate()
        if self.review:
            path = self.s.out / ".review.wav"
            sf.write(path, self.review[0], SR, subtype="PCM_16")
        else:
            cur = self.s.items[self.i] if self.i < len(self.s.items) else None
            prev = self.s.items[self.i - 1] if self.i > 0 else None
            it = cur if cur and self.s.done(cur) else prev if prev and self.s.done(prev) else None
            if it is None:
                return
            path = self.s.out / "clips" / f"{it['key']}.wav"
            self.msg = f"{DIM}playing {it['id']}{RST}"
        self.player = subprocess.Popen(["afplay", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def key(self, ch):
        it = self.s.items[self.i] if self.i < len(self.s.items) else None
        if ch == "q":
            if self.rec:
                self.stop(cancel=True)
            return False
        if self.rec:
            if ch in (" ", "\n", "\r"):
                self.stop()
            elif ch == "x":
                self.stop(cancel=True)
            return True
        if ch in (" ", "\n", "\r") and it is not None:
            self.start()
        elif ch == "k" and self.review:
            self.commit(*self.review)
        elif ch == "p":
            self.play()
        elif ch == "b" and self.i > 0:
            self.review, self.i, self.msg = None, self.i - 1, f"{DIM}back: SPACE re-records this one{RST}"
        elif ch == "s" and it is not None:
            self.review, self.i, self.msg = None, self.i + 1, f"{DIM}skipped {it['id']}{RST}"
        return True

    def tick(self):
        blocks = self.mic.pull()
        if blocks:
            x = np.concatenate(blocks)
            self.level = db(np.abs(x).max())
            if np.any(x != 0):
                self.any_signal = True
            self.pre.extend(blocks)
            if self.rec:
                self.take.extend(blocks)
        if self.rec and self.i < len(self.s.items):
            it = self.s.items[self.i]
            if it["kind"] == "roomtone" and time.time() - self.rec_t0 >= it["seconds"] + PREROLL_S:
                self.stop()

    def run(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            print("\033[?25l", end="")
            last = 0.0
            while True:
                self.tick()
                if time.time() - last > 0.08:
                    self.draw()
                    last = time.time()
                r, _, _ = select.select([sys.stdin], [], [], 0.02)
                if r and not self.key(os.read(fd, 1).decode(errors="ignore")):
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            print("\033[?25h\n")
            self.mic.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--speaker", default="s01")
    ap.add_argument("--data", type=Path, help="data dir with the test suites (default: main checkout's data/)")
    ap.add_argument("--out", type=Path, help="default: <data>/human/<speaker>")
    ap.add_argument("--suite", default="test_all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", help="input device index or name substring (--list-devices)")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--fake-wav", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    data = a.data or main_checkout() / "data"
    out = a.out or data / "human" / a.speaker
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "session.json").exists():
        plan = build_plan(data, a.speaker, a.suite, a.seed)
        (out / "session.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
        print(f"wrote {out / 'session.json'}")
    sess = Session(out)
    if a.plan_only:
        describe(sess.plan)
        return
    if a.status:
        describe(sess.plan)
        sess.status()
        return
    dev = int(a.device) if a.device and a.device.isdigit() else a.device
    UI(sess, Mic(dev, a.fake_wav)).run()
    sess.status()
    print(f"clips in {out / 'clips'}")


if __name__ == "__main__":
    main()
