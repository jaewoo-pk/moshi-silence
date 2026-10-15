"""Generate content packs with an LLM (Anthropic API).

  # key in moshi-silence/.env as ANTHROPIC_API_KEY=... (gitignored), or exported
  python -m silence.gen_packs --out packs/packs.jsonl --per-family 500 --workers 32

Rerunning with the same --out tops each family up to --per-family.

B2 side conversations carry their distractors as line tags (see SCHEMAS / validate).
Each pack gets a stable id and a train/test split (85/15, by hash of its text), so
no test script is ever seen in training. Packs are deduplicated on their key text.
SCHEMAS is the schema reference; validate() enforces what the episode builders rely on.
"""
import argparse
import hashlib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import specs

TOPICS = [
    "school", "work", "travel", "cooking", "sports", "music", "movies", "books", "pets", "money",
    "health and fitness", "hobbies", "family", "weekend plans", "technology", "science trivia",
    "math puzzles", "riddles", "history", "geography", "shopping", "apartments and roommates",
    "cars and commuting", "weather", "games", "art", "gardening", "holidays", "food recommendations",
]

# Every conversation opens with the user; Moshi never speaks first.
SCHEMAS = {
    "B1": """A question that the user plausibly needs time to think about (mental math, a riddle, recall, a personal decision, a would-you-rather).
{"opener": "the user's opening line, which starts the conversation, asking Moshi for something like this (e.g. asking for a riddle, a quick quiz, a math question, or help deciding), without stating the question itself; at most 15 words",
 "question": "Moshi asks it in reply to the opener, casually; at most 20 words",
 "reask": "Moshi asks again later, paraphrased, starting like 'Alright, ...' or 'So, ...'; at most 15 words",
 "answer": "the user's spoken answer; at most 10 words",
 "reaction": "Moshi's short reaction to the answer; at most 15 words",
 "mutter": ["2-3 things the user might mutter to themselves while thinking, e.g. partial reasoning or 'hmm'; at most 8 words each"]}""",
    "B2": """Moshi and the user are mid-conversation; then the user's friend interrupts and they talk briefly about something unrelated. Moshi has to stay silent during the side conversation, so it should contain moments that could tempt an assistant to jump in.
{"topic": [2 or 4 alternating lines between "user" and "moshi", starting with the user, who brings up something they want to talk about or need help with, and ending with Moshi's last line before the interruption, ideally a question; at most 15 words per line: {"spk": "user"|"moshi", "text": ...}],
 "side": [5 or 6 lines between "friend" and "user", mostly alternating, mundane logistics or chit-chat unrelated to the topic, at most 10 words per line: {"spk": "friend"|"user", "text": ..., "tag": "D1a"|"D1b"|"D1c"|"D3a"|"none"}],
 "friend_away": ["a line where the friend briefly steps away mid-chat, e.g. to grab something", "the friend's line on coming back, saying they're back"],
 "resume": "Moshi welcomes the user back and picks the topic back up, referring to it specifically; at most 18 words",
 "followup": "the user's short reply to the resume line; at most 10 words"}
Side-conversation tags ("none" on every other line):
 "D1a": the friend asks the user a question; the user answers it on the next line
 "D1b": the user asks the friend a question; the friend answers it on the next line
 "D1c": the friend asks a question a voice assistant could also answer (the time, the weather, a quick fact, a spelling); the user answers it on the next line
 "D3a": a user line that, out of context, could sound like the user coming back to Moshi (e.g. using "back", "anyway", "okay, so", "where were we", "I'm done"), but in context it is clearly still part of the side conversation; it is line 2, 3 or 4, so the side conversation clearly goes on after it
Every example has exactly one D3a line, early in the side conversation. Every question in the side conversation gets answered, the side conversation is coherent on its own, and nobody says either person's name. {b2_req}""",
    "B3": """The user is deciding between 3 options (sometimes 4) and describes each before asking Moshi's opinion.
{"context": "the user's first spoken sentence, which opens the conversation, in the first person, saying what they're deciding (e.g. 'I'm trying to pick a car for my new commute.'); at most 15 words",
 "options": [["option stated as a short noun phrase or action, e.g. 'A used family car.'", "one sentence of detail, at most 12 words"], ...3 or 4 options],
 "aside": "a short line the user could say right after describing one option that sounds a bit like they're finished, although more options follow (e.g. 'Okay, so that's that one.')",
 "comment": "Moshi's opinion that explicitly mentions EVERY option; at most 25 words",
 "comment_first_two": "Moshi's opinion mentioning only the first two options; at most 20 words",
 "comment_first": "Moshi's opinion on only the first option; at most 15 words",
 "followup": "user's short reply; at most 10 words"}""",
    "ORD": """An ordinary back-and-forth between Moshi (a friendly voice assistant) and a user.
{"turns": [4 or 6 alternating turns, starting with the user, who opens with a request or question for Moshi, and ending with Moshi: {"spk": "user"|"moshi", "text": ...}; user turns are 8-20 words and at least one contains a comma-separated clause; Moshi turns are at most 20 words]}""",
    "DECOY": """An ordinary back-and-forth between Moshi and a user, where one user turn naturally uses the word or phrase "{decoy}" WITHOUT asking Moshi to wait or be quiet.
{"turns": [4 or 6 alternating turns, starting with the user and ending with Moshi: {"spk": "user"|"moshi", "text": ...}; user turns are 8-20 words, Moshi turns at most 20 words]}""",
}


# Structured outputs: the API constrains each response to these JSON schemas, so batches
# always parse and items always have the right fields and speaker labels. The prose in
# SCHEMAS still says what goes in each field; validate() checks what JSON schema can't.
def _obj(props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _arr(items):
    return {"type": "array", "items": items}


_S = {"type": "string"}


def _line(speakers, tagged=False):
    props = {"spk": {"type": "string", "enum": speakers}, "text": _S}
    if tagged:
        props["tag"] = {"type": "string", "enum": ["none", "D1a", "D1b", "D1c", "D3a"]}
    return _obj(props)


_ITEMS = {
    "B1": _obj({"opener": _S, "question": _S, "reask": _S, "answer": _S, "reaction": _S, "mutter": _arr(_S)}),
    "B2": _obj({"topic": _arr(_line(["user", "moshi"])), "side": _arr(_line(["friend", "user"], tagged=True)),
                "friend_away": _arr(_S), "resume": _S, "followup": _S}),
    "B3": _obj({"context": _S, "options": _arr(_arr(_S)), "aside": _S, "comment": _S,
                "comment_first_two": _S, "comment_first": _S, "followup": _S}),
    "ORD": _obj({"turns": _arr(_line(["user", "moshi"]))}),
}
_ITEMS["DECOY"] = _ITEMS["ORD"]  # the decoy word is set by the caller, not echoed by the model
OUTPUT_SCHEMAS = {fam: _obj({"examples": _arr(item)}) for fam, item in _ITEMS.items()}

D1C_REQS = [
    (0.5, "Include one D1c question in about half of the examples."),
    (0.5, "Do not include any D1c question."),
]
DECOYS = ["back", "seconds", "options", "wait", "quiet", "think", "friend", "minute", "describe", "ask me again"]
# How the user talks in a batch (one per request), so user lines don't all share one voice.
STYLES = [
    "short and to the point, in brief sentences",
    "chatty, over-explains a little and adds small asides",
    "hesitant, with fillers like 'um', 'uh' and 'I mean', sometimes restarting a sentence",
    "polite and a bit formal",
    "casual and relaxed, like talking to a friend",
    "an older adult, unhurried, with plain old-fashioned phrasing",
    "a teenager, casual, says 'like' and 'kind of' now and then",
    "a non-native English speaker: simple, clear sentences with slightly unusual word choice",
    "upbeat and enthusiastic",
    "tired or a bit distracted, trailing off now and then",
]
# Real users rarely say the assistant's name. Half the batches allow it in about half the
# examples, so roughly a quarter of the packs name Moshi.
NAME_RULES = [
    (0.5, "In about half of the examples the user says the name Moshi once; in the others, never."),
    (0.5, "The user never says the name Moshi."),
]

PROMPT = """Write {n} distinct examples for a spoken-dialogue dataset, in the "examples" list.
Each example follows this schema:
{schema}

How the user talks in this batch: {style}. {name_rule}
Keep it brief: every line is something a person says out loud in one breath, and Moshi's
replies are short and conversational, never speeches. Respect the word limits.

Constraints: natural spoken English, contractions welcome, no emojis, no stage directions,
no numerals (spell numbers out), every example about a different situation.
The text will be read aloud by a small text-to-speech model that mispronounces unusual
words, so use only plain, common, easy-to-pronounce English: no foreign or loan words
(e.g. latte, croissant, quinoa, gnocchi, açaí, pho), no brand or product names at all
(say "a light laptop", not a brand or model name), no
acronyms or abbreviations, no uncommon place names (stick to well-known cities and
countries), and only common first names. Themes for this batch (spread the examples
across them): {topic}.
{avoid}"""


# Words Pocket TTS is known to garble, plus common loan words. Extend as you find more
# while listening; packs containing any of them are dropped.
HARD_WORDS = {
    "latte", "lattes", "espresso", "cappuccino", "macchiato", "croissant", "croissants",
    "quinoa", "gnocchi", "bruschetta", "pho", "gyro", "gyros", "acai", "açaí", "sriracha",
    "chipotle", "jalapeno", "jalapeño", "quesadilla", "edamame", "kombucha", "matcha",
    "boba", "bougie", "cafe", "café", "facade", "façade", "naive", "naïve", "resume",
    "genre", "entrepreneur", "segue", "hyperbole", "epitome", "alfama", "worcestershire",
    "macbook", "thinkpad", "chromebook",
}
_ALLOWED_CAPS = {"I", "OK", "TV"}


def is_speakable(p) -> tuple[bool, str]:
    """Reject packs a small TTS is likely to mispronounce."""
    def texts(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for k, v in x.items():
                if k not in ("id", "family", "split", "spk", "tag", "decoy", "style"):
                    yield from texts(v)
        elif isinstance(x, list):
            for v in x:
                yield from texts(v)
    for t in texts(p):
        if re.search(r"[0-9]", t):
            return False, "digit"
        if re.search(r"[^\x00-\x7F’‘“”—–]", t):
            return False, "non-ascii"
        for w in re.findall(r"[A-Za-z][A-Za-z'’]*", t):
            if w.lower().strip("'’") in HARD_WORDS:
                return False, w
            if len(w) >= 2 and w.isupper() and w not in _ALLOWED_CAPS:
                return False, f"acronym {w}"
            if re.search(r"[a-z][A-Z]", w):  # MacBook, iPhone, YouTube: product names
                return False, f"camelcase {w}"
    return True, ""


D1_SPK = {"D1a": "friend", "D1b": "user", "D1c": "friend"}


def _is_q(text):  # "Do you have a charger? My phone is dying." is a question too
    return "?" in text


def validate(fam, p) -> tuple[bool, str]:
    """Check the parts of the schema the episode builders rely on. B2: every side-line
    question is answered by the other person, tags fit their lines, and untagged
    questions get tagged D1a / D1b by speaker (they are questions to the other person)."""
    if fam == "B2":
        side = p["side"]
        if len(side) < 4 or any(t.get("spk") not in ("friend", "user") or not t.get("text") for t in side):
            return False, "schema: side"
        # Repairs instead of rejects: keep the first 6 lines (the prompt asks for 5-6, the
        # model often writes 7-9), then drop trailing lines that are a question (its answer
        # was cut) or a D3a (a return-cue lookalike right before the real cue would blur
        # where the hold ends). Rejected below if that leaves no D3a.
        del side[6:]
        while len(side) > 4 and (_is_q(side[-1]["text"]) or side[-1].get("tag") == "D3a"):
            side.pop()
        for i, t in enumerate(side):
            tag = t.get("tag")
            if tag == "none":
                tag = None
            if tag not in (None, "", "D1a", "D1b", "D1c", "D3a"):
                return False, f"schema: tag {tag}"
            if _is_q(t["text"]):
                if i == len(side) - 1 or side[i + 1]["spk"] == t["spk"]:
                    return False, "schema: unanswered question"
                if tag == "D1c" and t["spk"] != "friend":
                    tag = None
                if tag != "D1c":  # D1a / D1b follow the speaker
                    tag = "D1a" if t["spk"] == "friend" else "D1b"
            elif tag in D1_SPK:
                return False, "schema: D1 tag without a question"
            if tag == "D3a" and t["spk"] != "user":  # a friend cue lookalike would leak D4 (test-only)
                return False, "schema: D3a on a friend line"
            if tag == "D3a" and i == len(side) - 1:
                return False, "schema: D3a on the last line"
            if tag:
                t["tag"] = tag
            else:
                t.pop("tag", None)
        # The prompt asks for exactly one D3a; extra ones are kept (each is its own labeled
        # distractor event), none is a reject.
        if not any(t.get("tag") == "D3a" for t in side):
            return False, "schema: no D3a"
        fa = p.get("friend_away")
        if not (isinstance(fa, list) and len(fa) == 2 and all(isinstance(x, str) and x for x in fa)):
            return False, "schema: friend_away"
    def strs(x, lo=1):
        return isinstance(x, list) and len(x) >= lo and all(isinstance(v, str) and v.strip() for v in x)

    def alternates(turns):
        return all(t.get("spk") in ("moshi", "user") and t.get("text") for t in turns) and \
            all(a["spk"] != b["spk"] for a, b in zip(turns, turns[1:]))

    if fam == "B1":
        if not all(isinstance(p.get(k), str) and p[k] for k in ("opener", "question", "reask", "answer", "reaction")):
            return False, "schema: B1 fields"
        if not strs(p.get("mutter")):
            return False, "schema: mutter"
    if fam == "B2":
        top = p["topic"]
        if len(top) < 2 or top[0].get("spk") != "user" or top[-1].get("spk") != "moshi" or not alternates(top):
            return False, "schema: topic"
        if not all(isinstance(p.get(k), str) and p[k] for k in ("resume", "followup")):
            return False, "schema: B2 fields"
    if fam == "B3":
        if not (3 <= len(p["options"]) <= 4 and all(strs(o) for o in p["options"])):
            return False, "schema: options"
        if not all(isinstance(p.get(k), str) and p[k] for k in
                   ("context", "aside", "comment", "comment_first_two", "comment_first", "followup")):
            return False, "schema: B3 fields"
    if fam in ("ORD", "DECOY"):
        turns = p["turns"]
        if len(turns) < 4 or not alternates(turns) or turns[0]["spk"] != "user" or turns[-1]["spk"] != "moshi":
            return False, "schema: turns"
    if fam == "DECOY":
        d = p.get("decoy", "").lower()
        if not d or not any(re.search(rf"\b{re.escape(d)}\b", t["text"].lower()) for t in turns if t["spk"] == "user"):
            return False, "schema: decoy word missing"
    if re.search(r"\bthe user\b", json.dumps(p).lower()):  # spoken lines written as a description
        return False, "schema: third person"
    return True, ""


def _pick(r, reqs):
    x, acc = r.random(), 0.0
    for w, text in reqs:
        acc += w
        if x < acc:
            return text
    return reqs[-1][1]


def key_text(fam, p):
    if fam == "B1":
        return p["question"]
    if fam == "B2":
        return p["topic"][-1]["text"]
    if fam == "B3":
        return p["context"]
    return next(t["text"] for t in p["turns"] if t["spk"] == "user")


def norm(s):
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


_STOP = set("a an the i you to of and or is it in on for my me your this that do what how be with "
            "so are was we can just about at if its im youre".split())


def content_words(s):
    return frozenset(w for w in norm(s).split() if w not in _STOP)


def near_dup(words, pool, thresh=0.6):
    """Jaccard overlap of content words with any earlier key text of the same family."""
    return any(len(words & w) / max(1, len(words | w)) >= thresh for w in pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="packs/packs.jsonl")
    ap.add_argument("--per-family", type=int, default=500)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=32, help="concurrent API requests")
    ap.add_argument("--families", nargs="+", default=["B1", "B2", "B3", "ORD", "DECOY"])
    ap.add_argument("--rejects", default=None, help="also write rejected packs (with the reason) here")
    ap.add_argument("--seed", type=int, default=0, help="topic/style sequence seed (a new seed for a new top-up)")
    ap.add_argument("--per-family-json", default=None, help='per-family targets, e.g. \'{"B1": 2200, "B2": 1600}\'')
    a = ap.parse_args()

    import os

    import anthropic
    env = Path(__file__).resolve().parent.parent / ".env"
    if "ANTHROPIC_API_KEY" not in os.environ and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
    # Talk to the public API directly, whatever the shell environment says.
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.anthropic.com",
                                 max_retries=8)  # 429s / overloads back off and retry
    out = Path(a.out)
    seen, n_written, rejected = set(), 0, {}
    by_fam = {}    # family -> content-word sets of accepted key texts (near-dup check)
    recent = {}    # family -> accepted key texts; a random 40 are shown to the model to avoid
    usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    if out.exists():
        for l in out.read_text().splitlines():
            p = json.loads(l)
            fam = "DECOY" if p.get("decoy") else p["family"]
            k = key_text(fam, p)
            seen.add(norm(k))
            by_fam.setdefault(fam, []).append(content_words(k))
            recent.setdefault(fam, []).append(k)
    r = random.Random(a.seed)
    fams = a.families
    target = {fam: a.per_family for fam in fams}
    if a.per_family_json:
        target.update(json.loads(a.per_family_json))
    rej_f = open(a.rejects, "a") if a.rejects else None
    got = {fam: len(by_fam.get(fam, [])) for fam in fams}  # resume = top up to --per-family
    returned, accepted = 0, 0  # items the model returned / kept, for the yield estimate

    def make_request(fam):  # main thread only: draws from r and reads `recent`
        decoy = r.choice(DECOYS)
        schema = SCHEMAS[fam].replace("{decoy}", decoy).replace("{b2_req}", _pick(r, D1C_REQS))
        topic = ", ".join(r.sample(TOPICS, r.randint(2, 3)))
        style = r.choice(STYLES)
        prev = recent.get(fam, [])
        prev = r.sample(prev, min(40, len(prev)))
        avoid = ("Situations already covered (don't repeat them; find new ones):\n"
                 + "".join(f"- {x}\n" for x in prev)) if prev else ""
        prompt = PROMPT.format(n=a.batch, schema=schema, topic=topic, avoid=avoid, style=style,
                               name_rule=_pick(r, NAME_RULES))
        return fam, topic, decoy, style, prompt

    def call(fam, prompt):  # worker threads
        # Writing short scripts needs no reasoning: thinking off, low effort = fast and
        # no thinking tokens eating max_tokens (which truncated batches before). The JSON
        # schema means no unparseable batches, no preamble text, no malformed items.
        return client.messages.create(
            model=a.model, max_tokens=16000, thinking={"type": "between_tools"},
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMAS[fam]}},
            messages=[{"role": "user", "content": prompt}])

    def take(fam, topic, decoy, style, msg, f):
        nonlocal n_written, returned, accepted
        usage["calls"] += 1
        usage["input_tokens"] += msg.usage.input_tokens
        usage["output_tokens"] += msg.usage.output_tokens
        if msg.stop_reason != "end_turn":
            rejected[f"stop_reason {msg.stop_reason}"] = rejected.get(f"stop_reason {msg.stop_reason}", 0) + 1
        txt = next((b.text for b in msg.content if b.type == "text"), "")
        try:  # only a truncated (max_tokens) or refused response can fail here
            items = json.loads(txt)["examples"]
        except (ValueError, KeyError, TypeError):
            rejected["unparseable batch"] = rejected.get("unparseable batch", 0) + 1
            return
        for p in items:
            if got[fam] >= target[fam]:
                break
            if not isinstance(p, dict):
                continue
            returned += 1
            if fam == "DECOY":
                p["decoy"] = decoy  # the word we asked for, whatever the model echoed
            try:
                k = key_text(fam, p)
                ok, why = validate(fam, p)
            except (KeyError, IndexError, TypeError, AttributeError, StopIteration):
                ok, why = False, "schema: malformed"
            if ok:
                cw = content_words(k)
                if not norm(k) or norm(k) in seen:
                    ok, why = False, "duplicate"
                elif near_dup(cw, by_fam.get(fam, [])):
                    ok, why = False, "near-duplicate"
            if ok:
                ok, why = is_speakable(p)
            if ok:  # must fit one 60 s training window without cutting content
                est = specs.longest_episode(fam, p)
                if est > specs.BUDGET_DUR:
                    ok, why = False, f"{fam} too long for a 60 s episode"
            if not ok:
                rejected[why] = rejected.get(why, 0) + 1
                if rej_f:
                    rej_f.write(json.dumps({"why": why, "family": fam, "style": style, "pack": p}, ensure_ascii=False) + "\n")
                continue
            seen.add(norm(k))
            by_fam.setdefault(fam, []).append(cw)
            recent.setdefault(fam, []).append(k)
            h = int(hashlib.sha1(norm(k).encode()).hexdigest()[:8], 16)
            p["family"] = "ORD" if fam == "DECOY" else fam
            p["style"] = style
            p["est_max_s"] = round(est, 1)
            p["id"] = f"{fam.lower()}_{h:08x}"
            p["split"] = "test" if (h % 1000) / 1000 < a.test_frac else "train"
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            f.flush()
            got[fam] += 1
            accepted += 1
            n_written += 1

    # Keep `workers` requests in flight across all families; stop submitting for a family
    # once its in-flight requests are expected (at the running yield) to fill it.
    inflight = {}  # future -> (fam, topic, decoy)
    with out.open("a") as f, ThreadPoolExecutor(a.workers) as pool:
        while True:
            yld = max(0.2, accepted / returned) if returned >= 20 else 0.8
            pending = {fam: sum(1 for v in inflight.values() if v[0] == fam) for fam in fams}
            want = [fam for fam in fams if got[fam] + pending[fam] * a.batch * yld < target[fam]]
            while want and len(inflight) < a.workers:
                fam = min(want, key=lambda x: got[x] + pending[x] * a.batch * yld)
                _, topic, decoy, style, prompt = make_request(fam)
                inflight[pool.submit(call, fam, prompt)] = (fam, topic, decoy, style)
                pending[fam] += 1
                want = [x for x in fams if got[x] + pending[x] * a.batch * yld < target[x]]
            if not inflight:
                break
            done = next(as_completed(inflight))
            fam, topic, decoy, style = inflight.pop(done)
            try:
                msg = done.result()
            except anthropic.APIError as e:
                rejected[f"api error {type(e).__name__}"] = rejected.get(f"api error {type(e).__name__}", 0) + 1
                continue
            take(fam, topic, decoy, style, msg, f)
            print("  ".join(f"{x}: {got[x]}/{target[x]}" for x in fams) + f"  calls: {usage['calls']}", flush=True)
    print()
    print(f"wrote {n_written} packs to {out}")
    print("api usage:", usage)
    if rejected:
        print("rejected:", dict(sorted(rejected.items(), key=lambda x: -x[1])[:25]))


if __name__ == "__main__":
    main()
