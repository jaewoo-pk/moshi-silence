"""Generate content packs with an LLM (Anthropic API).

  # key in moshi-silence/.env as ANTHROPIC_API_KEY=... (gitignored), or exported
  python -m silence.gen_packs --out packs/packs.jsonl --per-family 1500

Each pack gets a stable id and a train/test split (85/15, by hash of its text), so
no test script is ever seen in training. Packs are deduplicated on their key text.
Schemas match packs/demo_packs.jsonl.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

TOPICS = [
    "school", "work", "travel", "cooking", "sports", "music", "movies", "books", "pets", "money",
    "health and fitness", "hobbies", "family", "weekend plans", "technology", "science trivia",
    "math puzzles", "riddles", "history", "geography", "shopping", "apartments and roommates",
    "cars and commuting", "weather", "games", "art", "gardening", "holidays", "food recommendations",
]

SCHEMAS = {
    "B1": """A question that the user plausibly needs time to think about (mental math, a riddle, recall, a personal decision, a would-you-rather).
{"question": "Moshi asks it, casually, 1-2 short sentences",
 "reask": "Moshi asks again later, paraphrased, starting like 'Alright, ...' or 'So, ...'",
 "answer": "the user's spoken answer, short",
 "reaction": "Moshi's short reaction to the answer",
 "mutter": ["2-3 things the user might mutter to themselves while thinking, e.g. partial reasoning or 'hmm'"]}""",
    "B2": """Moshi and the user are mid-conversation; then the user's friend interrupts and they talk briefly about something unrelated.
{"topic": [{"spk": "moshi", "text": ...}, {"spk": "user", "text": ...}, {"spk": "moshi", "text": "Moshi's last line before the interruption, ideally a question"}],
 "side": [6-8 alternating lines between "friend" and "user", mundane logistics or chit-chat, unrelated to the topic, each under 15 words],
 "resume": "Moshi welcomes the user back and picks the topic back up, referring to it specifically",
 "followup": "the user's short reply to the resume line"}""",
    "B3": """The user is deciding between 3 options (sometimes 4) and describes each before asking Moshi's opinion.
{"opener": "Moshi's opening line inviting the user to share",
 "context": "one sentence: what the user is deciding",
 "options": [["option stated as a short noun phrase or action, e.g. 'A used Honda Civic.'", "1-2 sentences of detail"], ...3 or 4 options],
 "comment": "Moshi's opinion that explicitly mentions EVERY option",
 "comment_first_two": "Moshi's opinion mentioning only the first two options",
 "comment_first": "Moshi's opinion on only the first option",
 "followup": "user's short reply"}""",
    "ORD": """An ordinary back-and-forth between Moshi (a friendly voice assistant) and a user.
{"turns": [5 alternating turns starting with moshi: {"spk": "moshi"|"user", "text": ...}; user turns are 8-25 words and at least one contains a comma-separated clause]}""",
    "DECOY": """An ordinary back-and-forth between Moshi and a user, where one user turn naturally uses the word or phrase "{decoy}" WITHOUT asking Moshi to wait or be quiet.
{"turns": [5 alternating turns starting with moshi], "decoy": "{decoy}"}""",
}
DECOYS = ["back", "seconds", "options", "wait", "quiet", "think", "friend", "minute", "describe", "ask me again"]

PROMPT = """Write {n} distinct examples for a spoken-dialogue dataset, as a JSON array.
Each element must follow this schema exactly:
{schema}

Constraints: natural spoken English, contractions welcome, no emojis, no stage directions,
no numerals (spell numbers out), every example about a different situation.
The text will be read aloud by a small text-to-speech model that mispronounces unusual
words, so use only plain, common, easy-to-pronounce English: no foreign or loan words
(e.g. latte, croissant, quinoa, gnocchi, açaí, pho), no unusual brand or product names, no
acronyms or abbreviations, no uncommon place names (stick to well-known cities and
countries), and only common first names. Theme for this batch: {topic}.
Return only the JSON array."""


# Words Pocket TTS is known to garble, plus common loan words. Extend as you find more
# while listening; packs containing any of them are dropped.
HARD_WORDS = {
    "latte", "lattes", "espresso", "cappuccino", "macchiato", "croissant", "croissants",
    "quinoa", "gnocchi", "bruschetta", "pho", "gyro", "gyros", "acai", "açaí", "sriracha",
    "chipotle", "jalapeno", "jalapeño", "quesadilla", "edamame", "kombucha", "matcha",
    "boba", "bougie", "cafe", "café", "facade", "façade", "naive", "naïve", "resume",
    "genre", "entrepreneur", "segue", "hyperbole", "epitome", "alfama", "worcestershire",
}
_ALLOWED_CAPS = {"I", "OK", "TV"}


def is_speakable(p) -> tuple[bool, str]:
    """Reject packs a small TTS is likely to mispronounce."""
    def texts(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for k, v in x.items():
                if k not in ("id", "family", "split", "spk", "decoy"):
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
    return True, ""


def key_text(fam, p):
    if fam == "B1":
        return p["question"]
    if fam == "B2":
        return p["topic"][-1]["text"]
    if fam == "B3":
        return p["context"]
    return p["turns"][1]["text"]


def norm(s):
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="packs/packs.jsonl")
    ap.add_argument("--per-family", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--test-frac", type=float, default=0.15)
    a = ap.parse_args()

    import os

    import anthropic
    env = Path(__file__).resolve().parent.parent / ".env"
    if "ANTHROPIC_API_KEY" not in os.environ and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
    # Talk to the public API directly, whatever the shell environment says.
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], base_url="https://api.anthropic.com")
    out = Path(a.out)
    seen, n_written, rejected = set(), 0, {}
    if out.exists():
        for l in out.read_text().splitlines():
            p = json.loads(l)
            seen.add(norm(key_text("DECOY" if p.get("decoy") else p["family"], p)))
    r = random.Random(0)
    with out.open("a") as f:
        for fam in ["B1", "B2", "B3", "ORD", "DECOY"]:
            got = 0
            while got < a.per_family:
                decoy = r.choice(DECOYS)
                schema = SCHEMAS[fam].replace("{decoy}", decoy)
                msg = client.messages.create(
                    model=a.model, max_tokens=8000,
                    messages=[{"role": "user", "content": PROMPT.format(n=a.batch, schema=schema, topic=r.choice(TOPICS))}])
                txt = msg.content[0].text
                try:
                    items = json.loads(txt[txt.index("["): txt.rindex("]") + 1])
                except ValueError:
                    continue
                for p in items:
                    try:
                        k = norm(key_text(fam, p))
                    except (KeyError, IndexError, TypeError):
                        continue
                    if not k or k in seen:
                        continue
                    ok, why = is_speakable(p)
                    if not ok:
                        rejected[why] = rejected.get(why, 0) + 1
                        continue
                    seen.add(k)
                    h = int(hashlib.sha1(k.encode()).hexdigest()[:8], 16)
                    p["family"] = "ORD" if fam == "DECOY" else fam
                    p["id"] = f"{fam.lower()}_{h:08x}"
                    p["split"] = "test" if (h % 1000) / 1000 < a.test_frac else "train"
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
                    got += 1
                    n_written += 1
                print(f"{fam}: {got}/{a.per_family}", end="\r")
            print()
    print(f"wrote {n_written} packs to {out}")
    if rejected:
        print("rejected as hard to pronounce:", dict(sorted(rejected.items(), key=lambda x: -x[1])[:20]))


if __name__ == "__main__":
    main()
