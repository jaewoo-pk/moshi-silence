"""Live-chat test scripts in the style of the training episodes (silence/specs.py), for chat_gui.

Each script: who says what, where to pause, and what Moshi should do. "train wording" scripts use
phrasings from the train lists in silence/phrasebank.py (in-distribution); "held-out" ones use
the test lists or conditions that only appear in the test suites (generalization). "friend" lines
are meant for a second person, or you in a clearly different voice / from further away.
  python -m silence.chat_scripts     # writes results/chat/test_scripts.md
"""
from pathlib import Path

# line = (speaker, text); speaker: "you", "friend", "pause", "moshi" (what Moshi should do there)
SCRIPTS = [
    # ---------------------------------------------------------------- training wording
    {"id": "B2_train", "group": "train wording", "title": "Side conversation",
     "lines": [("you", "Can you help me plan a birthday dinner for my mom?"),
               ("moshi", "answers"),
               ("you", "Hold on, Sam just came in. Don't say anything until I'm back."),
               ("moshi", "may say a short \"Sure.\", then silence"),
               ("friend", "Hey, do you have a sec?"),
               ("you", "Yeah, what's up?"),
               ("friend", "Are you free to help me move a table later?"),
               ("you", "Sure, after lunch. Okay, I'm done for today anyway."),
               ("pause", "5 s of silence (a stall in the side conversation)"),
               ("you", "Is it heavy, or can the two of us manage?"),
               ("friend", "We can manage. Thanks!"),
               ("you", "Okay, I'm back."),
               ("moshi", "resumes within ~1 s, back on the dinner")],
     "watch": "Silent through the friend's question, \"I'm done for today anyway\" (a return-cue lookalike), "
              "the stall and your question to the friend.",
     "expect": "Best-learned behavior: 96% on time for 2e-4 on dev. Base model talks through it."},
    {"id": "B3_train", "group": "train wording", "title": "Hear me out",
     "lines": [("you", "I'm trying to pick a weekend trip. Let me describe all three options before you comment."),
               ("moshi", "may say \"Go ahead.\", then silence"),
               ("you", "The first one is a cabin by a lake."),
               ("pause", "3 s"),
               ("you", "The second option is a city trip with museums and good food."),
               ("pause", "4 s (a pause between options)"),
               ("you", "That one sounds fun, honestly. And the third one, a quiet beach town."),
               ("you", "So those are the three options. What do you think?"),
               ("moshi", "gives an opinion right after")],
     "watch": "Silent through both pauses and \"that one sounds fun\" (a completion lookalike).",
     "expect": "Learned well at the higher LRs."},
    {"id": "B1_train", "group": "train wording", "title": "Timed hold",
     "lines": [("you", "Quiz me on capital cities."),
               ("moshi", "asks a question"),
               ("you", "Give me ten seconds to think, then ask me again."),
               ("pause", "stay silent and wait"),
               ("moshi", "re-asks after ~10 s")],
     "watch": "Does it re-ask on its own, and when?",
     "expect": "Likely FAILS: all models re-asked on their own only 13–26% of the time on dev. It usually waits for you."},
    {"id": "B1_early", "group": "train wording", "title": "Timed hold, answered early",
     "lines": [("you", "Ask me a hard riddle."),
               ("moshi", "asks a riddle"),
               ("you", "Let me think for twenty seconds, then ask me again."),
               ("pause", "~5 s"),
               ("you", "Oh, wait, I've got it. Is it a shadow?"),
               ("moshi", "responds right away to the answer")],
     "watch": "Responds to the early answer instead of waiting out the 20 s.",
     "expect": "Only 1–2 training examples at 1 h: uncertain."},
    {"id": "B1_rescind", "group": "train wording", "title": "Timed hold, cancelled",
     "lines": [("you", "Quiz me on animals."),
               ("moshi", "asks a question"),
               ("you", "Hold on, give me fifteen seconds to think about it, then ask again."),
               ("pause", "~4 s"),
               ("you", "Actually, never mind, ask me now."),
               ("moshi", "asks again right away")],
     "watch": "Responds to the cancel.",
     "expect": "Uncertain (1 training example at 1 h)."},
    {"id": "B3_rescind", "group": "train wording", "title": "Hear me out, cut short",
     "lines": [("you", "I can't decide what to cook. Hold your thoughts until I've told you all three options."),
               ("you", "The first one is a big pot of chili."),
               ("pause", "2 s"),
               ("you", "Actually, never mind the rest. What do you think about that first one?"),
               ("moshi", "answers about the chili right away")],
     "watch": "Speaks once you cut it short, without waiting for options two and three.",
     "expect": "Few training examples: uncertain."},
    {"id": "ORD_pause", "group": "train wording", "title": "Ordinary, with a mid-turn pause",
     "lines": [("you", "I've been thinking about learning to cook, and I was wondering…"),
               ("pause", "1.5 s mid-sentence"),
               ("you", "…what's a good first dish for a total beginner?"),
               ("moshi", "answers after you finish, not in the pause")],
     "watch": "No barge-in during the pause; answers promptly at the end.",
     "expect": "Fine-tuned (2e-4) answered 98% on dev; base 43%."},
    {"id": "DECOY", "group": "train wording", "title": "Decoy: hold-like words, no instruction",
     "lines": [("you", "I just got back from the gym. It took me ten seconds to realize I forgot my water bottle."),
               ("you", "Anyway, what's a good snack after a workout?"),
               ("moshi", "answers normally")],
     "watch": "\"Just got back\" and \"ten seconds\" must not trigger silence.",
     "expect": "Should answer normally."},
    # ---------------------------------------------------------------- held-out
    {"id": "B2_test", "group": "held-out wording", "title": "Side conversation, new wording",
     "lines": [("you", "What's a good gift for a coworker who's leaving?"),
               ("moshi", "answers"),
               ("you", "Mute yourself for a bit, I'm chatting with Taylor. I'll tell you when I'm back."),
               ("friend", "Yo, you got a second?"),
               ("you", "Sure. What's going on?"),
               ("friend", "Did you sign the card for Dana yet?"),
               ("you", "Not yet, I'll do it after this."),
               ("you", "Okay, you can talk again."),
               ("moshi", "resumes on the gift question")],
     "watch": "Same behavior with an instruction and a return cue it never heard in training.",
     "expect": "The test_word suite measures this; not scored yet."},
    {"id": "B3_test", "group": "held-out wording", "title": "Hear me out, new wording",
     "lines": [("you", "I need a new phone. Hear me out on all three before you jump in."),
               ("you", "The first one is the cheap one with a great battery."),
               ("pause", "3 s"),
               ("you", "The second option is the one with the best camera."),
               ("pause", "3 s"),
               ("you", "And the third one, the small one that fits in any pocket."),
               ("you", "Done listing. Tell me what you'd go with."),
               ("moshi", "gives a pick")],
     "watch": "Waits, then answers the held-out completion cue.",
     "expect": "Unknown (generalization)."},
    {"id": "KW", "group": "new condition", "title": "Keyword release (never trained)",
     "lines": [("you", "Let's play a game. Stay quiet until you hear me say the word pineapple."),
               ("pause", "8 s"),
               ("you", "I'm just going to talk for a bit. The weather's been strange this week."),
               ("pause", "4 s"),
               ("you", "Alright, pineapple. You can talk now."),
               ("moshi", "speaks only after \"pineapple\"")],
     "watch": "Silent through the unrelated talk; speaks after the keyword.",
     "expect": "test_cond condition: partial transfer at best."},
    {"id": "PHONE", "group": "new condition", "title": "Phone call (never trained)",
     "lines": [("you", "My phone's ringing. Don't say anything until I'm off."),
               ("you", "(as if on the phone) Hello? … Yeah, I'm home. … Can you bring milk? … Okay, bye."),
               ("you", "Alright, I'm off. Where were we?"),
               ("moshi", "resumes after \"I'm off\"")],
     "watch": "Silent during your half of the call (which sounds like you talking to it).",
     "expect": "test_cond condition: maybe."},
    {"id": "D4", "group": "new condition", "title": "The friend says \"I'm back\" (never trained)",
     "lines": [("you", "Give me a recipe idea for tonight."),
               ("moshi", "answers"),
               ("you", "One sec, my friend's here. Please stay quiet until I say I'm back."),
               ("friend", "Sorry, just grabbing my keys… okay, I'm back!"),
               ("moshi", "should STAY SILENT: it wasn't you"),
               ("you", "Okay, I'm back."),
               ("moshi", "resumes now")],
     "watch": "Only the user's return cue should release the hold. Needs a second person (or a clearly different voice).",
     "expect": "Likely fails: no training example rewards telling speakers apart."},
]


def to_markdown():
    out = ["# Live-chat test scripts", "",
           "In the style of the training episodes. **you** = you; **friend** = a second person (or you in a clearly "
           "different voice / from further away); *pause* = stay silent; → = what Moshi should do there. "
           "\"train wording\" = phrasings the model was trained on; \"held-out\" / \"new condition\" = never seen in training.",
           ""]
    for s in SCRIPTS:
        out += [f"## {s['title']}  ·  {s['group']}", ""]
        for who, text in s["lines"]:
            if who == "pause":
                out.append(f"- *pause: {text}*")
            elif who == "moshi":
                out.append(f"- → Moshi {text}")
            else:
                out.append(f"- **{who}:** {text}")
        out += ["", f"Watch for: {s['watch']}  ", f"Expect: {s['expect']}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    p = Path("results/chat/test_scripts.md")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(to_markdown())
    print(f"wrote {p} ({len(SCRIPTS)} scripts)")
