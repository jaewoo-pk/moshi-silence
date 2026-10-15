"""Hand-written phrase banks, each split into train / test entries.

Test entries never appear in training data. They are what the test_word suite
uses to check generalization to new instruction / cue wording.

Slots: {N} = spoken duration ("ten seconds"), {friend} = friend's name,
{k} = number of options as a word ("three").
"""

# ---------------------------------------------------------------- B1: timed hold
B1_INSTR = {
    "train": [
        "Give me {N} to think, then ask me again.",
        "Hold on, give me {N} to think about it, then ask again.",
        "Let me think for {N}, then ask me again.",
        "Wait, I need {N}. Ask me again after that.",
        "Can you give me {N}? Then repeat the question.",
        "Hmm, give me {N} and then ask me that again.",
        "I need {N} to think. Ask me again after.",
        "Okay, let me think. Give me {N}, then ask again.",
        "Give me {N}, then hit me with the question again.",
        "Pause for {N}, then ask me again.",
        "Hang on, {N}. Then ask me one more time.",
        "Ooh, tough one. Give me {N} to think, then ask again.",
        "Let me mull it over for {N}, then ask me again.",
        "Just give me {N}, and then ask the question again.",
        "Stay quiet for {N} while I think, then ask me again.",
        "Give me {N} to figure it out, then ask me again.",
        "I'll need about {N}. Ask me again after that.",
        "Wait {N}, then repeat the question.",
        "Let me think about that for {N}. Then ask again.",
        "Hold that thought for {N}, then ask me again.",
    ],
    "test": [
        "Could you wait {N} and then ask me that again?",
        "Don't say anything for {N}, then ask me again.",
        "I want to think for {N} first. Ask me again when it's up.",
        "Give me a good {N} on that one, then repeat it.",
        "Let me sit with that for {N}, then ask me once more.",
        "Count {N} in your head, then ask me again.",
    ],
}

# Spoken durations. Train values cover the range; test adds unseen values and
# unseen surface forms ("half a minute" = 30 s, a trained value said a new way).
B1_DURATIONS = {
    "train": {3: "three seconds", 5: "five seconds", 10: "ten seconds",
              15: "fifteen seconds", 20: "twenty seconds", 30: "thirty seconds"},
    "test": {8: "eight seconds", 12: "twelve seconds", 25: "twenty five seconds"},
    # Test-only waiting condition (T-cond): minute-scale holds.
    "test_cond": {30: "half a minute", 60: "a minute"},
}

B1_RESCIND = {
    "train": [
        "Actually, never mind, ask me now.",
        "Oh, wait, I'm ready. Ask me again.",
        "Actually, go ahead and ask me now.",
        "Never mind, I don't need the time. Ask away.",
    ],
    "test": [
        "Scratch that, you can ask me now.",
        "Okay, I'm ready sooner than I thought. Go ahead.",
    ],
}

# B1_double: after the first question is answered, the user asks for a second one.
B1_ANOTHER = {
    "train": [
        "Okay, give me another one.",
        "Nice. Hit me with another one.",
        "Alright, ask me another.",
        "Okay, one more.",
        "Let's do another one.",
        "That was fun. Got another?",
    ],
    "test": [
        "Go on, ask me a different one.",
        "Okay, next question.",
    ],
}

B1_EARLY_PREFIX = {
    "train": ["Oh, wait, I've got it.", "Okay, I know.", "Oh, actually,", "Got it."],
    "test": ["Hang on, I figured it out.", "Oh! I think I know."],
}

# ---------------------------------------------------------------- B2: side conversation
B2_INSTR = {
    "train": [
        "I'm going to talk to my friend for a bit. Stay quiet until I tell you I'm back.",
        "Hold on, {friend} just came in. Don't say anything until I'm back.",
        "I need to talk to {friend} real quick. Stay quiet till I tell you I'm back.",
        "One sec, my friend's here. Please stay quiet until I say I'm back.",
        "Can you hang on? I'm going to chat with {friend}. I'll tell you when I'm back.",
        "Pause for a bit, I'm talking to {friend}. Wait until I say I'm back.",
        "Hey, stay quiet for a moment, {friend} needs me. I'll let you know when I'm back.",
        "Hold that thought. I'm going to talk to {friend}, don't jump in until I'm back.",
        "Sorry, {friend}'s asking me something. Stay quiet till I'm back.",
        "Just a moment, I'm going to talk to my roommate. Don't say anything until I'm back.",
        "Hang on, I have to talk to {friend}. Stay silent until I tell you I'm back.",
        "Wait, my friend wants to talk. Keep quiet until I say I'm back, okay?",
        "Be quiet for a bit, I'm talking to {friend}. I'll say when I'm back.",
        "Can you wait? {friend} is here. Don't talk until I'm back.",
        "Hold on, I need a word with {friend}. Stay quiet until I come back to you.",
        "Okay, {friend} just walked in. Please don't say anything till I'm back.",
        "I'm going to step away to talk to {friend}. Stay quiet until I'm back.",
        "Give me a moment with {friend}. Don't interrupt until I tell you I'm back.",
        "Quick pause, I need to talk to my friend. Stay quiet till I'm back.",
        "Sorry, one second, {friend} needs something. Wait quietly until I'm back.",
    ],
    "test": [
        "Mute yourself for a bit, I'm chatting with {friend}. I'll tell you when I'm back.",
        "Stand by, {friend} and I need to sort something out. Don't speak until I'm back.",
        "Don't respond to anything until I say I'm back. I'm talking to {friend}.",
        "Let me deal with {friend} first. Keep quiet till I'm back with you.",
        "Hey, zip it for a minute, {friend}'s here. I'll tell you when I'm back.",
        "Stay out of this one, I'm talking to {friend}. Wait for me to say I'm back.",
    ],
}

B2_RETURN_CUE = {
    "train": [
        "Okay, I'm back.",
        "Alright, I'm back.",
        "Sorry about that, I'm back.",
        "Okay, sorry, where were we?",
        "I'm back, sorry about that.",
        "Okay, back to you.",
        "Alright, I'm all yours again.",
        "Okay, done. I'm back.",
        "Sorry, I'm back now.",
        "Okay, we can keep going.",
        "Back. Sorry for the wait.",
        "Alright, sorry, let's continue.",
        "Okay, I'm here again.",
        "Sorry about that. Where were we?",
        "Okay, that's sorted. I'm back.",
    ],
    "test": [
        "Okay, you can talk again.",
        "All good, I'm back with you.",
        "Okay, I'm free again. Go on.",
        "Right, all done over here. Go ahead.",
        "Sorry, back now. Continue.",
    ],
}

# Gender-neutral on purpose: the friend's voice is sampled independently of the name.
FRIEND_NAMES = {
    "train": ["Sam", "Alex", "Jordan", "Chris", "Jamie", "Casey"],
    "test": ["Taylor", "Riley", "Morgan"],
}

# ---------------------------------------------------------------- B3: hear me out
NUM_WORD = {2: "two", 3: "three", 4: "four"}

B3_INSTR = {
    "train": [
        "Let me describe all {k} options before you comment.",
        "I'm going to go through {k} options. Don't say anything until I've covered all of them.",
        "Hold your thoughts until I've told you all {k} options.",
        "Let me walk you through the {k} options first, then tell me what you think.",
        "Don't comment yet. Let me list all {k} options first.",
        "I'll describe {k} options. Wait until I'm done with all of them.",
        "Let me get through all {k} before you say anything.",
        "Okay, there are {k} options. Let me finish describing them before you weigh in.",
        "Just listen while I go through the {k} options, then tell me what you think.",
        "Save your opinion until I've described all {k} options.",
        "Let me lay out all {k} options first. No comments until I'm done.",
        "I'm going to describe {k} options, so hang on until I've finished.",
        "Wait until you've heard all {k} options before you comment.",
        "Let me explain all {k} choices before you respond.",
        "Hear all {k} options first, then give me your take.",
    ],
    "test": [
        "Hear me out on all {k} before you jump in.",
        "No comments until I've finished listing all {k}, okay?",
        "Bear with me while I run through {k} options, then react.",
        "I want to get all {k} options out before you say a word.",
        "Hold off until I've gone through every one of the {k}.",
    ],
}

B3_ORDINALS = {
    2: ["The first one is", "And the second one,"],
    3: ["The first one is", "The second option is", "And the third one,"],
    4: ["The first one is", "The second option is", "Option three is", "And the last one,"],
}

B3_COMPLETION = {
    "train": [
        "So those are the {k} options. What do you think?",
        "Okay, that's all of them. Thoughts?",
        "That's everything. Which one would you pick?",
        "So that's all {k}. What would you do?",
        "Okay, I'm done. What do you think?",
        "Those are my options. Which sounds best?",
        "That's the full list. Go ahead.",
        "Alright, that's all {k}. Your turn.",
        "Okay, that covers them. What's your take?",
        "So, those are the choices. What do you think?",
        "That's it, all {k}. Which one?",
        "Okay, that's everything I've got. Thoughts?",
    ],
    "test": [
        "And that's the lot. Weigh in.",
        "Done listing. Tell me what you'd go with.",
        "That's all of them, now you can comment.",
        "Okay, finished. Which is the smart pick?",
    ],
}

B3_EARLY = {
    "train": ["Actually, that's it, there are only two.", "Oh, wait, there are just two. What do you think?"],
    "test": ["Hmm, I guess it's really only two. Go ahead."],
}

B3_RESCIND = {
    "train": [
        "Actually, never mind the rest. What do you think about that first one?",
        "You know what, forget the others. Thoughts on the first?",
    ],
    "test": ["Eh, skip the rest. Is the first one any good?"],
}

# ---------------------------------------------------------------- Moshi acknowledgements (shared)
ACKS = {
    "B1": ["Sure.", "Take your time.", "Sure, take your time.", "Okay.", "No problem.", "Of course."],
    "B2": ["Sure.", "Okay, I'll wait.", "Of course.", "No problem.", "Sure, go ahead."],
    "B3": ["Go ahead.", "Sure, I'm listening.", "Okay.", "Sure.", "Of course, go ahead."],
    "KW": ["Okay.", "Got it.", "Sure."],
}

# Distractors are not phrase-banked: the LLM writes them into each pack in context
# (B2 side-conversation line tags, B2 friend_away, B3 aside, B1 mutter; see gen_packs),
# so they stay coherent and test packs carry unseen distractor wording.

FRIEND_ARRIVAL = {
    "train": ["Hey, do you have a sec?", "Hey! Are you busy?", "Hey, quick question.",
              "Sorry to interrupt, got a minute?"],
    "test": ["Yo, you got a second?", "Hey, can I steal you for a minute?"],
}

# ---------------------------------------------------------------- test-only waiting conditions (T-cond)
KEYWORDS = ["pineapple", "banana", "lighthouse", "marmalade"]
KW_INSTR = [
    "Don't say anything until I say {kw}.",
    "Stay quiet until you hear me say the word {kw}.",
]
KW_RELEASE = ["Okay. {kw_cap}.", "Alright, {kw}. You can talk now.", "{kw_cap}! Go ahead."]

PHONE_INSTR = [
    "I'm taking a call, stay quiet until I'm off the phone.",
    "My phone's ringing. Don't say anything until I'm off.",
]
PHONE_RELEASE = ["Okay, sorry, I'm off the phone.", "Alright, I'm off. Where were we?"]

STORY_INSTR = [
    "Don't interrupt until I finish my story.",
    "Let me tell you the whole story before you say anything.",
]


def bank(d: dict, which: str) -> list:
    """which = "train" or "test" (see splits.SPLITS for which suites use which)."""
    return d[which]
