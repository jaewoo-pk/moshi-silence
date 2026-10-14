"""What each split draws from. Every test suite changes ONE axis vs. training
(except test_all), so a failure points at its cause.

packs:    content packs (scripts)          phr:      instruction / cue phrasings
voices:   user + friend voices
dense:    every B2 gets a mid-side-conversation stall and every B3 a pause and a
          completion lookalike between options (test_distract also adds friend-says-
          "I'm back" D4 episodes; see build.py)

Distractor wording comes from the packs (written by the LLM in context), so it is
held out together with the scripts.
"""
SPLITS = {
    "train":         dict(packs="train", phr="train", voices="train"),
    "test_script":   dict(packs="test",  phr="train", voices="train"),
    "test_word":     dict(packs="test",  phr="test",  voices="train"),
    "test_voice":    dict(packs="test",  phr="train", voices="test"),
    "test_distract": dict(packs="test",  phr="train", voices="train", dense=True),
    "test_cond":     dict(packs="test",  phr="train", voices="train", cond=True),
    "test_ordinary": dict(packs="test",  phr="train", voices="train"),
    "test_all":      dict(packs="test",  phr="test",  voices="test"),
}
