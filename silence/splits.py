"""What each split draws from. Every test suite changes ONE axis vs. training
(except test_all), so a failure points at its cause.

packs:    content packs (scripts)          phr:      instruction / cue phrasings
voices:   user + friend voices             distract: distractor lexicon
"""
SPLITS = {
    "train":         dict(packs="train", phr="train", voices="train", distract="train"),
    "test_script":   dict(packs="test",  phr="train", voices="train", distract="train"),
    "test_word":     dict(packs="test",  phr="test",  voices="train", distract="train"),
    "test_voice":    dict(packs="test",  phr="train", voices="test",  distract="train"),
    "test_distract": dict(packs="test",  phr="train", voices="train", distract="test", dense=True),
    "test_cond":     dict(packs="test",  phr="train", voices="train", distract="train", cond=True),
    "test_ordinary": dict(packs="test",  phr="train", voices="train", distract="train"),
    "test_all":      dict(packs="test",  phr="test",  voices="test",  distract="test"),
}
