"""Voice pools per TTS backend, split by speaker into train / test.

Moshi always uses ONE fixed voice so that fine-tuning doesn't see Moshi's
voice change between conditions.

Pocket voices are the predefined embedding names shipped with the model.
espeak voices are "<language>+<variant>|<pitch>|<rate>". They are only for the
local demo (robotic, but free and offline). Kyutai voices are paths inside the
`kyutai/tts-voices` HF repo; run `python -m silence.list_kyutai_voices` on a
machine with HF access to fill KYUTAI from the real repo listing.
"""
import itertools
import json
from pathlib import Path

_ESPEAK_TRAIN_LANGS = ["en-us", "en-gb", "en-gb-x-rp", "en-us-nyc"]
_ESPEAK_TRAIN_VARIANTS = ["male1", "male3", "male4", "female1", "female2", "female3", "Annie", "Alicia"]
_ESPEAK_TEST_LANGS = ["en-029", "en-gb-scotland"]
_ESPEAK_TEST_VARIANTS = ["male6", "female4", "Linda", "John"]


def _espeak_pool(langs, variants, pitches=(40, 55), rates=(160, 175)):
    return [f"{l}+{v}|{p}|{r}" for l, v, p, r in itertools.product(langs, variants, pitches, rates)]


ESPEAK = {
    "moshi": "en-us+Andy|50|170",
    "train": _espeak_pool(_ESPEAK_TRAIN_LANGS, _ESPEAK_TRAIN_VARIANTS),
    "test": _espeak_pool(_ESPEAK_TEST_LANGS, _ESPEAK_TEST_VARIANTS),
}

_KYUTAI_FILE = Path(__file__).parent / "kyutai_voices.json"

# Pocket TTS predefined English voices. Moshi gets one fixed male voice (Moshiko is
# male); four voices are held out for the T-voice / T-all suites; the rest train.
POCKET = {
    "moshi": "michael",
    "test": ["eve", "george", "jane", "peter_yearsley"],
    "train": ["alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "cosette", "eponine",
              "fantine", "jean", "javert", "marius", "mary", "paul", "stuart_bell", "vera"],
}


# Multi-engine pool for the 50 h build (2026-10-15): user / friend voices come from three
# engines, chosen per speaker with these shares; Moshi stays Pocket "michael" (cache-compatible).
# ElevenLabs: the account's premade voices (the account's one cloned voice is left out).
# Fish: public library voices vetted by title / tags / description as generic (no voice named
# after or cloned from a real person or a fictional character). 2 F + 2 M per engine held out.
ELEVEN = {
    "train": ["CwhRBWXzGAHq8TQ4Fs17", "EXAVITQu4vr4xnSDxMaL", "FGY2WhTYpPnrIDTdsKH5", "IKne3meq5aSn9XLyUdCD",
              "JBFqnCBsd6RMkjVDRZzb", "N2lVS1w4EtoT3dr4eOWO", "SAz9YHcvj6GT2YYXdXww", "SOYHLrjzK2X1ezoPC6cr",
              "TX3LPaxmHKxFdv7VOQHJ", "Xb7hH8MSUJpSbSDYk0k2", "bIHbv24MWmeRgasZH58o", "cgSgspJ2msm6clMCkdW9",
              "cjVigY5qzO86Huf0OWal", "hpp4J3VqNfWAUOO0d1Us", "iP95p4xoKVk53GoZ742B", "nPczCjzI2devNBz1zQrb",
              "pNInz6obpgDQGcFmaJgB"],
    # Lily (F), Matilda (F), Daniel (M), Bill (M)
    "test": ["pFZP5JQG7iQjIQuC4Bku", "XrExE9yKIg1WjnnlVkGX", "onwK4e9ZLuTAKqWW03F9", "pqHfZKP75CvOlQylNhV4"],
}
FISH = {
    "train": [
        # female: Sarah, Laura, Paula, Measured Storyteller, Abby (US), Female Voice, Woman, Middle Aged
        # Woman, Amy (US), calm woman, Yorkshire Woman, african woman, australian women, British woman 1,
        # Old woman, 40 women
        "933563129e564b19a115bedd57b7406a", "e3cd384158934cc9a01029cd7d278634", "c2623f0c075b4492ac367989aee1576f",
        "8906b5268cae414fb9b8d3da6e84413d", "f6a19fe5ab494e1fa51bb1476d583a44", "2a9605eeafe84974b5b20628d42c0060",
        "57d4a7fa3d944ca5b2407f916ce5f4d5", "06366bd48deb46d98e0df6160f798cb3", "001262690f2a4eea84aa764cc536df24",
        "0f2279f5f3894673a675eb85a61e1e2e", "bebd3797a58b4b3d9934bf86776b7d2b", "5d3f36b4a0f84353980c15bdf186fadc",
        "4a49480b66434c538c5bd73198a42f49", "a30df91a10b346b9bcb2512010d2cb44", "8db547ea044f44838bf2f8a039f05b5a",
        "12b82c3a4e874e299e094defd39a986f",
        # male: Ethan, calm storyteller male, Documentary, Sleepless historian, CALM- NORMAL, alex, History
        # Voice, ELITE, WNBA VOICEOVER, horror, BOOK RECORD REGULAR, Voice DL, ton, Analog Horror Voice
        "536d3a5e000945adb7038665781a4aca", "e686ae649ee44f219a108aacba206c1a", "c3958f083c2d407982263c77cfa831e3",
        "beb44e5fac1e4b33a15dfcdcc2a9421d", "0b74ead073f2474a904f69033535b98e", "1d52151a55eb4878a997bd06e816b5f6",
        "fbac16744ffa49a9a51c29b1bf19242a", "d8a1340984ee4b63ad1ffae27a6a4339", "fb7ec16ca51a45a5a4db881244d7990a",
        "ef9c79b62ef34530bf452c0e50e3c260", "f8dfe9c83081432386f143e2fe9767ef", "1936333080804be19655c6749b2ae7b2",
        "b97618c195814c9fb7558ea34093cd28", "502eaf4758c348a6bf0b1686647f8c77"],
    # Woman (38de), old women (762f), Marcus the worm (8bed), Alok (b720)
    "test": ["38de3bb8dbf4469cafa3711183e8a748", "762fca8666d04f9a86fcd81af1c3d50a",
             "8bed0e9b444046e2bf72da4b251d9a1d", "b7204d4e40ef4a548c7c8547b7f73492"],
}
ENGINE_SHARES = {"pocket": 0.40, "el": 0.15, "fish": 0.45}
MULTI = {
    "moshi": "michael",
    "train": {"pocket": POCKET["train"], "el": [f"el:{v}" for v in ELEVEN["train"]], "fish": [f"fish:{v}" for v in FISH["train"]]},
    "test": {"pocket": POCKET["test"], "el": [f"el:{v}" for v in ELEVEN["test"]], "fish": [f"fish:{v}" for v in FISH["test"]]},
}


def engine_of(voice: str) -> str:
    """Voice ids are "el:<id>" / "fish:<id>"; plain names are Pocket voices."""
    return voice.split(":", 1)[0] if ":" in voice else "pocket"


def sample_multi(r, which, k=2):
    """k distinct voices, each from an engine drawn with ENGINE_SHARES."""
    out = []
    while len(out) < k:
        eng = r.choices(list(ENGINE_SHARES), weights=list(ENGINE_SHARES.values()))[0]
        v = r.choice(MULTI[which][eng])
        if v not in out:
            out.append(v)
    return out


def load_pool(backend: str) -> dict:
    if backend == "multi":
        return MULTI
    if backend == "pocket":
        return POCKET
    if backend == "espeak":
        return ESPEAK
    if backend == "kyutai":
        if not _KYUTAI_FILE.exists():
            raise FileNotFoundError(
                f"{_KYUTAI_FILE} missing. Run `python -m silence.list_kyutai_voices` "
                "on a machine with Hugging Face access first.")
        return json.loads(_KYUTAI_FILE.read_text())
    raise ValueError(backend)


def pool_for(backend: str, which: str) -> list[str]:
    """which = "train" or "test" (see splits.SPLITS)."""
    return load_pool(backend)[which]
