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


def load_pool(backend: str) -> dict:
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
