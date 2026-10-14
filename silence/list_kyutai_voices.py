"""Write silence/kyutai_voices.json from the real kyutai/tts-voices listing.

  python -m silence.list_kyutai_voices      # needs Hugging Face access

Uses VCTK voices (CC-BY 4.0) for users / friends, split 80/20 by speaker id,
plus one fixed voice for Moshi. Edit the result by hand if you want a different
Moshi voice.
"""
import json
import random
import re
from pathlib import Path

from huggingface_hub import list_repo_files

files = list_repo_files("kyutai/tts-voices")
wavs = sorted(f for f in files if f.startswith("vctk/") and f.endswith(".wav"))
by_spk = {}
for f in wavs:
    m = re.match(r"vctk/(p\d+)", f)
    if m:
        by_spk.setdefault(m.group(1), f)  # one reference clip per speaker
spk = sorted(by_spk)
random.Random(0).shuffle(spk)
moshi_spk, rest = spk[0], spk[1:]
n_test = max(1, len(rest) // 5)
out = {
    "moshi": by_spk[moshi_spk],
    "train": [by_spk[s] for s in rest[n_test:]],
    "test": [by_spk[s] for s in rest[:n_test]],
}
Path(__file__).with_name("kyutai_voices.json").write_text(json.dumps(out, indent=1))
print(f"moshi={out['moshi']}  train={len(out['train'])}  test={len(out['test'])}")
