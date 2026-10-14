"""Calibrate KyutaiTTS.word_offset_frames against an ASR's word timestamps (GPU box).

  python -m silence.calibrate_offset            # prints the median offset in frames

Renders 20 sentences with Kyutai TTS (offset 0), transcribes them with
whisper_timestamped (the same aligner moshi-finetune's annotate.py uses), matches
words, and reports median(asr_start - tts_start) in 80 ms frames. Put the rounded
value in KyutaiTTS(word_offset_frames=...). Expect something close to -2.
"""
import numpy as np
import whisper_timestamped as whisper

from .tts import FRAME_RATE, KyutaiTTS, SR
from .voices import load_pool

SENTS = [
    "Okay, quick one. What's seventeen times six?",
    "Give me ten seconds to think, then ask me again.",
    "I'm going to talk to my friend. Stay quiet until I tell you I'm back.",
    "Let me describe all three options before you comment.",
    "Welcome back! So, where are you thinking of staying in Lisbon?",
    "The keyboard is great and it's cheaper, but it's heavier.",
    "Probably around seven, if that works for everyone.",
    "Sorry about that, where were we?",
    "I just got back from visiting my sister in Boston.",
    "Kind of Blue, for sure. I've played it like ten times.",
]


def main():
    tts = KyutaiTTS(word_offset_frames=0)
    pool = load_pool("kyutai")
    voices = [pool["moshi"]] + pool["train"][:1]
    items = [(s, v) for s in SENTS for v in voices]
    asr = whisper.load_model("medium", device="cuda")
    diffs = []
    for (text, _v), (audio, words) in zip(items, tts.synth_batch(items)):
        a16 = np.interp(np.arange(0, len(audio), SR / 16000), np.arange(len(audio)), audio).astype(np.float32)
        res = whisper.transcribe(asr, a16, language="en")
        asr_words = [w for seg in res["segments"] for w in seg.get("words", [])]
        for (w, s, _e), aw in zip(words, asr_words):
            if w.strip(".,!?").lower() == aw["text"].strip(".,!?").lower():
                diffs.append(aw["start"] - s)
    d = np.array(diffs)
    print(f"{len(d)} matched words; median offset {np.median(d):+.3f}s = {np.median(d) * FRAME_RATE:+.2f} frames "
          f"(IQR {np.percentile(d, 25):+.3f}..{np.percentile(d, 75):+.3f}s)")


if __name__ == "__main__":
    main()
