"""TTS backends with word timestamps, behind one interface, plus an on-disk cache.

Every backend returns, per (text, voice):
  audio: float32 mono at 24 kHz (Mimi's rate)
  words: [[word, start_s, end_s], ...] relative to the clip start, one entry per
         whitespace-separated word of the input text (punctuation kept).

Backends
  pocket  Kyutai Pocket TTS (~100M params, CPU, 24 kHz) via the pocket-tts-timestamped
          fork, which reads word timestamps off the model's own attention. Default.
  kyutai  kyutai/tts-1.6b-en_fr on a GPU. Higher quality, needs calibration.
  espeak  offline, robotic. Plumbing tests only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 24_000
FRAME_RATE = 12.5


def _speech_bounds(audio, sr=SR, thresh_db=-40.0, win=0.02):
    """First / last time the short-term energy is within thresh_db of the peak."""
    n = max(1, int(win * sr))
    m = len(audio) // n
    if m == 0:
        return 0.0, len(audio) / sr
    e = (audio[: m * n].reshape(m, n) ** 2).mean(1) + 1e-12
    db = 10 * np.log10(e / e.max())
    idx = np.nonzero(db > thresh_db)[0]
    if len(idx) == 0:
        return 0.0, len(audio) / sr
    return idx[0] * win, (idx[-1] + 1) * win


def speech_span(audio, sr=SR, rel_db=-30.0, win=0.01):
    """Onset / offset of actual speech: 10 ms frames within rel_db of the clip's peak.
    -30 dB sits above Pocket TTS's noise floor (~-42 dBFS) but below quiet consonants.

    About 5% of Pocket clips open with up to a second of breath / noise that pokes above
    -30 dB before the first word. So the onset is the first such frame that is followed
    by real speech: a frame above -15 dB within 150 ms, or a dense 100 ms (40% of frames
    above rel_db) reaching -23 dB. The noise never gets louder than about -26 dB except
    for isolated clicks; a short word before a pause ("The ... clock") passes."""
    n = int(win * sr)
    m = len(audio) // n
    if m == 0:
        return 0.0, len(audio) / sr
    e = (audio[: m * n].reshape(m, n) ** 2).mean(1) + 1e-12
    db = 10 * np.log10(e / e.max())
    idx = np.nonzero(db > rel_db)[0]
    if len(idx) == 0:
        return 0.0, len(audio) / sr
    on = next((i for i in idx if (db[i: i + 16] > -15).any()
               or ((db[i: i + 10] > rel_db).mean() >= 0.4 and (db[i: i + 10] > -23).any())), idx[0])
    return on * win, (idx[-1] + 1) * win


def retime(audio, words, tol=0.03):
    """Fix word times that fall before the speech actually starts.

    Pocket TTS's attention timestamps put the first word at ~0 s when the clip opens
    with silence (median error 0.3 s, up to 1 s). Words stamped before the energy onset
    are moved onto it; ends are rebuilt as next-start. Returns (words, onset, offset)."""
    on, off = speech_span(audio)
    w = [[x[0], float(x[1]), float(x[2])] for x in words]
    k = 0
    for x in w:
        if x[1] < on - tol:
            x[1] = on + 0.04 * k
            k += 1
    for i in range(1, len(w)):
        w[i][1] = max(w[i][1], w[i - 1][1] + 0.02)
    for i in range(len(w)):
        w[i][2] = w[i + 1][1] if i + 1 < len(w) else max(off, w[i][1] + 0.1)
    return [[x[0], round(x[1], 3), round(x[2], 3)] for x in w], on, off


def _words_from_starts(text, starts, speech_end):
    """Map whitespace words to start times; end = next start (or speech end)."""
    words = text.split()
    assert len(words) == len(starts), (text, starts)
    out = []
    for i, (w, s) in enumerate(zip(words, starts)):
        e = starts[i + 1] if i + 1 < len(starts) else max(speech_end, s + 0.1)
        out.append([w, round(float(s), 3), round(float(max(e, s + 0.02)), 3)])
    return out


# ------------------------------------------------------------------ espeak-ng (demo)
class EspeakTTS:
    name = "espeak"

    def __init__(self):
        import ctypes

        self.ct = ctypes
        self.lib = ctypes.CDLL("libespeak-ng.so.1")

        class _Id(ctypes.Union):
            _fields_ = [("number", ctypes.c_int), ("name", ctypes.c_char_p), ("string", ctypes.c_char * 8)]

        class _Event(ctypes.Structure):
            _fields_ = [("type", ctypes.c_int), ("unique_identifier", ctypes.c_uint),
                        ("text_position", ctypes.c_int), ("length", ctypes.c_int),
                        ("audio_position", ctypes.c_int), ("sample", ctypes.c_int),
                        ("user_data", ctypes.c_void_p), ("id", _Id)]

        self._Event = _Event
        CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.POINTER(_Event))
        self.lib.espeak_Initialize.restype = ctypes.c_int
        self.src_sr = self.lib.espeak_Initialize(2, 0, None, 0)  # AUDIO_OUTPUT_SYNCHRONOUS
        self._buf, self._events = [], []

        def _cb(wav, n, events):
            if wav and n > 0:
                self._buf.append(np.ctypeslib.as_array(wav, shape=(n,)).copy())
            i = 0
            while events[i].type != 0:  # espeakEVENT_LIST_TERMINATED
                ev = events[i]
                if ev.type == 1:  # espeakEVENT_WORD
                    self._events.append((ev.text_position, ev.length, ev.audio_position))
                i += 1
            return 0

        self._cb = CB(_cb)  # keep a reference
        self.lib.espeak_SetSynthCallback(self._cb)

    def _one(self, text, voice):
        name, pitch, rate = voice.split("|")
        self.lib.espeak_SetVoiceByName(name.encode())
        self.lib.espeak_SetParameter(1, int(rate), 0)   # espeakRATE
        self.lib.espeak_SetParameter(3, int(pitch), 0)  # espeakPITCH
        self._buf, self._events = [], []
        b = text.encode("utf-8")
        self.lib.espeak_Synth(b, len(b) + 1, 0, 1, 0, 1, None, None)  # POS_CHARACTER, espeakCHARS_UTF8
        self.lib.espeak_Synchronize()
        pcm = np.concatenate(self._buf).astype(np.float32) / 32768.0
        audio = resample_poly(pcm, SR, self.src_sr).astype(np.float32)
        # Character offsets of whitespace words; text_position is 1-based.
        spans, pos = [], 0
        for w in text.split():
            s = text.index(w, pos)
            spans.append((s, s + len(w)))
            pos = s + len(w)
        starts = [None] * len(spans)
        for tp, _ln, ms in self._events:
            c = tp - 1
            for j, (a, z) in enumerate(spans):
                if a <= c < z and starts[j] is None:
                    starts[j] = ms / 1000.0
                    break
        # Words without an event (rare): interpolate between neighbours.
        s0, s1 = _speech_bounds(audio)
        known = [i for i, s in enumerate(starts) if s is not None]
        if not known:
            starts = list(np.linspace(s0, s1, len(spans), endpoint=False))
        else:
            for i in range(len(starts)):
                if starts[i] is None:
                    prev = max([k for k in known if k < i], default=None)
                    nxt = min([k for k in known if k > i], default=None)
                    a = starts[prev] if prev is not None else s0
                    z = starts[nxt] if nxt is not None else s1
                    starts[i] = (a + z) / 2
        return audio, _words_from_starts(text, starts, s1)

    def synth_batch(self, items):
        return [self._one(t, v) for t, v in items]


# ------------------------------------------------------------------ Kyutai TTS (GPU)
class KyutaiTTS:
    """Batched Kyutai TTS. Mirrors moshi.models.tts.TTSModel.simple_generate, but also
    returns word timestamps from TTSResult.all_transcripts.

    `word_offset_frames`: transcript steps are on the TTS *text* timeline. Audio is
    decoded from frames[delay_steps:] and the first 2 decoded frames are dropped (as in
    simple_generate), so a word consumed at text step s should start at audio frame
    ~ s - 2. CALIBRATE THIS on the GPU box (silence/calibrate_offset.py) before
    generating at scale.
    """
    name = "kyutai"

    def __init__(self, hf_repo="kyutai/tts-1.6b-en_fr", device="cuda", cfg_coef=2.0,
                 batch_size=32, word_offset_frames=-2, temp=0.6):
        import torch
        from moshi.models.loaders import CheckpointInfo
        from moshi.models.tts import TTSModel

        self.torch = torch
        ci = CheckpointInfo.from_hf_repo(hf_repo)
        self.m = TTSModel.from_checkpoint_info(ci, n_q=32, temp=temp, device=device)
        assert self.m.multi_speaker, "expected a cross-attention (multi-speaker) TTS checkpoint"
        self.batch_size, self.offset = batch_size, word_offset_frames
        if self.m.valid_cfg_conditionings:
            self.cfg_attr = cfg_coef
        else:
            self.m.cfg_coef, self.cfg_attr = cfg_coef, None

    def synth_batch(self, items):
        torch, m, out = self.torch, self.m, []
        for i0 in range(0, len(items), self.batch_size):
            chunk = items[i0: i0 + self.batch_size]
            entries = [m.prepare_script([t], padding_between=1) for t, _ in chunk]
            attrs = [m.make_condition_attributes([m.get_voice_path(v)], cfg_coef=self.cfg_attr) for _, v in chunk]
            res = m.generate(entries, attrs)
            with m.mimi.streaming(len(chunk)), torch.no_grad():
                pcms = [torch.clip(m.mimi.decode(f[:, 1:, :]), -1, 1) for f in res.frames[m.delay_steps:]]
            pcms = pcms[2:]
            for i, (text, _v) in enumerate(chunk):
                end = res.end_steps[i]
                if end is None:
                    raise RuntimeError(f"TTS ran out of budget for: {text!r}")
                audio = torch.cat([p[i, 0] for p in pcms[:end]]).float().cpu().numpy()
                steps = [s for _w, s in res.all_transcripts[i]]
                starts = [max(0.0, (s + self.offset) / FRAME_RATE) for s in steps]
                n_words = len(text.split())
                if len(starts) != n_words:  # TTS splits on whitespace too; guard anyway
                    starts = list(np.interp(np.linspace(0, len(starts) - 1, n_words),
                                            np.arange(len(starts)), starts))
                _, s1 = _speech_bounds(audio)
                out.append((audio, _words_from_starts(text, starts, s1)))
        return out



# ------------------------------------------------------------------ Kyutai Pocket TTS (CPU)
class PocketTTS:
    """Pocket TTS with word timestamps (pocket-tts-timestamped, MIT).

    Loads weights from a local download of kyutai/pocket-tts-without-voice-cloning
    (no HF access needed at run time):
      hf download kyutai/pocket-tts-without-voice-cloning \\
          --include "languages/english_2026-09/*" --local-dir models/pocket-tts
    Voices are the predefined embeddings in .../embeddings/<name>.safetensors.
    """
    name = "pocket"

    def __init__(self, model_dir="models/pocket-tts", language="english_2026-09", temp=None):
        import tempfile

        import yaml
        import pocket_tts_timestamped as ptt
        from pocket_tts_timestamped import TTSModel
        from pocket_tts_timestamped.timestamps.text import _lexical_words

        self._lexical_words = _lexical_words
        d = Path(model_dir).resolve() / "languages" / language
        if not (d / "model.safetensors").exists():
            raise FileNotFoundError(f"{d}/model.safetensors missing; see the PocketTTS docstring")
        cfg = yaml.safe_load((Path(ptt.__file__).parent / "config" / f"{language}.yaml").read_text())
        cfg["weights_path"] = str(d / "model.safetensors")
        cfg["weights_path_without_voice_cloning"] = None
        cfg["flow_lm"]["lookup_table"]["tokenizer_path"] = str(d / "tokenizer.json")
        tmp = Path(tempfile.mkdtemp()) / f"{language}.yaml"
        tmp.write_text(yaml.safe_dump(cfg))
        self.m = TTSModel.load_model(config=tmp, temp=temp)
        assert self.m.sample_rate == SR, self.m.sample_rate
        self.emb_dir = d / "embeddings"
        self._states = {}

    def voices(self):
        return sorted(p.stem for p in self.emb_dir.glob("*.safetensors"))

    def _state(self, voice):
        if voice not in self._states:
            self._states[voice] = self.m.get_state_for_audio_prompt(self.emb_dir / f"{voice}.safetensors")
        return self._states[voice]

    def _one(self, text, voice):
        res = self.m.generate_audio_with_timestamps(self._state(voice), text)
        audio = res.audio.detach().float().cpu().numpy().reshape(-1)
        lex = self._lexical_words(text)  # same word split the fork timestamps
        t_by_idx = {w.word_index: (w.start_time, w.end_time) for w in res.words}
        # Whitespace word -> lexical words inside its character span.
        spans, pos = [], 0
        for w in text.split():
            s = text.index(w, pos)
            spans.append((s, s + len(w)))
            pos = s + len(w)
        starts = []
        for a, z in spans:
            ts = [t_by_idx[lw.word_index] for lw in lex if a <= lw.begin < z and lw.word_index in t_by_idx]
            starts.append(ts[0][0] if ts else None)
        s0, s1 = _speech_bounds(audio)
        known = [i for i, s in enumerate(starts) if s is not None]
        for i in range(len(starts)):
            if starts[i] is None:
                prev = max([k for k in known if k < i], default=None)
                nxt = min([k for k in known if k > i], default=None)
                lo = starts[prev] if prev is not None else s0
                hi = starts[nxt] if nxt is not None else s1
                starts[i] = (lo + hi) / 2
        for i in range(1, len(starts)):  # keep monotone
            starts[i] = max(starts[i], starts[i - 1] + 0.02)
        return audio, _words_from_starts(text, starts, s1)

    def synth_batch(self, items):
        return [self._one(t, v) for t, v in items]


def get_backend(name, **kw):
    return {"pocket": PocketTTS, "espeak": EspeakTTS, "kyutai": KyutaiTTS}[name](**kw)


# ------------------------------------------------------------------ cache
class ClipCache:
    """One wav + one json per unique (backend, voice, text)."""

    def __init__(self, root, backend):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = backend

    def key(self, text, voice):
        return hashlib.sha1(f"{self.backend.name}|{voice}|{text}".encode()).hexdigest()[:16]

    def has(self, text, voice):
        return (self.root / f"{self.key(text, voice)}.json").exists()

    def render_missing(self, items):
        todo = sorted({(t, v) for t, v in items if not self.has(t, v)})
        for i0 in range(0, len(todo), 256):
            chunk = todo[i0: i0 + 256]
            for (t, v), (audio, words) in zip(chunk, self.backend.synth_batch(chunk)):
                k = self.key(t, v)
                sf.write(self.root / f"{k}.wav", audio, SR, subtype="FLOAT")
                (self.root / f"{k}.json").write_text(json.dumps({"text": t, "voice": v, "words": words}))
        return len(todo)

    def get(self, text, voice):
        k = self.key(text, voice)
        audio, sr = sf.read(self.root / f"{k}.wav", dtype="float32")
        assert sr == SR
        return k, audio, json.loads((self.root / f"{k}.json").read_text())["words"]
