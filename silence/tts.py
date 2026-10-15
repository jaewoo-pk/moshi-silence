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


# ------------------------------------------------------------------ hosted APIs (50 h build)
def _api_key(name):
    import os
    if name in os.environ:
        return os.environ[name]
    for f in (Path("/dev/shm/silence_keys"), Path(__file__).resolve().parent.parent / ".env"):
        if f.exists():
            for line in f.read_text().splitlines():
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise KeyError(f"{name} not set")


def _api_words(text, audio):
    """Hosted engines give no word times. Only user / friend lines go through them, and
    only their onset / offset matter (Moshi's text comes from Pocket), so word starts are
    spread over the speech span in proportion to word length."""
    on, off = speech_span(audio)
    ws = text.split()
    lens = np.array([len(w) + 1 for w in ws], float)
    starts = on + (off - on) * np.concatenate([[0], np.cumsum(lens)[:-1]]) / lens.sum()
    return _words_from_starts(text, list(starts), off)


class FatalEngineError(RuntimeError):
    """Auth / quota / balance errors: every later request would fail too, so stop the engine."""


class _HTTPTTS:
    """Shared request / retry / concurrency logic; subclasses build the request."""
    concurrency = 4
    PAD_S = 0.15  # silence around each clip so assemble.gate's margins and fades fit

    def _request(self, text, voice):
        raise NotImplementedError

    # Plausible speaking rate over the speech span, in s/word (Pocket: median 0.35, p99 0.70).
    # Outside it the engine most likely dropped, repeated or garbled words; retried, then failed.
    RATE_OK = (0.12, 1.0)

    def _check(self, text, raw):
        if raw[:4] == b"RIFF" or raw[:4] == b"ID3\x03" or raw[:1] == b"{":
            raise ValueError(f"not raw PCM (starts {raw[:4]!r})")
        raw = raw[: len(raw) // 2 * 2]
        audio = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
        if len(audio) < SR // 10 or np.abs(audio).max() < 1e-3:
            raise ValueError(f"empty / silent audio ({len(audio)} samples)")
        on, off = speech_span(audio)
        nw = len(text.split())
        rate = (off - on) / nw
        if nw >= 3 and not (self.RATE_OK[0] <= rate <= self.RATE_OK[1]):
            raise ValueError(f"implausible rate {rate:.2f} s/word over {nw} words")
        return audio

    def _one(self, text, voice):
        import http.client
        import time
        import urllib.error
        import urllib.request
        url, headers, body = self._request(text, voice)
        bad, last = 0, None
        for attempt in range(40):
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read()
                audio = self._check(text, raw)
                pad = np.zeros(int(self.PAD_S * SR), np.float32)
                audio = np.concatenate([pad, audio, pad])
                return audio, _api_words(text, audio)
            except ValueError as e:  # bad audio: resample a couple of times, then give up on the line
                bad, last = bad + 1, e
                if bad >= 3:
                    raise
            except urllib.error.HTTPError as e:
                last = e
                if e.code == 429 or e.code >= 500:  # rate limit / overload: wait, honour Retry-After (<= 60 s)
                    ra = e.headers.get("Retry-After") if e.headers else None
                    time.sleep(min(60.0, float(ra)) if ra and ra.replace(".", "").isdigit() else min(20, 1.0 * 1.5 ** min(attempt, 8)))
                    continue
                msg = f"{self.name} {e.code} for {voice!r}: {e.read()[:300]!r}"
                if e.code in (401, 402, 403):  # key / quota / balance: fatal for the whole engine
                    raise FatalEngineError(msg) from e
                raise RuntimeError(msg) from e
            except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as e:
                last = e
                time.sleep(min(20, 1.0 * 1.5 ** min(attempt, 8)))
        raise RuntimeError(f"{self.name}: giving up on {text!r} / {voice}: {last!r}")

    def synth_iter(self, items):
        """Yields ((text, voice), (audio, words) | Exception) as requests complete."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(self.concurrency) as ex:
            futs = {ex.submit(self._one, t, v): (t, v) for t, v in items}
            consecutive = 0
            for f in as_completed(futs):
                try:
                    res = f.result()
                    consecutive = 0
                except FatalEngineError:
                    for g in futs:
                        g.cancel()
                    raise
                except Exception as e:  # noqa: BLE001 - recorded per clip, the build decides
                    consecutive += 1
                    if consecutive >= 25:  # something systemic, not one bad line
                        for g in futs:
                            g.cancel()
                        raise FatalEngineError(f"{self.name}: 25 consecutive failures, last: {e!r}") from e
                    res = e
                yield futs[f], res

    def synth_batch(self, items):
        out = dict(self.synth_iter(items))
        for tv in items:
            if isinstance(out[tv], Exception):
                raise out[tv]
        return [out[tv] for tv in items]


class ElevenLabsTTS(_HTTPTTS):
    """ElevenLabs Flash v2.5, raw 24 kHz PCM. Voice = premade voice id."""
    name = "el"
    concurrency = 3  # Starter plan limit

    def __init__(self, model="eleven_flash_v2_5"):
        self.key, self.model = _api_key("ELEVENLABS_API_KEY"), model

    def _request(self, text, voice):
        return (f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_24000",
                {"xi-api-key": self.key, "Content-Type": "application/json"},
                {"text": text, "model_id": self.model})


class FishTTS(_HTTPTTS):
    """Fish Audio s2-pro, raw 24 kHz PCM. Voice = public model id (reference_id)."""
    name = "fish"
    concurrency = 5  # limit below $100 prepaid

    def __init__(self, model="s2-pro"):
        self.key, self.model = _api_key("FISH_API_KEY"), model

    def _request(self, text, voice):
        return ("https://api.fish.audio/v1/tts",
                {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "model": self.model},
                {"text": text, "reference_id": voice, "format": "pcm", "sample_rate": SR})


def get_backend(name, **kw):
    return {"pocket": PocketTTS, "espeak": EspeakTTS, "kyutai": KyutaiTTS,
            "el": ElevenLabsTTS, "fish": FishTTS}[name](**kw)


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

    def _write(self, t, v, audio, words):
        """wav first, then json, each via a temp file + rename: has() (json exists) implies
        both files are complete, even after a crash or with several writers on one volume."""
        import os
        k = self.key(t, v)
        import uuid
        tmp = self.root / f".{k}.{uuid.uuid4().hex}"
        # Hosted engines deliver 16-bit PCM: store it losslessly as 16-bit FLAC (half the bytes
        # to move to the training volume). Pocket clips stay float wav, as before.
        # SILENCE_CACHE_FLAC=1 does the same for Pocket (clips rendered off the volume and uploaded).
        ext, kw = ("flac", {"subtype": "PCM_16", "format": "FLAC"}) \
            if hasattr(self.backend, "synth_iter") or os.environ.get("SILENCE_CACHE_FLAC") \
            else ("wav", {"subtype": "FLOAT", "format": "WAV"})
        sf.write(f"{tmp}.{ext}", audio, SR, **kw)
        os.replace(f"{tmp}.{ext}", self.root / f"{k}.{ext}")
        Path(f"{tmp}.json").write_text(json.dumps({"text": t, "voice": v, "words": words}))
        os.replace(f"{tmp}.json", self.root / f"{k}.json")

    def render_missing(self, items):
        """Returns the number rendered. Hosted engines (synth_iter) write each clip as it
        arrives and collect failures in self.failed instead of losing the batch."""
        todo = sorted({(t, v) for t, v in items if not self.has(t, v)})
        if hasattr(self.backend, "synth_iter"):
            self.failed = getattr(self, "failed", [])
            for (t, v), res in self.backend.synth_iter(todo):
                if isinstance(res, Exception):
                    self.failed.append((t, v, repr(res)[:300]))
                else:
                    self._write(t, v, *res)
            return len(todo)
        for i0 in range(0, len(todo), 256):
            chunk = todo[i0: i0 + 256]
            for (t, v), (audio, words) in zip(chunk, self.backend.synth_batch(chunk)):
                self._write(t, v, audio, words)
        return len(todo)

    def get(self, text, voice):
        k = self.key(text, voice)
        f = self.root / f"{k}.wav"
        audio, sr = sf.read(f if f.exists() else self.root / f"{k}.flac", dtype="float32")
        assert sr == SR
        return k, audio, json.loads((self.root / f"{k}.json").read_text())["words"]


class _Name:
    def __init__(self, name):
        self.name = name


class MultiCache:
    """Routes (text, voice) to one ClipCache per engine under root/<engine>/ by the voice's
    prefix ("el:", "fish:", none = Pocket). Keys are engine|voice-id|text, so Pocket clips
    keep the keys of the single-engine cache/tts/pocket."""

    def __init__(self, root, backends=None, model_dir="models/pocket-tts"):
        self.root, self.model_dir = Path(root), model_dir
        self.backends = backends or {}
        self.caches = {}

    @staticmethod
    def split(voice):
        from .voices import engine_of
        eng = engine_of(voice)
        return eng, (voice.split(":", 1)[1] if eng != "pocket" else voice)

    def _cache(self, eng, need_backend=False):
        c = self.caches.get(eng)
        if c is None or (need_backend and isinstance(c.backend, _Name)):
            if need_backend:
                b = self.backends.get(eng) or get_backend(eng, **({"model_dir": self.model_dir} if eng == "pocket" else {}))
                self.backends[eng] = b
            else:
                b = _Name(eng)
            c = self.caches[eng] = ClipCache(self.root / eng, b)
        return c

    def key(self, text, voice):
        eng, v = self.split(voice)
        return self._cache(eng).key(text, v)

    def has(self, text, voice):
        eng, v = self.split(voice)
        return self._cache(eng).has(text, v)

    def get(self, text, voice):
        eng, v = self.split(voice)
        return self._cache(eng).get(text, v)

    @property
    def failed(self):
        return [f for c in self.caches.values() for f in getattr(c, "failed", [])]

    def render_missing(self, items):
        by = {}
        for t, voice in items:
            eng, v = self.split(voice)
            by.setdefault(eng, []).append((t, v))
        return sum(self._cache(eng, need_backend=True).render_missing(xs) for eng, xs in by.items())
