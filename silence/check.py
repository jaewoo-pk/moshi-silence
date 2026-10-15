"""Sanity checks on built episodes, run through moshi-finetune's own Interleaver.

  python -m silence.check data/train --finetune ../moshi-finetune [--plots]

For every episode:
  1. wav is stereo 24 kHz; every alignment lies inside the audio
  2. Moshi's channel is digitally silent across the hold (after the ack)
  3. release timing: B1 re-ask lands at instr_end + N; cue releases 0.25-1.5 s after the cue
  4. every human turn is audible on the user channel above the noise floor
  5. the trainer's Interleaver turns the alignments into the expected text stream:
     a run of PAD across the hold and EPAD on the frame right before Moshi resumes
  6. no human turn overlaps a Moshi turn (e.g. a B1 filler running into the re-ask)
  7. train episodes fit one 60 s training window
  8. the user speaks first, never Moshi

The real Moshi tokenizer lives on Hugging Face; when it isn't available a stand-in
tokenizer is used (it only changes how many tokens each word becomes, not where
PAD / EPAD go). Pass --spm path/to/tokenizer_spm_32k_3.model to use the real one.
"""
import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .assemble import POST_S, PRE_S

FR = 12.5
PAD, EPAD, ZERO = 3, 0, -1  # moshi LMModel: existing_text_padding_id, existing_text_end_padding_id, zero_token_id


class StandInTokenizer:
    """Implements the 3 sentencepiece calls Interleaver makes. Words -> 4-char pieces."""

    def __init__(self):
        self.vocab, self.inv = {}, {}

    def bos_id(self):
        return 1

    def eos_id(self):
        return 2

    def _ids(self, s):
        if s == "\n":
            return [4]
        out = []
        for w in s.split():
            for j, ch in enumerate(w[i: i + 4] for i in range(0, len(w), 4)):
                p = ("▁" if j == 0 else "") + ch
                if p not in self.vocab:
                    self.vocab[p] = 10 + len(self.vocab)
                    self.inv[self.vocab[p]] = p
                out.append(self.vocab[p])
        return out

    def encode(self, x, **kw):
        return self._ids(x) if isinstance(x, str) else [self._ids(s) for s in x]

    def piece(self, i):
        return self.inv.get(i, str(i))


def load_interleaver(finetune_repo, spm=None):
    sys.path.insert(0, str(Path(finetune_repo).resolve()))
    from finetune.data.interleaver import Interleaver  # the trainer's own code

    if spm:
        import sentencepiece
        tok = sentencepiece.SentencePieceProcessor(model_file=spm)
        tok.piece = tok.id_to_piece
    else:
        tok = StandInTokenizer()
    il = Interleaver(tok, FR, text_padding=PAD, end_of_text_padding=EPAD, zero_padding=ZERO,
                     keep_main_only=True, device="cpu")
    return il, tok


def text_stream(il, alignments, dur):
    al = [(w, tuple(ts), spk) for w, ts, spk in alignments]
    return il.prepare_item(al, dur).view(-1).tolist()


def rle(tokens, tok, a, b):
    """Run-length string of tokens[a:b], e.g. 'PAD×42 EPAD ▁What 's'."""
    out, i = [], a
    while i < b:
        t, j = tokens[i], i
        while j < b and tokens[j] == t:
            j += 1
        name = {PAD: "PAD", EPAD: "EPAD", ZERO: "∅"}.get(t) or tok.piece(t)
        out.append(f"{name}×{j - i}" if j - i > 1 else name)
        i = j
    return " ".join(out)


def check_episode(wav_path, il, tok):
    d = json.loads(wav_path.with_suffix(".json").read_text())
    ev, al = d["events"], d["alignments"]
    audio, sr = sf.read(wav_path, dtype="float32")
    res, ok = {"id": wav_path.stem, "type": ev["type"]}, True
    fails = []

    # 1. format
    dur = len(audio) / sr
    if sr != 24000 or audio.ndim != 2 or audio.shape[1] != 2:
        fails.append(f"format sr={sr} shape={audio.shape}")
    if any(not (0 <= s < e <= dur + 1e-3) for _w, (s, e), _ in al):
        fails.append("alignment outside audio")
    moshi, user = audio[:, 0], audio[:, 1]

    # 2. Moshi silent across each hold (after the ack, if any). Episodes built before
    #    events["holds"] existed carry only the single top-level hold.
    holds = ev.get("holds")
    if holds is None:
        holds = [{"hold": ev["hold"], "release": ev.get("release", {})}] if ev.get("hold") else []
    res["n_holds"] = len(holds)
    hold_starts = []  # per hold: end of the ack (or the instruction), where silence must begin
    for k, h in enumerate(holds):
        hold, pre = h["hold"], "" if k == 0 else f"hold {k + 1}: "
        acks = [t["t"][1] for t in ev["moshi_turns"] if t["role"] == "ack" and hold[0] <= t["t"][0] < hold[1]]
        hold_starts.append(max([hold[0]] + acks))
        # Skip the ack's fade-out and the release turn's fade-in (assemble.gate margins).
        a = hold_starts[-1] + POST_S + 0.02
        b = hold[1] - PRE_S - 0.02
        seg = moshi[int(a * sr): int(b * sr)]
        mx = float(np.abs(seg).max()) if len(seg) else 0.0
        if k == 0:
            res["hold_s"] = round(hold[1] - hold[0], 2)
            res["moshi_max_in_hold"] = mx
        if mx > 1e-3:
            fails.append(f"{pre}Moshi channel not silent in hold (max {mx:.4f})")

    # 2b. alignments agree with the audio: each Moshi turn's first word starts where
    #     Moshi's channel actually becomes loud (catches TTS timestamp drift)
    env_win = int(0.01 * sr)
    m = len(moshi) // env_win
    e = (moshi[: m * env_win].reshape(m, env_win) ** 2).mean(1)
    loud = e > (e.max() * 10 ** (-30 / 10))
    offs = []
    for t in ev["moshi_turns"]:
        a0 = max(0, int((t["t"][0] - 1.0) * 100))
        hits = np.nonzero(loud[a0: int((t["t"][1]) * 100)])[0]
        if len(hits):
            offs.append(float((a0 + hits[0]) / 100 - t["t"][0]))
    if offs:
        res["align_vs_audio_s"] = [round(min(offs), 3), round(max(offs), 3)]
        if max(abs(o) for o in offs) > 0.15:
            fails.append(f"Moshi word times off from audio onset by up to {max(offs, key=abs):+.2f}s")

    # 3. release timing
    for k, h in enumerate(holds):
        rel, pre = h["release"], "" if k == 0 else f"hold {k + 1}: "
        if rel.get("kind") == "timer":
            err = rel["moshi_onset"] - rel["ref"]
            res.setdefault("timer_err_s", round(err, 3))
            if abs(err) > 0.3:
                fails.append(f"{pre}B1 re-ask off by {err:+.2f}s")
        elif rel.get("kind") == "cue":
            lat = rel["moshi_onset"] - rel["ref"]
            res.setdefault("cue_latency_s", round(lat, 3))
            if not (0.25 <= lat <= 1.5):  # moshi_gap is 0.45-1.35 s
                fails.append(f"{pre}cue latency {lat:.2f}s")

    # 4. human turns audible over the noise floor
    quiet = [t for t in ev["human_turns"]]
    gaps = np.ones(len(user), bool)
    for t in quiet:
        gaps[int(t["t"][0] * sr): int(t["t"][1] * sr)] = False
    floor = np.sqrt((user[gaps] ** 2).mean()) if gaps.any() else 1e-6
    snrs = []
    for t in quiet:
        seg = user[int(t["t"][0] * sr): int(t["t"][1] * sr)]
        snrs.append(20 * np.log10(np.sqrt((seg ** 2).mean()) / floor + 1e-9))
    res["min_turn_snr_db"] = round(float(min(snrs)), 1) if snrs else None
    if snrs and min(snrs) < 6:
        fails.append(f"a human turn is only {min(snrs):.1f} dB over the floor")

    # 6. humans and Moshi never talk over each other
    for h in ev["human_turns"]:
        for mt in ev["moshi_turns"]:
            ov = min(h["t"][1], mt["t"][1]) - max(h["t"][0], mt["t"][0])
            if ov > 0.05:
                fails.append(f"{h['spk']} overlaps Moshi by {ov:.2f}s at {max(h['t'][0], mt['t'][0]):.1f}s")

    # 7. one training window
    if d["meta"]["split"] == "train" and dur > 60.0:
        fails.append(f"train episode is {dur:.1f}s (> 60 s window)")

    # 8. the user speaks first, never Moshi
    if ev["moshi_turns"] and ev["human_turns"] and ev["moshi_turns"][0]["t"][0] < ev["human_turns"][0]["t"][0]:
        fails.append("Moshi speaks first")

    # 5. text stream from the trainer's Interleaver
    toks = text_stream(il, al, dur)
    res["frames"] = len(toks)
    res["n_pad"] = sum(t == PAD for t in toks)
    res["n_epad"] = sum(t == EPAD for t in toks)
    for k, (h, start) in enumerate(zip(holds, hold_starts)):
        pre = "" if k == 0 else f"hold {k + 1}: "
        onset = h["release"]["moshi_onset"]
        f_on = next(i for i, t in enumerate(toks) if i >= int(onset * FR) - 1 and t not in (PAD, EPAD))
        f_hold = int(np.ceil(start * FR)) + 1
        epad = toks[f_on - 1] == EPAD
        pads, frames = sum(t == PAD for t in toks[f_hold: f_on]), f_on - f_hold
        if k == 0:
            res["release_frame"] = f_on
            res["epad_before_release"] = epad
            res["pads_in_hold"] = pads
            res["hold_frames"] = frames
            res["stream_at_release"] = rle(toks, tok, max(f_on - 6, 0), min(f_on + 4, len(toks)))
        if not epad:
            fails.append(f"{pre}no EPAD right before the release frame")
        if pads < frames - 1:
            fails.append(f"{pre}non-PAD tokens inside the hold")
    res["fails"] = fails
    res["ok"] = not fails
    return res, toks, d, audio


def plot_episode(png, d, audio, toks, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = d["events"]
    holds = ev.get("holds") or ([{"hold": ev["hold"], "release": ev.get("release", {})}] if ev.get("hold") else [])
    sr = 24000
    t = np.arange(len(audio)) / sr
    fig, axes = plt.subplots(3, 1, figsize=(14, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.35]})
    colors = {"moshi": "#2b6cb0", "user": "#2f855a", "friend": "#b7791f"}
    axes[0].plot(t[::40], audio[::40, 0], lw=0.4, color=colors["moshi"])
    axes[1].plot(t[::40], audio[::40, 1], lw=0.4, color="#777")
    for h in ev["human_turns"]:
        axes[1].axvspan(*h["t"], color=colors[h["spk"]], alpha=0.18, lw=0)
    for m in ev["moshi_turns"]:
        axes[0].text(m["t"][0], 0.9, m["role"] or "", fontsize=7, va="top")
    for ax in axes[:2]:
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        for h in holds:
            ax.axvspan(*h["hold"], color="#e53e3e", alpha=0.07, lw=0)
    axes[0].set_ylabel("Moshi\n(left)", rotation=0, ha="right", va="center", fontsize=8)
    axes[1].set_ylabel("user+friend\n(right)", rotation=0, ha="right", va="center", fontsize=8)
    for dd in ev["distractors"]:
        axes[1].axvspan(*dd["t"], ymin=0.88, ymax=1.0, color="#9b2c2c", alpha=0.8, lw=0)
        axes[1].text(dd["t"][0], -0.95, dd["code"], fontsize=7, color="#9b2c2c")
    for h in holds:
        if h["release"].get("ref") is not None:
            for ax in axes:
                ax.axvline(h["release"]["ref"], color="#e53e3e", lw=0.8, ls="--")
    # text stream strip: grey = PAD, red = EPAD, blue = word tokens
    ft = np.arange(len(toks)) / FR
    cols = ["#e2e8f0" if x == PAD else "#e53e3e" if x == EPAD else "#2b6cb0" for x in toks]
    axes[2].bar(ft, 1, width=1 / FR, color=cols, align="edge", lw=0)
    axes[2].set_yticks([])
    axes[2].set_ylabel("text\nstream", rotation=0, ha="right", va="center", fontsize=8)
    axes[2].set_xlabel("seconds")
    status = "OK" if res["ok"] else "FAIL: " + "; ".join(res["fails"])
    fig.suptitle(f"{res['id']}  ·  {res['type']}  ·  {status}", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


def write_index(out_dir, results):
    rows = []
    for r, d in results:
        ev = d["events"]
        script = "".join(
            f"<div class='{t['spk']}'><b>{t['spk']}</b> <span class=ts>{t['t'][0]:.1f}s</span> {html.escape(t['text'])}</div>"
            for t in sorted([{**m, 'spk': 'moshi'} for m in ev['moshi_turns']] + ev['human_turns'], key=lambda x: x['t'][0]))
        facts = ", ".join(f"{k}={v}" for k, v in r.items() if k in
                          ("hold_s", "timer_err_s", "cue_latency_s", "align_vs_audio_s", "pads_in_hold", "hold_frames", "epad_before_release", "min_turn_snr_db"))
        rows.append(f"""<section><h2>{r['id']} <small>{r['type']} · {'✅' if r['ok'] else '❌ ' + html.escape('; '.join(r['fails']))}</small></h2>
<audio controls preload="none" src="episodes/{r['id']}.wav"></audio>
<img src="plots/{r['id']}.png"><p class=facts>{facts}<br><code>{html.escape(r.get('stream_at_release', ''))}</code></p>
<details><summary>script</summary>{script}</details></section>""")
    (out_dir / "index.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Episodes</title>
<style>body{{font:14px system-ui;max-width:1200px;margin:24px auto;padding:0 16px}}img{{width:100%}}audio{{width:100%}}
section{{border-top:1px solid #ddd;padding:12px 0}}small{{color:#666;font-weight:normal}}.facts{{color:#444;font-size:12px}}
.moshi{{color:#2b6cb0}}.user{{color:#2f855a}}.friend{{color:#b7791f}}.ts{{color:#999;font-size:11px}}</style>
<h1>Episodes</h1><p>Stereo: <b>left = Moshi</b>, <b>right = user + friend + noise</b>. Use headphones.
Red band = hold (Moshi must stay silent). Dashed line = release reference (cue end, or instr_end + N). Dark red ticks = distractors.
Bottom strip = Moshi's text stream as built by moshi-finetune's Interleaver: grey PAD, red EPAD, blue word tokens.</p>
{''.join(rows)}""")


_WK = {}


def _init_worker(finetune, spm):
    _WK["il"], _WK["tok"] = load_interleaver(finetune, spm)


def _check_one(w):
    r, toks, d, audio = check_episode(w, _WK["il"], _WK["tok"])
    return r, None, d, None  # toks / audio only needed for plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--finetune", default="../moshi-finetune", help="path to a kyutai-labs/moshi-finetune checkout")
    ap.add_argument("--spm", default=None)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (no --plots)")
    a = ap.parse_args()
    out = Path(a.data_dir)
    wavs = sorted((out / "episodes").glob("*.wav"))
    results = []
    if a.plots:
        (out / "plots").mkdir(exist_ok=True)
    if a.workers > 1 and not a.plots:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.workers, _init_worker, (a.finetune, a.spm)) as pool:
            checked = pool.imap(_check_one, wavs, chunksize=8)
            checked = list(checked)
    else:
        il, tok = load_interleaver(a.finetune, a.spm)
        checked = (check_episode(w, il, tok) for w in wavs)
    for w, res in zip(wavs, checked):
        r, toks, d, audio = res
        results.append((r, d))
        flag = "ok  " if r["ok"] else "FAIL"
        extra = "" if r["ok"] else "  <- " + "; ".join(r["fails"])
        print(f"{flag} {r['id']:28s} {r['type']:14s} hold={r.get('hold_s', 0):5.1f}s "
              f"pads={r.get('pads_in_hold', '-')!s:>4}/{r.get('hold_frames', '-')!s:<4} "
              f"epad={r.get('epad_before_release', '-')!s:5} "
              f"timer={r.get('timer_err_s', '')!s:6} cue={r.get('cue_latency_s', '')!s:6} "
              f"align={r.get('align_vs_audio_s', '')!s:16}{extra}")
        if a.plots:
            plot_episode(out / "plots" / f"{w.stem}.png", d, audio, toks, r)
    (out / "checks.json").write_text(json.dumps([r for r, _ in results], indent=1))
    if a.plots:
        write_index(out, results)
    n_bad = sum(not r["ok"] for r, _ in results)
    print(f"\n{len(results) - n_bad}/{len(results)} episodes pass")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
