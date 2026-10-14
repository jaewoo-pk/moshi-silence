"""Generate Moshi's side of evaluation episodes, with the script forced outside scored windows.

Runs on the GPU box, inside moshi-finetune's uv env (moshi 061cc4c), from this repo's root:

  PY=/workspace/moshi-finetune/.venv/bin/python
  # alignment self-test first (one episode, base model): must report shift 1 as matching
  $PY -m silence.eval_gen selftest --episodes data/dev/dev_80.jsonl
  # base model (R0) on dev_80
  $PY -m silence.eval_gen run --episodes data/dev/dev_80.jsonl --out /workspace/eval/R0/dev_80
  # every checkpoint of a run + the base model, one process (base weights load once, LoRA
  # swapped per checkpoint; finished ones are skipped on a rerun):
  $PY -m silence.eval_gen run --episodes data/dev/dev_80.jsonl --out /workspace/eval/R3_1h_lr5e-05/dev_80 \
      --ckpt base /workspace/runs/R3_1h_lr5e-05/checkpoints/checkpoint_*
  # then score each output dir:
  $PY -m silence.eval_score /workspace/eval/R3_1h_lr5e-05/dev_80/checkpoint_000070

What happens per episode (see CLAUDE.md "Eval harness"):
- The right channel (user + friend + noise) is Mimi-encoded and fed frame by frame.
- Moshi's own stream (text tokens + 8 audio codebooks) is FORCED to the script everywhere
  except inside the free windows from `eval_score.free_windows` (one per hold, from the end
  of the instruction to the next scripted human turn after the release; ordinary episodes:
  everything after the first human turn). Inside them Moshi generates freely. Re-forcing
  after each window puts the model back on the script, so every hold is scored from a
  scripted context.
- The script's text stream is built with moshi-finetune's own Interleaver from the
  episode's alignments (exactly what training saw); the script audio is the Mimi encoding of
  the left channel.

Alignment with training (checked by `selftest`): training feeds the model, at step t, the
delayed codes of frame t-1 (`LMModel.forward` prepends the initial token), so Moshi's text at
frame t is predicted from user audio up to frame t-1. LMGen writes the user frame given at
step s into the column read at step s, i.e. the slot training fills with frame s-1. So step s
gets user frame s-1 (`--user-shift 1`); the frame given at step 0 is discarded by LMGen anyway.

Outputs in --out: per episode <id>.json (words from the text stream, VAD segments, forced
mask summary, text tokens), codes.npz (Moshi's text + audio codes per episode, so audio can be
re-decoded later), meta.json, and with --save-wav a stereo wav per episode (left = generated
Moshi, right = the episode's user channel) for listening.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .eval_score import FR, free_windows, vad_segments, words_from_text_tokens

SR = 24000
W = Path("/workspace")
DEFAULTS = dict(moshi_dir=W / "moshiko", finetune=W / "moshi-finetune")


# ---------------------------------------------------------------- loading
def load_models(moshi_dir, ckpt=None, device="cuda", fuse_lora=True):
    from moshi.models import loaders
    moshi_dir = Path(moshi_dir)
    kw = dict(moshi_weights=moshi_dir / loaders.MOSHI_NAME, mimi_weights=moshi_dir / loaders.MIMI_NAME,
              tokenizer=moshi_dir / loaders.TEXT_TOKENIZER_NAME)
    if ckpt:
        cons = Path(ckpt) / "consolidated"
        cfg = json.loads((cons / "config.json").read_text())
        assert cfg.get("lora") and cfg.get("lora_rank"), f"{cons}/config.json has no LoRA settings"
        kw.update(config_path=cons / "config.json", lora_weights=cons / "lora.safetensors")
    # The base repo has no config.json: moshi falls back to its built-in 7B config (as training did).
    info = loaders.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16", **kw)
    mimi = info.get_mimi(device=device)
    mimi.set_num_codebooks(8)
    lm = info.get_moshi(device=device, dtype=torch.bfloat16, fuse_lora=fuse_lora)
    lm.eval()
    tok = info.get_text_tokenizer()
    return mimi, lm, tok


def make_interleaver(finetune_dir, tok, lm):
    sys.path.insert(0, str(Path(finetune_dir).resolve()))
    from finetune.data.interleaver import Interleaver  # the trainer's own code
    return Interleaver(tok, FR, lm.text_padding_token_id, lm.end_of_text_padding_id, lm.zero_token_id,
                       keep_main_only=True, device="cpu")


# ---------------------------------------------------------------- episodes
def read_jsonl(path):
    path = Path(path)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        p = Path(r["path"])
        if not p.is_absolute():
            p = (path.parent / p).resolve()
        out.append(p)
    return out


def read_wav(path):
    from scipy.io import wavfile
    sr, x = wavfile.read(path)
    assert sr == SR, (path, sr)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    return x.astype(np.float32)


def write_wav(path, x):
    from scipy.io import wavfile
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def prepare(wav_paths, mimi, il, device, th=None):
    """Per-batch tensors: user codes, script text + audio codes, force mask; padded to T_max."""
    eps = []
    for p in wav_paths:
        d = json.loads(p.with_suffix(".json").read_text())
        x = read_wav(p)
        dur = len(x) / SR
        eps.append(dict(id=p.stem, wav=x, dur=dur, events=d["events"], alignments=d["alignments"]))
    n_max = max(len(e["wav"]) for e in eps)
    n_max = int(math.ceil(n_max / 1920) * 1920)
    B = len(eps)
    audio = np.zeros((B, 2, n_max), np.float32)
    for i, e in enumerate(eps):
        audio[i, :, : len(e["wav"])] = e["wav"].T
    with torch.no_grad():
        a = torch.from_numpy(audio).to(device)
        moshi_codes = mimi.encode(a[:, 0:1])  # [B, 8, T]
        user_codes = mimi.encode(a[:, 1:2])
    T = moshi_codes.shape[-1]
    text = torch.full((B, T), 3, dtype=torch.long)  # PAD; frames past an episode's end are free anyway
    force = torch.zeros((B, T), dtype=torch.bool)
    for i, e in enumerate(eps):
        al = [(w, tuple(ts), spk) for w, ts, spk in e["alignments"]]
        tt = il.prepare_item(al, e["dur"]).view(-1)  # ceil(dur * 12.5) frames
        n = min(len(tt), T)
        text[i, :n] = tt[:n]
        e["T"] = int(math.ceil(e["dur"] * FR))
        e["windows"] = free_windows(e["events"], e["dur"]) if th is None else free_windows(e["events"], e["dur"], th)
        f = torch.ones(T, dtype=torch.bool)
        for w0, w1 in e["windows"]:
            f[int(math.ceil(w0 * FR)): int(math.floor(w1 * FR))] = False
        f[e["T"]:] = False
        force[i] = f
    text[text < 0] = 3
    return eps, user_codes, moshi_codes, text.to(device), force.to(device)


# ---------------------------------------------------------------- generation
class ForcedLMGen:
    """LMGen (moshi 061cc4c) with per-element, per-step forcing of Moshi's own stream.

    Mirrors LMGen.step, except that (a) the sampled text token is replaced by the script's
    where forced, BEFORE the depformer runs, so forced audio is conditioned on forced text as
    in training; (b) the depformer's audio codes are replaced by the script's where forced;
    (c) it can return the text logits (selftest)."""

    def __init__(self, lm, temp=0.8, temp_text=0.7, top_k=250, top_k_text=25):
        from moshi.models.lm import LMGen
        self.gen = LMGen(lm, temp=temp, temp_text=temp_text, top_k=top_k, top_k_text=top_k_text)
        self.lm = lm
        self.delays = lm.delays
        self.dep_q = lm.dep_q
        assert self.delays[: self.dep_q + 1] == [0, 0] + [1] * (self.dep_q - 1), self.delays
        assert max(self.delays) == 1

    @torch.no_grad()
    def step(self, user, force_text=None, text_script=None, force_a0=None, a0_script=None,
             force_a17=None, a17_script=None, want_logits=False):
        """user [B, 8]; force_* [B] bool; text_script [B]; a0_script [B]; a17_script [B, 7].
        Returns (text [B], audio [B, 8], text_logits or None). audio[:, 0] is Moshi's cb0 for
        this frame; audio[:, 1:] are cb1-7 for the PREVIOUS frame (acoustic delay 1)."""
        from moshi.utils.sampling import sample_token
        g, lm = self.gen, self.lm
        st = g._streaming_state
        CT = st.cache.shape[2]
        for q in range(user.shape[1]):
            k = self.dep_q + 1 + q
            st.cache[:, k, (st.offset + self.delays[k]) % CT] = user[:, q]
        pos = st.offset % CT
        for k, delay in enumerate(self.delays):
            if st.offset <= delay:
                st.cache[:, k, pos] = st.initial[:, k, 0]
        inp = st.cache[:, :, pos: pos + 1]
        transformer_out, text_logits = st.graphed_main(inp, st.condition_sum)
        text = sample_token(text_logits.float(), g.use_sampling, g.temp_text, g.top_k_text)[:, 0, 0]
        if force_text is not None:
            text = torch.where(force_text, text_script, text)
        audio = st.graphed_depth(text, transformer_out)
        if force_a0 is not None:
            audio[:, 0] = torch.where(force_a0, a0_script, audio[:, 0])
        if force_a17 is not None:
            audio[:, 1:] = torch.where(force_a17[:, None], a17_script, audio[:, 1:])
        st.offset += 1
        pos = st.offset % CT
        st.cache[:, 0, pos] = text
        st.cache[:, 1: self.dep_q + 1, pos] = audio
        return text, audio, (text_logits[:, 0, 0].float().clone() if want_logits else None)


def generate(fg, user_codes, moshi_codes, text_script, force, user_shift=1, want_logits=False):
    """Run T+1 steps. Returns text [B, T], audio [B, 8, T] (undelayed), logits [B, T, V] or None."""
    B, _, T = user_codes.shape
    dev = user_codes.device
    text_out = torch.full((B, T), 3, dtype=torch.long, device=dev)
    audio_out = torch.zeros((B, 8, T), dtype=torch.long, device=dev)
    logits = [] if want_logits else None
    no = torch.zeros(B, dtype=torch.bool, device=dev)
    with fg.gen.streaming(B):
        for s in range(T + 1):
            uf = min(max(s - user_shift, 0), T - 1)
            ft = force[:, s] if s < T else no
            fa17 = force[:, s - 1] if s >= 1 else no
            text, audio, lg = fg.step(
                user_codes[:, :, uf],
                force_text=ft, text_script=text_script[:, min(s, T - 1)],
                force_a0=ft, a0_script=moshi_codes[:, 0, min(s, T - 1)],
                force_a17=fa17, a17_script=moshi_codes[:, 1:, max(s - 1, 0)],
                want_logits=want_logits)
            if s < T:
                text_out[:, s] = text
                audio_out[:, 0, s] = audio[:, 0]
                if want_logits:
                    logits.append(lg)
            if s >= 1:
                audio_out[:, 1:, s - 1] = audio[:, 1:]
    return text_out, audio_out, (torch.stack(logits, 1) if want_logits else None)


def set_context(lm, n_frames):
    """Size the KV cache to the batch (default 3000 frames = ~1.6 GB per sequence). With
    capacity >= sequence length nothing is ever evicted, so attention is unchanged."""
    from moshi.modules.transformer import set_attention_context
    set_attention_context(lm.transformer, max(n_frames + 8, 64))


# ---------------------------------------------------------------- commands
def cmd_selftest(a):
    """Does LMGen-with-forcing reproduce the training forward pass? Forces the whole script of
    one episode, and compares the per-step text logits with LMModel.forward(codes) for user
    shifts 0 and 1. The matching shift must give near-identical logits (bf16: small diffs).
    (Checked exactly in fp32 on a tiny random model with moshi 061cc4c: shift 1 reproduces every
    training input column and the logits to 3e-6; shift 0 matches 2.5% of columns.)"""
    dev = a.device
    ck = a.ckpt[0] if a.ckpt and a.ckpt[0] != "base" else None
    mimi, lm, tok = load_models(a.moshi_dir, ck, dev)
    il = make_interleaver(a.finetune, tok, lm)
    paths = read_jsonl(a.episodes)[: 1]
    eps, user, moshi, text, force = prepare(paths, mimi, il, dev)
    force[:] = True
    T = user.shape[-1]
    codes = torch.cat([text[:, None], moshi, user], 1)  # [1, 17, T], training layout
    with torch.no_grad():
        ref = lm.forward(codes).text_logits[0, 0].float()  # [T, V]
    set_context(lm, T)
    fg = ForcedLMGen(lm)
    print(f"episode {eps[0]['id']}: {T} frames")
    res = {}
    for shift in (0, 1):
        torch.manual_seed(0)
        _, _, lg = generate(fg, user, moshi, text, force, user_shift=shift, want_logits=True)
        lg = lg[0]
        ok = torch.isfinite(ref).all(-1)
        diff = (lg[ok] - ref[ok]).abs().max(-1).values
        agree = (lg[ok].argmax(-1) == ref[ok].argmax(-1)).float().mean().item()
        tgt = text[0][ok]
        ce_gen = torch.nn.functional.cross_entropy(lg[ok], tgt).item()
        ce_ref = torch.nn.functional.cross_entropy(ref[ok], tgt).item()
        res[shift] = agree
        print(f"user shift {shift}: argmax agreement {100 * agree:.1f}%, max|dlogit| median {diff.median():.3f} "
              f"p99 {diff.quantile(0.99):.3f}; script text CE gen {ce_gen:.3f} vs training forward {ce_ref:.3f}")
    best = max(res, key=res.get)
    print(f"=> shift {best} matches training" + ("" if best == a.user_shift else
          f"  !! differs from --user-shift {a.user_shift}; fix the default before evaluating"))
    if res[best] < 0.97:
        print("!! agreement < 97%: something else is misaligned; do not trust generation evals")


def swap_lora(lm, ckpt, device):
    """Load a checkpoint's LoRA weights into an (unfused) LoRA model in place; ckpt None = base
    model (lora_B = 0, so W' = W exactly)."""
    from safetensors.torch import load_file
    lora = {n: p for n, p in lm.named_parameters() if ".lora_A." in n or ".lora_B." in n}
    assert lora, "model has no LoRA layers (load it from a LoRA checkpoint with fuse_lora=False)"
    with torch.no_grad():
        if ckpt is None:
            for n, p in lora.items():
                if ".lora_B." in n:
                    p.zero_()
            return
        sd = load_file(Path(ckpt) / "consolidated" / "lora.safetensors", device=str(device))
        extra = sorted(set(sd) - set(lora))
        missing = sorted(set(lora) - set(sd))
        assert not extra and not missing, f"{ckpt}: {len(extra)} unexpected, {len(missing)} missing LoRA keys"
        for n, p in lora.items():
            p.copy_(sd[n].to(p.dtype))


def ckpt_name(c):
    return "base" if c is None else Path(c).name


def cmd_run(a):
    """Evaluate one or more checkpoints ("base" = the unmodified model) on one episode list.
    No --ckpt: the base model, written to --out. With --ckpt (one or more): each to
    --out/<checkpoint_XXXXXX | base>/ (the layout eval_report expects). The base
    weights load once; only the LoRA tensors are swapped per checkpoint (all checkpoints must
    share rank and scaling). Finished checkpoints (meta.json present) are skipped, so an
    interrupted job resumes."""
    dev = a.device
    t0 = time.time()
    ckpts = [None if c == "base" else c for c in (a.ckpt or ["base"])]
    multi = len(ckpts) > 1
    outs = {c: (Path(a.out) / ckpt_name(c) if a.ckpt else Path(a.out)) for c in ckpts}
    todo = [c for c in ckpts if not (outs[c] / "meta.json").exists()]
    if len(todo) < len(ckpts):
        print(f"skipping {len(ckpts) - len(todo)} finished checkpoint(s)")
    if not todo:
        return
    lora_ckpts = [c for c in todo if c is not None]
    if multi and lora_ckpts:
        cfgs = {json.dumps({k: json.loads((Path(c) / "consolidated" / "config.json").read_text())[k]
                            for k in ("lora_rank", "lora_scaling")}) for c in lora_ckpts}
        assert len(cfgs) == 1, f"checkpoints differ in LoRA rank / scaling: {cfgs}"
        mimi, lm, tok = load_models(a.moshi_dir, lora_ckpts[0], dev, fuse_lora=False)
    else:
        mimi, lm, tok = load_models(a.moshi_dir, todo[0], dev)
    il = make_interleaver(a.finetune, tok, lm)
    paths = read_jsonl(a.episodes)
    if a.limit:
        paths = paths[: a.limit]
    # Batches are fixed by the episode list (sorted by length) and seeded per batch, so every
    # checkpoint sees the same sampling noise per episode (common random numbers).
    order = [i for _, i in sorted((p.stat().st_size, i) for i, p in enumerate(paths))]
    batches = []
    for b0 in range(0, len(order), a.batch):
        bt = prepare([paths[i] for i in order[b0: b0 + a.batch]], mimi, il, dev)
        if a.force_all:  # sanity mode: output = Mimi round trip of the script
            bt[4][:] = True
        batches.append((b0, bt))
    fg = ForcedLMGen(lm, temp=a.temp, temp_text=a.temp_text)
    piece = tok.id_to_piece
    print(f"loaded + prepared in {time.time() - t0:.0f} s; {len(todo)} checkpoint(s) x {len(paths)} episodes "
          f"x {a.samples} sample(s), batch {a.batch}", flush=True)

    for c in todo:
        tc = time.time()
        if multi:
            if lora_ckpts:
                swap_lora(lm, c, dev)
            elif c is not None:
                raise AssertionError("unreachable")
        out = outs[c]
        out.mkdir(parents=True, exist_ok=True)
        codes_npz, ids, ms = {}, [], []
        for j in range(a.samples):
            for b0, (eps, user, moshi, text, force) in batches:
                tb = time.time()
                T = user.shape[-1]
                set_context(lm, T)
                torch.manual_seed(a.seed + 1000 * j + b0)
                text_out, audio_out, _ = generate(fg, user, moshi, text, force, user_shift=a.user_shift)
                ms.append(1000 * (time.time() - tb) / (T + 1))
                with torch.no_grad():
                    wav = mimi.decode(audio_out).float().cpu().numpy()[:, 0]  # [B, N]
                for k, e in enumerate(eps):
                    n = int(round(e["dur"] * SR))
                    x = wav[k, :n]
                    eid = e["id"] if a.samples == 1 else f"{e['id']}_s{j}"
                    toks = text_out[k, : e["T"]].tolist()
                    segs = vad_segments(x, SR)
                    rec = {"id": e["id"], "sample": j, "duration": e["dur"], "free": e["windows"],
                           "forced_frames": int(force[k, : e["T"]].sum()), "frames": e["T"],
                           "words": words_from_text_tokens(toks, piece),
                           "segments": [[round(s_, 3), round(t_, 3)] for s_, t_ in segs],
                           "text_tokens": toks}
                    (out / f"{eid}.json").write_text(json.dumps(rec))
                    codes_npz[eid] = np.concatenate([np.array(toks)[None],
                                                     audio_out[k, :, : e["T"]].cpu().numpy()]).astype(np.int16)
                    if a.save_wav:
                        write_wav(out / f"{eid}.wav", np.stack([x, e["wav"][:n, 1]], 1))
                    ids.append(eid)
        np.savez_compressed(out / "codes.npz", **codes_npz)
        meta = {"episodes": str(a.episodes), "episodes_dir": str(paths[0].parent), "ids": ids,
                "ckpt": str(c) if c else "base", "samples": a.samples, "seed": a.seed,
                "temp": a.temp, "temp_text": a.temp_text, "user_shift": a.user_shift, "batch": a.batch,
                "force_all": a.force_all, "ms_per_step": round(float(np.mean(ms)), 1),
                "seconds": round(time.time() - tc, 1)}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))  # written last = "finished"
        print(f"{ckpt_name(c)}: {len(ids)} episodes in {time.time() - tc:.0f} s "
              f"({np.mean(ms):.0f} ms/step) -> {out}", flush=True)
    print(f"all done in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("selftest", "run"):
        p = sub.add_parser(name)
        p.add_argument("--episodes", required=True, help="jsonl of episodes (dev_80.jsonl, test_*/episodes.jsonl)")
        p.add_argument("--ckpt", nargs="*", help="checkpoint_XXXXXX dir(s), or 'base' (default: base model)")
        p.add_argument("--moshi-dir", default=DEFAULTS["moshi_dir"])
        p.add_argument("--finetune", default=DEFAULTS["finetune"], help="moshi-finetune checkout (Interleaver)")
        p.add_argument("--user-shift", type=int, default=1)
        p.add_argument("--device", default="cuda")
    r = sub.choices["run"]
    r.add_argument("--out", required=True)
    r.add_argument("--batch", type=int, default=40)
    r.add_argument("--samples", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--temp", type=float, default=0.8)
    r.add_argument("--temp-text", type=float, default=0.7)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--save-wav", action="store_true")
    r.add_argument("--force-all", action="store_true",
                   help="force the whole script (sanity check: scores must match the oracle)")
    a = ap.parse_args()
    {"selftest": cmd_selftest, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
