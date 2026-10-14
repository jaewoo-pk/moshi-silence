"""moshi-finetune yamls for the run matrix (see CLAUDE.md, "Run matrix").

  # phase 1: 1 h, 5 LRs, 20 epochs, checkpoint + dev loss every epoch
  python -m silence.make_configs --phase 1 --train data/train/train_1h.jsonl \\
      --eval data/dev/dev_200.jsonl --moshi-dir /workspace/moshiko --runs /workspace/runs

  # phase 2: 10 h, top 2 LRs from phase 1, E epochs
  python -m silence.make_configs --phase 2 --train data/train/train_10h.jsonl --epochs E \\
      --lrs 5e-5 2e-4 --eval data/dev/dev_200.jsonl --moshi-dir ... --runs ...

Epochs are fixed, not steps: max_steps = ceil(epochs * n / batch), with n counted from the
train jsonl. Checkpoint and eval every round(n / batch) steps (~1 epoch); the step -> epoch
map is written next to each yaml. Fixed for every run: LoRA r=64 scaling 2, batch 16,
duration_sec 60, AdamW wd 0.1, OneCycle pct_start 0.05, first codebook x100, text padding
weight 0.5, mask_main_speech_audio on (needs patches/moshi-finetune-audio-mask.patch), seed 0,
every checkpoint kept (moshi-finetune's default num_ckpt_keep=3 deletes older ones).

Phase 1 also writes R0_devloss.yaml: 2 steps at lr 0 with no checkpoint, i.e. the base
model's dev loss (LoRA B starts at zero), the step-0 point of every loss curve.

Needs all three patches in patches/ (gpu_bootstrap.sh applies them): audio-mask (the
mask_main_speech_audio key; unpatched, the yaml fails to load), eval-reload (unpatched, every
evaluation after the first logs NaN; also logs text / audio dev loss separately) and
text-tail (no text loss on the zero-padded tail of each 60 s window).
"""
import argparse
import json
import math
from pathlib import Path

PHASE1_LRS = [2e-6, 1e-5, 5e-5, 2e-4, 5e-4]
PHASE1_EPOCHS = 20

TEMPLATE = """\
# {name}: {note}
# {n} train episodes, {epochs} epochs = {max_steps} steps at batch {batch};
# checkpoint + dev loss every {every} steps (~{every_ep:.2f} epochs)
data:
  train_data: {train}
  eval_data: {eval}
  shuffle: true

moshi_paths:
  hf_repo_id: kyutai/moshiko-pytorch-bf16
  moshi_path: {moshi_dir}/model.safetensors
  mimi_path: {moshi_dir}/{mimi}
  tokenizer_path: {moshi_dir}/{tokenizer}

full_finetuning: false
lora:
  enable: true
  rank: 64
  scaling: 2.
  ft_embed: false

first_codebook_weight_multiplier: 100.
text_padding_weight: .5
mask_main_speech_audio: true

duration_sec: 60
batch_size: {batch}
max_steps: {max_steps}
gradient_checkpointing: true
optim:
  lr: {lr:.1e}
  weight_decay: 0.1
  pct_start: 0.05

seed: 0
log_freq: 1
eval_freq: {every}
do_eval: true
do_ckpt: {do_ckpt}
ckpt_freq: {every}
num_ckpt_keep: null
save_adapters: true

run_dir: {run_dir}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2], required=True)
    ap.add_argument("--train", required=True, help="train jsonl (absolute wav paths on the GPU box)")
    ap.add_argument("--eval", required=True, help="dev jsonl for the per-epoch validation loss")
    ap.add_argument("--epochs", type=int, help=f"default {PHASE1_EPOCHS} for phase 1; required for phase 2")
    ap.add_argument("--lrs", type=float, nargs="+", help="default: the 5 phase-1 LRs; phase 2: the top 2")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--moshi-dir", required=True, help="local snapshot of kyutai/moshiko-pytorch-bf16 (pinned)")
    ap.add_argument("--mimi", default="tokenizer-e351c8d8-checkpoint125.safetensors")
    ap.add_argument("--tokenizer", default="tokenizer_spm_32k_3.model")
    ap.add_argument("--runs", required=True, help="parent dir for run_dir (must not exist yet per run)")
    ap.add_argument("--out", default="configs")
    a = ap.parse_args()

    epochs = a.epochs or (PHASE1_EPOCHS if a.phase == 1 else None)
    if epochs is None:
        ap.error("phase 2 needs --epochs (E, chosen from phase 1)")
    lrs = a.lrs or (PHASE1_LRS if a.phase == 1 else None)
    if lrs is None:
        ap.error("phase 2 needs --lrs (the top 2 from phase 1)")

    n = sum(1 for l in Path(a.train).read_text().splitlines() if l.strip())
    max_steps = math.ceil(epochs * n / a.batch)
    every = max(1, round(n / a.batch))
    data_tag = Path(a.train).stem.replace("train_", "")
    first = 1 if a.phase == 1 else 6
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    common = dict(n=n, batch=a.batch, train=a.train, eval=a.eval, moshi_dir=a.moshi_dir.rstrip("/"),
                  mimi=a.mimi, tokenizer=a.tokenizer)
    if a.phase == 1:
        # R0's dev loss: LoRA B starts at zero and AdamW at lr 0 changes nothing, so this
        # evaluates the base model exactly (at the end of 2 no-op steps; nothing saved).
        name = "R0_devloss"
        (out / f"{name}.yaml").write_text(TEMPLATE.format(
            name=name, note="base model dev loss (lr 0, no checkpoint)", epochs="—", max_steps=2,
            every=0, every_ep=0.0, lr=0.0, do_ckpt="false", run_dir=f"{a.runs.rstrip('/')}/{name}", **common))
        print(f"{out / name}.yaml  lr=0  2 no-op steps, dev loss at the end")
    for i, lr in enumerate(lrs):
        name = f"R{first + i}_{data_tag}_lr{lr:g}"
        run_dir = f"{a.runs.rstrip('/')}/{name}"
        (out / f"{name}.yaml").write_text(TEMPLATE.format(
            name=name, note=f"phase {a.phase}", epochs=epochs, max_steps=max_steps,
            every=every, every_ep=every * a.batch / n, lr=lr, do_ckpt="true", run_dir=run_dir, **common))
        steps = sorted(set(range(every, max_steps + 1, every)) | {max_steps})
        (out / f"{name}.epochs.json").write_text(json.dumps(
            {"episodes": n, "batch": a.batch, "epochs": epochs, "max_steps": max_steps,
             "checkpoint_epoch": {f"checkpoint_{s:06d}": round(s * a.batch / n, 3) for s in steps}}, indent=1))
        print(f"{out / name}.yaml  lr={lr:g}  {max_steps} steps, {len(steps)} checkpoints")


if __name__ == "__main__":
    main()
