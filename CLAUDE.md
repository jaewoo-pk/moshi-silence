# CLAUDE.md: moshi-silence

Project memory for a one-day research sprint (started 2026-10-14). Read this fully before
touching code or data. Last full review: 2026-10-14, evening (after the second pre-launch
review and the eval harness).

## The task

Investigate **how much synthetic fine-tuning data it takes to teach Moshi when to stay
silent and when to speak again**. Use a public duplex checkpoint (`kyutai/moshiko-pytorch-bf16`)
and Kyutai's fine-tuning code (https://github.com/kyutai-labs/moshi-finetune), with public
or self-generated data. The brief, condensed:

1. Concrete behaviors: "Give me ten seconds to think, then ask me again." / "I'm going to
   talk to my friend. Stay quiet until I tell you I'm back." / "Let me describe all three
   options before you comment."
2. The interesting question: does it learn the underlying instruction and generalize to
   new wording and conditions?
3. Synthetic conversations: TTS, varied speakers, realistic pauses. Both humans on the
   incoming stream, silence on Moshi's output while it should wait, then an appropriate
   response. Include ordinary conversations so it keeps responding normally. Silence must
   come from the model itself (no external mute rule).
4. Measure learning vs. data size and learning rate. Baseline the unmodified model; start
   with 1 h and 10 h and a small LR sweep; extend promising settings to 100 h; 1,000 h is a
   stretch goal. Every comparison starts from the same checkpoint. Track unique examples,
   hours, and training exposure separately (repeating examples or adding longer silences
   must not look like more data).
5. Test generalization incl. failure cases: unseen scripts, voices, and some waiting
   conditions held out; a small human-recorded test set; check interruptions during the
   requested silence, resume timing, and that ordinary requests still get answered.
   Distractors: questions addressed to the other person, pauses before the conversation is
   finished, phrases resembling the return cue. Evaluate generated audio; listen to
   representative successes and failures.
6. Deliverable: short experimental report + demo. Behavior vs. data size, LR effect,
   reproducible code, a few audio examples, what generalized, what failed, next experiment.
   Optional if time allows: a "talk about trees more often" experiment as a second, simpler
   test of how fast fine-tuning changes behavior.

## Status at a glance (2026-10-14, evening)

| Piece | State |
|---|---|
| Generator (`silence/`) | Working. Demo 14/14 pass check.py. |
| Content packs | Done: 2,500 (`packs/packs.jsonl`), 348 screened test packs. **v2 (2026-10-15):** + 5,470 new train packs, reverse-screened against dev/test → `packs/packs_v2_screened.jsonl` = 7,664 packs (see "50 h train set") |
| Train set `data/train` | Done: 1,150 episodes / 10.49 h, all pass check.py; `train_1h` = 113 eps, `train_10h` = 1,101 eps |
| 50 h train set `data50/train50` (volume only) | **Done 2026-10-15 07:14 UTC**: 5,585 episodes / 51.24 h (12 check.py failures + 3 assembly drops removed), nested `train_1h` / `_10h` / `_25h` / `_50h` jsonls (abs paths) + `train_all.jsonl`; 17 GB. v2 mix, 3 TTS engines. Built on the CPUs of the EPAD-10h pods + the Mac; both pods terminated 07:14. Cost ≈ $30 (packs ≈ $20, Fish $9.39, ElevenLabs within plan). **R22** (50 h, ×10, lr 5e-5, 345 steps = 1 epoch, 2026-10-15) did not beat R21 (10 h, same compute) on dev or the test suites: more talkative, 2–6 points lower; confounded by the v2 recipe: `results/run50h/summary.md`. Details: "50 h train set" below, `docs/training_record.md` §12, metadata `results/build50/` |
| Dev set `data/dev` | Done: 200 episodes / 1.97 h (`dev_80` ⊂ `dev_200`), all pass, no overlap with train |
| Test suites `data/test_*` | Done: 7 × 90 = 630 episodes / 6.1 h, all pass check.py, no script overlap with train |
| moshi-finetune patches | 3 patches, tested to stack on upstream main `2acc879f` |
| Training configs + GPU bootstrap | Written; pre-launch review done 2026-10-14 (configs load with the real `TrainArgs`, patches apply to pristine upstream, data audited). First GPU attempt (2026-10-14, ~7:50 pm ET, wrong personal account, terminated) **failed at `uv sync`** (see "Traps": openai-whisper / pkg_resources). **Second attempt on the API-key account: bootstrap complete (2026-10-14, ~8:20 pm ET).** Env (two-step sync fix), weights, data, configs on volume `np7c8l94dh`; SMOKE OK (2 checkpoints, no NaN, both dev evals logged); eval selftest + R0 benchmark OK. Setup pod terminated |
| GPU account | **Use the RunPod API-key account, never the user's personal one** (see step 2). Current resources there (2026-10-15 07:15 UTC): volume `np7c8l94dh` (`moshi-silence`, **250 GB** Standard, US-NE-1; 139 GB used at 06:10 UTC, plus since then the 17 GB 50 h set, its clip cache and the R20/R21 checkpoints — measure with `du` before the next run); **no pods running.** Phase 2 ran on 4× pod `sq9r1dzsa8l6rc` (02:33–04:08 UTC, ≈ $25, terminated); its 84 checkpoints are on the volume (`results/phase2/setup/checkpoints_manifest.txt`). Earlier: 4× pod `z77njns1kqy38r` ran phase 1 (00:20–01:52 UTC, ≈ $25) and is terminated; setup pod `ldkqt7xnpy9pxe` terminated. The volume holds all 126 phase-1 LoRA checkpoints (`results/phase1/setup/checkpoints_manifest.txt`: sizes + sha256) and all eval outputs. The volume holds ~32 GB (env, weights, data, configs, `runs/_smoke`, `eval/_bench`). The personal-account pod and volume are deleted |
| Eval harness | `eval_gen.py` / `eval_score.py` / `eval_report.py` / `scripts/eval_dev.sh` built and CPU-tested; **on the 7B (H100, 2026-10-14): selftest passes (100% identical input columns), R0 benchmark ran** (see "Eval harness"). Whisper + LLM judge for final tests not built |
| Human-recorded test set | **Recorded + evaluated 2026-10-15.** Solo, speaker s01, 20 min (`silence/record.py`); `silence/human.py` builds test_human_tts (24 TTS twins) / test_human (24, verbatim) / test_human_free (16 holds, own-words instruction + cue), all pass check.py. Eval (3 samples, pod `odf3sua1adban4` 10:59–11:18 UTC ≈ $1.30, terminated): score TTS → human → own words: R21 ep 2.2 0.83 → 0.79 → 0.68 (16 holds: 0.81 / 0.75 / 0.68), R22 0.79 → 0.74 → 0.67, base 0.41 → 0.33 → 0.28. Human voice alone: −0.04 / −0.05 (CIs include 0); own words − TTS: R21 −0.14 [−0.25, −0.02]. Losses mostly in cue releases (R21 cue spoke 91% → 73% → 67%) and silence with own-words instructions. `results/human/summary.md`, `scripts/test_human.sh` |
| Phase 1 (1 h LR sweep) | **Done 2026-10-15 01:50 UTC.** 6 LRs × 21 checkpoints, all scored on dev_80 (see `docs/training_record.md` §10). Behavior improves monotonically with LR (2e-4 best: plateau 11% violated / 74% on time / 98% answered vs base 86 / 40 / 43); dev loss ranks LRs ~backwards. Copied to `results/phase1/` |
| Phase 2 (10 h) | **Done 2026-10-15 04:07 UTC** (user's choice: all four LRs 2e-5 / 5e-5 / 2e-4 / 5e-4 at 10 h, E = 5, checkpoints every ¼ epoch). Best 10 h LR is 2e-5–5e-5 (plateau score 0.84–0.85); 2e-4 slightly worse; 5e-4 collapses into a silent model. 10 h holds silence better (96–98%) but doesn't speak again better (66–74%) than 1 h. Plateau ≈ 70 optimizer steps at both sizes. Timed holds still unlearned. Results in `results/phase2/`, `docs/training_record.md` §10 |
| Live chat | Chat pod `vc42i3dtvzv7s5` (05:25–~06:30 UTC 2026-10-15, 1× H100, ≈ $4.30) **terminated at the user's request** (ssh alias `chat` removed). It had slot A = 10 h 2e-5 ep 4.2 (`R8_10h_lr2e-05/checkpoint_000289`), slot B = 1 h 5e-5 EPAD×10 ep 5.9 (`R17_1h_lr5e-05_epad10/checkpoint_000042`). (Earlier chat pod `ywpfqsbqerh53c` ran 02:40–03:07, ≈ $1.80.) To restart: create a 1× H100 pod on the volume, add ssh alias `chat`, run `python -m silence.chat_gui`, press Load in both slots (the GUI writes `/root/chat.sh` + `load.sh`). `/root/chat.sh <port> <ckpt dir|base>` runs `moshi.server` (env on the volume, `--lora-weight` + `--config-path`) on 127.0.0.1:8998 (slot A) / 8999 (slot B); `/root/load.sh` swaps a slot. The GUI keeps the tunnel. Cheaper GPUs in US-NE-1 are Blackwell, which the torch 2.6 env can't run; a fresh moshi 0.2.13 install (supports LoRA) would work anywhere. The Mac (M5, 24 GB) would need 4-bit MLX (`moshi_mlx`); not pursued. |
| Disk incident (2026-10-15 04:54 UTC) | The 150 GB volume filled up (checkpoints alone were 104 GB; ~0.37 GB each) while the EPAD sweep saved step 133: all four runs crashed at step 132–133 of 142, keeping 18 complete checkpoints each (epochs 1–17.8; the partial `checkpoint_000133` temp dirs were deleted). Freed uv-cache (7.6 GB) + `runs/_smoke`; **volume grown to 250 GB** (user's suggestion, ~$0.07/GB-month). **Check free space before launching runs**: `du -sh /workspace/*` (df shows the whole cluster, not our quota) vs the volume size |
| Silence-bias finding + EPAD sweep | The training signal is silence-heavy by construction: ~90% of text frames are PAD, the "start talking" frame (EPAD) is 0.6% of frames and ~0.6% of the text-loss weight (it shares text_padding_weight 0.5 with PAD), and the audio loss covers only Moshi's silent frames (speech audio masked). **EPAD sweep done** (pod `7f5yuy4acmhxzp`, 04:19–05:16 UTC, ≈ $15, terminated): R16–R19 = 1 h, lr 5e-5, EPAD ×3/×10/×30/×100, baseline R3; runs truncated at epoch 17.8 by the disk incident. At the plateau EPAD trades silence for speaking (score flat ≈ 0.80); mid-training (ep 4–9) timed-hold re-asks reach 60–80% but decay later; best balanced checkpoint ×10 ep 6 (0.86: 81% silent / 91% speaks / 61% timed re-ask). Next: moderate EPAD with a shorter schedule or on 10 h. Results in `results/epad/` (incl. `analysis/epad_sweep.png`), `docs/training_record.md` §10. |
| EPAD at 10 h | **Done 2026-10-15** (training 05:33–06:49, dev_80 evals until 07:12 UTC) on two 1× H100 pods, `086j0wex48cmlp` (R20_10h_lr5e-05_epad3, EPAD ×3) and `xk9kzxl2eev9of` (R21_10h_lr5e-05_epad10, ×10); 10 h, lr 5e-5, 5 epochs, ckpt every 17 steps; baseline R9. Both pods terminated 07:14 UTC (ssh aliases `tr3` / `tr10` removed); ≈ 1.7 h × 2 × $3.99 ≈ $13.4. Plateau score (dev_80) 0.87 (×3) / 0.88 (×10) vs 0.85 for R9; ×10 speaks 94% / silent 81% / timed re-ask 73%; best ckpts R20 and R21 `checkpoint_000153` (ep 2.2) at 0.91, within noise: confirm on dev_200. Lost silence is in timed holds (mutter reactions up to 53%). Results: `results/epad10h/summary.md`. Their CPUs also built the 50 h set (≈ 8% training slowdown) |
| Test suites (held out) | **First results 2026-10-15** (1 sample, 7 × 90 episodes, pod `i1wn60oi6h7n8g` 07:29–09:31 UTC, ≈ $8.1 incl. R22, terminated): base / R17 1 h ×10 / R21 10 h ×10 / R22 50 h ×10. 10 h ×10 best on 6 of 7 suites (test_script 0.97, test_all 0.84, test_distract 0.98 incl. the friend's "I'm back" ignored 18/18); weak spots: timed holds with new wording / durations, and new conditions ("a minute" holds fail, keyword/phone releases missed). `results/test/summary.md`; scorer `results/test/analysis/score_tests.py`; `scripts/test_suites.sh` **Live chat (user, ~10:05 UTC): R22 (50 h) felt much better than R21 in real conversation; R21 a bit quiet** — the reverse of the TTS-voiced suites (they never test a real voice; R22 trained on 63 voices / 3 engines). Logged in `docs/training_record.md` §9 + `results/chat/notes.jsonl`; the human-recorded test set is the way to settle it |
| Next decisions (user) | (1) checkpoint-selection rule (proposed: equal-weight mean of 1−violated, on time, ordinary answered, on dev_200; ties → earlier epoch); (2) whether to run 5e-4 to bracket the LR; (3) phase-2 LRs and E. Then dev_200 for the candidates (needs a pod) |
| Report, deliverable demo | Not started |
| Demo listening pass | Not recorded anywhere; ask the user whether it happened |

**Git:**
- Local repo, no remote.
- `master` has only the first 2 commits.
- **Branch `worktree-docs-progress` is the complete, current state**: a snapshot of the main
  checkout's (then uncommitted) pipeline code, plus everything built since (dev/test
  tooling, configs, bootstrap, patches, generated packs).
- The main checkout's working copy holds the same pipeline code as uncommitted changes, plus
  the built `data/`.
- Neither `master` nor the main checkout's working copy has everything
  `scripts/gpu_bootstrap.sh` needs (no `scripts/`, 1 of 3 patches, no `make_configs.py`).
  **Upload the code from the worktree** (`.claude/worktrees/docs-progress`, which has no
  `.env` and no `data/`) **and `data/` from the main checkout**: two rsyncs, commands in the
  `scripts/gpu_bootstrap.sh` header. No git operation needed. (Checking the branch out in the
  main checkout fails: the branch is already checked out in the worktree.)

**Only one session at a time may write to `data/`.** Two sessions once extended `data/train`
at the same moment (`--n 1110` / `--n 1150`); the manifest had to be restored.

## Next steps, in order

1. ~~Build and check the test suites~~ (done, see "Test suites").
2. ~~Rent the GPU~~ (volume + setup done 2026-10-14; the 4× pod is next, step 4). **Agreed plan** (user, 2026-10-14):
   - RunPod **Secure Cloud, on-demand** (never spot: `train.py` can't resume). **$3.99/hr per
     H100 SXM** (live catalog price, 2026-10-14; the earlier ~$3 was a guess), so 4× = $15.96/hr.
   - Create a **network volume ≥ 150 GB** first, mounted at `/workspace` (Standard, ~$0.07/GB
     per month). Volume `np7c8l94dh` is now 250 GB (≈ $0.58/day).
   - **1× H100 SXM pod for setup**, with SSH over exposed TCP so rsync works.
   - Then a **4× H100 SXM pod on the same volume** for the training burst and the phase-1
     generation evals.
   - Terminate each pod as soon as its work is done (idle 4× costs ~$16/hr); everything that
     matters is on the volume.

   **Driving RunPod from Claude Code.** RunPod's hosted MCP server is registered at user scope.
   **Since 2026-10-14 it authenticates with a RunPod API key** (`claude mcp add --transport
   http runpod -s user https://mcp.getrunpod.io/ --header "Authorization: Bearer <key>"`; the
   key sits in plain text in `~/.claude.json`). It was OAuth on the user's **personal**
   account before. The user doesn't want to pay for this project personally: **all
   pods and volumes go on the API-key account**. If `list-pods` shows the empty personal
   account, or anything else looks off, stop and ask.
   - **The API-key account is shared:** it already holds other people's pods and volumes
     (e.g. SSH key `rfu@Ryans-MacBook-Pro`). **Never stop, modify or delete anything this
     project didn't create**, and don't edit the account's SSH keys. The user's key
     (`~/.ssh/id_ed25519.pub`) is **not** registered there, so pass it as the pod's
     `PUBLIC_KEY` env var on create (startSsh alone would install only the account's keys).
   - In the first session on that account, Claude Code's auto-mode classifier blocked a
     RunPod catalog read ("Containment Escape") right after the old pod was terminated from
     inside with its own key (below). The user may need a permission rule for `mcp__runpod__*`.
   - Plan: check H100 SXM stock per data center with `count: 4`; pick one where **4×** is
     available and that offers Standard network volumes (2026-10-14: US-NE-1, US-MO-1,
     EUR-IS-3 qualified, all LOW). Create the 150 GB volume there; deploy the 1× pod
     (Secure Cloud, on-demand, template `runpod-torch-v280` = image
     `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, `gpu.minCudaVersion` 12.8, the volume
     at `/workspace`, ports `22/tcp` only, `startJupyter: false`).
   - `ssh.direct` shows up in `get-pod` ~20 s after RUNNING. Write a `box` alias into
     `~/.ssh/config` from it (not the `ssh.runpod.io` proxy: it can't rsync), run the uploads
     and the bootstrap (step 3), watch `/workspace/bootstrap.log`, report smoke / selftest /
     bench results. Remove the `box` alias when the pod is gone: its IP:port gets reused.
   - **Before every billable action (volume, each pod) state the spec and price and get the
     user's OK.** Terminate pods when their work is done; the volume keeps the state.
   - The Mac's `rsync` is macOS openrsync (protocol 29); it works with the pod's GNU rsync.
   - Inside a pod, `/proc/1/environ` has a pod-scoped `RUNPOD_API_KEY` and `RUNPOD_POD_ID`
     (a non-interactive ssh shell doesn't export them), and the image has `runpodctl`.
     `runpodctl remove pod $RUNPOD_POD_ID` from inside terminates the pod. This was used once
     (2026-10-14), at the user's request, for a pod the MCP could no longer see.
3. ~~**On the 1× pod**~~ (done 2026-10-14; results under "Compute" and "Eval harness"): the two rsyncs (worktree code, main-checkout `data/`, ~6.2 GB; commands in
   the `scripts/gpu_bootstrap.sh` header). Use `rsync -rlpt`, not `-a`: the network volume
   (MooseFS) refuses chown, so `-a` copies everything but exits 23. The image already has
   rsync. Then `bash scripts/gpu_bootstrap.sh`. Steps 1–2 (env, weights) don't need `data/`
   and can run while it uploads; step 3 rewrites jsonls in `data/`, so start the full script
   only after the upload finishes (it's safe to rerun). Everything (uv, its Python 3.12, the env, weights,
   configs) lands on the volume. Its step 6 runs the eval selftest + a 40-episode R0
   generation benchmark. Note s/step and peak memory (`smoke.log`) and ms/step
   (`eval_bench.log`) and update the compute table below. Terminate the pod.
4. **Next: on a 4× H100 SXM pod** (same volume, US-NE-1; $15.96/hr; get the user's OK; pass
   `PUBLIC_KEY`; new `box` alias): `cd /workspace/moshi-silence && mkdir -p /workspace/logs &&
   nohup bash scripts/run_parallel.sh configs/R0_devloss.yaml
   configs/R?_1h_*.yaml > /workspace/logs/parallel.log 2>&1 &`. That's one run per GPU,
   with the next queued config starting as a GPU frees up: R0–R3 start together, R4 takes
   R0's GPU a few minutes later, R5 starts when the first of R1–R3 finishes.
5. **Phase-1 generation evals** as soon as runs finish, on the same 4× pod:
   `nohup bash scripts/eval_dev.sh dev_80 R0 R1_1h_lr2e-06 R2_1h_lr1e-05 R3_1h_lr5e-05
   R4_1h_lr0.0002 R5_1h_lr0.0005 > /workspace/logs/eval_dev.log 2>&1 &`, then
   `eval_report` (see "Eval harness"). Stop the 4× pod when the evals are done. Then choose E
   and the top 2 LRs; phase 2 (R8–R9), test suites on the final checkpoints, listening log, report.
   **Change (user, 2026-10-15 00:51 UTC):** R5 (5e-4) dropped before it started, because R3/R4
   memorized the 1 h scripts (train text loss → 0.01; dev text loss best at epoch 3–4, then
   doubling). Added R6_1h_lr5e-06 and R7_1h_lr2e-05 (same 20-epoch recipe; `make_configs
   --first 6 --no-base --lrs 5e-6 2e-5`). Phase 2 is now R8–R9 (`make_configs` default
   `--first 8`). The original scheduler was killed by hand (R1–R4 kept running);
   `results/phase1/setup/phase1b.sh` runs R6/R7 and the dev_80 evals so no GPU idles.
6. Human-recorded test set: label each episode with the same `events` schema.

## Environment on this Mac

- `.venv/` (uv, Python 3.12): torch 2.9.1, moshi 0.2.13, pocket-tts-timestamped 1.2.0,
  anthropic 1.14.0, sphn, sentencepiece, matplotlib. `source .venv/bin/activate`.
- `../moshi-finetune/`: clone of kyutai-labs/moshi-finetune on local branch
  `silence-audio-mask`, i.e. upstream main `2acc879f` plus the audio-mask patch as uncommitted
  changes. check.py imports its Interleaver. Training happens on a rented GPU, not here.
- `models/pocket-tts/languages/english_2026-09/` (372 MB): Pocket TTS weights from the
  ungated `kyutai/pocket-tts-without-voice-cloning`. They load from disk, so no HF access is
  needed at run time.
- `.env`: `ANTHROPIC_API_KEY` (gitignored). Never print or log it.
- `cache/tts/pocket/`: rendered clips, keyed by `sha1(backend|voice|text)[:16]`. They're
  stored raw; retime and gate run at assembly. Rebuilds reuse them.
- Pocket TTS throughput: **~12× realtime with 4 workers** (up to 17× on small extensions).
  The 10 h build was 7,661 clips ≈ 7.8 h of speech in 37.6 min.
- `--tts espeak` loads `libespeak-ng.so.1`, which is Linux-only, so it fails on this Mac.
- The Moshi weights (~15 GB) aren't here; the GPU bootstrap downloads them.
- `silero-vad` 6.2.3 is installed in `.venv` (for `eval_score --oracle`); it isn't in
  `requirements.txt`.
- **Temporary, will vanish with the background job that made them:** the CPU eval tests
  (`test_eval_gen.py`, `test_eval_pipeline.py`, `test_eval_swap.py`, `test_selftest_cpu.py`,
  `test_report.sh`, `test_eval_dev.sh`, `test_parallel.sh`), a scratch venv with moshi 061cc4c
  + torch 2.6.0 (`m061venv`), and the Mimi weights + tokenizer (`hfmoshi/`), all under
  `~/.claude/jobs/35d2ba60/tmp/`. Move them into the repo (e.g. `tests/`) if they should be
  kept; they hard-code those paths.

## Data: what's built and how

All builds use seed 0. Episodes are stereo 24 kHz PCM_16 wavs (**left = Moshi, right = user +
friend + noise**), each with a JSON (`alignments`, `events`, `meta`) and a `.spec.json`, plus
`manifest.jsonl`, `episodes.jsonl` and `checks.json` per set. `python -m silence.transcript
<ep.json>` prints an episode as a timed script.

### Content packs (`packs/`)

- `packs.jsonl`: 2,500 packs, 500 per family (B1, B2, B3, ORD, DECOY), written by
  `claude-sonnet-5-5`. That took 322 API calls (~0.69M input / ~0.81M output tokens).
  - DECOY packs are stored as `family: "ORD"` with a `decoy` field.
  - `split` is decided by hash: `sha1(key_text) % 1000 < 150` → test. That gave 387 test
    packs: B1 94, B2 75, B3 61, ORD 79, DECOY 78.
- 620 candidates were rejected, logged with reasons in `rejects.jsonl`:
  - schema checks: no D3a 206; D3a on the last line 40; decoy word missing 26; unanswered
    question 16
  - too long for 57 s: B3 106, B2 99
  - near-duplicates (content-word Jaccard ≥ 0.6 within a family): 93
  - a few speakability failures
- `smoke.jsonl`: the 40-pack smoke test.
- `gen_packs.log`: the run log.
- Top-up: rerun with the same `--out`. Each family is topped up to `--per-family`. The
  topic/style sequence is `random.Random(0)` and replays on every run, so only the avoid
  list differs.
- `packs_test_screened.jsonl`: the 348 test packs the test suites are built from (see below).

### Train (`data/train`)

`python -m silence.build train --packs packs/packs.jsonl --n 1150 --out data/train`

- `--n 1000` reached only 9.07 h (mean episode 32.6 s, not ~36 s), so it was extended to 1150
  with the same seed. Types and packs come out in a fixed order, so the first 1,000 episodes
  are byte-identical.
- `python -m silence.make_splits data/train --hours 1 10` writes first-K prefixes
  (1 h ⊂ 10 h) and `ledger.json`:

| Subset | Episodes | Packs | TTS clips | Hours | Speech h | Moshi speech h | Hold-silence h |
|---|---|---|---|---|---|---|---|
| train_1h | 113 | 116 | 885 | 1.01 | 0.70 | 0.25 | 0.39 |
| train_10h | 1,101 | 1,135 | 8,463 | 10.01 | 7.05 | 2.48 | 3.73 |
| full build | 1,150 | 1,185 | 8,851 | 10.49 | 7.36 | 2.59 | 3.92 |

- `train_1h` by type: B1 std 17 / double 3 / early 1 / rescind 1; B2 24; B3 std 18 / early 2 /
  rescind 2; ORD 31; DECOY 14. That's only 1–2 contrast episodes per type at 1 h. It's a
  random draw from TRAIN_MIX; say so in the report.
- No pack is used twice in train.
- Logs: `data/build_train.log` (first 1,000) and `data/build_train_1150.log`.

### Dev (`data/dev`), for model selection only

`python -m silence.build_dev --exclude data/train --out data/dev`

- 200 episodes / 1.97 h. `dev_80.jsonl` (0.77 h) is the first 80 of `dev_200.jsonl`.
- Exactly balanced: 20 / 50 per family (B1, B2, B3, ORD+DECOY), each family split by its
  TRAIN_MIX sub-mix, families round-robin.
- Train distribution (train voices, phrasings, conditions) on **unseen scripts**. Episodes
  are ≤ 60 s.
- **No overlap with train** (the user's requirement):
  - It excludes every pack used anywhere in `data/train`, including B1 `pack2`.
  - A screen also drops candidates that share any pack string with a training pack, or are
    near-duplicates (Jaccard ≥ 0.5). That dropped 200 of 928 candidates. It's stricter than
    needed, because it compares raw pack strings.
- `check_overlap` result: 0 shared packs or episodes; closest script Jaccard 0.47 (median
  0.22); 1 shared pack line ("Okay."). 407 phrase-bank line instances (137 distinct) are
  shared by design.
- Use: validation loss on all 200 every epoch. Generation eval on `dev_80` in phase 1 and on
  all 200 in phase 2 and for the final checkpoint choice. Never trained on, never reported
  as the result.
- `episodes.jsonl` says `split: "dev"`; each episode's `meta.split` says `"train"` (the
  phrase / voice pools it used).
- **Trap:** `build.py train` doesn't know about dev. Extending train past 1,150 with the same
  seed draws exactly the leftover packs dev took. Any future train build must skip the dev
  packs (`meta.pack` / `pack2` in `data/dev`): pass `build.py --exclude data/dev` (added
  2026-10-15; reads `pack` / `pack2` from each set's `episodes.jsonl`).

### Test suites (`data/test_*`), reported once on final checkpoints

```
python -m silence.screen_packs --split test --against data/train data/dev --out packs/packs_test_screened.jsonl
for s in test_script test_word test_voice test_distract test_cond test_ordinary test_all; do
  python -m silence.build $s --packs packs/packs_test_screened.jsonl --n 90 --out data/$s; done
```

- The screen drops test packs containing a line **actually spoken** in train/dev episodes,
  or near-duplicating a train/dev pack (Jaccard ≥ 0.5). It dropped 39 of 387, leaving B1 81,
  B2 60, B3 51, ORD 79, DECOY 77. It compares spoken lines, not raw pack strings, because
  e.g. `friend_away` "Okay, I'm back" lines are spoken only in test-only D4 episodes.

| Suite | Packs | Wording | Voices | Mix / extra |
|---|---|---|---|---|
| test_script | test | train | train | TRAIN_MIX |
| test_word | test | **test** | train | TRAIN_MIX |
| test_voice | test | train | **test** | TRAIN_MIX |
| test_distract | test | train | train | B2_std 50 / B2_D4 20 / B3_std 30; dense: every B2 gets a D2a stall, every B3 a D2b pause + D3b |
| test_cond | test | train | train | COND_keyword 35 / COND_phone 35 / B1_minutes 30 |
| test_ordinary | test | train | train | ORD 70 / DECOY 30 |
| test_all | test | **test** | **test** | TRAIN_MIX; no dense distractors, D4 or new conditions |

- Test episodes have no 60 s cap; B1 "a minute" runs past 60 s. That's fine for generation
  eval, since the context is 240 s.
- **The suites are NOT paired** (checked 2026-10-14). `PackSampler` seeds on
  `packs:{seed}:test`, but it uses one RNG for all families and shuffles each family's queue
  lazily, in the order the suite's own type sequence (`types:{seed}:{split}`) first asks for
  it. So each suite draws its own episode types and its own pack order. test_script /
  test_word / test_voice / test_all share only 42–68 of their 90 packs pairwise, and their
  type mixes differ (B1: 6 / 11 / 14 / 14 episodes). Differences between suites therefore
  mix the held-out axis with script and type-mix differences. Comparisons between models on
  the same suite are fully paired.
- test_word and test_all use the test B1 durations (8 / 12 / 25 s, never trained), so B1
  timing there confounds new wording with new N.
- test_distract and test_cond need ~63 B2 packs from 60. The sampler silently reshuffles and
  reuses (`uses` is counted but never printed), so ~3 B2 scripts repeat in each.
- test_word holds out the **user's** wording. Moshi's acks ("Sure.", "Okay, I'll wait.") come
  from a shared list, and a couple of B3 option lines coincide ("The first one is taking the
  train."). That's expected.
- **Built and checked (2026-10-14, ~6 pm):** all 630 episodes pass check.py.

| Suite | Hours | Disk | Test packs | Closest train script (Jaccard) | Phrase-bank lines shared with train | Pack lines shared |
|---|---|---|---|---|---|---|
| test_script | 0.80 | 266 MB | 92 | 0.44 | 144 (82 distinct) | 0 |
| test_word | 0.82 | 271 MB | 92 | 0.43 | 43 (11; Moshi acks + 2 B3 option lines) | 0 |
| test_voice | 0.83 | 275 MB | 93 | 0.43 | 172 (94) | 1 ("Sure.") |
| test_distract | 1.15 | 382 MB | 89 | 0.43 | 262 (91) | 1 (D4 friend "Okay, I'm back.", by design) |
| test_cond | 1.14 | 378 MB | 90 | 0.43 | 60 (8) | 0 |
| test_ordinary | 0.54 | 179 MB | 90 | 0.43 | 0 | 0 |
| test_all | 0.82 | 271 MB | 95 | 0.43 | 40 (12) | 1 ("Take your time.") |

  0 shared packs or episodes with train in every suite. test_voice and test_all use only the
  4 test voices; the others use train voices. `data/` totals 6.2 GB (train 3.6, dev 0.65,
  tests 2.0).
- Pocket TTS logged "Maximum generation length reached without EOS" twice (test_script,
  test_distract). A duration scan of all 14,581 distinct clips used in train/dev/test found
  no run-on clip: median 0.35 s/word, p99 0.70, max 1.76 (one-word acks, raw clips before
  gating).

### 50 h train set (`data50/train50` on the volume), built 2026-10-15 ~06:05–07:20 UTC

Goal: data scaling beyond 10 h, with more voice diversity; the user asked for it in under
~1 h wall clock. It lives only on the volume (`/workspace/data50/train50`, clip cache
`/workspace/cache50`), not in the Mac's `data/`. **Result:** 5,585 episodes / 51.24 h (35.9 h speech,
13.2 h Moshi speech, 17.6 h hold silence), 5,799 packs, 40,921 clips; dropped 1 (failed Fish line) + 2
(over 60 s) at assembly and 12 check.py failures (7 Moshi word-time offsets of ~−1 s, 6 of them
B1_double: look into it before the next build). Subsets `train_{1,10,25,50}h.jsonl` + `train_all.jsonl`.
Metadata copied to `results/build50/`. Numbers + costs: `docs/training_record.md` §12.

- **Packs:** `packs/packs_v2.jsonl` = `packs.jsonl` (2,500) + 5,470 new Claude Sonnet 5.5
  packs (`gen_packs --seed 1`, then `--seed 2`; `--test-frac 0 --per-family-json` B1 2120 /
  B2 1580 / B3 1240 / ORD 2250 / DECOY 780; ≈ 822 calls, ≈ $20 est.; log
  `packs/gen_packs_v2.log`, rejects `packs/rejects_v2.jsonl`). The seed-1 run crashed at call
  578 on a StopIteration inside validation (now caught as "schema: malformed"); resumed.
- **Reverse screen** (`silence/screen_train_packs.py`): drops a NEW train pack that contains a
  line spoken in `data/dev` or any of the 7 test suites, or whose key text has Jaccard ≥ 0.5
  with a test pack or a dev-used pack (same family). Dropped 306 (shared line + similar: B2
  172 + 4, B1 15 + 75, B3 13 + 18, ORD 2 + 7), kept 5,164 → `packs/packs_v2_screened.jsonl`
  (7,664 packs). Verified: 0 planned lines contain a held-out (test) phrase-bank wording.
  The exact-line rule is strict on short lines ("Okay.").
- **Mix** `--mix v2` = `specs.TRAIN_MIX_V2`: B1 25 (std 17, double 4, early 2, rescind 2), B2
  20, B3 15 (std 11, early 2, rescind 2), ORD 30, DECOY 10; `specs.MUTTER_P` 0.7 (was 0.5).
  Reason: timed holds were the unlearned behavior.
- **Voices** `--tts multi` (`voices.MULTI`; engine drawn per speaker with `ENGINE_SHARES`
  pocket .40 / el .15 / fish .45; Moshi is always Pocket `michael`):
  - ElevenLabs Flash v2.5: 17 premade train voices; test: Lily, Matilda, Daniel, Bill. The
    account's one cloned voice is excluded.
  - Fish Audio s2-pro: 30 public-library train voices + 4 test, vetted by title / tags /
    description as generic. The library has many voices named after or cloned from real
    people or characters; all excluded.
  - Pocket: 16 train / 4 test, as before. 63 train voices in all.
  - The held-out EL / Fish voices aren't in any existing test suite.
- **Code:**
  - `tts.py`: `ElevenLabsTTS` / `FishTTS` on `_HTTPTTS`. Concurrency 3 / 5 (account limits, so
    one process per engine); 429 / 5xx retried, Retry-After capped at 60 s; `FatalEngineError`
    on 401/402/403 or 25 consecutive failures. Audio checks: reject RIFF / JSON / ID3 prefixes,
    silent or < 0.1 s audio, a rate outside 0.12–1.0 s/word (≥ 3 words); up to 3 resamples.
    0.15 s padding. Word times are synthetic (spread over `speech_span` by word length): only
    human-turn onset / offset are used downstream, and only Moshi's words become text tokens.
  - `MultiCache` routes `el:` / `fish:` voices to `<cache>/<engine>/`, key engine|voice-id|text
    (Pocket keys unchanged). Atomic writes (uuid temp + `os.replace`, audio before json).
    Hosted clips are 16-bit FLAC; `SILENCE_CACHE_FLAC=1` does the same for Pocket.
  - API keys `ELEVENLABS_API_KEY` / `FISH_API_KEY` come from the env, `.env` or
    `/dev/shm/silence_keys`. Hosted engines run from the Mac only; keys never leave it.
  - `build.py`: `--tts multi`, `--mix v2`, `--exclude <sets>`, `--render-only <engines>
    --shard i/n [--reverse]`, `--max-drop` (3%). Prints a plan hash (hash of the clip set);
    every shard must print the same. The final (assembly) run renders nothing: missing hosted
    clips drop their episodes (bounded by `--max-drop`), missing Pocket clips abort. In multi
    mode an over-60 s episode gets only the B2 shrink, else it's dropped and its files deleted.
  - `check.py --workers N`. `silence/filter_checked.py <dir>`: failing episodes → `rejected/`,
    `manifest.jsonl` / `episodes.jsonl` rewritten in order (originals `*.all.jsonl`, so first-K
    prefixes stay nested), `rejected.json`, `train_all.jsonl` (absolute paths).
- **Plan:** `--n 5600`, seed 0, est ~50 h; 41,070 unique clips (Pocket ~24.1k ≈ 23.2 h speech,
  Fish ~12.7k ≈ 10.3 h, EL ~4.2k ≈ 3.4 h). Mac (macOS libm) and Linux plans differ by 1 ulp in
  25 timing floats; the clip sets are identical (verified). **Only one machine assembles.**
- **Execution:** Pocket shards 0/3 on tr3, 1/3 on tr10, 2/3 on the Mac (into the old Pocket
  cache via symlink, reusing the 10 h build's `michael` clips); the pods then take over the
  Mac's shard from the far end (`--reverse`). EL + Fish from the Mac. A sync script uploads
  Mac clips as FLAC to `/workspace/cache50` (`rsync --ignore-existing`, so a pod's own render
  is never overwritten). Assembly + `check.py` + `filter_checked` on tr3. Shard 0 crashed once
  at 06:41 (BrokenPipe after its launching ssh session was killed); relaunched detached
  (`ssh -f -n` + `setsid`). A 4× RTX PRO 6000 CPU-burst pod was blocked by the permission
  classifier (no CPU-only pods in US-NE-1).
- **Measured:** Fish ~20× realtime at 5 concurrent (~0.8 s latency); EL Flash ~20× at 3
  concurrent, ≈ 0.5 credit/char (Starter, 116,845 credits; plan ~209k chars ≈ 105k credits);
  Fish $15 / 1M chars, plan ~622k chars ≈ $9 (credit $25.83). Pocket on the H100 pods' CPUs
  (22-vCPU quota, nice 19, 18 workers) 15–21× per pod; Mac ~11×. It slowed R20/R21 training
  ~8% (10.2–10.4k vs ~11.2k words/s). Mac → pod ~1.2 MB/s for one big file; clip sync ~25
  clips/s.
- **Caveat:** not "10 h but bigger": the recipe changed too (v2 mix, MUTTER_P, 3 engines / 63
  train voices, new packs, dev exclusion), and its first-K prefixes are not the old
  `train_1h` / `train_10h`. Durations are Pocket-fitted. See `docs/training_record.md` §12.

### Human test set (recorded and evaluated 2026-10-15; results in `results/human/summary.md`)

Plan (user, 2026-10-15): solo, ≤ 30 min. The user voices the USER side of held-out test
episodes; the friend stays TTS. Recorded clips replace the TTS user clips and the episodes are
re-assembled, so labels / check.py / scoring are unchanged and each episode pairs with its TTS
twin. Final models only, reported separately.

- `python -m silence.record` (run it in a real terminal; it needs mic access for that app):
  plans on first use → `data/human/s01/session.json` in the main checkout (gitignored), then
  prompts. 24 `test_all` episodes (B1 std 4 / double 1 / rescind 1, B2 6, B3 std 2 / early 1 /
  rescind 1, ORD 6, DECOY 2; seed 0), ≈ 23 min:
  - R: 10 s room tone.
  - B: 28 own-words clips. Each hold's instruction and cue / rescind, from an intent card
    (never the scripted wording), recorded before A so the scripted lines can't prime it.
    Key `free_<episode>_<anchor>`.
  - A: 116 verbatim user lines (1,066 words) with 2 turns of context, addressee (Moshi /
    friend / mutter / aside) and mid-turn-pause hints. Key `sha1("human|s01|<text>")[:16]`
    = `ClipCache.key` for a backend named `human`; recurring lines are recorded once.
  - Clips: `clips/<key>.wav` (24 kHz, trimmed to speech + 0.2 / 0.3 s) + `.json` (speech
    span, peak, SNR); every take in `raw/`, log in `takes.jsonl`. Space toggles recording
    (0.5 s pre-roll); takes with warnings (quiet, clipped, SNR < 20 dB, cut-off) wait for
    redo or keep. `--status`, `--list-devices`, `--device`. Resumable.
- `python -m silence.human`: `HumanCache` serves `human:s01` voices from the clips (block B by
  `free:<key>` text), everything else from the Pocket cache; writes `data/test_human_tts` (the 24
  TTS originals, copied), `data/test_human` (24, verbatim) and `data/test_human_free` (16 hold
  episodes, own-words instruction + cue). Original episode ids, so noise / RIR are identical to
  the TTS twin. Human episodes run ~7% longer (0.97–1.18×). Block B wording isn't transcribed
  (Whisper later if needed); the card fixes N for B1.
- Eval: `OUT=/workspace/eval/human bash scripts/test_human.sh base <ckpt> ...` (3 samples; the
  suites are on the volume); scoring + paired bootstrap: `results/human/analysis/score_human.py`.
- Limits: one voice; friend is TTS; no real two-person timing.

## Generator design (agreed with the user; don't relitigate without asking)

**Training mix by episode count, family shares 15 / 25 / 20 / 40** (`specs.TRAIN_MIX`):

| Family | Share | Variants (weights) |
|---|---|---|
| B1 timed hold ("give me N seconds") | 15 | std 9, double 2 (two holds in one episode), early answer 2, rescind 2 |
| B2 side conversation ("stay quiet until I'm back") | 25 | std 25 |
| B3 hear me out ("let me describe all three options") | 20 | std 16, finishes early 2, rescind 2 |
| Ordinary | 40 | ORD 30 (⅓ of user turns have a 0.8–2 s mid-turn pause), DECOY 10 |

- B2 rescind was deliberately cut (a cue release is already the release).
- Contrast episodes (early answer, rescind, finishes early) teach following the instruction
  rather than "go deaf for a while".
- Lexical decoys use hold words with no hold intent ("I just got back from Boston", "took
  ten seconds").
- **B1 N:**
  - Train N ∈ {3, 5, 10, 15, 20, 30} s, capped so the episode fits the 57 s budget.
  - early / rescind use N ≥ 10; double uses N ≤ 12.
  - Test-only: {8, 12, 25} s; "half a minute" and "a minute" only in test_cond.
- **Distractor codes:**

  | Code | What it is |
  |---|---|
  | D1a | friend → user question |
  | D1b | user → friend question |
  | D1c | friend question that could be for Moshi |
  | D2a | 2–8 s stall in a side conversation (50% of B2; always in dense suites) |
  | D2b | 3–5 s pause between options |
  | D2c | fillers during a B1 hold (50% of B1 std with N ≥ 5) |
  | D3a | return-cue lookalike |
  | D3b | completion lookalike |
  | **D4** (test only) | the *friend* says "Okay, I'm back!"; Moshi must stay quiet |

  Distractor lines are **written by the LLM in context**: they're tagged lines in each
  pack's `side`, D3b is `aside`, D2c is `mutter`, D4 is `friend_away`. So they're held out
  together with the scripts. They are not inserted from a lexicon.
- **Voices (Pocket TTS):** Moshi is always `michael`. Test-only: eve, george, jane,
  peter_yearsley. Train: the other 16 English voices. Only 4 unseen voices is a known
  limitation; say so in the report.
- **Phrase bank** (`phrasebank.py`): instructions, return cues, rescinds, early answers,
  completions, B1 durations, friend arrivals and names each have train vs test lists.
  Moshi's acks are one shared list. `STORY_INSTR` is unused.
- **Timing and mixing** (`specs.py`, `assemble.py`):
  - Turn gap: lognormal(−0.9, 0.45), clipped to [0.12, 1.2] s, × `GAP_SCALE` 1.5. This is a
    placeholder; ideally fit it on DailyTalkContiguous.
  - Moshi gap 0.45–1.35 s.
  - Side-conversation gaps overlap by up to 0.25 s 15% of the time.
  - Moshi acks the instruction in 70% of holds.
  - Friend gain −9 to −4 dB with a synthetic RIR (rt60 0.2–0.6 s).
  - Noise (pink / brown / white_lp) on the user channel only, at SNR 15–35 dB against a
    −20 dBFS speech target. Fillers at −6 dB; peak limit 0.99.
- **Length:** episode budget `BUDGET_DUR` 57 s. Train episodes must be ≤ `MAX_TRAIN_DUR`
  60.0 s, so each is exactly one 60 s window. Over-length B2 episodes get a shorter side
  conversation, otherwise the next pack.
- **No episode opens with Moshi** (`load_packs` drops packs that do).

**Timing fixes in tts.py / assemble.py. Keep all three:**
- `tts.retime()`: Pocket's attention timestamps put the first word at ~0 s when a clip opens
  with silence. Words stamped before the energy onset move onto the onset.
- `assemble.gate()`: zeroes each clip outside [onset − 50 ms, offset + 100 ms] with 10 ms
  fades. This removes Pocket's ~−42 dBFS noise floor from Moshi's channel during holds.
- `speech_span()` onset must be followed by real speech (a frame > −15 dB within 150 ms, or a
  dense 100 ms reaching −23 dB), so breath or noise at the start of ~5% of clips isn't taken
  as the onset.

**check.py** (moshi-finetune's own Interleaver; a stand-in tokenizer unless `--spm`; writes
`checks.json`, plus plots and `index.html` with `--plots`). Fails on:
- hold-silence amplitude > 1e-3, outside the gate margins
- word vs audio onset misalignment > ±0.15 s
- B1 timer error > 0.3 s
- cue latency outside [0.25, 1.5] s
- a human turn < 6 dB above the floor
- human/Moshi overlap > 0.05 s
- a train episode > 60 s
- Moshi speaking first
- no EPAD right before the release, or non-PAD text inside a hold

It globs `episodes/*.wav`, not the manifest, so orphans from an earlier larger `--n` get
counted.

## Pronunciation (Pocket TTS garbles unusual words; the user heard "latte" break)

- The `gen_packs.py` prompt asks for plain, common English: no loan words, unusual brands,
  acronyms, rare place names or uncommon first names.
- `gen_packs.is_speakable()` drops packs with digits, non-ASCII characters, all-caps
  acronyms (except I / OK / TV), or a word in `HARD_WORDS`. Add to `HARD_WORDS` when
  listening turns up a bad word.
- Phrase banks pass the filter today, but nothing enforces it.
- Not built: ASR every rendered clip (e.g. whisper) and drop clips above ~10% WER. Worth
  adding if listening keeps turning up garbled words.

## Layout

```
silence/phrasebank.py      instruction / cue / ack phrasings, train vs test lists
silence/splits.py          which axis each test suite holds out
silence/voices.py          voice pools (pocket default; kyutai, espeak)
silence/specs.py           episode builders, TRAIN_MIX, timing
silence/tts.py             TTS backends (pocket | kyutai | espeak) + ClipCache + retime
silence/assemble.py        timeline -> stereo wav + {alignments, events, meta} json; gate, mixing
silence/build.py           CLI: specs -> TTS (parallel) -> episodes, manifest.jsonl, episodes.jsonl
silence/check.py           checks via moshi-finetune's Interleaver; --plots for plots + index.html
silence/transcript.py      print episodes as timed transcripts
silence/gen_packs.py       LLM content packs (Anthropic API, reads .env); schema + speakability filters
silence/make_splits.py     nested 1h ⊂ 10h jsonl subsets + ledger.json; `exposure` subcommand
silence/build_dev.py       dev set: unused + screened train packs, balanced, dev_80 ⊂ dev_200
silence/screen_packs.py    held-out pack file minus packs overlapping train/dev
silence/check_overlap.py   train vs held-out audit (packs, near-dup scripts, shared lines, voices)
silence/screen_train_packs.py  reverse screen: new train packs minus those overlapping dev / test (50 h build)
silence/filter_checked.py  move check.py failures to rejected/, rewrite manifests (originals *.all.jsonl), train_all.jsonl
silence/make_configs.py    moshi-finetune yamls per run (fixed epochs) + checkpoint -> epoch maps
silence/eval_gen.py        GPU: batched generation with the script forced outside scored windows; selftest
silence/eval_score.py      VAD segments + events -> per-hold / per-response behavior scores (+ oracle mode)
silence/eval_report.py     bootstrap CIs, summary tables, per-epoch curves, threshold sensitivity
silence/plot_training.py   loss curves / LR schedule / dev loss vs LR + training_summary from metrics.*.jsonl
silence/demo_phase1.py     listening demo: base + every 1 h / 10 h run's lowest-dev-loss and final checkpoints on 9 dev episodes -> results/demo/index.html
silence/chat_gui.py        local control panel (http://127.0.0.1:8770) for live chat with checkpoints on the chat pod: 2 slots, 10 representative models (CURATED, with good/bad notes, incl. R21 10 h EPAD ×10 ep 2.2 / ep 5.0 and R22 50 h; 'show all' for the rest), SSH tunnel, test scripts, notes -> results/chat/notes.jsonl
silence/human.py           human test set: recorded clips swapped into the test_all episodes -> data/test_human{,_tts,_free}
silence/record.py          prompter for the human test set (solo): plan from test_all, room tone / own-words / verbatim blocks -> data/human/<speaker>/clips
silence/chat_scripts.py    13 live-chat test scripts in training-episode style (train wording / held-out / new conditions) -> chat_gui + results/chat/test_scripts.md
silence/transcript_gen.py  readable transcripts: script merged with each model's generated words, holds, outcomes -> results/transcripts.md; --html results/transcripts.html = interactive viewer (10 representative models x 9 demo conversations incl. R22 50 h, audio from demo_curated). Serve with `cd results && python3 -m http.server 8780`
silence/plotstyle.py       shared plot style: runs colored by peak LR (ordinal blue ramp), base dashed
silence/list_kyutai_voices.py, calibrate_offset.py   only for the optional 1.6B Kyutai TTS
scripts/gpu_bootstrap.sh   GPU box: moshi-finetune + 3 patches (uv.lock env), pinned weights, configs, smoke
scripts/run_parallel.sh    run configs one per GPU (own CUDA device + master port + log); skips .done, refuses leftover run_dirs
scripts/eval_dev.sh        generation eval of whole runs on dev_80 / dev_200, one run per GPU, then scoring
patches/moshi-finetune-audio-mask.patch    mask_main_speech_audio (no audio loss on Moshi's TTS speech)
patches/moshi-finetune-eval-reload.patch   fresh eval loader per eval + text/audio dev loss logged
patches/moshi-finetune-text-tail.patch     no text loss on the zero-padded tail of a window
patches/moshi-finetune-loss-weights.patch  (4th, 2026-10-15) optional text_epad_weight (EPAD's own weight) + audio_loss_weight; training only, dev loss unchanged
scripts/ablation.sh        EPAD-weight sweep (1 h, lr 5e-5, EPAD x3/x10/x30/x100 = R16-R19)
scripts/phase2.sh          phase 2 (10 h x 4 LRs)
scripts/epad10h.sh         EPAD x3 / x10 at 10 h, lr 5e-5 (R20, R21); arg 3|10|both
packs/                     packs.jsonl, packs_test_screened.jsonl, rejects.jsonl, smoke.jsonl, gen_packs.log
packs/ (v2, 50 h build)    packs_v2.jsonl (7,970), packs_v2_screened.jsonl (7,664, the build's input), rejects_v2.jsonl, gen_packs_v2.log
docs/training_record.md    write-up record of training (setup, hyperparameters, compute, cost, issues, results)
results/phase1/, phase2/   configs, metrics, logs, setup logs, eval outputs, plots, analysis pulled back from the box (no checkpoints)
results/demo/              combined listening demo (index.html + audio)
results/demo_curated/      same 9 conversations, only the 10 representative models with good/bad notes (`demo_phase1 --curated`; audio needs eval_raw codes.npz, which `results/epad10h/setup/pull_run.sh` excludes: pull them by hand for new models)
results/overview.md        one-page overview of every run, findings, failure modes, pointers
results/epad/              EPAD sweep results (+ analysis/epad_sweep.png)
demo/                      14 example episodes (json, plots, index.html, transcripts.txt; wavs gitignored)
data/                      gitignored: train, dev, test_*, build logs
```

**Demo:** 14 episodes (one per type, plus B1 double, phone call, B1 minutes), 14/14 pass.
It's a listening / plumbing check only. It **can't be rebuilt**: `build.py` has no `demo`
split anymore, `packs/demo_packs.jsonl` is deleted, and its pack ids aren't in `packs.jsonl`.
The committed `index.html` points at wavs that only exist in the main checkout.

## How moshi-finetune consumes the data (verified in the source)

- JSON next to each wav (`splitext(path)[0] + ".json"`) with
  `{"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...]}`. Extra keys are ignored.
- jsonl rows are `{path, duration}`. sphn resolves relative paths against the jsonl's own
  directory; the bootstrap writes absolute ones anyway.
- `keep_main_only=True`: **only Moshi's words become text tokens**. User speech is
  audio-only input.
- Text stream: word tokens at their start frame, PAD (3) elsewhere, EPAD (0) on the frame
  before a word that follows padding. zero token = −1. 12.5 Hz frames.
- Loss = text CE + audio CE over Moshi's 8 codebooks. The user's codebooks are input only.
  - `text_padding_weight` 0.5 down-weights **both** PAD and EPAD.
  - `first_codebook_weight_multiplier` 100 makes the audio loss ≈ 93% semantic codebook.
  - A separate higher EPAD weight would be a worthwhile ablation, since EPAD is the "start
    talking now" decision.
- Files are cut into consecutive `duration_sec` windows from t=0, each an independent
  context. Hence the ≤ 60 s episode cap.
- Moshi's context is 3000 frames = 240 s.
- Training iterates **real epochs**: an infinite stream, reshuffled without replacement each
  pass (seed +1 per epoch), with no drop_last loss. So epoch = steps × batch / n.
- `InterleavedTokenizer` passes a frame count where `prepare_item` expects seconds. It works
  only because the result is cropped; don't "fix" it.
- Optimizer: AdamW, betas (0.9, 0.95). Adam normalizes the gradient scale, so loss weights
  change emphasis, not step size.
- LR schedule: `OneCycleLR(max_lr=lr, total_steps=max_steps, pct_start=0.05)`: warmup from
  lr/25, cosine decay to ~0. The swept "LR" is the peak. **A mid-run checkpoint isn't a
  shorter run**: it was taken at high LR, before annealing.
- LoRA: `W' = W + scaling·B·A` on every `nn.Linear` (main transformer, depformer, heads). B
  starts at zero, so step 0 is the base model. r = 64 is ~194M parameters, ~0.39 GB per bf16
  checkpoint.
- **`train.py` has no resume** (no optimizer state saved) and refuses an existing `run_dir`.
- YAML is loaded strictly: unknown keys raise. An unpatched tree fails loudly on
  `mask_main_speech_audio`.
- Recommended upstream: LoRA r=128, lr 2e-6 for 2000 steps (the code default is 1e-4).
  ~40 GB peak at batch 16 × 100 s on one H100.

**Traps found in upstream and how we handle them:**
- `pip install -e .` can't resolve: pyproject pins `sphn==0.1.12` but takes moshi from git
  main, which now needs sphn ≥ 0.2. Use `uv sync --frozen`. uv.lock pins moshi `061cc4c`
  (0.2.4a1), torch 2.6.0, huggingface-hub 0.28.1 (it has `huggingface-cli`; there's no `hf`).
- **`uv sync --frozen` fails on openai-whisper 20240930** (found on the GPU, 2026-10-14;
  pulled in by `whisper_timestamped`). It's an sdist whose setup.py imports `pkg_resources`,
  and build isolation installs the newest setuptools (84), which dropped it (setuptools < 81
  still has it). These did **not** help under `uv sync --frozen` with uv 0.13.1:
  `UV_BUILD_CONSTRAINT`, and `build-constraint-dependencies` in a user `uv.toml` or a project
  `uv.toml`. A project `uv.toml` also replaces moshi-finetune's own `[tool.uv]` settings. The
  bootstrap now syncs in two steps: `--no-install-package openai-whisper` (installs the
  locked setuptools 78.1.0), then `--no-build-isolation-package openai-whisper`. **Verified on the GPU 2026-10-14**
  (whisper builds; the env imports).
- On the first (personal-account) run, the bootstrap logged "already applied" for all three patches on
  a fresh clone (the second run's fresh clone logged "applied", as it should). The tree was checked afterwards: it's exactly `2acc879` + the 3 patches, each
  applied once (compared with a pristine worktree). Cause unknown, maybe a stale log; if it
  recurs, run the same check.
- **The eval loader is built once but is a finite generator.** Every evaluation after the
  first sees zero batches and logs NaN. It also drops the last partial batch, and never logs
  the text/audio split. → eval-reload patch. Upstream has a `fix_eval` branch, not checked.
- **Text past the end of a short file is PAD and counts in the text loss.** The audio there is
  −1 and masked. At 1 h that's ~46% of all text frames (113 × 750 window frames vs ~45k real).
  → text-tail patch sets them to −1.
- `num_ckpt_keep` defaults to 3 and deletes older checkpoints. → `num_ckpt_keep: null`.
- No revision argument for the weights. → the bootstrap downloads a snapshot at
  `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289` and the configs point at the files. The repo has
  no `config.json`, so moshi uses its built-in 7B config: the architecture comes from the
  uv-locked moshi package. `hf_repo_id` still makes one (404) request; all training and
  eval jobs run with `HF_HUB_OFFLINE=1`, so it fails instantly and is caught (see below).
- sphn skips unreadable wavs with only a log line (`on_error=log`). The smoke check warns on
  "error" lines.
- PyYAML reads `2e-06` (no dot) as a string. make_configs writes `2.000000e-06` (`.6e`, so
  phase-2 LRs keep full precision).
- `finetune/distributed.py` reads `os.environ['CUDA_VISIBLE_DEVICES']` with no default, so a
  torchrun without it set crashes with KeyError. The bootstrap smoke run and
  `run_parallel.sh` both set it.
- moshi-finetune says only `requires-python >= 3.10` and has no `.python-version`, so uv
  picks the newest Python (3.14), for which torch 2.6.0 has no wheels. The bootstrap pins
  `UV_PYTHON=3.12`.
- moshi's loader always probes the hub for the base repo's `config.json` (there is none → 404,
  caught). A hub outage / 429 would crash a job at load, so every training and eval command
  runs with `HF_HUB_OFFLINE=1` (weights are local); the probe then fails instantly and is
  caught. Only the bootstrap's weight download (step 2) needs the hub.
- The dev loss averages per-batch means, and 200 = 12 × 16 + 8, so the last 8 dev episodes
  count about double (~4% of the weight). It's the same at every checkpoint, so comparisons
  are unaffected; mention it if absolute dev-loss values are reported.

**Loading a checkpoint for inference** (eval harness):
`CheckpointInfo.from_hf_repo(repo, moshi_weights=…, mimi_weights=…, tokenizer=…,
config_path=ckpt/config.json, lora_weights=ckpt/lora.safetensors).get_moshi(...)`, then
`LMGen`. **Pass the checkpoint's `config.json`**: it carries `lora: true`, `lora_rank: 64`,
`lora_scaling: 2`. Without it, moshi asserts or defaults to rank 128 and fails on shapes.
`fuse_lora=True` for a single checkpoint; `eval_gen` keeps LoRA unfused when it evaluates
several checkpoints in one process, so it can swap the LoRA tensors. Checkpoints live in
`run_dir/checkpoints/checkpoint_XXXXXX/consolidated/{lora.safetensors, config.json}`.

## Run matrix (agreed 2026-10-14)

**Epochs are fixed per phase, not steps.** 1 h is a proof of concept: does the model start
picking up the behavior at all?

Fixed for every run (`silence/make_configs.py`):
- the pinned weights
- LoRA r=64, scaling 2; batch 16; `duration_sec: 60` (one episode = one window)
- AdamW weight decay 0.1; OneCycle `pct_start` 0.05
- first codebook ×100; `text_padding_weight` 0.5
- **`mask_main_speech_audio: true`** (main-speech mask pre 0.08 s / post 0.24 s)
- all three patches applied
- seed 0, one seed per run
- every checkpoint kept

| Phase | Run | Data | Peak LR | Epochs | Steps | Checkpoints |
|---|---|---|---|---|---|---|
| 0 | R0_devloss | none (base model) | 0 | — | 2 no-op | dev loss only |
| 1 | R1–R4 | train_1h (113 eps) | 2e-6, 1e-5, 5e-5, 2e-4 | 20 | 142 each | 21 each (every 7 steps ≈ 0.99 epoch, plus the last) |
| 1b | R6–R7 | train_1h | 5e-6, 2e-5 (added 2026-10-15) | 20 | 142 each | 21 each |
| 2 | R8–R9 | train_10h (1,101 eps) | top 2 from phase 1 | E (from phase 1) | ⌈E·1101/16⌉: 345 at E=5, 689 at E=10 | every 69 steps (≈ 1 epoch), plus the last |

- `max_steps = ceil(E · n / 16)`, with n counted from the jsonl. `configs/<run>.epochs.json`
  maps each checkpoint to its exact epoch.
- **Phase 1 chooses E and the top 2 LRs** from dev. 2e-6 is expected to be a no-op at 142
  steps. It doubles as a pipeline check: if it doesn't match R0, something in the
  LoRA / eval path changes behavior by itself.
- **Report caveat:** the 1 h point at E epochs is the epoch-E checkpoint of a 20-epoch
  OneCycle run, not a run trained to exactly E.
- **Dropped or on hold** (user's call): a separate 1 h rerun at exactly E; the 100 h build
  and run; the stretched-holds ablation (×3 would break the 60 s cap); the 1,000 h stretch; a
  behavior-loss validation script; a second seed.

**Compute. Measured on 1× H100 SXM, 2026-10-14** (bootstrap smoke run + eval benchmark; the
plan had assumed upstream's ~12k tokens/s ≈ 9 s/step):
- **Training: ~10 s/step** at batch 16 × 60 s (≈ 11.3k words/s steady); peak memory 29.1 GB.
  Model load + first step ≈ 2.6 min per run.
- **Dev loss on 200 episodes: ~47 s per evaluation.**
- **Generation: 28 ms/step at batch 40**, i.e. 40 dev episodes in 35 s (67 s with load and
  scoring) → dev_80 ≈ 70 s per checkpoint. Under the 2-minute bar, so phase-1 generation evals
  run **every epoch**.
- Phase 1, per 1 h run: 142 × 10 s ≈ 24 min + ~20 dev evals × 47 s ≈ 16 min + load ≈ **43 min**.
  On 4×: R0–R3 start together, R4 takes R0's GPU after ~4 min, R5 starts when R1–R3 finish
  (~43 min) → training done after **≈ 1.5 h** (4 × 1.5 = 6 GPU-h billed ≈ $24).
- Every dev-loss eval has a saved LoRA checkpoint (`eval_freq` = `ckpt_freq` = 7, both also on the
  last step; `save_adapters: true`, `num_ckpt_keep: null`), as the user asked (2026-10-14).
- Phase-1 generation evals: ~105 checkpoints × ~75 s ≈ 2.2 GPU-h ≈ 35 min on 4 GPUs. So the
  whole phase-1 burst is ≈ 2 h on the 4× pod ≈ **$32**.
- Phase 2: E=5: 345 steps ≈ 58 min, E=10: 689 steps ≈ 1.9 h per run (+ ~1 epoch of dev evals,
  ~47 s each). R6 and R7 side by side, while the other 2 GPUs run generation evals.
- Setup run (1× pod, 23:58–00:18 UTC): ~20 min ≈ $1.35. The personal-account attempt before it
  ran ~9 min ≈ $0.60.
- Don't split one run across GPUs: moshi-finetune's `batch_size` is per GPU, so that would
  change the recipe, and 142-step runs gain little.
- Disk: phase 1 checkpoints ≈ 105 × 0.39 GB ≈ 41 GB.

**Exposure accounting:** each episode is one window, so exposure = steps × batch × GPUs.
`python -m silence.make_splits exposure --episodes N --mean-dur S --steps K` gives epochs
and audio-hours seen. Report unique episodes, packs and TTS clips, and total / speech /
hold-silence hours, separately (`ledger.json`).

## Eval harness (built 2026-10-14; CPU-tested, then run on the 7B on an H100)

**Agreed decisions (user, 2026-10-14):**
1. Timing from Silero VAD on Moshi's decoded audio; words from Moshi's own text stream for
   dev sweeps. Whisper ASR only on final test evals and the listening log (not built yet).
2. 1 sample per episode for dev sweeps, 3 for final test evals (`--samples 3`). Same seed per
   episode across models (common random numbers).
3. LLM judge for resume appropriateness: test evals only, 1 sample (~2.5k calls ≈ $10; ask
   before running). Not built yet.
4. Generation on dev_80 every epoch in phase 1 if the R0 benchmark shows ≤ 2 min per
   checkpoint, else every 2; dev_200 for the top candidates and in phase 2.
5. Re-force the script between scored windows (per-frame forcing mask), so every hold is
   scored, not just the first. Ordinary episodes: free-run after the first user turn.
6. Thresholds: 2.5 s ack grace; violation = ≥ 1 s speech or > 3 words; 3 s resume / response
   / distractor windows; B1 tolerance max(1.5 s, 0.2N). Sensitivity is reported.

**Pipeline** (GPU box, moshi-finetune's env `$PY=/workspace/moshi-finetune/.venv/bin/python`):
1. `silence/eval_gen.py run` (GPU): per batch of episodes, Mimi-encodes both channels, builds
   the script's text stream with moshi-finetune's own Interleaver, and generates with
   `ForcedLMGen`, a copy of `LMGen.step` (moshi 061cc4c) that forces Moshi's text + 8 audio
   codebooks to the script on every frame outside the free windows (per batch element; forced
   text is substituted BEFORE the depformer, so forced audio is conditioned on it, as in
   training). Then Mimi-decodes Moshi's codes, runs VAD, and writes `<id>.json` (words,
   segments, text tokens), `codes.npz` (int16 codes, to re-decode later), `meta.json` (written
   last = finished). `--ckpt base ckpt1 ckpt2 …`: base weights load once (unfused LoRA), only
   the LoRA tensors are swapped per checkpoint, output to `--out/<checkpoint_XXXXXX|base>/`;
   finished checkpoints are skipped on a rerun. `--save-wav` writes stereo listening wavs
   (left = generated Moshi, right = user). `--force-all` = sanity mode (output = the script).
2. `silence/eval_score.py <gen dir>`: scores → `scores.jsonl` + printed summary.
3. `silence/eval_report.py <eval root> --configs configs --runs /workspace/runs --out …`:
   per checkpoint, pooled rates with 95% bootstrap CIs over episodes, epoch from
   `configs/<run>.epochs.json`, text/audio dev loss from `<run_dir>/metrics.eval.jsonl` →
   `summary.csv`, `summary.md`, curves per set (matplotlib; it's not in the GPU env, so
   rsync `/workspace/eval` back and plot on the Mac). `--sensitivity` rescores from the saved
   segments at other thresholds (no regeneration).
4. `scripts/eval_dev.sh dev_80 R0 R1_… R5_…`: steps 1–2 for whole runs, one run per GPU.
5. `gpu_bootstrap.sh` step 6 runs `eval_gen selftest` and a 40-episode R0 benchmark on the 1×
   pod (ms/step → the every-epoch vs every-2-epochs decision).

**User-frame alignment (important, verified):** training feeds the model, at step t, the
delayed codes of frame t−1 (`LMModel.forward` prepends the initial token), so Moshi's text at
frame t is predicted from user audio up to frame t−1. `LMGen.step` writes the user frame
given at step s into the column the model reads at step s, i.e. the slot training fills with
frame s−1. Feeding a wav straight into `LMGen` therefore hands the model the user's audio one
frame (80 ms) earlier than training did. `eval_gen` feeds frame s−1 at step s
(`--user-shift 1`). Checked in fp32 on a tiny random LM with moshi 061cc4c: shift 1 reproduces
every training input column (text, Moshi audio, user audio) and the text logits to 3e-6;
shift 0 matches 2.5% of columns. `eval_gen selftest` repeats the check on the real model.

**Free windows** (`eval_score.free_windows`): per hold, [instr_end, next scripted human turn
after the release], at least release ref + 4 s, capped 1 s before the next hold's instruction.
Ordinary episodes (no holds): from the end of the first human turn to the end.

**Scoring** (`eval_score.py`): per hold, `violated` (≥ 1 s or > 3 words inside
[instr_end + grace, ref − tolerance]; None when that region is empty, i.e. B1 N = 3 s, whose
timing is still scored), `outcome` on_time | late | no_onset (no new onset in
[ref − tol, ref + 3 s]: silent, or still talking from before; see `talking_through`),
`latency`, `timer_err`, `words_in_hold`, per distractor `reacted` (a segment STARTED within 3 s
of it: attributable) and `overlapped` (≥ 0.5 s of speech after it, however it started).
Ordinary responses only on hold-free episodes; barge-in into mid-turn pauses; onsets inside
human turns.

**Tests so far (CPU, Mac):**
- Oracle (ground-truth Moshi channel, dev_80): 0% violations (n=58), 100% on time (n=63), cue
  latency median 0.92 s, timer error median −0.11 s, 100% ordinary responses, 0 distractor
  reactions. Negative control (user channel as Moshi): 76% violations, 94% no onset.
- `eval_gen run` end to end with real Mimi + tokenizer + Interleaver and a tiny random LM
  (moshi 061cc4c, torch 2.6.0): `--force-all` output scores identically to the oracle on the
  same 6 episodes (the Mimi round trip keeps timing); free mode runs and a babbling model
  scores 100% violations; `selftest` on real data: shift 1 = 100% identical input columns
  (shift 0: 0.3%).
- LoRA swap: loads checkpoints exactly; `base` (lora_B = 0) reproduces the LoRA-free model
  token for token; multi-checkpoint layout + resume work. `eval_dev.sh` tested with a fake
  python.
- **On the 7B, H100, bf16 (bootstrap step 6, 2026-10-14):** selftest shift 1 = 100% identical
  input columns; text argmax agreement 99.7% (max|Δlogit| median 0.094, p99 0.232: bf16
  streaming vs full forward); script text CE 0.491 streaming vs 0.490 full. Benchmark: 28 ms/step
  at batch 40.
- **R0 baseline on 40 dev episodes (base model, 1 sample, benchmark only, not the dev_80
  result):** hold violated 82.8% (n=29), talking through 19.4%, resume on time 41.9% (no
  onset 58.1%, n=31), words in hold mean 16.5, ordinary responded 42.9% (n=21), 33 interrupt
  onsets; distractor triggered 67–100% across D1a–D3b. As expected, the base model doesn't
  hold silence. Output: `/workspace/eval/_bench`.
- Second review (2026-10-14, eval code): found the Mimi batch-memory blocker above; CUDA-graph
  buffer reuse, graph-safe RNG + per-batch `manual_seed`, KV capacity, the alignment, the
  checkpoint `config.json` keys and the LoRA key naming (world size 1: plain module names)
  all verified OK. Scanned free windows on dev + 4 test suites: none empty / overlapping /
  ending inside the scripted release response.

**Memory:** the KV cache is sized by the attention context (3000 frames ≈ 1.6 GB per
sequence for the 7B). `set_context` shrinks it to the batch's frame count (≥ the sequence,
so attention is unchanged): ~0.4 GB per 60 s episode (B=40: 16 GB, B=80: 32 GB) next to the
15 GB model. Default `--batch 40`; try 80 after the benchmark. **Mimi is the memory hog, not
the LM:** it's fp32 with 64-channel activations at the sample rate, ~3 GB per 60 s episode
for encode or decode, so `eval_gen` runs Mimi 4 episodes at a time (`MIMI_CHUNK`); encoding
a whole batch of 40 at once would need > 100 GB (found in review, 2026-10-14).

**`selftest` criterion:** it records every input column the model actually read and compares
them integer-for-integer with the training forward's input (`cat([initial, delayed codes])`).
Pass = 100% identical columns for shift 1. The logit agreement is printed only as a secondary
check (bf16 streaming vs full-sequence attention differ slightly).

**Still to build:** Whisper ASR pass + LLM judge for the final test evals; the listening log.

## Write-up record and plots

- **`docs/training_record.md`** is the write-up-oriented record (pinned versions, hardware,
  data and exposure, full hyperparameters, run matrix, eval protocol, measured compute,
  timeline and spend, issues and fixes, observations, results). The user asked (2026-10-14) for
  everything a submission write-up would need to be documented; keep it current as runs finish.
- Raw artifacts go to `results/phase1/` (configs, per-run `args.yaml` + `metrics.*.jsonl`,
  logs, setup logs + `pip-freeze.txt`, plots, eval summaries). Checkpoints stay on the volume.
- `silence/plot_training.py`: loss curves (train per step + 1-epoch mean, dev per checkpoint;
  total / text / audio), LR schedule, dev loss vs LR, `training_summary.{csv,md}`.
  `silence/eval_report.py` behavior curves use the same style (`silence/plotstyle.py`: runs
  colored by peak LR on one ordinal blue ramp, base model dashed gray; validated with the
  dataviz ordinal check). Both start every curve at epoch 0 with the base model (LoRA B = 0).
- Audio dev loss collapses within ~1 epoch (it's almost all on Moshi's silent frames, since
  Moshi's speech audio is masked): treat text dev loss and behavior as the signal.

## Known gaps / TODO

- Whether anyone listened to the demo isn't recorded; ask.
- Turn gaps are a placeholder lognormal. Noise is synthetic.
- The B1 fillers (D2c) come from each pack's `mutter`.
- Pack reuse is silent when a family runs out (test_distract / test_cond, ~3 B2 scripts).
- Dev-pack exclusion now exists: `build.py --exclude data/dev` (2026-10-15). Use it (plus a
  reverse screen like `screen_train_packs`) for any further train build.
- `requirements.txt` misses `whisper_timestamped` (calibrate_offset.py), `silero-vad`
  (eval_score) and `huggingface_hub[cli]`; `sentencepiece` and `torch` come in only through moshi /
  pocket-tts. The GPU box doesn't use it (uv.lock there).
- The demo wavs are gitignored, so `demo/index.html` plays only in the main checkout.
  `configs/` is gitignored, since it's generated on the box with absolute paths.
- The KyutaiTTS (1.6B, GPU) backend is optional and untested; calibrate its word offset first.
- 100 h, if it comes back: ~78 h of speech ≈ 6.5 h CPU here, and it needs more packs (B2
  first); `--exclude` and the multi-engine sharding from the 50 h build cover the rest.
- The 50 h set's held-out ElevenLabs / Fish voices aren't in any test suite yet.

## Working with the user (Jaewoo)

- Casual and direct. They push back when estimates look inflated; give real numbers with
  the arithmetic.
- Render math in LaTeX in explanations.
- Ask before generating at scale or spending money (API calls for thousands of packs, GPU
  rentals).
