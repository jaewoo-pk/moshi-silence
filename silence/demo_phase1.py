"""Listening demo (1 h + 10 h runs): an HTML page to hear and see how each fine-tuned model behaves.

  python -m silence.demo_phase1 --dev ~/repo/moshi-silence/data/dev \
      --mimi ~/repo/moshi-silence/models/moshiko/tokenizer-e351c8d8-checkpoint125.safetensors

Models: the base model, and for every phase-1 LR the checkpoint with the lowest total dev loss
plus the final checkpoint (epoch 20). Episodes: one dev_80 episode per behavior type (two for the
side conversation), each the one on which the models' scored outcomes differ the most, so
successes and failures both show up.

Audio: each model's generated Moshi channel is decoded from the saved Mimi codes (CPU) and mixed
with the episode's user channel (Moshi panned left, user + friend right), AAC in an .m4a.
Outside the scored "free" windows the eval forces Moshi's script (text + audio), so only the
free windows are the model's own behavior; the page marks them.

Output: results/demo/index.html (self-contained apart from audio/ and the copied plots).
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import plotstyle as ps
from .eval_score import Thresholds

GROUPS = [  # (types, title, what the model should do)
    (["B1_std"], "Timed hold",
     "The user asks for N seconds to think. Moshi should stay silent for about N seconds, then re-ask."),
    (["B1_rescind", "B1_early"], "Timed hold, cut short",
     "The user asks for time, then cancels it or answers early. Moshi should respond right away "
     "instead of waiting out the timer."),
    (["B1_double"], "Two timed holds",
     "Two 'give me N seconds' requests in one conversation."),
    (["B2_std"], "Side conversation",
     "'Stay quiet until I say I'm back.' The user talks with a friend, including questions and "
     "lookalike phrases. Moshi must stay silent until the user's real return cue."),
    (["B2_std"], "Side conversation (2)",
     "A second side-conversation example."),
    (["B3_std"], "Hear me out",
     "The user lists three options. Moshi must wait for all of them, including through pauses, "
     "then give its view."),
    (["B3_rescind", "B3_early"], "Hear me out, cut short",
     "The user starts listing options, then says 'actually, just tell me' or finishes early. "
     "Moshi should respond then."),
    (["ORD"], "Ordinary conversation",
     "No instruction. Moshi should answer normally, and not barge into the user's mid-turn pauses."),
    (["DECOY"], "Decoy (ordinary)",
     "Ordinary conversation with hold-like words ('I just got back', 'took ten seconds') but no "
     "instruction. Moshi should answer normally."),
]


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def data_tag(run):
    m = re.search(r"_(\d+h)_", run)
    return m.group(1).replace("h", " h") if m else ""


def epad_tag(run):
    """' · EPAD ×10' for a run with an up-weighted EPAD loss (R16–R21), else ''."""
    m = re.search(r"_epad(\d+)", run)
    return f" · EPAD ×{m.group(1)}" if m else ""


def pick_models(results):
    """[(results_dir, run, ckpt, role)]: per run (by LR) the lowest-dev-loss and final checkpoint."""
    models = []
    runs = sorted((d.name for d in (results / "runs").iterdir()
                   if d.name.startswith("R") and data_tag(d.name) and (results / "eval_raw" / d.name).is_dir()),
                  key=lambda r: ps.run_lr(r))
    for run in runs:
        ev = [json.loads(l) for l in (results / "runs" / run / "metrics.eval.jsonl").read_text().splitlines() if l]
        # only checkpoints that exist and were scored (a crashed run can log a dev eval it never saved)
        ev = [e for e in ev if (results / "eval_raw" / run).glob(f"*/checkpoint_{e['step']:06d}/scores.jsonl")
              and any((results / "eval_raw" / run).glob(f"*/checkpoint_{e['step']:06d}/scores.jsonl"))]
        if not ev:
            continue
        best = min(ev, key=lambda e: e["eval_loss"])
        final = max(ev, key=lambda e: e["step"])
        b, f = f"checkpoint_{best['step']:06d}", f"checkpoint_{final['step']:06d}"
        if b == f:
            models.append((results, run, f, "lowest dev loss = final"))
        else:
            models += [(results, run, b, "lowest dev loss"), (results, run, f, "final")]
    return models


def signature(row):
    hs = tuple((h["violated"], h["outcome"]) for h in row["holds"])
    rs = tuple(r["responded"] for r in row["responses"])
    return hs + rs + (min(row["interrupt_onsets"], 3),)


def pick_episodes(scores, ep_types, models):
    """One episode per group: the one whose scored outcomes differ most across the models."""
    chosen = []
    for g in GROUPS:
        types = g[0]
        cands = [e for e, t in ep_types.items() if t in types and e not in [c for c, _ in chosen]]
        if not cands:
            continue
        def diversity(e):
            sigs = {signature(scores[m][e]) for m in models if e in scores[m]}
            return (len(sigs), -int(e.split("_")[1]))
        chosen.append((max(cands, key=diversity), g))
    return chosen


def episode_payload(eid, ev, dur, group):
    th = Thresholds()
    holds = []
    for h in ev.get("holds", []):
        rel = h["release"]
        timer = rel["kind"] == "timer"
        tol = th.timer_tol(rel["N"]) if timer else th.cue_pre
        holds.append({"instr_end": h["instr_end"], "ref": rel["ref"], "kind": rel["kind"], "N": rel.get("N"),
                      "region": [h["instr_end"] + th.grace, rel["ref"] - tol],
                      "resume": [rel["ref"] - tol, rel["ref"] + th.resume_win]})
    return {"id": eid, "type": ev["type"], "title": group[1], "desc": group[2], "duration": round(dur, 3),
            "human_turns": ev.get("human_turns", []), "moshi_turns": ev.get("moshi_turns", []),
            "holds": holds, "distractors": ev.get("distractors", []),
            "grace": th.grace}


def decode_and_mix(mimi, codes, user_wav, sr, out_m4a):
    import torch
    with torch.no_grad():
        m = mimi.decode(torch.from_numpy(codes[1:].astype(np.int64))[None]).float().numpy()[0, 0]
    u = user_wav
    n = min(len(m), len(u))
    m, u = m[:n], u[:n]
    stereo = np.stack([m + 0.3 * u, u + 0.3 * m])
    peak = np.abs(stereo).max()
    if peak > 0.95:
        stereo *= 0.95 / peak
    import sphn
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "x.wav"
        sphn.write_wav(str(tmp), stereo.astype(np.float32), sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:a", "aac", "-b:a", "96k",
                        str(out_m4a)], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", default=["results/phase1", "results/phase2", "results/epad", "results/epad10h", "results/run50h"],
                    help="results dirs; the first one's models choose the episodes (keeps them stable)")
    ap.add_argument("--out", default="results/demo")
    ap.add_argument("--dev", required=True, help="data/dev (episodes/<id>.wav + .json)")
    ap.add_argument("--mimi", required=True, help="Mimi weights (tokenizer-e351c8d8-checkpoint125.safetensors)")
    ap.add_argument("--set", default="dev_80")
    ap.add_argument("--curated", action="store_true",
                    help="only the representative models from silence.chat_gui.CURATED (with their good/bad notes)")
    a = ap.parse_args()
    dirs = [Path(r) for r in a.results if (Path(r) / "eval" / "summary.csv").exists()]
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    summary = {}
    for R in dirs:
        summary.update({(r["run"], r["ckpt"]): r for r in read_csv(R / "eval" / "summary.csv") if r["set"] == a.set})
    models = [(dirs[0], "R0", "base", "base model")] + [m for R in dirs for m in pick_models(R)]
    scores = {}
    for R, run, ck, _ in models:
        p = R / "eval_raw" / run / a.set / ck / "scores.jsonl"
        scores[(run, ck)] = {r["id"]: r for r in map(json.loads, p.read_text().splitlines()) if r}
    first = [(run, ck) for R, run, ck, _ in models if R == dirs[0]]
    ep_types = {e: r["type"] for e, r in scores[first[0]].items()}
    picked = pick_episodes(scores, ep_types, first)
    eids = [e for e, _ in picked]
    groups = dict(picked)
    notes = {}
    if a.curated:  # same episodes, but only the representative models, in the panel's order
        from .chat_gui import CURATED
        cur = []
        for key, title, note in CURATED:
            run, ck = ("R0", "base") if key == "base" else key.split("/")
            # a curated run may have eval_raw scores but no eval/summary.csv yet (e.g. R22): look in every results dir
            R = next(R for R in map(Path, a.results) if (R / "eval_raw" / run / a.set / ck / "scores.jsonl").exists())
            cur.append((R, run, ck, title))
            notes[(run, ck)] = note
            p_ = R / "eval_raw" / run / a.set / ck / "scores.jsonl"
            scores[(run, ck)] = {r["id"]: r for r in map(json.loads, p_.read_text().splitlines()) if r}
        models = cur

    import sphn
    from moshi.models import loaders
    mimi = loaders.get_mimi(a.mimi, device="cpu")
    mimi.set_num_codebooks(8)

    dev = Path(a.dev).expanduser()
    episodes = []
    rows = {}
    missing_audio = set()
    for e in eids:
        ev = json.loads((dev / "episodes" / f"{e}.json").read_text())["events"]
        wav, sr = sphn.read(str(dev / "episodes" / f"{e}.wav"))
        user = wav[1]
        episodes.append(episode_payload(e, ev, wav.shape[1] / sr, groups[e]))
        for R, run, ck, _ in models:
            d = R / "eval_raw" / run / a.set / ck
            gen = json.loads((d / f"{e}.json").read_text())
            sc = scores[(run, ck)][e]
            fn = f"{e}__{run}__{ck}.m4a"
            if not (out / "audio" / fn).exists():
                if (d / "codes.npz").exists():
                    codes = np.load(d / "codes.npz")[e]
                    decode_and_mix(mimi, codes, user, sr, out / "audio" / fn)
                else:  # codes not pulled back from the GPU volume yet: page shows the timeline without audio
                    missing_audio.add(f"{run}/{ck}")
            rows.setdefault(e, {})[f"{run}|{ck}"] = {
                "audio": f"audio/{fn}" if (out / "audio" / fn).exists() else None,
                "segments": gen["segments"], "words": gen["words"], "free": gen["free"],
                "holds": [{k: h.get(k) for k in ("violated", "outcome", "latency", "timer_err", "words_in_hold",
                                                  "first_violation_t", "talking_through")}
                          | {"distractors": h.get("distractors", [])} for h in sc["holds"]],
                "responses": sc["responses"], "interrupt_onsets": sc["interrupt_onsets"],
                "barge_ins": sc.get("barge_ins", [])}
        print(f"  {e}: {len(models)} models")
    for m in sorted(missing_audio):
        print(f"  no audio for {m}: eval_raw/.../codes.npz missing (pull it from the volume and rerun)")

    model_payload = []
    for R, run, ck, role in models:
        s = summary.get((run, ck), {})
        ej = R / "configs" / f"{run}.epochs.json"
        epoch = json.loads(ej.read_text())["checkpoint_epoch"].get(ck) if ej.exists() else 0.0
        data = data_tag(run)
        model_payload.append({
            "key": f"{run}|{ck}", "run": run, "ckpt": ck, "role": role, "data": data, "note": notes.get((run, ck), ""),
            "lr": ps.run_lr(run),
            "lr_label": f"{data} · {ps.run_label(run)}{epad_tag(run)}" if run != "R0" else "base model",
            "color": ps.run_color(run), "epoch": epoch,
            "dev_text": num(s.get("text_dev_loss")), "viol": num(s.get("hold_violated")),
            "on_time": num(s.get("on_time")), "ord": num(s.get("ord_responded"))})

    plots = []
    for R in dirs:
        tag = data_tag(next((run for RR, run, ck, _ in models if RR == R and run != "R0"), "")) or R.name
        for name, what in (("eval/curves_dev_80.png", "behavior vs epoch"), ("plots/loss_curves.png", "loss vs epoch")):
            src = R / name
            if src.exists():
                dst = f"{R.name}_{src.name}"
                shutil.copy(src, out / dst)
                plots.append({"src": dst, "alt": f"{tag}: {what} per learning rate"})
    data = {"models": model_payload, "episodes": episodes, "rows": rows, "set": a.set, "plots": plots}
    (out / "index.html").write_text(HTML.replace("__DATA__", json.dumps(data)))
    print(f"wrote {out}/index.html ({len(episodes)} episodes x {len(models)} models)")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Moshi silence fine-tuning: listening demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
.viz-root { color-scheme: light;
  --surface-1:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --good:#0ca30c; --good-text:#006300; --warn:#fab219; --critical:#d03b3b;
  --hold:#ecebe5; --forced:#c3c2b7; --user:#52514e; --friend:#a8a69e; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root {
  color-scheme: dark; --surface-1:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10); --good-text:#0ca30c;
  --hold:#2a2a28; --forced:#4a4a46; --user:#c3c2b7; --friend:#6f6d66; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--page); }
.viz-root { font-family:system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:var(--page);
  max-width:1280px; margin:0 auto; padding:24px 24px 64px; font-size:14px; line-height:1.45; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:28px 0 8px; } h3 { font-size:15px; margin:0; }
.sub { color:var(--ink-2); margin:0 0 16px; }
.card { background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:16px; }
.how { display:grid; grid-template-columns:1fr 1fr; gap:6px 24px; color:var(--ink-2); margin:0; padding-left:18px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--grid); white-space:nowrap; }
th { color:var(--ink-2); font-weight:600; font-size:12px; }
td.num, th.num { text-align:right; }
.sw { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:baseline; }
.sw.base { background:transparent; border:2px dashed var(--muted); width:11px; height:11px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 12px; }
.chip { border:1px solid var(--border); background:var(--surface-1); color:var(--ink); border-radius:999px;
  padding:6px 12px; cursor:pointer; font:inherit; }
.chip[aria-pressed="true"] { background:var(--ink); color:var(--surface-1); border-color:var(--ink); }
.ep-head { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:10px; }
.ep-desc { color:var(--ink-2); margin:4px 0 0; max-width:820px; }
details summary { cursor:pointer; color:var(--ink-2); }
.script { margin:8px 0 0; font-size:13px; color:var(--ink-2); columns:2; column-gap:28px; }
.script div { break-inside:avoid; margin-bottom:3px; }
.script b { color:var(--ink); font-weight:600; }
.legend { display:flex; flex-wrap:wrap; gap:14px; font-size:12px; color:var(--ink-2); margin:10px 0 6px; align-items:center; }
.key { display:inline-block; width:18px; height:8px; border-radius:2px; vertical-align:middle; margin-right:5px; }
.row { display:grid; grid-template-columns:230px 1fr; gap:4px 16px; padding:10px 0; border-top:1px solid var(--grid); }
.row.playing { background:linear-gradient(90deg, rgba(42,120,214,0.06), transparent 60%); }
.lab { font-size:13px; } .lab .role { color:var(--ink-2); font-size:12px; } .lab .agg { color:var(--muted); font-size:11.5px; margin-top:2px; }
.lab audio { width:100%; height:30px; margin-top:6px; }
svg.tl { width:100%; height:64px; display:block; cursor:pointer; }
.words { font-size:12px; color:var(--ink-2); margin-top:2px; line-height:1.5; }
.words .f { color:var(--muted); font-style:italic; }
.badges { display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; }
.badge { font-size:11.5px; border-radius:6px; padding:2px 7px; border:1px solid var(--border); color:var(--ink); background:var(--surface-1); }
.badge .ic { font-weight:700; margin-right:4px; }
.badge.good .ic { color:var(--good); } .badge.bad .ic { color:var(--critical); } .badge.warn .ic { color:#b07a00; }
.note { color:var(--muted); font-size:12px; margin-top:8px; }
img.plot { width:100%; border:1px solid var(--border); border-radius:8px; background:#fcfcfb; margin-top:8px; }
</style></head>
<body><div class="viz-root">
<h1>Teaching Moshi when to stay silent: listening demo</h1>
<p class="sub">Fine-tuned on <b>1 h</b> (6 learning rates, 20 epochs) and <b>10 h</b> (4 learning rates, 5 epochs) of synthetic conversations, plus runs with an up-weighted "start talking" token (<b>EPAD ×k</b>) at 1 h and 10 h, and one 50 h run (v2 data, EPAD ×10, 1 epoch). Representative checkpoints and the unmodified base model, all on the same unseen dev conversations.</p>
<div class="chips" id="datafilter" role="group" aria-label="Training data"></div>

<div class="card">
<h3>How to read this</h3>
<ul class="how">
<li>Pick a conversation below. Each row is one model; press play to hear it (headphones help: Moshi is on the
left, the user and friend on the right).</li>
<li><b>Timeline:</b> top lane = user / friend speech; bottom lane = Moshi's speech in that model's audio.</li>
<li>The <b>shaded band</b> is the requested silence; the vertical line is where Moshi should speak again.</li>
<li><span style="color:var(--critical);font-weight:600">Red</span> Moshi speech = talking where it should be silent
(after a 2.5 s grace for a short "okay").</li>
<li>Only the parts under <b>"model generates"</b> are the model's own choice. Elsewhere the eval replays
the scripted Moshi (gray), so every hold starts from the same context.</li>
<li>Click a timeline to jump there. The text under each row is Moshi's own words stream
(gray italics = replayed script).</li>
</ul>
<p class="note" style="margin:10px 0 0">The conversations were chosen automatically, one per behavior type,
as the dev episode on which these models' outcomes differ the most. So they are harder and more contrasting than a
typical episode: use the table above for the average picture and these for what successes and failures sound like.</p>
</div>

<h2>The models</h2>
<div class="card" style="overflow-x:auto"><table id="models"></table>
<p class="note">Rates are over all 80 dev_80 conversations (58 scored holds, 63 resumes, 42 ordinary questions; ±10–12 points at 95%).
Dev loss = teacher-forced text loss on 200 dev conversations.</p></div>

<h2>Conversations</h2>
<div class="chips" id="chips" role="group" aria-label="Conversation"></div>
<div class="card" id="episode"></div>

<h2>The curves behind it</h2>
<div class="card">
<p class="sub" style="margin:0">Behavior per epoch (top) and loss (bottom). Note how the runs with the lowest dev loss
are not the ones that behave best.</p>
<div id="plots"></div>
</div>
</div>

<script>
const DATA = __DATA__;
let FILTER = "all";
const shown = m => m.run === "R0" || FILTER === "all" || m.data === FILTER;
document.getElementById("plots").innerHTML = DATA.plots.map(p => `<p class="note" style="margin:14px 0 0">${p.alt}</p><img class="plot" src="${p.src}" alt="${p.alt}">`).join("");
const pct = v => v == null ? "–" : Math.round(v * 100) + "%";
const fmt = (v, d = 2) => v == null ? "–" : v.toFixed(d);
const lat = v => v == null ? "" : (v < 0 ? ` (${(-v).toFixed(1)} s early)` : ` (+${v.toFixed(1)} s)`);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const css = n => getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(n).trim();

// ---------- models table
function renderModels() {
  const t = document.getElementById("models");
  let h = `<tr><th>Model</th><th>Checkpoint</th><th class="num">Epoch</th><th class="num">Dev text loss</th>
    <th class="num">Hold violated ↓</th><th class="num">Resumed on time ↑</th><th class="num">Ordinary answered ↑</th></tr>`;
  for (const m of DATA.models.filter(shown)) {
    const sw = m.run === "R0" ? `<span class="sw base"></span>` : `<span class="sw" style="background:${m.color}"></span>`;
    h += `<tr><td>${sw}${esc(m.lr_label)}</td><td>${esc(m.role)}${m.note ? `<div class="note" style="white-space:normal;max-width:420px">${esc(m.note)}</div>` : ""}</td><td class="num">${m.run === "R0" ? "–" : fmt(m.epoch, 1)}</td>
      <td class="num">${fmt(m.dev_text, 3)}</td><td class="num">${pct(m.viol)}</td><td class="num">${pct(m.on_time)}</td><td class="num">${pct(m.ord)}</td></tr>`;
  }
  t.innerHTML = h;
}
renderModels();
const df = document.getElementById("datafilter");
[["all", "All models"], ["1 h", "1 h runs"], ["10 h", "10 h runs"]].forEach(([v, label]) => {
  const b = document.createElement("button"); b.className = "chip"; b.textContent = label; b.setAttribute("aria-pressed", v === FILTER);
  b.onclick = () => { FILTER = v; [...df.children].forEach(c => c.setAttribute("aria-pressed", c === b)); renderModels();
    const cur = DATA.episodes.find(e => "#" + e.id === location.hash) || DATA.episodes[0]; showEpisode(cur); };
  df.appendChild(b);
});

// ---------- episode chips
const chips = document.getElementById("chips");
DATA.episodes.forEach((ep, i) => {
  const b = document.createElement("button");
  b.className = "chip"; b.textContent = ep.title; b.setAttribute("aria-pressed", i === 0);
  b.dataset.id = ep.id;
  b.onclick = () => { [...chips.children].forEach(c => c.setAttribute("aria-pressed", c === b)); history.replaceState(null, "", "#" + ep.id); showEpisode(ep); };
  chips.appendChild(b);
});

function badgesFor(ep, r) {
  const out = [];
  r.holds.forEach((h, i) => {
    const tag = r.holds.length > 1 ? ` (hold ${i + 1})` : "";
    if (h.violated === true) out.push(["bad", "✗", `talked during hold${tag}: ${h.words_in_hold} words`]);
    else if (h.violated === false) out.push(["good", "✓", `stayed silent${tag}`]);
    if (h.outcome === "on_time") out.push(["good", "✓", `resumed on time${tag}` + lat(h.latency)]);
    else if (h.outcome === "late") out.push(["warn", "!", `resumed late${tag}` + lat(h.latency)]);
    else if (h.outcome === "no_onset") out.push(["bad", "✗", h.talking_through ? `talked through the release${tag}` : `didn't resume${tag}`]);
    (h.distractors || []).forEach(d => { if (d.reacted) out.push(["bad", "✗", `reacted to ${d.code} distractor`]); });
  });
  r.responses.forEach((x, i) => {
    out.push(x.responded ? ["good", "✓", `answered question ${i + 1}` + lat(x.latency)]
                         : ["bad", "✗", `didn't answer question ${i + 1}`]);
  });
  if (r.interrupt_onsets > 0) out.push(["bad", "✗", `${r.interrupt_onsets} interruption${r.interrupt_onsets > 1 ? "s" : ""} of the user`]);
  (r.barge_ins || []).forEach(b => { if (b.barged) out.push(["bad", "✗", "barged into a pause"]); });
  return out.map(([c, ic, t]) => `<span class="badge ${c}"><span class="ic">${ic}</span>${esc(t)}</span>`).join("");
}

function overlap(a0, a1, b0, b1) { return [Math.max(a0, b0), Math.min(a1, b1)]; }

function timeline(ep, r, color) {
  const W = 1000, D = ep.duration, x = t => (t / D) * W;
  const s = [];
  // hold bands + release lines
  ep.holds.forEach(h => {
    s.push(`<rect x="${x(h.instr_end)}" y="0" width="${x(h.ref) - x(h.instr_end)}" height="50" fill="${css("--hold")}"/>`);
    s.push(`<line x1="${x(h.ref)}" x2="${x(h.ref)}" y1="0" y2="50" stroke="${css("--ink")}" stroke-width="1.5"/>`);
  });
  // free windows (model generates)
  r.free.forEach(([a, b]) => {
    s.push(`<line x1="${x(a)}" x2="${x(b)}" y1="56" y2="56" stroke="${css("--ink-2")}" stroke-width="2" stroke-linecap="round"/>`);
  });
  s.push(`<text x="2" y="63" font-size="8" fill="${css("--muted")}">model generates ↑ (elsewhere: replayed script)</text>`);
  // human lane
  ep.human_turns.forEach(t => {
    const c = t.spk === "friend" ? css("--friend") : css("--user");
    s.push(`<rect x="${x(t.t[0])}" y="6" width="${Math.max(1, x(t.t[1]) - x(t.t[0]))}" height="12" rx="3" fill="${c}"><title>${esc((t.spk || "") + ": " + (t.text || ""))}</title></rect>`);
  });
  ep.distractors.forEach(d => {
    const t0 = d.t[0];
    s.push(`<path d="M${x(t0) - 4},2 L${x(t0) + 4},2 L${x(t0)},7 Z" fill="${css("--ink-2")}"><title>${esc(d.code + (d.text ? ": " + d.text : " (stall)"))}</title></path>`);
  });
  // moshi lane: generated (row color), replayed script (gray), violation (red)
  const inFree = t => r.free.some(([a, b]) => t >= a && t < b);
  r.segments.forEach(([a, b]) => {
    const mid = (a + b) / 2;
    const c = inFree(mid) ? color : css("--forced");
    s.push(`<rect x="${x(a)}" y="28" width="${Math.max(1, x(b) - x(a))}" height="14" rx="3" fill="${c}"/>`);
    ep.holds.forEach(h => {
      const [o0, o1] = overlap(a, b, h.region[0], h.region[1]);
      if (o1 > o0) s.push(`<rect x="${x(o0)}" y="28" width="${x(o1) - x(o0)}" height="14" fill="${css("--critical")}"><title>speech during hold</title></rect>`);
    });
  });
  // axis ticks
  for (let t = 0; t <= D; t += 5) {
    s.push(`<line x1="${x(t)}" x2="${x(t)}" y1="47" y2="50" stroke="${css("--axis")}"/>`);
    s.push(`<text x="${x(t) + 2}" y="49" font-size="7.5" fill="${css("--muted")}">${t}s</text>`);
  }
  s.push(`<line class="ph" x1="0" x2="0" y1="0" y2="50" stroke="${css("--ink")}" stroke-width="1.2" opacity="0"/>`);
  return `<svg class="tl" viewBox="0 0 ${W} 64" preserveAspectRatio="none">${s.join("")}</svg>`;
}

function wordsLine(r) {
  const inFree = t => r.free.some(([a, b]) => t >= a && t < b);
  if (!r.words.length) return `<span class="f">(silent)</span>`;
  return r.words.map(([w, t]) => inFree(t) ? esc(w) : `<span class="f">${esc(w)}</span>`).join(" ");
}

function scriptHTML(ep) {
  const items = [...ep.human_turns.map(t => ({t: t.t[0], who: t.spk === "friend" ? "Friend" : "User", text: t.text})),
                 ...ep.moshi_turns.map(t => ({t: t.t[0], who: "Moshi (script)", text: t.text}))]
    .sort((a, b) => a.t - b.t);
  ep.holds.forEach(h => items.push({t: h.instr_end + 0.001, who: "", text: `⟶ hold starts (${h.kind === "timer" ? h.N + " s timer" : "until cue"}); Moshi should speak again at ${h.ref.toFixed(1)} s`}));
  items.sort((a, b) => a.t - b.t);
  return items.map(i => `<div>${i.t.toFixed(1)}s <b>${esc(i.who)}</b> ${esc(i.text || "")}</div>`).join("");
}

function showEpisode(ep) {
  document.querySelectorAll("audio").forEach(a => a.pause());
  const el = document.getElementById("episode");
  let h = `<div class="ep-head"><div><h3>${esc(ep.title)} <span style="color:var(--muted);font-weight:400">· ${esc(ep.id)} · ${ep.type} · ${ep.duration.toFixed(0)} s</span></h3>
    <p class="ep-desc">${esc(ep.desc)}</p></div></div>
    <details><summary>Script (what the user and friend say; what the scripted Moshi said in the training-style data)</summary>
    <div class="script">${scriptHTML(ep)}</div></details>
    <div class="legend">
      <span><span class="key" style="background:var(--user)"></span>user</span>
      <span><span class="key" style="background:var(--friend)"></span>friend</span>
      <span><span class="key" style="background:var(--hold)"></span>requested silence</span>
      <span><span class="key" style="background:var(--ink);width:2px;height:12px"></span>Moshi should speak again</span>
      <span><span class="key" style="background:#2a78d6"></span>Moshi, generated (row color; base model = black)</span>
      <span><span class="key" style="background:var(--forced)"></span>Moshi, replayed script</span>
      <span><span class="key" style="background:var(--critical)"></span>Moshi talking during the hold</span>
      <span>▾ distractor</span>
    </div>`;
  for (const m of DATA.models.filter(shown)) {
    const r = DATA.rows[ep.id][m.key];
    const color = m.run === "R0" ? css("--ink") : m.color;
    const sw = m.run === "R0" ? `<span class="sw base"></span>` : `<span class="sw" style="background:${m.color}"></span>`;
    h += `<div class="row" data-key="${m.key}">
      <div class="lab">${sw}<b>${esc(m.lr_label)}</b>${m.run === "R0" ? "" : ` · ep ${fmt(m.epoch, 1)}`}
        <div class="role">${esc(m.role)}${m.dev_text != null ? ` · dev loss ${fmt(m.dev_text, 3)}` : ""}</div>
        ${m.note ? `<div class="agg" style="color:var(--ink-2)">${esc(m.note)}</div>` : ""}
        <div class="agg">all dev_80: violated ${pct(m.viol)} · on time ${pct(m.on_time)} · answered ${pct(m.ord)}</div>
        ${r.audio ? `<audio controls preload="none" src="${r.audio}"></audio>` : `<div class="agg" style="margin-top:6px">audio not decoded yet (Mimi codes still on the GPU volume)</div>`}</div>
      <div>${timeline(ep, r, color)}<div class="badges">${badgesFor(ep, r)}</div><div class="words">${wordsLine(r)}</div></div>
    </div>`;
  }
  el.innerHTML = h;
  el.querySelectorAll(".row").forEach(row => {
    const audio = row.querySelector("audio"), svg = row.querySelector("svg"), ph = svg.querySelector(".ph");
    const D = ep.duration;
    if (!audio) return;  // audio not decoded yet for this model
    audio.addEventListener("play", () => {
      document.querySelectorAll("audio").forEach(a => { if (a !== audio) a.pause(); });
      document.querySelectorAll(".row").forEach(r => r.classList.toggle("playing", r === row));
      ph.setAttribute("opacity", "1");
    });
    audio.addEventListener("timeupdate", () => {
      const xx = (audio.currentTime / D) * 1000;
      ph.setAttribute("x1", xx); ph.setAttribute("x2", xx);
    });
    svg.addEventListener("click", ev => {
      const b = svg.getBoundingClientRect();
      audio.currentTime = ((ev.clientX - b.left) / b.width) * D;
      audio.play();
    });
  });
}
const start = DATA.episodes.find(e => "#" + e.id === location.hash) || DATA.episodes[0];
[...chips.children].forEach(c => c.setAttribute("aria-pressed", c.dataset.id === start.id));
showEpisode(start);
</script>
</body></html>
"""

if __name__ == "__main__":
    main()
