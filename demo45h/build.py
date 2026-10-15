"""Build the "Moshi Fine-tuning Demo" site: base vs the 45h model on chosen held-out conversations.

    ~/repo/moshi-silence/.venv/bin/python demo45h/build.py            # final site  -> demo45h/site/
    ~/repo/moshi-silence/.venv/bin/python demo45h/build.py --review   # all candidates (A/B/C) -> demo45h/review/

Choices per card: demo45h/choices.json ({"1_side_conversation": "A", ...}); candidates: demo45h/candidates.json
(from pick.py). The output folder is self-contained: index.html with every number inlined (no fetch, no server,
no CDN, no web fonts) + audio/*.mp3 + figs/*.png. Open index.html directly on any computer, or host it anywhere.

Per conversation and model: Mimi-decode the saved codes (sample 0, CPU), mix L = Moshi + 0.3 x input,
R = input + 0.3 x Moshi into a 64 kbps mp3, and keep waveform peaks of the two sides separately.
"""
import argparse
import html
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

R = Path("results")
DATA = Path.home() / "repo/moshi-silence/data"
MIMI = Path.home() / "repo/moshi-silence/models/moshiko/tokenizer-e351c8d8-checkpoint125.safetensors"
K45 = "checkpoint_000527"
BINS = 500
GRACE, CUE_PRE = 2.5, 0.25

CARDS = {
    "1_side_conversation": ("Talking to a friend",
                            "“Stay quiet until I tell you I'm back.” The user and a friend talk; Moshi should wait, then come back on the cue."),
    "2_cue_lookalike": ("A line that sounds like the cue",
                        "During the side conversation the user says something that resembles “I'm back” to the friend. Moshi should keep waiting."),
    "3_hear_me_out_pauses": ("Hear me out",
                             "“Let me describe all three options before you comment,” with long pauses between options. Moshi should not jump into a pause."),
    "4_hear_me_out_cancelled": ("Instruction taken back",
                                "The user asks Moshi to hear them out, then changes their mind. Moshi should speak again."),
    "5_timed_hold": ("“Give me N seconds, then ask me again”",
                     "No one gives a cue: Moshi has to keep time itself and re-ask when the time is up."),
    "6_ordinary_with_pause": ("An ordinary question",
                              "A normal request with a pause mid-sentence. Moshi should still answer, without cutting in during the pause."),
    "7a_my_voice_scripted": ("My own voice",
                             "The user's side recorded by me, reading the held-out script."),
    "7b_my_voice_own_words": ("My own voice, my own wording",
                              "Recorded by me, with the instruction and the return cue in my own words (not the script's)."),
    "8a_FAIL_friend_says_im_back": ("Failure: the friend says “I'm back”",
                                    "Only the user's “I'm back” should count. Here the friend says it, and the 45h model often takes the bait."),
    "8b_FAIL_minute_hold": ("Failure: a minute-long hold",
                            "“Give me a minute.” The longest hold in training was 30 s, and this goes past the 60 s training window."),
}
SUITE_NOTE = {"test_script": "unseen script", "test_word": "unseen wording", "test_voice": "unseen voices",
              "test_distract": "dense distractors", "test_cond": "unseen condition", "test_ordinary": "ordinary",
              "test_all": "unseen script, wording and voices", "test_human": "my voice",
              "test_human_free": "my voice, own words"}
_mimi = None


def gen_dir(suite, model):
    sub = "base" if model == "base" else K45
    if suite.startswith("test_human"):
        return R / "human/raw" / suite / sub
    return R / ("test/raw" if model == "base" else "test/raw_R24") / suite / sub


def peaks(x):
    n = len(x) // BINS * BINS
    return np.abs(x[:n]).reshape(BINS, -1).max(1)


def decode(job):
    global _mimi
    import soundfile as sf
    import torch
    torch.set_num_threads(2)
    if _mimi is None:
        from moshi.models import loaders
        _mimi = loaders.get_mimi(str(MIMI), device="cpu")
        _mimi.set_num_codebooks(8)
    npz, key, wav, out = job
    codes = np.load(npz)[key]
    with torch.no_grad():
        m = _mimi.decode(torch.from_numpy(codes[1:].astype(np.int64))[None]).float().numpy()[0, 0]
    u, sr = sf.read(wav, dtype="float32")
    u = u[:, 1]
    n = min(len(m), len(u))
    m, u = m[:n], u[:n]
    st = np.stack([m + 0.3 * u, u + 0.3 * m], 1)
    pk = np.abs(st).max()
    if pk > 0.95:
        st *= 0.95 / pk
    out = Path(out)
    if not out.exists():
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "x.wav"
            sf.write(tmp, st, sr, subtype="PCM_16")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:a", "libmp3lame", "-b:a", "64k",
                            str(out)], check=True)
    return out.name, peaks(m).tolist(), peaks(u).tolist(), n / sr


def intents():
    p = DATA / "human/s01/session.json"
    if not p.exists():
        return {}
    plan = json.loads(p.read_text())
    return {(it["episode"], it["anchor"]): it["intent"] for it in plan["items"] if it["block"] == "B"}


def script(suite, eid, intent):
    ep = json.loads((DATA / suite / "episodes" / f"{eid}.json").read_text())
    ev = ep["events"]
    spec = json.loads((DATA / suite / "episodes" / f"{eid}.spec.json").read_text())
    anchors = [t.get("anchor") for t in spec["turns"] if t["spk"] != "moshi"]
    human = []
    for k, t in enumerate(ev["human_turns"]):
        txt = t["text"]
        if txt.startswith("free:"):
            a = anchors[k] if k < len(anchors) else None
            txt = f"[own words: {intent.get((eid, a), 'instruction / cue')}]"
        human.append({"spk": t["spk"], "t": t["t"], "text": txt})
    holds = []
    for h in ev["holds"]:
        rel = h["release"]
        timer = rel["kind"] == "timer"
        tol = max(1.5, 0.2 * rel["N"]) if timer else CUE_PRE
        holds.append({"t0": h["instr_end"], "ref": rel["ref"], "kind": rel["kind"], "N": rel.get("N"),
                      "region": [h["instr_end"] + GRACE, rel["ref"] - tol]})
    dist = [{"code": d["code"], "t": d["t"], "text": d.get("text", "")} for d in ev.get("distractors", [])]
    return {"human": human, "holds": holds, "dist": dist, "type": ev["type"]}


def verdict(row):
    out = []
    for h in row["holds"]:
        if h["violated"] is None:
            pass
        elif h["violated"]:
            out.append([0, f"talked during the requested silence ({h['words_in_region']} words)"])
        else:
            out.append([1, "stayed silent"])
        if h["outcome"] == "on_time":
            lat = h.get("latency")
            out.append([1, "spoke again on time" + (f" ({lat:+.1f} s)" if lat is not None and h["kind"] == "timer" else "")])
        elif h["outcome"] == "late":
            out.append([0, "spoke again too late"])
        else:
            out.append([0, "did not speak again"])
        for d in h.get("distractors", []):
            if d["reacted"]:
                out.append([0, f"reacted to a distractor ({d['code']})"])
    if row["responses"]:
        k = sum(x["responded"] for x in row["responses"])
        out.append([int(k == len(row["responses"])), f"answered {k} of {len(row['responses'])} questions"])
    for b in row["barge_ins"]:
        if b["barged"]:
            out.append([0, "cut into the user's pause"])
    return out


def moshi_lines(gen):
    """Every word Moshi's text stream produced, grouped into lines at gaps > 1 s or at a generated/replayed
    boundary (nothing dropped, unlike grouping by VAD segments)."""
    free = gen["free"]
    is_gen = lambda t: any(f0 <= t < f1 for f0, f1 in free)
    lines = []
    for w, t in gen["words"]:
        if w in ("<s>", "</s>"):
            continue
        g = is_gen(t)
        if lines and t - lines[-1]["t"][1] <= 1.0 and lines[-1]["gen"] == g:
            lines[-1]["text"] += " " + w
            lines[-1]["t"][1] = t + 0.3
        else:
            lines.append({"t": [t, t + 0.3], "text": w, "gen": g})
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true", help="every candidate, labeled A/B/C")
    a = ap.parse_args()
    out = Path("demo45h/review" if a.review else "demo45h/site")
    (out / "audio").mkdir(parents=True, exist_ok=True)
    cands = json.loads(Path("demo45h/candidates.json").read_text())
    choice = {} if a.review else json.loads(Path("demo45h/choices.json").read_text())
    intent = intents()

    items = []
    for card, cs in cands.items():
        for k, c in enumerate(cs):
            letter = "ABC"[k]
            if not a.review and choice.get(card, "A") != letter:
                continue
            items.append((card, letter, c))

    jobs, meta = [], []
    for card, letter, c in items:
        suite, eid = c["suite"], c["id"]
        human = suite.startswith("test_human")
        key = f"{eid}_s0" if human else eid
        wav = DATA / suite / "episodes" / f"{eid}.wav"
        rows = {}
        for model in ("base", "45h"):
            d = gen_dir(suite, model)
            gen = json.loads((d / f"{key}.json").read_text())
            sc = next(json.loads(l) for l in (d / "scores.jsonl").read_text().splitlines() if json.loads(l)["id"] == key)
            jobs.append((str(d / "codes.npz"), key, str(wav), str(out / "audio" / f"{suite}_{eid}_{model}.mp3")))
            rows[model] = {"gen": gen, "score": sc}
        meta.append((card, letter, c, rows))
    with ProcessPoolExecutor(5) as ex:
        decoded = list(ex.map(decode, jobs))

    eps = []
    for i, (card, letter, c, rows) in enumerate(meta):
        sc = script(c["suite"], c["id"], intent)
        models = []
        for j, model in enumerate(("base", "45h")):
            mp3, pm, pu, dur = decoded[2 * i + j]
            g, s = rows[model]["gen"], rows[model]["score"]
            region_hits = [[max(x0, h["region"][0]), min(x1, h["region"][1])]
                           for x0, x1 in g["segments"] for h in sc["holds"]
                           if min(x1, h["region"][1]) > max(x0, h["region"][0])]
            models.append({"name": "Base model" if model == "base" else "Fine-tuned (45h)", "audio": "audio/" + mp3,
                           "pm": [round(v, 3) for v in pm], "pu": [round(v, 3) for v in pu],
                           "segs": g["segments"], "free": g["free"], "bad": region_hits,
                           "moshi": moshi_lines(g), "verdict": verdict(s)})
        scale_m = max(max(m["pm"]) for m in models) or 1
        scale_u = max(max(m["pu"]) for m in models) or 1
        title, desc = CARDS[card]
        eps.append({"card": card, "letter": letter, "title": title, "desc": desc, "fail": "FAIL" in card,
                    "src": f"{SUITE_NOTE.get(c['suite'], c['suite'])} · {c['id']}", "dur": round(dur, 2),
                    "human": sc["human"], "holds": sc["holds"], "dist": sc["dist"], "models": models,
                    "sm": scale_m, "su": scale_u})

    bars = []
    if not a.review:
        ts = json.loads((R / "test/summary.json").read_text())
        hs = json.loads((R / "human/summary.json").read_text())
        k45 = next(k for k in ts if "R24" in k)
        h45 = next(k for k in hs if "R24" in k)
        for su, lab in [("test_script", "new scripts"), ("test_word", "new instruction wording"),
                        ("test_voice", "unseen voices"), ("test_distract", "dense distractors"),
                        ("test_cond", "new conditions (keyword, phone call, a minute)"),
                        ("test_ordinary", "ordinary conversation"), ("test_all", "everything held out")]:
            bars.append([lab, round(ts["base"][su]["score"], 2), round(ts[k45][su]["score"], 2), "synthetic, 90 conversations"])
        for su, lab in [("test_human", "my voice, scripted"), ("test_human_free", "my voice, my own wording")]:
            bars.append([lab, round(hs["base"][su]["score"], 2), round(hs[h45][su]["score"], 2), "recorded, 24 conversations"])
        shutil.rmtree(out / "figs", ignore_errors=True)
    page = TEMPLATE.replace("__DATA__", json.dumps({"eps": eps, "review": a.review, "bars": bars}, separators=(",", ":")))
    (out / "index.html").write_text(page)
    used = {Path(m["audio"]).name for e in eps for m in e["models"]}
    for f in (out / "audio").iterdir():
        if f.name not in used:
            f.unlink()
    print(f"wrote {out}/index.html with {len(eps)} conversations, {len(used)} clips")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Moshi Fine-tuning Demo</title>
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--card:#ffffff;
--user:#9a998f;--moshi:#2a78d6;--bad:#c8382c;--hold:rgba(42,120,214,.08);--good:#1d7a4a}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--ink:#f2f1ec;--ink2:#c3c2b7;--muted:#898781;--line:#2c2b28;
--card:#1c1c1a;--user:#6f6e67;--moshi:#5d9be6;--bad:#e5675b;--hold:rgba(93,155,230,.12);--good:#5bbf88}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:28px;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--ink2);margin:0 0 22px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:0 0 26px}
.stat{border:1px solid var(--line);border-radius:10px;padding:10px 12px;background:var(--card)}
.stat b{display:block;font-size:20px}.stat span{color:var(--ink2);font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--ink2);font-size:13px;margin:0 0 28px}
.legend i{display:inline-block;width:14px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
section.ep{border-top:1px solid var(--line);padding:26px 0 8px}
section.ep h2{font-size:18px;margin:0}
.desc{color:var(--ink2);margin:4px 0 2px}.src{color:var(--muted);font-size:12px;margin-bottom:14px}
.fail h2::after{content:"limitation";font-size:11px;font-weight:600;color:var(--bad);border:1px solid var(--bad);
border-radius:99px;padding:1px 7px;margin-left:8px;vertical-align:3px}
.row{margin:12px 0 18px}
.rowhead{display:flex;align-items:center;gap:10px;margin-bottom:4px}
.rowhead b{font-size:14px;min-width:120px}
button.play{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:99px;width:30px;height:30px;
cursor:pointer;font-size:12px;line-height:1}
button.play:focus-visible{outline:2px solid var(--moshi);outline-offset:2px}
.time{color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
svg.wf{width:100%;height:92px;display:block;cursor:pointer;background:var(--card);border:1px solid var(--line);border-radius:8px}
.verdict{display:flex;flex-wrap:wrap;gap:6px 12px;font-size:13px;margin-top:6px}
.ok{color:var(--good)}.no{color:var(--bad)}
details{margin:6px 0 0;font-size:13px}summary{cursor:pointer;color:var(--ink2)}
.tr{margin:6px 0 0;padding:0;list-style:none}.tr li{display:grid;grid-template-columns:44px 64px 1fr;gap:6px;padding:2px 0}
.tr .t{color:var(--muted);font-variant-numeric:tabular-nums}.tr .w{color:var(--ink2)}
.tr li.m .x{color:var(--moshi)}.tr li.m.f .x{color:var(--muted);font-style:italic}.tr li.m.bad .x{color:var(--bad)}
footer{border-top:1px solid var(--line);margin-top:30px;padding-top:20px;color:var(--ink2);font-size:13px}
.bars{display:grid;grid-template-columns:minmax(150px,1.2fr) 2fr;gap:8px 14px;margin:14px 0 18px;align-items:center}
.bars .l{color:var(--ink);font-size:13px}.bars .l small{display:block;color:var(--muted);font-size:11px}
.bar{display:flex;align-items:center;gap:8px;height:12px;margin:3px 0}.bar i{height:10px;border-radius:0 3px 3px 0}
.bar span{font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums;white-space:nowrap}
@media (max-width:560px){.stats{grid-template-columns:1fr}.rowhead b{min-width:0}}
</style>
</head>
<body>
<main>
<h1>Moshi Fine-tuning Demo</h1>
<p class="sub">Teaching a full-duplex speech model when to stay silent and when to speak again. Each conversation
below was never seen in training; the same recording is played to the <b>base model</b> and to Moshi
<b>fine-tuned on 45 h of synthetic conversations</b>. The silence comes from the model itself; there is no mute rule.</p>
<div class="stats" id="stats"></div>
<div class="legend">
<span><i style="background:var(--user)"></i>user + friend</span>
<span><i style="background:var(--moshi)"></i>Moshi</span>
<span><i style="background:var(--hold);outline:1px solid var(--line)"></i>Moshi asked to stay silent</span>
<span><i style="background:var(--ink);width:2px"></i>when Moshi should speak again</span>
<span><i style="background:var(--bad)"></i>Moshi talking when it should be silent</span>
</div>
<p class="sub" style="font-size:13px">Transcripts: user and friend lines are the script; <span style="color:var(--moshi)">blue</span>
is what Moshi generated, <span style="color:var(--bad)">red</span> is Moshi talking during the requested silence, and
<i style="color:var(--muted)">gray italics</i> are Moshi's scripted lines replayed before the part being tested.</p>
<div id="eps"></div>
<footer id="foot"></footer>
</main>
<script>
const D = __DATA__;
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = t => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

if (!D.review) {
  document.getElementById("stats").innerHTML = [
    ["0.29 → 0.94", "behavior score on unseen conversations (dev)"],
    ["0.32 → 0.86", "everything held out: new scripts, wording, voices"],
    ["0.33 → 0.83", "on my own recorded voice"]
  ].map(([n, t]) => `<div class="stat"><b>${n}</b><span>${t}<br>base → 45h</span></div>`).join("");
} else document.getElementById("stats").remove();

function wave(ep, m) {
  const W = 1000, H = 92, D_ = ep.dur, x = t => (t / D_) * W, n = m.pm.length, bw = W / n;
  const s = [];
  ep.holds.forEach(h => {
    s.push(`<rect x="${x(h.t0)}" y="0" width="${Math.max(0, x(h.ref) - x(h.t0))}" height="${H}" fill="var(--hold)"/>`);
  });
  // user + friend (top), Moshi (bottom), as mirrored peak bars
  m.pu.forEach((v, i) => {
    const a = Math.min(1, v / ep.su) * 19;
    if (a > 0.4) s.push(`<rect x="${i * bw}" y="${24 - a}" width="${bw * 0.8}" height="${2 * a}" fill="var(--user)"/>`);
  });
  const inBad = t => m.bad.some(([a, b]) => t >= a && t < b);
  m.pm.forEach((v, i) => {
    const a = Math.min(1, v / ep.sm) * 19, t = (i + 0.5) * D_ / n;
    if (a > 0.4) s.push(`<rect x="${i * bw}" y="${68 - a}" width="${bw * 0.8}" height="${2 * a}" fill="${inBad(t) ? "var(--bad)" : "var(--moshi)"}"/>`);
  });
  ep.holds.forEach(h => s.push(`<line x1="${x(h.ref)}" x2="${x(h.ref)}" y1="0" y2="${H}" stroke="var(--ink)" stroke-width="2"/>`));
  ep.dist.forEach(d => s.push(`<path d="M${x(d.t[0]) - 5},1 L${x(d.t[0]) + 5},1 L${x(d.t[0])},7 Z" fill="var(--ink2)"><title>${esc(d.code + ": " + (d.text || "pause"))}</title></path>`));
  s.push(`<text x="6" y="12" font-size="10" fill="var(--muted)">user + friend</text>`);
  s.push(`<text x="6" y="88" font-size="10" fill="var(--muted)">Moshi</text>`);
  s.push(`<line class="ph" x1="0" x2="0" y1="0" y2="${H}" stroke="var(--ink)" stroke-width="1.2" opacity="0"/>`);
  return `<svg class="wf" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="waveform and timeline">${s.join("")}</svg>`;
}

function transcript(ep, m) {
  const inBad = (a, b) => m.bad.some(([x0, x1]) => Math.min(b, x1) > Math.max(a, x0));
  const items = [...ep.human.map(h => ({t: h.t[0], who: h.spk === "friend" ? "Friend" : "User", text: h.text, cls: ""})),
                 ...m.moshi.map(l => ({t: l.t[0], who: "Moshi", text: l.text, cls: "m" + (l.gen ? "" : " f") + (inBad(l.t[0], l.t[1]) ? " bad" : "")}))]
    .sort((a, b) => a.t - b.t);
  return `<ul class="tr">${items.map(i => `<li class="${i.cls}"><span class="t">${i.t.toFixed(1)}s</span><span class="w">${i.who}</span><span class="x">${esc(i.text)}</span></li>`).join("")}</ul>`;
}

const host = document.getElementById("eps");
D.eps.forEach(ep => {
  const sec = document.createElement("section");
  sec.className = "ep" + (ep.fail ? " fail" : "");
  sec.innerHTML = `<h2>${D.review ? esc(ep.card) + " · " + ep.letter + " — " : ""}${esc(ep.title)}</h2>
    <p class="desc">${esc(ep.desc)}</p><div class="src">${esc(ep.src)} · ${fmt(ep.dur)}</div>` +
    ep.models.map((m, k) => `<div class="row" data-k="${k}">
      <div class="rowhead"><button class="play" aria-label="play ${esc(m.name)}">▶</button><b>${esc(m.name)}</b><span class="time">0:00 / ${fmt(ep.dur)}</span></div>
      ${wave(ep, m)}
      <div class="verdict">${m.verdict.map(([g, t]) => `<span class="${g ? "ok" : "no"}">${g ? "✓" : "✗"} ${esc(t)}</span>`).join("")}</div>
      <details open><summary>full transcript</summary>${transcript(ep, m)}</details>
      <audio preload="none" src="${m.audio}"></audio></div>`).join("");
  host.appendChild(sec);
});

const all = [...document.querySelectorAll("audio")];
document.querySelectorAll(".row").forEach(row => {
  const au = row.querySelector("audio"), btn = row.querySelector(".play"), svg = row.querySelector("svg"),
        ph = svg.querySelector(".ph"), tm = row.querySelector(".time"), total = tm.textContent.split(" / ")[1];
  btn.onclick = () => { if (au.paused) { all.forEach(o => o !== au && o.pause()); au.play(); } else au.pause(); };
  au.onplay = () => btn.textContent = "❚❚";
  au.onpause = au.onended = () => btn.textContent = "▶";
  au.ontimeupdate = () => {
    const f = au.currentTime / (au.duration || 1);
    ph.setAttribute("x1", f * 1000); ph.setAttribute("x2", f * 1000); ph.setAttribute("opacity", 1);
    tm.textContent = `${fmt(au.currentTime)} / ${total}`;
  };
  svg.onclick = ev => {
    const b = svg.getBoundingClientRect(), f = (ev.clientX - b.left) / b.width;
    const go = () => { au.currentTime = f * (au.duration || 0); all.forEach(o => o !== au && o.pause()); au.play(); };
    if (au.readyState >= 1) go(); else { au.addEventListener("loadedmetadata", go, {once: true}); au.load(); }
  };
});

const foot = document.getElementById("foot");
if (!D.review) foot.innerHTML = `<p><b>Scores on every held-out set</b> (score = mean of “silent when asked” and “speaks again when it should”; 1.0 = perfect):</p>
<div class="bars">${D.bars.map(([l, b, m, note]) => `<div class="l">${esc(l)}<small>${esc(note)}</small></div>
<div><div class="bar"><i style="width:${b * 100}%;background:var(--user)"></i><span>base ${b.toFixed(2)}</span></div>
<div class="bar"><i style="width:${m * 100}%;background:var(--moshi)"></i><span>45h ${m.toFixed(2)}</span></div></div>`).join("")}</div>
<p>Model: <code>kyutai/moshiko-pytorch-bf16</code> + LoRA (rank 64), fine-tuned with moshi-finetune. Left ear: Moshi; right ear: user + friend.</p>`;
else foot.remove();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
