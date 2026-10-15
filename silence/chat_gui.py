"""Local control panel for talking to fine-tuned Moshi checkpoints on the chat pod.

  python -m silence.chat_gui            # then open http://127.0.0.1:8770

To start: create a 1x H100 pod on the volume (US-NE-1; the torch 2.6 env can't run Blackwell), add an
ssh alias `chat`, run this, and press Load in each slot (it writes /root/chat.sh + load.sh on the pod).
The chat pod runs two moshi servers, slot A on port 8998 and
slot B on 8999, via /root/chat.sh <port> <checkpoint dir | base>. This app:
  - keeps an SSH tunnel open (localhost:8998/8999 -> the pod), so the moshi web UI runs on
    http://localhost:<port> (the browser allows the mic on localhost);
  - lets you load any scored checkpoint (base, phase 1, phase 2) into either slot (~1 min);
  - lists test scenarios and saves your observations to results/chat/notes.jsonl.
Standard library only.
"""
import argparse
import csv
import json
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import plotstyle as ps
from .chat_scripts import SCRIPTS

SLOTS = {"A": 8998, "B": 8999}
REMOTE_RUNS = "/workspace/runs"
LOAD_SH = r'''cat > /root/chat.sh <<"EOF"
#!/usr/bin/env bash
# $1 = port, $2 = checkpoint dir (or "base"): moshi.server from the training env on the volume
W=/workspace; M=$W/moshiko
ARGS="--host 127.0.0.1 --port $1 --moshi-weight $M/model.safetensors --mimi-weight $M/tokenizer-e351c8d8-checkpoint125.safetensors --tokenizer $M/tokenizer_spm_32k_3.model"
if [ "$2" != base ]; then ARGS="$ARGS --lora-weight $2/consolidated/lora.safetensors --config-path $2/consolidated/config.json"; fi
cd $W/moshi-finetune && exec .venv/bin/python -m moshi.server $ARGS
EOF
chmod +x /root/chat.sh
cat > /root/load.sh <<"EOF"
#!/usr/bin/env bash
# $1 = port, $2 = checkpoint dir or "base": restart the moshi server on that port
for p in $(pgrep -f -- "--port $1 "); do kill $p; done
sleep 2
nohup /root/chat.sh $1 $2 > /root/chat_$1.log 2>&1 &
echo "$2" > /root/slot_$1.txt
EOF
chmod +x /root/load.sh'''



# Representative models for live testing (dev_80 numbers), each with what it's good / bad at.
CURATED = [
    ("R24_v3_b1-15_lr5e-05_epad10/checkpoint_000527", "45h model (v3 data, EPAD ×10): final",
     "Best model overall: dev 0.94 (97% silent / 92% speaks), best or tied on 4 of 7 held-out suites, and the most "
     "robust to a real voice (0.83 on my recordings). Weak spots: the friend saying \"I'm back\", keyword releases, "
     "minute-long holds."),
    ("R21_10h_lr5e-05_epad10/checkpoint_000153", "10 h, EPAD ×10: best checkpoint (ep 2.2)",
     "Highest dev_80 score so far (0.91). Speaks when it should (93%: answers every question, resumes on every "
     "\"I'm back\" cue) and re-asks after \"give me N seconds\" 61% of the time. Silence 90%: what it loses is in "
     "timed holds, where it sometimes speaks before N is up or answers your own muttering."),
    ("R21_10h_lr5e-05_epad10/checkpoint_000345", "10 h, EPAD ×10: final checkpoint (ep 5.0)",
     "Same run, fully annealed (score 0.90). Re-asks after timed holds most often of any model (78%) and speaks "
     "95% of the time when due; silence slips to 84%, again almost entirely in timed holds."),
    ("R22_50h_lr5e-05_epad10/checkpoint_000345", "50 h (v2 data), EPAD ×10: final (1 epoch)",
     "Trained on the 50 h set (more timed holds, 3 TTS engines). Speaks readily (93–97%) and re-asks after timed holds, "
     "but jumps in after pauses: it often talks after the first option or a short silence (test_all 0.80 vs 0.84 for R21)."),
    ("R8_10h_lr2e-05/checkpoint_000289", "Best 10 h (EPAD ×1)",
     "Most reliable silence (97%) and ignores return-cue lookalikes (\"I'm done…\"). A bit cautious about speaking "
     "again (79%); won't re-ask after \"give me N seconds\"."),
    ("R4_1h_lr0.0002/checkpoint_000112", "Best 1 h",
     "Holds well (90%), resumes on \"I'm back\" every time and answers everything. Falls for lookalike cues "
     "(\"I'm done…\" → \"Welcome back!\") and answers your own filler during timed holds."),
    ("R17_1h_lr5e-05_epad10/checkpoint_000042", "EPAD ×10 (loss fix)",
     "The only model that re-asks after \"give me N seconds\" (61%), and it speaks up readily (91%). Talks during "
     "holds more often (81% silent) and falls for lookalike cues."),
    ("base", "Base model (no fine-tuning)",
     "Reference. Talks through holds and over you (14% silent), answers only about half your questions."),
    ("R1_1h_lr2e-06/checkpoint_000142", "Too-low LR: the 'quiet model'",
     "Failure example. Learned to stay quiet (59%) but rarely speaks again: answers about 10% of questions."),
    ("R11_10h_lr0.0005/checkpoint_000345", "Too-high LR at 10 h: collapsed",
     "Failure example. Almost never talks (100% silent, speaks 17%), from the other end of the LR range."),
    ("R19_1h_lr5e-05_epad100/checkpoint_000028", "EPAD ×100: over-corrected",
     "Failure example. Chatty: talks through most holds (9% silent) but answers everything."),
]


def load_models(results_dirs):
    """Choice list: base + every scored checkpoint (dev_80), with its metrics."""
    models = {"base": {"key": "base", "run": "R0", "ckpt": "base", "label": "base model", "lr": None,
                       "epoch": 0.0, "data": "", "viol": None, "on_time": None, "ord": None, "score": None,
                       "color": ps.BASE, "remote": "base"}}
    for R in results_dirs:
        R = Path(R)
        summ = R / "eval" / "summary.csv"
        if not summ.exists():
            continue
        speak = {}
        sp = R / "analysis" / "speak_silence.json"
        if sp.exists():
            for o in json.loads(sp.read_text()):
                speak[(o["run"], o["ck"])] = o
        for r in csv.DictReader(open(summ)):
            if r["set"] != "dev_80" or r["run"] == "R0":
                continue
            f = lambda k: float(r[k]) if r.get(k) not in (None, "", "nan") else None
            run, ck = r["run"], r["ckpt"]
            data = "10 h" if "_10h_" in run else "1 h"
            key = f"{run}/{ck}"
            if key in models:
                continue
            epad = run.split("_epad")[1] if "_epad" in run else None
            models[key] = {"key": key, "run": run, "ckpt": ck, "lr": ps.run_lr(run), "epoch": f("epoch"),
                           "data": data, "label": f"{data} · {ps.run_label(run)}" + (f" · EPAD ×{epad}" if epad else "") + f" · ep {f('epoch'):.1f}",
                           "viol": f("hold_violated"), "on_time": f("on_time"), "ord": f("ord_responded"),
                           "score": (lambda o: (o["silent"] + o["speak"]) / 2 if o else None)(speak.get((run, ck))),
                           "silent_due": (speak.get((run, ck)) or {}).get("silent"), "speaks_due": (speak.get((run, ck)) or {}).get("speak"),
                           "timed": (speak.get((run, ck)) or {}).get("tim_any"), "color": ps.run_color(run),
                           "remote": f"{REMOTE_RUNS}/{run}/checkpoints/{ck}"}
    for i, (key, title, note) in enumerate(CURATED):
        if key not in models and key != "base":  # curated but no eval summary yet (e.g. a fresh run)
            run, ck = key.split("/")
            models[key] = {"key": key, "run": run, "ckpt": ck, "lr": ps.run_lr(run), "epoch": None, "data": "",
                           "label": title, "viol": None, "on_time": None, "ord": None, "score": None,
                           "silent_due": None, "speaks_due": None, "timed": None, "color": ps.run_color(run),
                           "remote": f"{REMOTE_RUNS}/{run}/checkpoints/{ck}"}
        if key in models:
            models[key].update(curated=i, title=title, note=note)
    return models


class State:
    def __init__(self, host, results_dirs, notes):
        self.host = host
        self.results_dirs = results_dirs
        self.notes = Path(notes)
        self.notes.parent.mkdir(parents=True, exist_ok=True)
        self.models = load_models(results_dirs)
        self.slot_model = {"A": None, "B": None}
        self.slot_loading_since = {"A": None, "B": None}
        self.tunnel = None
        self.lock = threading.Lock()

    def ssh(self, cmd, timeout=30):
        return subprocess.run(["ssh", "-o", "ConnectTimeout=10", self.host, cmd], capture_output=True,
                              text=True, timeout=timeout)

    def ensure_tunnel(self):
        with self.lock:
            if self.tunnel and self.tunnel.poll() is None:
                return True
            args = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30"]
            for p in SLOTS.values():
                args += ["-L", f"{p}:127.0.0.1:{p}"]
            self.tunnel = subprocess.Popen(args + [self.host], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1.5)
            return self.tunnel.poll() is None

    def sync_slots(self):
        """Which checkpoint each remote slot runs (from the pod's slot files / process list)."""
        r = self.ssh("for p in 8998 8999; do echo \"$p $(cat /root/slot_$p.txt 2>/dev/null || "
                     "(ps -ef | grep -v grep | grep -- \"--port $p \" | grep -o '/workspace/runs/[^/ ]*/checkpoints/checkpoint_[0-9]*' "
                     "| head -1) )\"; done; ps -ef | grep -v grep | grep -c moshi.server")
        by_remote = {m["remote"]: k for k, m in self.models.items()}
        for line in r.stdout.splitlines()[:2]:
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            port, what = int(parts[0]), parts[1].strip()
            slot = next(s for s, p in SLOTS.items() if p == port)
            key = by_remote.get(what)
            if key is None and what in ("", "base"):
                ps_ = self.ssh(f"ps -ef | grep -v grep | grep -- '--port {port} ' | grep -c lora-weight").stdout.strip()
                key = "base" if ps_ == "0" else None
            self.slot_model[slot] = key

    def ready(self, port):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1.5) as r:
                return r.status == 200
        except Exception:
            return False

    def load(self, slot, key):
        m = self.models[key]
        port = SLOTS[slot]
        self.ssh(LOAD_SH)
        r = self.ssh(f"/root/load.sh {port} {m['remote']}")
        self.slot_model[slot] = key
        self.slot_loading_since[slot] = time.time()
        return r.returncode == 0, r.stderr[-500:]

    def log_tail(self, slot):
        return self.ssh(f"tail -15 /root/chat_{SLOTS[slot]}.log").stdout


def make_handler(st):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            b = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/":
                return self.send(200, PAGE, "text/html; charset=utf-8")
            if self.path == "/api/models":
                return self.send(200, json.dumps(list(st.models.values())))
            if self.path == "/api/scenarios":
                return self.send(200, json.dumps(SCRIPTS))
            if self.path == "/api/status":
                tun = st.ensure_tunnel()
                out = {"tunnel": tun, "slots": {}}
                for s, p in SLOTS.items():
                    rd = st.ready(p)
                    if rd:
                        st.slot_loading_since[s] = None
                    since = st.slot_loading_since[s]
                    out["slots"][s] = {"port": p, "model": st.slot_model[s], "ready": rd,
                                       "loading_for": None if since is None else round(time.time() - since)}
                return self.send(200, json.dumps(out))
            if self.path.startswith("/api/log/"):
                return self.send(200, json.dumps({"log": st.log_tail(self.path.rsplit("/", 1)[1])}))
            if self.path == "/api/notes":
                rows = [json.loads(l) for l in st.notes.read_text().splitlines() if l] if st.notes.exists() else []
                return self.send(200, json.dumps(rows))
            self.send(404, "{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/load":
                ok, err = st.load(data["slot"], data["key"])
                return self.send(200, json.dumps({"ok": ok, "err": err}))
            if self.path == "/api/note":
                m = st.models.get(data.get("model") or "", {})
                row = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), **data,
                       "lr": m.get("lr"), "epoch": m.get("epoch"), "data": m.get("data")}
                with st.notes.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                return self.send(200, json.dumps({"ok": True}))
            self.send(404, "{}")
    return H


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Moshi chat control</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
 --grid:#e1e0d9;--border:rgba(11,11,11,.10);--good:#0ca30c;--good-ink:#006300;--warn:#fab219;--critical:#d03b3b;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-1:#1a1a19;
 --page:#0d0d0d;--ink:#f0efec;--ink-2:#c3c2b7;--grid:#2c2c2a;--border:rgba(255,255,255,.10);--accent:#3987e5;--good-ink:#0ca30c}}
*{box-sizing:border-box}html,body{margin:0;background:#f9f9f7}@media (prefers-color-scheme:dark){html,body{background:#0d0d0d}}
.viz-root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--page);max-width:1240px;
 margin:0 auto;padding:22px 22px 60px;font-size:14px;line-height:1.45}
h1{font-size:21px;margin:0 0 2px}h2{font-size:16px;margin:24px 0 8px}.sub{color:var(--ink-2);margin:0 0 14px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:12px;border:1px solid var(--border);border-radius:999px;padding:2px 9px}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block}
select,button,textarea,input{font:inherit;color:var(--ink);background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:6px 10px}
select{width:100%}button{cursor:pointer}button.primary{background:var(--ink);color:var(--surface-1);border-color:var(--ink)}
button:disabled{opacity:.5;cursor:default}
.slot h3{margin:0 0 6px;font-size:15px}.row{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
.metrics{font-size:12.5px;color:var(--ink-2);margin-top:6px;font-variant-numeric:tabular-nums}
pre.log{font-size:11px;color:var(--ink-2);background:var(--page);border:1px solid var(--border);border-radius:6px;padding:8px;
 max-height:160px;overflow:auto;white-space:pre-wrap;display:none;margin:8px 0 0}
.scen{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.ln{font-size:13px;margin:2px 0;display:grid;grid-template-columns:62px 1fr;gap:6px}
.ln .who{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.03em;padding-top:2px}
.ln.you .txt{color:var(--ink)}.ln.friend .txt{color:var(--ink-2)}.ln.pause .txt{color:var(--muted);font-style:italic}
.ln.moshi .txt{color:var(--good-ink,#006300);font-weight:500}.tag{font-size:11px;border:1px solid var(--border);border-radius:999px;padding:1px 7px;color:var(--ink-2);margin-left:6px;font-weight:400}
.tag.held{border-color:var(--accent);color:var(--accent)}.scen .card{padding:12px}
.scen h4{margin:0 0 4px;font-size:14px}.scen p{margin:4px 0;font-size:13px;color:var(--ink-2)}.scen b{color:var(--ink)}
.scen .exp{border-top:1px solid var(--grid);padding-top:5px;margin-top:6px}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{border-radius:999px;padding:4px 10px;font-size:12.5px}
.chip[aria-pressed="true"]{background:var(--ink);color:var(--surface-1);border-color:var(--ink)}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid);vertical-align:top}
th{color:var(--ink-2);font-weight:600}.note{color:var(--muted);font-size:12px}
label{font-size:12.5px;color:var(--ink-2)}
</style></head><body><div class="viz-root">
<h1>Talk to the fine-tuned Moshi models</h1>
<p class="sub">Two model slots run on the chat pod (1× H100, $3.99/hr: ask Claude to terminate it when you're done).
Load any checkpoint into a slot, press <b>Open chat</b>, allow the microphone, and talk. Headphones avoid echo.</p>
<div class="card" id="conn"></div>

<h2>Model slots</h2>
<div class="grid2" id="slots"></div>
<p class="note"><label><input type="checkbox" id="showall"> Show all checkpoints (283) instead of the representative models</label></p>
<div class="card" style="margin-top:8px"><b style="font-size:13px">Representative models</b><table id="curated" style="margin-top:6px"></table></div>
<p class="note">"score" = mean(silent when it should be, speaks when it should) on dev_80; violated / on time / answered are the
dev_80 rates. Loading a checkpoint restarts that slot's server (~1–2 min); the other slot keeps running.</p>

<h2>Test scripts</h2>
<p class="sub" style="margin-top:-4px">In the style of the training conversations. <b>friend</b> = a second person (or you in a clearly different voice); <i>pause</i> = stay silent; green = what Moshi should do. <span class="tag">train wording</span> = phrasings it was trained on; <span class="tag held">held-out</span> = never seen. Also in results/chat/test_scripts.md.</p>
<div class="scen" id="scen"></div>

<h2>Log what happened</h2>
<div class="card">
 <div class="row"><label>Slot</label><span class="chips" id="n-slot"></span>
  <label style="margin-left:12px">Scenario</label><select id="n-scen" style="width:auto"></select></div>
 <div class="row"><label>Outcome</label><span class="chips" id="n-out"></span></div>
 <div class="row" style="align-items:flex-start"><textarea id="n-text" rows="2" style="flex:1" placeholder="What did it do? (e.g. stayed silent 25 s, said 'welcome back' to my friend's line, answered after 1 s)"></textarea>
  <button class="primary" id="n-save">Save note</button></div>
 <p class="note" id="n-msg">Saved to results/chat/notes.jsonl (with the model, LR, epoch and time).</p>
 <table id="notes"></table>
</div>
</div>
<script>
const $ = s => document.querySelector(s);
const pct = v => v == null ? "–" : Math.round(v * 100) + "%";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
let MODELS = {}, SCEN = [], STATUS = null, noteSlot = "A", noteOut = new Set();
const OUTCOMES = ["✓ stayed silent", "✗ talked during hold", "✓ resumed on cue", "✗ didn't resume", "! resumed late",
  "✓ answered", "✗ didn't answer", "✗ interrupted me", "✗ reacted to friend / lookalike", "~ odd / garbled audio"];

let SHOW_ALL = false;
function optionLabel(m) {
  if (m.curated != null && !SHOW_ALL) return `${m.title}  —  ${m.key === "base" ? "base model" : m.label}`;
  if (m.key === "base") return "base model (no fine-tuning)";
  const s = m.score != null ? `score ${m.score.toFixed(2)} · ` : "";
  return `${m.label}   (${s}viol ${pct(m.viol)} · on time ${pct(m.on_time)} · answered ${pct(m.ord)})`;
}
function selectOptions() {
  if (!SHOW_ALL) {
    const cur = Object.values(MODELS).filter(m => m.curated != null).sort((a, b) => a.curated - b.curated);
    return `<optgroup label="Representative models">${cur.map(m => `<option value="${esc(m.key)}">${esc(optionLabel(m))}</option>`).join("")}</optgroup>`;
  }
  const list = Object.values(MODELS).sort((a, b) => (a.data || "").localeCompare(b.data || "") || (a.lr ?? -1) - (b.lr ?? -1) || a.epoch - b.epoch);
  const groups = {};
  list.forEach(m => { const g = m.key === "base" ? "Base" : `${m.data} · ${m.label.split(" · ")[1]}`; (groups[g] = groups[g] || []).push(m); });
  return Object.entries(groups).map(([g, ms]) => `<optgroup label="${esc(g)}">${ms.map(m => `<option value="${esc(m.key)}">${esc(optionLabel(m))}</option>`).join("")}</optgroup>`).join("");
}
function buildSlots() {
  const opts = selectOptions();
  $("#slots").innerHTML = ["A", "B"].map(s => `<div class="card slot" id="slot-${s}">
    <h3>Slot ${s} <span class="note">localhost:${s === "A" ? 8998 : 8999}</span></h3>
    <div class="row"><span class="pill" id="st-${s}"><span class="dot"></span>…</span><b id="cur-${s}"></b></div>
    <div class="metrics" id="met-${s}"></div>
    <div class="metrics" id="note-${s}" style="color:var(--ink)"></div>
    <div class="row"><select id="sel-${s}">${opts}</select></div>
    <div class="row"><button id="load-${s}">Load into slot ${s}</button>
      <button class="primary" id="open-${s}">Open chat ↗</button><button id="logb-${s}">Server log</button></div>
    <pre class="log" id="log-${s}"></pre></div>`).join("");
  ["A", "B"].forEach(s => {
    $(`#load-${s}`).onclick = async () => {
      const key = $(`#sel-${s}`).value; $(`#load-${s}`).disabled = true;
      await fetch("/api/load", {method: "POST", body: JSON.stringify({slot: s, key})});
      $(`#load-${s}`).disabled = false; refresh();
    };
    $(`#open-${s}`).onclick = () => window.open(`http://localhost:${s === "A" ? 8998 : 8999}/`, "_blank");
    $(`#logb-${s}`).onclick = async () => {
      const el = $(`#log-${s}`); if (el.style.display === "block") { el.style.display = "none"; return; }
      el.textContent = "loading…"; el.style.display = "block";
      el.textContent = (await (await fetch(`/api/log/${s}`)).json()).log || "(empty)";
    };
  });
}
function renderStatus() {
  if (!STATUS) return;
  $("#conn").innerHTML = `<span class="pill"><span class="dot" style="background:${STATUS.tunnel ? "var(--good)" : "var(--critical)"}"></span>
    ${STATUS.tunnel ? "✓ connected to the chat pod (SSH tunnel up)" : "✗ no tunnel to the chat pod: is it running?"}</span>`;
  for (const [s, v] of Object.entries(STATUS.slots)) {
    const m = MODELS[v.model];
    const st = v.ready ? ["var(--good)", "✓ ready"] : v.loading_for != null ? ["var(--warn)", `loading… ${v.loading_for}s`] : ["var(--muted)", "starting / not running"];
    $(`#st-${s}`).innerHTML = `<span class="dot" style="background:${st[0]}"></span>${st[1]}`;
    $(`#cur-${s}`).innerHTML = m ? `<span class="dot" style="background:${m.color};margin-right:4px"></span>${esc(m.key === "base" ? "base model" : m.label)}` : "(unknown model)";
    $(`#met-${s}`).textContent = m && m.key !== "base" ? `dev_80: violated ${pct(m.viol)} · resumed on time ${pct(m.on_time)} · answered ${pct(m.ord)}` + (m.score != null ? ` · score ${m.score.toFixed(2)}` : "") : (m ? "dev_80: violated 86% · resumed on time 40% · answered 43%" : "");
    $(`#note-${s}`).textContent = m && m.note ? `${m.title}: ${m.note}` : "";
    $(`#open-${s}`).disabled = !v.ready;
    if (m && document.activeElement !== $(`#sel-${s}`) && !$(`#sel-${s}`).dataset.touched) $(`#sel-${s}`).value = m.key;
  }
}
async function refresh() { try { STATUS = await (await fetch("/api/status")).json(); renderStatus(); } catch (e) {} }
async function loadNotes() {
  const rows = await (await fetch("/api/notes")).json();
  $("#notes").innerHTML = `<tr><th>Time</th><th>Model</th><th>Scenario</th><th>Outcome</th><th>Note</th></tr>` +
    rows.slice().reverse().map(r => `<tr><td>${esc(r.t.slice(11, 16))}</td><td>${esc(MODELS[r.model]?.label || r.model)}</td><td>${esc(r.scenario)}</td><td>${esc((r.outcome || []).join(", "))}</td><td>${esc(r.text)}</td></tr>`).join("");
}
(async () => {
  MODELS = Object.fromEntries((await (await fetch("/api/models")).json()).map(m => [m.key, m]));
  SCEN = await (await fetch("/api/scenarios")).json();
  buildSlots();
  ["A", "B"].forEach(s => $(`#sel-${s}`).addEventListener("change", e => e.target.dataset.touched = "1"));
  $("#curated").innerHTML = `<tr><th>Model</th><th>dev_80: silent / speaks / timed re-ask</th><th>Good / bad at</th></tr>` +
    Object.values(MODELS).filter(m => m.curated != null).sort((a, b) => a.curated - b.curated).map(m =>
      `<tr><td><span class="dot" style="background:${m.color};margin-right:6px"></span><b>${esc(m.title)}</b><br><span class="note">${esc(m.key === "base" ? "base model" : m.label)}</span></td>
       <td>${m.key === "base" ? "14% / 45% / 33%" : `${pct(m.silent_due)} / ${pct(m.speaks_due)} / ${pct(m.timed)}`}</td><td>${esc(m.note)}</td></tr>`).join("");
  $("#showall").onchange = e => {
    SHOW_ALL = e.target.checked;
    ["A", "B"].forEach(s => { const sel = $(`#sel-${s}`); const v = sel.value; sel.innerHTML = selectOptions(); if ([...sel.options].some(o => o.value === v)) sel.value = v; });
  };
  const WHO = {you: "you", friend: "friend", pause: "pause", moshi: "→ Moshi"};
  $("#scen").innerHTML = SCEN.map(c => `<div class="card"><h4>${esc(c.title)}<span class="tag ${c.group === "train wording" ? "" : "held"}">${esc(c.group)}</span></h4>
    ${c.lines.map(([w, t]) => `<div class="ln ${w}"><span class="who">${WHO[w]}</span><span class="txt">${esc(t)}</span></div>`).join("")}
    <p class="exp"><b>Watch for:</b> ${esc(c.watch)}<br><b>Expect:</b> ${esc(c.expect)}</p></div>`).join("");
  $("#n-scen").innerHTML = SCEN.map(c => `<option value="${c.id}">${esc(c.title)}</option>`).join("") + `<option value="other">Other</option>`;
  $("#n-slot").innerHTML = ["A", "B"].map(s => `<button class="chip" data-s="${s}" aria-pressed="${s === "A"}">Slot ${s}</button>`).join("");
  $("#n-slot").onclick = e => { const b = e.target.closest("button"); if (!b) return; noteSlot = b.dataset.s; [...$("#n-slot").children].forEach(c => c.setAttribute("aria-pressed", c === b)); };
  $("#n-out").innerHTML = OUTCOMES.map(o => `<button class="chip" aria-pressed="false">${esc(o)}</button>`).join("");
  $("#n-out").onclick = e => { const b = e.target.closest("button"); if (!b) return; const on = b.getAttribute("aria-pressed") !== "true"; b.setAttribute("aria-pressed", on); on ? noteOut.add(b.textContent) : noteOut.delete(b.textContent); };
  $("#n-save").onclick = async () => {
    const model = STATUS?.slots?.[noteSlot]?.model;
    await fetch("/api/note", {method: "POST", body: JSON.stringify({slot: noteSlot, model, scenario: $("#n-scen").value, outcome: [...noteOut], text: $("#n-text").value})});
    $("#n-text").value = ""; noteOut.clear(); [...$("#n-out").children].forEach(c => c.setAttribute("aria-pressed", "false"));
    $("#n-msg").textContent = "Saved ✓ (results/chat/notes.jsonl)"; loadNotes();
  };
  refresh(); loadNotes(); setInterval(refresh, 3000);
})();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="chat", help="ssh alias of the chat pod")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--results", nargs="+", default=["results/phase1", "results/phase2", "results/epad", "results/epad10h"])
    ap.add_argument("--notes", default="results/chat/notes.jsonl")
    a = ap.parse_args()
    st = State(a.host, a.results, a.notes)
    st.ensure_tunnel()
    try:
        st.sync_slots()
    except Exception as e:
        print("could not read the pod's slots:", e)
    print(f"{len(st.models)} models; slots: {st.slot_model}")
    print(f"open http://127.0.0.1:{a.port}")
    try:
        ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(st)).serve_forever()
    finally:
        if st.tunnel:
            st.tunnel.terminate()


if __name__ == "__main__":
    main()
