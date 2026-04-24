"""
ClinDetect: FastAPI server (WebSocket primary, HTTP for debug).

Port is read from os.environ["PORT"], default 7860 for HF Spaces.
WORKERS=1 is mandatory — sessions are in-memory per WebSocket connection.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..graph import get_graph
from ..models import ClinDetectAction, ClinDetectState, StepResult
from .environment import ClinDetectEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ClinDetect starting — loading graph ...")
    graph = get_graph()  # blocks until loaded; ~5s on cold start
    logger.info(
        "Graph ready: %d nodes loaded",
        len(graph.nodes),
    )
    yield
    logger.info("ClinDetect shutting down.")


app = FastAPI(
    title="ClinDetect",
    description=(
        "OpenEnv: LLM agent navigates a gene-disease knowledge graph to diagnose "
        "rare diseases. Three task tiers: monogenic (easy), oligogenic (medium), "
        "phenotype_mismatch (hard)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "version": "1.0.0", "environment": "clindetect"}


# ── WebSocket (primary transport) ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Persistent WebSocket session.
    Protocol:
      Client → Server:
        {"type": "reset", "task_type": "monogenic|oligogenic|phenotype_mismatch", "seed": int|null}
        {"type": "step",  "action": {...}}
        {"type": "state"}
      Server → Client:
        {"type": "observation", "data": StepResult}
        {"type": "state",       "data": ClinDetectState}
        {"type": "error",       "message": "..."}
    """
    await websocket.accept()
    env = ClinDetectEnvironment()
    logger.info("WS session opened")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as e:
                await websocket.send_text(json.dumps({"type": "error", "message": f"Bad JSON: {e}"}))
                continue

            mtype = msg.get("type")

            if mtype == "reset":
                try:
                    result = env.reset(
                        task_type=msg.get("task_type", "monogenic"),
                        seed=msg.get("seed"),
                    )
                    await websocket.send_text(
                        json.dumps({"type": "observation", "data": result.model_dump()})
                    )
                except Exception as e:
                    logger.exception("reset error")
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

            elif mtype == "step":
                try:
                    action = ClinDetectAction.model_validate(msg.get("action", {}))
                    result = env.step(action)
                    await websocket.send_text(
                        json.dumps({"type": "observation", "data": result.model_dump()})
                    )
                except Exception as e:
                    logger.exception("step error")
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

            elif mtype == "state":
                try:
                    state = env.state()
                    await websocket.send_text(
                        json.dumps({"type": "state", "data": state.model_dump()})
                    )
                except Exception as e:
                    logger.exception("state error")
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))

            else:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": f"Unknown type: {mtype}"})
                )

    except WebSocketDisconnect:
        logger.info("WS session closed")
    except Exception as e:
        logger.exception("WS unexpected error: %s", e)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


# ── HTTP debug endpoints (stateless, not for concurrent use) ─────────────────

_http_env = ClinDetectEnvironment()


@app.post("/reset", response_model=StepResult)
async def http_reset(task_type: Optional[str] = None, seed: Optional[int] = None) -> StepResult:
    return _http_env.reset(task_type=task_type or "monogenic", seed=seed)


@app.post("/step", response_model=StepResult)
async def http_step(action: ClinDetectAction) -> StepResult:
    try:
        return _http_env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=ClinDetectState)
async def http_state() -> ClinDetectState:
    return _http_env.state()


# ── Web UI ────────────────────────────────────────────────────────────────────

_WEB_UI = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ClinDetect | OpenEnv</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0a0e1a; color: #cdd9e5; font-family: 'Courier New', monospace; padding: 20px; }
    h1 { color: #58a6ff; font-size: 1.4em; margin-bottom: 4px; }
    .sub { color: #8b949e; font-size: 0.82em; margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
    .stat { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px; text-align: center; }
    .stat .lbl { color: #8b949e; font-size: 0.72em; margin-bottom: 4px; }
    .stat .val { color: #58a6ff; font-size: 1.1em; font-weight: bold; }
    .terminal { background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
                padding: 14px; min-height: 350px; max-height: 500px; overflow-y: auto;
                font-size: 0.82em; line-height: 1.7; }
    .ts { color: #484f58; }
    .info { color: #58a6ff; }
    .reward { color: #56d364; }
    .warn { color: #e3b341; }
    .err { color: #f85149; }
    .controls { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; }
    select, button { background: #161b22; color: #58a6ff; border: 1px solid #58a6ff;
                     padding: 7px 14px; cursor: pointer; border-radius: 6px; font-family: monospace; }
    button:hover { background: #58a6ff; color: #0d1117; }
    .phenotypes { margin: 10px 0; padding: 10px; background: #161b22;
                  border: 1px solid #30363d; border-radius: 6px; font-size: 0.82em; }
    .phe-tag { display: inline-block; background: #1f6feb; color: #cdd9e5;
               padding: 2px 8px; border-radius: 12px; margin: 2px; font-size: 0.78em; }
  </style>
</head>
<body>
  <h1>ClinDetect — Rare Disease Diagnosis Agent</h1>
  <p class="sub">OpenEnv | Navigate the gene-disease knowledge graph to find the causal variant</p>

  <div class="grid">
    <div class="stat"><div class="lbl">STEP</div><div class="val" id="s-step">—</div></div>
    <div class="stat"><div class="lbl">REWARD</div><div class="val" id="s-reward">—</div></div>
    <div class="stat"><div class="lbl">TASK</div><div class="val" id="s-task">—</div></div>
    <div class="stat"><div class="lbl">STATUS</div><div class="val" id="s-status">IDLE</div></div>
  </div>

  <div class="phenotypes" id="pheno-box">Patient phenotypes will appear here after RESET.</div>

  <div class="terminal" id="term">
    <span class="info">[CLINDETECT] Ready. Select a task and press RESET.</span><br>
  </div>

  <div class="controls">
    <select id="task-sel">
      <option value="monogenic">monogenic — Easy (single gene)</option>
      <option value="oligogenic">oligogenic — Medium (2-3 genes)</option>
      <option value="phenotype_mismatch">phenotype_mismatch — Hard (resist decoy)</option>
    </select>
    <button onclick="doReset()">RESET</button>
    <button onclick="doSummarise()">SUMMARISE TRAIL</button>
    <button onclick="doBacktrack()">BACKTRACK</button>
    <button onclick="doRequestLab()">REQUEST LAB</button>
  </div>
  <div class="controls" style="margin-top:8px">
    <input id="hop-input" placeholder="Node ID to hop to..." style="flex:1; padding:7px; border-radius:6px; border:1px solid #30363d; background:#161b22; color:#cdd9e5; font-family:monospace;">
    <button onclick="doHop()">HOP</button>
    <input id="flag-input" placeholder="Variant ID to flag (VAR:xxxxx)..." style="flex:1; padding:7px; border-radius:6px; border:1px solid #30363d; background:#161b22; color:#cdd9e5; font-family:monospace;">
    <button onclick="doFlag()">FLAG CAUSAL</button>
  </div>

  <script>
    let ws = null;
    const term = document.getElementById('term');

    function log(msg, cls='') {
      const ts = new Date().toLocaleTimeString();
      term.innerHTML += `<span class="ts">[${ts}]</span> <span class="${cls}">${msg}</span><br>`;
      term.scrollTop = term.scrollHeight;
    }

    function connect() {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => log('Connected to ClinDetect environment.', 'info');
      ws.onclose = () => { log('Disconnected.', 'err'); ws = null; };
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'observation') {
          const r = msg.data;
          const obs = r.observation;
          document.getElementById('s-step').textContent = obs.step + '/' + obs.max_steps;
          document.getElementById('s-reward').textContent = r.reward.toFixed(4);
          document.getElementById('s-task').textContent = obs.task_type;
          document.getElementById('s-status').textContent = obs.done ? 'DONE' : 'ACTIVE';

          if (obs.patient_phenotypes && obs.step === 0) {
            const box = document.getElementById('pheno-box');
            box.innerHTML = '<strong style="color:#58a6ff">Patient phenotypes:</strong> ' +
              obs.phenotype_names.map((n,i) =>
                `<span class="phe-tag" title="${obs.patient_phenotypes[i]}">${n}</span>`
              ).join('');
          }

          if (obs.current_node) {
            const n = obs.current_node;
            log(`📍 Now at: [${n.type.toUpperCase()}] ${n.name} (${n.id})`, 'info');
            log(`   Neighbors: ${n.connected_node_ids.slice(0,6).join(', ')}${n.connected_node_ids.length > 6 ? ' ...' : ''}`, 'ts');
          }
          if (r.reward !== 0) {
            const cls = r.reward > 0 ? 'reward' : 'warn';
            log(`Reward: ${r.reward > 0 ? '+' : ''}${r.reward.toFixed(4)}`, cls);
          }
          if (obs.done) {
            log('══ EPISODE COMPLETE ══', 'reward');
            if (obs.info && obs.info.ground_truth_hint)
              log('Causal genes: ' + obs.info.ground_truth_hint.join(', '), 'reward');
          }
        } else if (msg.type === 'error') {
          log('ERROR: ' + msg.message, 'err');
        }
      };
    }

    function send(p) {
      if (!ws || ws.readyState !== 1) { connect(); setTimeout(() => send(p), 600); return; }
      ws.send(JSON.stringify(p));
    }

    function doReset() {
      const t = document.getElementById('task-sel').value;
      document.getElementById('pheno-box').innerHTML = 'Loading case...';
      log('── RESET task=' + t + ' ──', 'info');
      send({ type: 'reset', task_type: t });
    }
    function doHop() {
      const nid = document.getElementById('hop-input').value.trim();
      if (!nid) { log('Enter a node ID first.', 'warn'); return; }
      send({ type: 'step', action: { action_type: 'hop', node_id: nid, reasoning: 'Manual hop via UI.' } });
    }
    function doFlag() {
      const vid = document.getElementById('flag-input').value.trim();
      if (!vid) { log('Enter a variant ID first.', 'warn'); return; }
      send({ type: 'step', action: { action_type: 'flag_causal', variant_id: vid, reasoning: 'Flagging via UI.' } });
    }
    function doBacktrack() { send({ type: 'step', action: { action_type: 'backtrack', reasoning: 'Backtracking.' } }); }
    function doSummarise() { send({ type: 'step', action: { action_type: 'summarise_trail', reasoning: 'Requesting trail summary.' } }); }
    function doRequestLab() { send({ type: 'step', action: { action_type: 'request_lab', test_type: 'gene_panel', reasoning: 'Requesting lab test.' } }); }

    connect();
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(content=_WEB_UI)


@app.get("/web", response_class=HTMLResponse)
async def web_ui() -> HTMLResponse:
    return HTMLResponse(content=_WEB_UI)


def run() -> None:
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    workers = int(os.environ.get("WORKERS", 1))
    uvicorn.run("clindetect.server.app:app", host=host, port=port, workers=workers)


if __name__ == "__main__":
    run()
