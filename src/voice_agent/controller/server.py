from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from ..state import SharedState


def create_api(state: SharedState) -> FastAPI:
    app = FastAPI()

    @app.get("/healthz")
    async def healthz() -> JSONResponse:  # noqa: D401
        return JSONResponse({"ok": True, "listening": state.listening, "speaking": state.tts_speaking})

    @app.post("/api/listen/start")
    async def listen_start() -> JSONResponse:  # noqa: D401
        state.set_listening(True)
        return JSONResponse({"ok": True, "listening": state.listening})

    @app.post("/api/listen/stop")
    async def listen_stop() -> JSONResponse:  # noqa: D401
        state.set_listening(False)
        return JSONResponse({"ok": True, "listening": state.listening})

    @app.post("/api/listen/toggle")
    async def listen_toggle() -> JSONResponse:  # noqa: D401
        state.set_listening(not state.listening)
        return JSONResponse({"ok": True, "listening": state.listening})

    @app.get("/")
    async def index() -> HTMLResponse:  # noqa: D401
        return HTMLResponse(_INDEX_HTML)

    return app


async def run_http_server(app: FastAPI, host: str = "127.0.0.1", port: int = 8710) -> None:
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    await server.serve()


_INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Voice Agent Control</title>
  <style>
    :root { --bg: #0f1226; --card: #141834; --primary: #7c5cff; --text: #e8e9ee; --muted: #9aa0b4; --ok: #2ecc71; --bad: #ff5c7c; }
    html, body { height: 100%; }
    body {
      margin: 0; background: radial-gradient(1200px 800px at 10% 10%, #1b2050, var(--bg)), linear-gradient(160deg, #0e0f22, #0f1226);
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
      color: var(--text); display: grid; place-items: center;
    }
    .card {
      width: min(520px, 92vw); background: linear-gradient(180deg, #171b3e, var(--card));
      border: 1px solid #2a2f62; box-shadow: 0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
      border-radius: 16px; padding: 28px; transform: translateY(-4px);
    }
    h1 { font-size: 22px; margin: 0 0 10px 0; letter-spacing: .3px; }
    p  { margin: 0; color: var(--muted); }
    .row { display: flex; align-items: center; justify-content: space-between; margin-top: 18px; }
    .status {
      display: inline-flex; align-items: center; gap: 8px; font-weight: 600; letter-spacing: .2px;
      padding: 10px 14px; border-radius: 999px; border: 1px solid #2b2f59; background: #0f1330;
    }
    .dot { width: 10px; height: 10px; border-radius: 99px; background: var(--bad); box-shadow: 0 0 12px rgba(255,92,124,.55); }
    .dot.on { background: var(--ok); box-shadow: 0 0 12px rgba(46,204,113,.55); }
    .btn {
      appearance: none; border: 0; border-radius: 999px; padding: 12px 18px; font-weight: 700; letter-spacing: .3px;
      background: linear-gradient(180deg, #8f74ff, var(--primary)); color: white; cursor: pointer;
      box-shadow: 0 8px 20px rgba(124,92,255,.35), inset 0 1px 0 rgba(255,255,255,.2);
      transition: transform .05s ease-out, box-shadow .2s ease;
    }
    .btn:active { transform: translateY(1px); box-shadow: 0 4px 10px rgba(124,92,255,.25); }
    .grid { display: grid; grid-template-columns: 1fr auto; gap: 14px; margin-top: 16px; align-items: center; }
    footer { margin-top: 18px; color: var(--muted); font-size: 12px; text-align: center; }
    code { color: #c5c9ff; }
  </style>
</head>
<body>
  <main class="card">
    <h1>Voice Agent Controller</h1>
    <p>Mic gate lets the agent run 24/7 while remotely toggling listening.</p>

    <div class="grid">
      <div class="status"><span id="dot" class="dot"></span> <span id="label">Listening: Off</span></div>
      <button id="toggle" class="btn">Toggle Listening</button>
    </div>

    <footer>
      <div>Health: <code id="health">checking…</code></div>
    </footer>
  </main>

  <script>
    const healthEl = document.getElementById('health');
    const labelEl = document.getElementById('label');
    const dotEl = document.getElementById('dot');
    const btn = document.getElementById('toggle');

    async function getJSON(url, opts) {
      try {
        const r = await fetch(url, opts);
        return await r.json();
      } catch (e) { return null; }
    }

    async function update() {
      const data = await getJSON('/healthz');
      if (!data) { healthEl.textContent = 'offline'; return; }
      healthEl.textContent = data.ok ? 'ok' : 'error';
      setListening(Boolean(data.listening));
    }

    function setListening(on) {
      labelEl.textContent = 'Listening: ' + (on ? 'On' : 'Off');
      dotEl.classList.toggle('on', on);
    }

    btn.addEventListener('click', async () => {
      const data = await getJSON('/api/listen/toggle', { method: 'POST' });
      if (data && 'listening' in data) setListening(Boolean(data.listening));
    });

    update();
    setInterval(update, 1000);
  </script>
</body>
</html>
"""


__all__ = ["create_api", "run_http_server"]

