from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from ..state import SharedState
from ..sfx import SFXManager


def create_api(state: SharedState, sfx_manager: Optional[SFXManager] = None) -> FastAPI:
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

    @app.get("/api/sfx")
    async def list_sfx() -> JSONResponse:
        if not sfx_manager:
            raise HTTPException(status_code=404, detail="SFX playback not configured")
        tracks = sfx_manager.list_tracks()
        return JSONResponse({"tracks": tracks, "playing": sfx_manager.current_track()})

    @app.post("/api/sfx/play")
    async def play_sfx(payload: dict = Body(...)) -> JSONResponse:
        if not sfx_manager:
            raise HTTPException(status_code=404, detail="SFX playback not configured")
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise HTTPException(status_code=400, detail="Missing SFX name")
        try:
            await sfx_manager.play(name)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="SFX not found") from None
        return JSONResponse({"ok": True, "playing": sfx_manager.current_track()})

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
      width: min(540px, 92vw); background: linear-gradient(180deg, #171b3e, var(--card));
      border: 1px solid #2a2f62; box-shadow: 0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
      border-radius: 16px; padding: 30px; transform: translateY(-4px);
    }
    h1 { font-size: 22px; margin: 0 0 10px 0; letter-spacing: .3px; }
    h2 { font-size: 18px; margin: 0; letter-spacing: .2px; }
    p  { margin: 0; color: var(--muted); }
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
    .btn.ghost {
      background: rgba(124,92,255,.14); color: var(--primary); box-shadow: none; border: 1px solid rgba(124,92,255,.35);
      padding: 10px 16px;
    }
    .btn.ghost:active { box-shadow: none; }
    .grid { display: grid; grid-template-columns: 1fr auto; gap: 14px; margin-top: 16px; align-items: center; }
    .section-head { margin-top: 26px; display: flex; align-items: center; justify-content: space-between; gap: 14px; }
    .sfx-grid { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 10px; }
    .pill {
      border: 1px solid #2a2f62; border-radius: 999px; background: #161a3a; color: var(--text);
      padding: 10px 14px; font-weight: 600; letter-spacing: .2px; cursor: pointer;
      transition: background .15s ease, transform .05s ease;
    }
    .pill:hover { background: #1f2350; }
    .pill:disabled { opacity: .55; cursor: not-allowed; }
    .pill.playing { border-color: var(--primary); box-shadow: 0 0 12px rgba(124,92,255,.45); }
    .muted { color: var(--muted); }
    .empty { padding: 12px 0; width: 100%; }
    footer { margin-top: 22px; color: var(--muted); font-size: 12px; text-align: center; }
    code { color: #c5c9ff; }
  </style>
</head>
<body>
  <main class="card">
    <h1>Voice Agent Controller</h1>
    <p>Mic gate lets the agent run 24/7 while remotely toggling listening.</p>

    <div class="grid">
      <div class="status"><span id="dot" class="dot"></span> <span id="label">Listening: Off</span></div>
      <button id="toggle" class="btn" type="button">Toggle Listening</button>
    </div>

    <section class="sfx">
      <div class="section-head">
        <h2>Sound FX</h2>
        <button id="refresh-sfx" class="btn ghost" type="button">Refresh List</button>
      </div>
      <div id="sfx-list" class="sfx-grid">
        <div class="empty muted">Drop audio files into <code>./sfx</code></div>
      </div>
    </section>

    <footer>
      <div>Health: <code id="health">checking…</code></div>
    </footer>
  </main>

  <script>
    const healthEl = document.getElementById('health');
    const labelEl = document.getElementById('label');
    const dotEl = document.getElementById('dot');
    const toggleBtn = document.getElementById('toggle');
    const sfxListEl = document.getElementById('sfx-list');
    const refreshSfxBtn = document.getElementById('refresh-sfx');

    async function getJSON(url, opts) {
      try {
        const r = await fetch(url, opts);
        if (!r.ok) return null;
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

    toggleBtn.addEventListener('click', async () => {
      const data = await getJSON('/api/listen/toggle', { method: 'POST' });
      if (data && 'listening' in data) setListening(Boolean(data.listening));
    });

    function renderSfx(tracks, playing) {
      if (!Array.isArray(tracks) || tracks.length === 0) {
        sfxListEl.innerHTML = '<div class="empty muted">Drop audio files into <code>./sfx</code></div>';
        return;
      }
      sfxListEl.innerHTML = '';
      tracks.forEach((name) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'pill' + (name === playing ? ' playing' : '');
        button.textContent = name;
        button.addEventListener('click', () => triggerSfx(name, button));
        sfxListEl.appendChild(button);
      });
    }

    async function fetchSfx() {
      const data = await getJSON('/api/sfx');
      if (!data) {
        renderSfx([], null);
        return;
      }
      renderSfx(data.tracks || [], data.playing || null);
    }

    async function triggerSfx(name, button) {
      if (button) button.disabled = true;
      const data = await getJSON('/api/sfx/play', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (button) {
        setTimeout(() => { button.disabled = false; }, 350);
      }
      if (data && data.ok) {
        setTimeout(fetchSfx, 150);
      }
    }

    refreshSfxBtn.addEventListener('click', () => { fetchSfx(); });

    update();
    fetchSfx();
    setInterval(update, 1000);
    setInterval(fetchSfx, 3000);
  </script>
</body>
</html>
"""


__all__ = ["create_api", "run_http_server"]
