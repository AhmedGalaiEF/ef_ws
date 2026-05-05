"""
Small HTTP status monitor for the hand pose navigation runner.

This intentionally uses only the Python standard library so the direct
runner does not need Flask/FastAPI or a package install step.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Tuple
from urllib.parse import urlparse


StatusProvider = Callable[[], Dict]


def start_status_server(
    status_provider: StatusProvider,
    host: str = "0.0.0.0",
    port: int = 8088,
) -> Tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer((host, port), _make_handler(status_provider))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _make_handler(status_provider: StatusProvider):
    class StatusHandler(BaseHTTPRequestHandler):
        server_version = "HandPoseNavStatus/1.0"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._send_html(_INDEX_HTML)
            elif path == "/api/status":
                self._send_json(status_provider())
            else:
                self.send_error(404, "Not found")

        def log_message(self, fmt: str, *args) -> None:
            return

        def _send_json(self, payload: Dict) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return StatusHandler


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hand Pose Navigation</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #20231f;
      --muted: #667064;
      --border: #d9ddd4;
      --accent: #1b7f6b;
      --warn: #b7791f;
      --bad: #b42318;
      --good: #157347;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111411;
        --panel: #191d19;
        --text: #eef2ec;
        --muted: #a7b0a4;
        --border: #30382f;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--border);
      background: var(--panel);
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }
    main {
      width: min(1120px, 100%);
      margin: 0 auto;
      padding: 18px;
      display: grid;
      gap: 16px;
    }
    .status-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: var(--muted);
      display: inline-block;
      margin-right: 8px;
    }
    .status-dot.running { background: var(--accent); }
    .status-dot.done { background: var(--good); }
    .status-dot.error { background: var(--bad); }
    .summary {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
    }
    .metric label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .metric strong {
      display: block;
      font-size: 24px;
      font-weight: 680;
      overflow-wrap: anywhere;
    }
    .grid {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 16px;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 15px;
      font-weight: 650;
    }
    dl {
      margin: 0;
      display: grid;
      grid-template-columns: 130px 1fr;
      row-gap: 8px;
      column-gap: 12px;
      font-size: 14px;
    }
    dt { color: var(--muted); }
    dd { margin: 0; overflow-wrap: anywhere; }
    pre {
      margin: 0;
      min-height: 360px;
      max-height: 58vh;
      overflow: auto;
      white-space: pre-wrap;
      font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    @media (max-width: 820px) {
      header { align-items: flex-start; flex-direction: column; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Hand Pose Navigation</h1>
    <div><span id="dot" class="status-dot"></span><span id="state">Connecting</span></div>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><label>Iteration</label><strong id="iteration">-</strong></div>
      <div class="metric"><label>Elapsed</label><strong id="elapsed">-</strong></div>
      <div class="metric"><label>Position Error</label><strong id="poserr">-</strong></div>
      <div class="metric"><label>Rotation Error</label><strong id="roterr">-</strong></div>
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Runtime</h2>
        <dl id="runtime"></dl>
      </div>
      <div class="panel">
        <h2>Log</h2>
        <pre id="log"></pre>
      </div>
    </section>
  </main>
  <script>
    const ids = Object.fromEntries(["dot", "state", "iteration", "elapsed", "poserr", "roterr", "runtime", "log"].map(id => [id, document.getElementById(id)]));
    function fmtNum(value, digits, suffix) {
      return Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "-";
    }
    function render(data) {
      ids.dot.className = "status-dot " + (data.running ? "running" : (data.converged ? "done" : ""));
      ids.state.textContent = data.running ? "Running" : (data.converged ? "Converged" : "Stopped");
      ids.iteration.textContent = data.iteration ?? "-";
      ids.elapsed.textContent = fmtNum(data.total_elapsed_s, 1, "s");
      ids.poserr.textContent = fmtNum(data.last_error_pos_m, 4, "m");
      ids.roterr.textContent = fmtNum(data.last_error_rot_rad, 3, "rad");
      const cfg = data.config || {};
      const rows = {
        Robot: data.robot_mode || "-",
        "SDK Error": data.sdk_error || "-",
        Arm: cfg.arm || "-",
        Detector: cfg.detection_method || "-",
        "IK Solver": cfg.ik_solver || "-",
        Rate: cfg.rate_hz ? `${cfg.rate_hz} Hz` : "-",
        Timeout: cfg.timeout_s ? `${cfg.timeout_s} s` : "unlimited",
        "Detection Fails": data.detection_failures ?? 0,
        "IK Fails": data.ik_failures ?? 0,
        "Safety Rejects": data.safety_rejections ?? 0
      };
      ids.runtime.innerHTML = Object.entries(rows).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
      ids.log.textContent = (data.log || []).join("\\n");
      ids.log.scrollTop = ids.log.scrollHeight;
    }
    async function poll() {
      try {
        const res = await fetch("/api/status", { cache: "no-store" });
        render(await res.json());
      } catch (err) {
        ids.dot.className = "status-dot error";
        ids.state.textContent = "Monitor disconnected";
      }
    }
    poll();
    setInterval(poll, 1000);
  </script>
</body>
</html>
"""
