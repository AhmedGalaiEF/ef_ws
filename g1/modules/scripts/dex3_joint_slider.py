#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sdk_hand import Dex3HandController, HAND_MAX_LIMITS, HAND_MIN_LIMITS, hand_open_targets


JOINT_NAMES = [
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
]

DEFAULT_WEB_HOST = "0.0.0.0"
DEFAULT_WEB_PORT = 8095


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Web Dex3 joint pose slider.")
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--iface", default="eth0", help="Network interface for DDS traffic.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=50.0,
        help="Low-level command publish rate.",
    )
    parser.add_argument(
        "--speed-rad-s",
        type=float,
        default=0.45,
        help="Maximum commanded joint transition speed.",
    )
    parser.add_argument("--kp", type=float, default=0.5, help="Joint proportional gain.")
    parser.add_argument("--kd", type=float, default=0.1, help="Joint derivative gain.")
    parser.add_argument("--tau", type=float, default=0.0, help="Feed-forward torque.")
    parser.add_argument("--web-host", default=DEFAULT_WEB_HOST, help="HTTP server bind host.")
    parser.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT, help="HTTP server port.")
    return parser.parse_args()


class Dex3JointSlider:
    def __init__(self, args: argparse.Namespace) -> None:
        self.iface = str(args.iface)
        self.domain_id = int(args.domain_id)
        self.rate_hz = max(1.0, float(args.rate_hz))
        self.speed_rad_s = max(0.01, float(args.speed_rad_s))
        self.kp = float(args.kp)
        self.kd = float(args.kd)
        self.tau = float(args.tau)

        self.hand = str(args.hand)
        self.controller: Dex3HandController | None = None
        self.current_targets = hand_open_targets(self.hand)
        self.desired_targets = list(self.current_targets)
        self.last_tick_s = time.monotonic()
        self.matched_once = False
        self.status = ""
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._publish_thread: threading.Thread | None = None

        self._set_controller(self.hand)

    def start(self) -> None:
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._publish_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._publish_thread is not None:
            self._publish_thread.join(timeout=1.0)

    def _set_controller(self, hand: str) -> None:
        try:
            self.controller = Dex3HandController(
                hand=hand,
                iface=self.iface,
                domain_id=self.domain_id,
            )
            self.status = f"Connected: {hand} on {self.iface}"
        except Exception as exc:
            self.controller = None
            self.status = f"Controller init failed: {exc}"

    def set_hand(self, hand: str) -> None:
        side = str(hand).strip().lower()
        if side not in ("right", "left"):
            raise ValueError("hand must be 'right' or 'left'")
        with self._lock:
            if side == self.hand:
                return
            self.hand = side
            self.current_targets = hand_open_targets(self.hand)
            self.desired_targets = list(self.current_targets)
            self.matched_once = False
            self._set_controller(self.hand)

    def _joint_bounds(self, joint_idx: int) -> tuple[float, float]:
        return (
            float(HAND_MIN_LIMITS[self.hand][joint_idx]),
            float(HAND_MAX_LIMITS[self.hand][joint_idx]),
        )

    def set_joint_target(self, joint_idx: int, value: float) -> None:
        if joint_idx < 0 or joint_idx >= len(JOINT_NAMES):
            raise ValueError("invalid joint index")
        with self._lock:
            lo, hi = self._joint_bounds(joint_idx)
            self.desired_targets[joint_idx] = max(lo, min(hi, float(value)))

    def update_settings(
        self,
        *,
        speed_rad_s: float | None = None,
        kp: float | None = None,
        kd: float | None = None,
        tau: float | None = None,
    ) -> None:
        with self._lock:
            if speed_rad_s is not None:
                self.speed_rad_s = max(0.01, float(speed_rad_s))
            if kp is not None:
                self.kp = max(0.0, float(kp))
            if kd is not None:
                self.kd = max(0.0, float(kd))
            if tau is not None:
                self.tau = float(tau)

    def state(self, joint_idx: int = 0) -> dict[str, Any]:
        joint_idx = max(0, min(len(JOINT_NAMES) - 1, int(joint_idx)))
        with self._lock:
            lo, hi = self._joint_bounds(joint_idx)
            joints = []
            for idx, name in enumerate(JOINT_NAMES):
                joint_lo, joint_hi = self._joint_bounds(idx)
                joints.append(
                    {
                        "index": idx,
                        "name": name,
                        "current": self.current_targets[idx],
                        "desired": self.desired_targets[idx],
                        "min": joint_lo,
                        "max": joint_hi,
                    }
                )
            return {
                "hand": self.hand,
                "jointIndex": joint_idx,
                "jointName": JOINT_NAMES[joint_idx],
                "current": self.current_targets[joint_idx],
                "desired": self.desired_targets[joint_idx],
                "min": lo,
                "max": hi,
                "speedRadS": self.speed_rad_s,
                "kp": self.kp,
                "kd": self.kd,
                "tau": self.tau,
                "status": self.status,
                "joints": joints,
            }

    def open_hand(self) -> None:
        with self._lock:
            self.desired_targets = hand_open_targets(self.hand)

    def zero_gains_once(self) -> bool:
        with self._lock:
            controller = self.controller
            targets = list(self.current_targets)
        if controller is None:
            return False
        ok = controller.write_targets_once(
            targets,
            kp=0.0,
            kd=0.0,
            tau=0.0,
            timeout=1,
            first_write_timeout_s=1.0,
        )
        with self._lock:
            self.status = f"Zero-gain stop sent: {ok}"
        return bool(ok)

    def _publish_loop(self) -> None:
        interval_s = 1.0 / self.rate_hz
        while not self._stop_event.wait(interval_s):
            self._publish_step()

    def _publish_step(self) -> None:
        with self._lock:
            if self.controller is None:
                return

            now = time.monotonic()
            dt = max(1.0 / self.rate_hz, min(0.2, now - self.last_tick_s))
            self.last_tick_s = now
            max_delta = self.speed_rad_s * dt

            next_targets = list(self.current_targets)
            for idx, (current, desired) in enumerate(zip(self.current_targets, self.desired_targets)):
                error = desired - current
                if abs(error) <= max_delta:
                    next_value = desired
                else:
                    next_value = current + max_delta * (1.0 if error > 0.0 else -1.0)
                next_targets[idx] = next_value

            self.current_targets = next_targets
            controller = self.controller
            targets = list(self.current_targets)
            kp = self.kp
            kd = self.kd
            tau = self.tau
            first_write_timeout_s = None if self.matched_once else 1.0

        ok = controller.write_targets_once(
            targets,
            kp=kp,
            kd=kd,
            tau=tau,
            timeout=0,
            first_write_timeout_s=first_write_timeout_s,
        )

        with self._lock:
            self.matched_once = self.matched_once or ok
            if not ok:
                self.status = "DDS command subscriber not matched"


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dex3 Joint Slider</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; min-height: 100vh; background: #111315; color: #eef2f3; display: grid; place-items: center; }
    main { width: min(760px, calc(100vw - 32px)); }
    h1 { font-size: 1.35rem; font-weight: 650; margin: 0 0 18px; letter-spacing: 0; }
    .panel { border: 1px solid #30383d; border-radius: 8px; background: #181c1f; padding: 20px; box-shadow: 0 18px 48px rgba(0,0,0,.28); }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-bottom: 20px; }
    label { display: grid; gap: 7px; color: #aeb8bd; font-size: .86rem; }
    select, input[type="number"] { width: 100%; box-sizing: border-box; border: 1px solid #3b464c; border-radius: 6px; background: #0f1214; color: #eef2f3; padding: 10px 11px; font: inherit; }
    .readout { display: grid; gap: 8px; border-top: 1px solid #2d3539; border-bottom: 1px solid #2d3539; padding: 18px 0; margin-bottom: 18px; }
    .pose { font-size: 1.55rem; font-weight: 700; }
    .limits, .status { color: #aeb8bd; font-size: .92rem; overflow-wrap: anywhere; }
    input[type="range"] { width: 100%; margin: 14px 0 6px; accent-color: #4fb3a4; }
    .settings { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    button { border: 1px solid #3d4a50; border-radius: 6px; background: #263036; color: #f7fafb; padding: 10px 14px; font: inherit; cursor: pointer; }
    button:hover { background: #303c43; }
    @media (max-width: 620px) { body { place-items: start center; padding: 16px 0; } .grid, .settings { grid-template-columns: 1fr; } .pose { font-size: 1.2rem; } }
  </style>
</head>
<body>
  <main>
    <h1>Dex3 Joint Slider</h1>
    <section class="panel">
      <div class="grid">
        <label>Hand
          <select id="hand">
            <option value="right">right</option>
            <option value="left">left</option>
          </select>
        </label>
        <label>Joint
          <select id="joint"></select>
        </label>
      </div>
      <div class="readout">
        <div class="pose" id="pose">--</div>
        <div class="limits" id="limits">--</div>
        <input id="slider" type="range" step="0.001">
      </div>
      <div class="settings">
        <label>Ramp speed rad/s <input id="speed" type="number" step="0.05" min="0.01"></label>
        <label>kp <input id="kp" type="number" step="0.1" min="0"></label>
        <label>kd <input id="kd" type="number" step="0.01" min="0"></label>
      </div>
      <div class="actions">
        <button id="openHand" type="button">Open Hand</button>
        <button id="zeroGains" type="button">Zero Gains</button>
      </div>
      <p class="status" id="status"></p>
    </section>
  </main>
  <script>
    const hand = document.querySelector("#hand");
    const joint = document.querySelector("#joint");
    const slider = document.querySelector("#slider");
    const pose = document.querySelector("#pose");
    const limits = document.querySelector("#limits");
    const status = document.querySelector("#status");
    const speed = document.querySelector("#speed");
    const kp = document.querySelector("#kp");
    const kd = document.querySelector("#kd");
    let selectedJoint = 0;
    let syncing = false;

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function fmt(value) {
      return Number(value).toFixed(3);
    }

    function render(state) {
      syncing = true;
      selectedJoint = state.jointIndex;
      hand.value = state.hand;
      if (joint.options.length !== state.joints.length) {
        joint.replaceChildren(...state.joints.map((item) => {
          const option = document.createElement("option");
          option.value = item.index;
          option.textContent = `${item.index}: ${item.name}`;
          return option;
        }));
      }
      joint.value = String(selectedJoint);
      slider.min = state.min;
      slider.max = state.max;
      slider.value = state.desired;
      speed.value = state.speedRadS;
      kp.value = state.kp;
      kd.value = state.kd;
      pose.textContent = `${state.hand} ${state.jointName}: current ${fmt(state.current)} rad, desired ${fmt(state.desired)} rad`;
      limits.textContent = `limits [${fmt(state.min)}, ${fmt(state.max)}]`;
      status.textContent = state.status;
      syncing = false;
    }

    async function refresh() {
      try {
        render(await api(`/api/state?joint=${selectedJoint}`));
      } catch (error) {
        status.textContent = String(error);
      }
    }

    async function patch(payload) {
      render(await api("/api/state", { method: "POST", body: JSON.stringify(payload) }));
    }

    hand.addEventListener("change", () => patch({ hand: hand.value, jointIndex: selectedJoint }));
    joint.addEventListener("change", () => { selectedJoint = Number(joint.value); refresh(); });
    slider.addEventListener("input", () => {
      if (syncing) return;
      patch({ jointIndex: selectedJoint, desired: Number(slider.value) });
    });
    for (const input of [speed, kp, kd]) {
      input.addEventListener("change", () => patch({
        jointIndex: selectedJoint,
        speedRadS: Number(speed.value),
        kp: Number(kp.value),
        kd: Number(kd.value),
      }));
    }
    document.querySelector("#openHand").addEventListener("click", async () => {
      render(await api("/api/open", { method: "POST", body: "{}" }));
    });
    document.querySelector("#zeroGains").addEventListener("click", async () => {
      render(await api("/api/zero_gains", { method: "POST", body: "{}" }));
    });
    refresh();
    setInterval(refresh, 250);
  </script>
</body>
</html>
"""


class Dex3HttpServer(ThreadingHTTPServer):
    daemon_threads = True
    slider: Dex3JointSlider


def make_handler(slider: Dex3JointSlider) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server: Dex3HttpServer

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(
                "%s - - [%s] %s\n"
                % (self.client_address[0], self.log_date_time_string(), fmt % args)
            )

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("", "/"):
                self._send_text(INDEX_HTML, "text/html; charset=utf-8")
                return
            if path == "/api/state":
                joint_idx = self._query_joint_index(default=0)
                self._send_json(slider.state(joint_idx))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                payload = self._read_json()
                joint_idx = int(payload.get("jointIndex", 0))
                if path == "/api/state":
                    if "hand" in payload:
                        slider.set_hand(str(payload["hand"]))
                    if "desired" in payload:
                        slider.set_joint_target(joint_idx, float(payload["desired"]))
                    slider.update_settings(
                        speed_rad_s=payload.get("speedRadS"),
                        kp=payload.get("kp"),
                        kd=payload.get("kd"),
                        tau=payload.get("tau"),
                    )
                    self._send_json(slider.state(joint_idx))
                    return
                if path == "/api/open":
                    slider.open_hand()
                    self._send_json(slider.state(joint_idx))
                    return
                if path == "/api/zero_gains":
                    slider.zero_gains_once()
                    self._send_json(slider.state(joint_idx))
                    return
            except Exception as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _query_joint_index(self, default: int) -> int:
            if "?" not in self.path:
                return default
            query = self.path.split("?", 1)[1]
            for pair in query.split("&"):
                key, _, value = pair.partition("=")
                if key == "joint":
                    return int(value)
            return default

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_text(self, text: str, content_type: str) -> None:
            data = text.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve(slider: Dex3JointSlider, host: str, port: int) -> Dex3HttpServer:
    server = Dex3HttpServer((host, port), make_handler(slider))
    server.slider = slider
    return server


def main() -> int:
    args = parse_args()
    slider = Dex3JointSlider(args)
    slider.start()
    server = serve(slider, str(args.web_host), int(args.web_port))
    host_for_url = "127.0.0.1" if str(args.web_host) in ("", "0.0.0.0") else str(args.web_host)
    print(f"Dex3 joint slider web UI: http://{host_for_url}:{int(args.web_port)}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        slider.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
