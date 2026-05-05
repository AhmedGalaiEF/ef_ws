#!/usr/bin/env python3
"""
Terminal control and probe UI for the Unitree G1 Robot wrapper.

Uses ../modules/sdk_client.py through modules.sdk_client.Robot.

Keys
----
  Tab                 cycle focus: FSM -> locomotion -> probes
  q / Esc             quit, sending stop first

  FSM:
    z                 zero torque
    d                 damp
    a                 FSM id 2 / airborne placeholder
    w                 walk mode (FSM id 501)
    r                 run mode (FSM id 802)
    v                 dev gait mode
    f                 refresh FSM/state

  Locomotion:
    Space             stop and zero command
    m                 toggle continuous send
    Up/Down           adjust vx
    Left/Right        adjust vyaw
    h/l               adjust vy
    0                 zero command
    +/-               adjust speed step

  Probes:
    j/k               select probe page
    p                 force refresh
"""

from __future__ import annotations

import argparse
import curses
import json
import math
import os
import sys
import time
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODULES_DIR = os.path.join(ROOT_DIR, "modules")
for _p in (ROOT_DIR, MODULES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dds_env import ensure_cyclonedds_environment

ensure_cyclonedds_environment()

from modules.sdk_client import BODY_JOINT_NAME_BY_INDEX, Robot


FOCUS_FSM = 0
FOCUS_LOCO = 1
FOCUS_PROBES = 2
FOCUS_NAMES = ("FSM", "Locomotion", "Probes")

C_NORMAL = 0
C_GREEN = 1
C_YELLOW = 2
C_RED = 3
C_CYAN = 4
C_SEL = 5
C_FOCUS = 6
C_BLUE = 7

PROBE_PAGES = ("summary", "sensors", "joints", "raw")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.{digits}f}"
    if isinstance(value, tuple):
        return "(" + ", ".join(_fmt(v, digits) for v in value) + ")"
    return str(value)


def _json_default(value: Any) -> Any:
    if hasattr(value, "__dict__"):
        return {
            key: val
            for key, val in vars(value).items()
            if not key.startswith("_") and isinstance(val, (str, int, float, bool, list, tuple, dict, type(None)))
        }
    return repr(value)


class RobotTUI:
    def __init__(self, args: argparse.Namespace) -> None:
        self.iface = str(args.iface)
        self.domain_id = int(args.domain_id)
        self.rate_hz = max(1.0, float(args.rate_hz))
        self.send_hz = max(1.0, float(args.send_hz))
        self.max_vx = abs(float(args.max_vx))
        self.max_vy = abs(float(args.max_vy))
        self.max_vyaw = abs(float(args.max_vyaw))
        self.step_v = abs(float(args.step_v))
        self.step_yaw = abs(float(args.step_yaw))

        self.focus = FOCUS_FSM
        self.probe_idx = 0
        self.status = "Initializing Robot..."
        self._running = True

        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.loco_enabled = False
        self.last_send_s = 0.0
        self.last_probe_s = 0.0
        self.probe_interval_s = max(0.05, float(args.probe_interval))
        self.snapshot: dict[str, Any] = {}
        self.last_error: str | None = None

        self.robot = Robot(
            iface=self.iface,
            domain_id=self.domain_id,
            auto_start_sensors=True,
        )
        sport_ok = self.robot.wait_for_sport_state(timeout=float(args.wait_timeout))
        low_ok = self.robot.wait_for_low_state(timeout=float(args.wait_timeout))
        self.status = (
            f"Connected on {self.iface} domain {self.domain_id} "
            f"(sport={'ok' if sport_ok else 'waiting'}, lowstate={'ok' if low_ok else 'waiting'})"
        )
        self.refresh_probe(force=True)

    @staticmethod
    def _safe_addstr(win, y: int, x: int, text: str, attr: int = 0) -> None:
        try:
            win.addstr(y, x, text, attr)
        except curses.error:
            pass

    @staticmethod
    def _safe_addnstr(win, y: int, x: int, text: str, n: int, attr: int = 0) -> None:
        try:
            win.addnstr(y, x, text, max(0, n), attr)
        except curses.error:
            pass

    def _cp(self, pair: int) -> int:
        return curses.color_pair(pair) if curses.has_colors() else 0

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, float(value)))

    def _set_status(self, text: str, *, error: bool = False) -> None:
        self.status = text
        self.last_error = text if error else None

    def _call_robot(self, label: str, func, *args) -> None:
        try:
            result = func(*args)
            suffix = "" if result is None else f" -> {result}"
            self._set_status(f"{label}{suffix}")
            self.refresh_probe(force=True)
        except Exception as exc:
            self._set_status(f"{label} failed: {exc}", error=True)

    def _stop_loco(self) -> None:
        self.vx = self.vy = self.vyaw = 0.0
        self.loco_enabled = False
        try:
            self.robot.stop()
            self._set_status("Locomotion stopped")
        except Exception as exc:
            self._set_status(f"Stop failed: {exc}", error=True)

    def _zero_loco_cmd(self) -> None:
        self.vx = self.vy = self.vyaw = 0.0
        self._set_status("Locomotion command zeroed")

    def _send_loco(self) -> None:
        try:
            result = self.robot.loco_move(self.vx, self.vy, self.vyaw)
            self.last_send_s = time.monotonic()
            self._set_status(
                f"Sent vx={self.vx:+.2f} vy={self.vy:+.2f} vyaw={self.vyaw:+.2f} -> {result}"
            )
        except Exception as exc:
            self.loco_enabled = False
            self._set_status(f"Move failed: {exc}", error=True)

    def tick(self) -> None:
        now = time.monotonic()
        if self.loco_enabled and (now - self.last_send_s) >= (1.0 / self.send_hz):
            self._send_loco()
        self.refresh_probe(force=False)

    def refresh_probe(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self.last_probe_s < self.probe_interval_s:
            return True
        self.last_probe_s = now
        try:
            imu = self.robot.get_imu()
            self.snapshot = {
                "fsm": self.robot.get_fsm(),
                "mode": self.robot.get_mode(),
                "gait": self.robot.get_gait(),
                "body_height": self.robot.get_body_height(),
                "position": self.robot.get_position(),
                "velocity": self.robot.get_velocity(),
                "odom_pose": self.robot.get_odom_pose(),
                "imu": {
                    "rpy": None if imu is None else imu.rpy,
                    "gyro": None if imu is None else imu.gyro,
                    "acc": None if imu is None else imu.acc,
                    "temp": None if imu is None else imu.temp,
                },
                "sensor_timestamps": self.robot.get_sensor_timestamps(),
                "sensor_stale": self.robot.sensors_stale(max_age=1.0),
                "joint_states": self.robot.get_joint_states(),
                "is_moving": self.robot.is_moving(),
            }
            return True
        except Exception as exc:
            self._set_status(f"Probe refresh failed: {exc}", error=True)
            return False

    def _draw_header(self, win, h: int, w: int) -> None:
        self._safe_addstr(win, 0, 0, "-" * w, self._cp(C_CYAN))
        title = "Robot TUI"
        self._safe_addstr(win, 0, max(0, (w - len(title)) // 2), title, self._cp(C_CYAN) | curses.A_BOLD)
        self._safe_addnstr(
            win,
            1,
            0,
            f" iface={self.iface} domain={self.domain_id} focus={FOCUS_NAMES[self.focus]}",
            w,
            self._cp(C_CYAN) | curses.A_BOLD,
        )

        fsm = self.snapshot.get("fsm") or {}
        imu = self.snapshot.get("imu") or {}
        state = (
            f" FSM id={_fmt(fsm.get('id'))} mode={_fmt(fsm.get('mode'))}"
            f" sport_mode={_fmt(self.snapshot.get('mode'))} gait={_fmt(self.snapshot.get('gait'))}"
            f" moving={_fmt(self.snapshot.get('is_moving'))}"
        )
        self._safe_addnstr(win, 2, 0, state, w)
        pose = (
            f" pos={_fmt(self.snapshot.get('position'))}"
            f" vel={_fmt(self.snapshot.get('velocity'))}"
            f" rpy={_fmt(imu.get('rpy'))}"
        )
        self._safe_addnstr(win, 3, 0, pose, w)
        self._safe_addstr(win, 4, 0, "-" * w, self._cp(C_CYAN))

    def _draw_fsm_panel(self, win, top: int, left: int, height: int, width: int) -> None:
        attr = self._cp(C_FOCUS) | curses.A_BOLD if self.focus == FOCUS_FSM else curses.A_BOLD
        self._safe_addnstr(win, top, left, " FSM [z]zero [d]damp [a]id2 [w]walk [r]run [v]dev [f]refresh ", width, attr)
        rows = [
            ("Zero torque", "z", "robot.zero_torque() / fsm_0_zt()"),
            ("Damp", "d", "robot.damp() / fsm_1_damp()"),
            ("FSM id 2", "a", "robot.fsm_2_airborne()"),
            ("Walk mode", "w", "robot.walk_mode() -> SetFsmId(501)"),
            ("Run mode", "r", "robot.run_mode() -> SetFsmId(802)"),
            ("Dev gait", "v", "robot.dev_mode()"),
        ]
        for idx, (name, key, detail) in enumerate(rows[: max(0, height - 2)]):
            y = top + 1 + idx
            self._safe_addnstr(win, y, left, f" [{key}] {name:<12} {detail}", width)

    def _draw_loco_panel(self, win, top: int, left: int, height: int, width: int) -> None:
        attr = self._cp(C_FOCUS) | curses.A_BOLD if self.focus == FOCUS_LOCO else curses.A_BOLD
        enabled = "ON" if self.loco_enabled else "OFF"
        enabled_attr = self._cp(C_GREEN if self.loco_enabled else C_RED) | curses.A_BOLD
        self._safe_addnstr(win, top, left, " Locomotion [m]send [space]stop arrows/h/l adjust +/- step ", width, attr)
        self._safe_addstr(win, top + 1, left, f" continuous send: {enabled}", enabled_attr)
        self._safe_addnstr(win, top + 2, left, f" vx   {self.vx:+.3f} m/s    limit +/-{self.max_vx:.2f}", width)
        self._safe_addnstr(win, top + 3, left, f" vy   {self.vy:+.3f} m/s    limit +/-{self.max_vy:.2f}", width)
        self._safe_addnstr(win, top + 4, left, f" vyaw {self.vyaw:+.3f} rad/s  limit +/-{self.max_vyaw:.2f}", width)
        self._safe_addnstr(win, top + 5, left, f" step_v={self.step_v:.3f} m/s  step_yaw={self.step_yaw:.3f} rad/s", width, self._cp(C_YELLOW))
        self._safe_addnstr(win, top + 6, left, " Up/Down=vx  h/l=vy  Left/Right=vyaw  0=zero command", width)

    def _draw_probe_panel(self, win, top: int, left: int, height: int, width: int) -> None:
        attr = self._cp(C_FOCUS) | curses.A_BOLD if self.focus == FOCUS_PROBES else curses.A_BOLD
        page = PROBE_PAGES[self.probe_idx]
        self._safe_addnstr(win, top, left, f" Probes page={page} [j/k]page [p]refresh ", width, attr)
        lines = self._probe_lines(page, max(0, height - 1), width)
        for idx, line in enumerate(lines):
            self._safe_addnstr(win, top + 1 + idx, left, line, width)

    def _probe_lines(self, page: str, max_lines: int, width: int) -> list[str]:
        if max_lines <= 0:
            return []
        if page == "summary":
            imu = self.snapshot.get("imu") or {}
            return [
                f"fsm: {self.snapshot.get('fsm')}",
                f"mode={_fmt(self.snapshot.get('mode'))} gait={_fmt(self.snapshot.get('gait'))} height={_fmt(self.snapshot.get('body_height'))}",
                f"position={_fmt(self.snapshot.get('position'))}",
                f"velocity={_fmt(self.snapshot.get('velocity'))}",
                f"odom_pose={_fmt(self.snapshot.get('odom_pose'))}",
                f"imu.rpy={_fmt(imu.get('rpy'))}",
                f"imu.gyro={_fmt(imu.get('gyro'))}",
                f"imu.acc={_fmt(imu.get('acc'))}",
            ][:max_lines]
        if page == "sensors":
            ts = self.snapshot.get("sensor_timestamps") or {}
            stale = self.snapshot.get("sensor_stale") or {}
            now = time.time()
            rows = []
            for name in sorted(ts):
                stamp = float(ts.get(name) or 0.0)
                age = "-" if stamp <= 0.0 else f"{now - stamp:.2f}s"
                mark = "STALE" if stale.get(name) else "ok"
                rows.append(f"{name:<38} age={age:<7} {mark}")
            return rows[:max_lines]
        if page == "joints":
            state = self.snapshot.get("joint_states") or {}
            joints = state.get("joints") or {}
            rows = [f"timestamp={_fmt(state.get('timestamp'))} count={len(joints)}"]
            for index in sorted(BODY_JOINT_NAME_BY_INDEX):
                name = BODY_JOINT_NAME_BY_INDEX[index]
                entry = joints.get(name) or {}
                rows.append(
                    f"{index:02d} {name:<25} q={_fmt(entry.get('position')):>8} "
                    f"dq={_fmt(entry.get('velocity')):>8} tau={_fmt(entry.get('torque')):>8}"
                )
            return rows[:max_lines]
        raw = json.dumps(
            {k: v for k, v in self.snapshot.items() if k != "joint_states"},
            indent=2,
            sort_keys=True,
            default=_json_default,
        ).splitlines()
        return [line[:width] for line in raw[:max_lines]]

    def _draw_footer(self, win, h: int, w: int) -> None:
        hints = "Tab:focus  q/Esc:quit  FSM z/d/a/w/r/v  loco m/space/arrows/h/l/0  probes j/k/p"
        self._safe_addnstr(win, h - 2, 0, hints, w, self._cp(C_YELLOW))
        st_attr = self._cp(C_RED if self.last_error else C_GREEN)
        self._safe_addnstr(win, h - 1, 0, f" {self.status}"[:w], w, st_attr)

    def draw(self, win, h: int, w: int) -> None:
        if h < 18 or w < 80:
            self._safe_addstr(win, 0, 0, f"Terminal too small ({w}x{h}). Need 80x18.")
            return
        self._draw_header(win, h, w)
        content_top = 5
        content_bottom = h - 3
        content_h = max(1, content_bottom - content_top)
        left_w = max(38, w // 2)
        right_x = left_w + 1
        right_w = max(1, w - right_x)
        top_h = min(8, content_h // 2)
        self._draw_fsm_panel(win, content_top, 0, top_h, left_w)
        self._draw_loco_panel(win, content_top + top_h + 1, 0, content_h - top_h - 1, left_w)
        for y in range(content_top, content_bottom):
            self._safe_addstr(win, y, left_w, "|", self._cp(C_CYAN))
        self._draw_probe_panel(win, content_top, right_x, content_h, right_w)
        self._draw_footer(win, h, w)

    def handle_key(self, key: int) -> None:
        if key in (ord("q"), 27):
            self._running = False
            return
        if key == 9:
            self.focus = (self.focus + 1) % 3
            return

        if key == ord(" "):
            self._stop_loco()
            return
        if key == ord("m"):
            self.loco_enabled = not self.loco_enabled
            self._set_status(f"Continuous locomotion send {'enabled' if self.loco_enabled else 'disabled'}")
            if self.loco_enabled:
                self._send_loco()
            return
        if key == ord("0"):
            self._zero_loco_cmd()
            return

        if key in (curses.KEY_UP, curses.KEY_DOWN, curses.KEY_LEFT, curses.KEY_RIGHT, ord("h"), ord("l")):
            if key == curses.KEY_UP:
                self.vx = self._clamp(self.vx + self.step_v, self.max_vx)
            elif key == curses.KEY_DOWN:
                self.vx = self._clamp(self.vx - self.step_v, self.max_vx)
            elif key == ord("h"):
                self.vy = self._clamp(self.vy + self.step_v, self.max_vy)
            elif key == ord("l"):
                self.vy = self._clamp(self.vy - self.step_v, self.max_vy)
            elif key == curses.KEY_LEFT:
                self.vyaw = self._clamp(self.vyaw + self.step_yaw, self.max_vyaw)
            elif key == curses.KEY_RIGHT:
                self.vyaw = self._clamp(self.vyaw - self.step_yaw, self.max_vyaw)
            self._set_status(f"Command vx={self.vx:+.2f} vy={self.vy:+.2f} vyaw={self.vyaw:+.2f}")
            if self.loco_enabled:
                self._send_loco()
            return

        if key in (ord("+"), ord("=")):
            self.step_v = min(0.5, self.step_v * 2.0)
            self.step_yaw = min(1.0, self.step_yaw * 2.0)
            self._set_status(f"Steps set to v={self.step_v:.3f}, yaw={self.step_yaw:.3f}")
            return
        if key == ord("-"):
            self.step_v = max(0.005, self.step_v / 2.0)
            self.step_yaw = max(0.01, self.step_yaw / 2.0)
            self._set_status(f"Steps set to v={self.step_v:.3f}, yaw={self.step_yaw:.3f}")
            return

        if key == ord("z"):
            self._call_robot("Zero torque", self.robot.zero_torque)
            return
        if key == ord("d"):
            self._call_robot("Damp", self.robot.damp)
            return
        if key == ord("a"):
            self._call_robot("FSM id 2", self.robot.fsm_2_airborne)
            return
        if key == ord("w"):
            self._call_robot("Walk mode", self.robot.walk_mode)
            return
        if key == ord("r"):
            self._call_robot("Run mode", self.robot.run_mode)
            return
        if key == ord("v"):
            self._call_robot("Dev gait", self.robot.dev_mode)
            return
        if key in (ord("f"), ord("p")):
            if self.refresh_probe(force=True):
                self._set_status("Probe refreshed")
            return

        if key in (ord("j"), ord("k")) and self.focus == FOCUS_PROBES:
            delta = 1 if key == ord("j") else -1
            self.probe_idx = (self.probe_idx + delta) % len(PROBE_PAGES)
            self.refresh_probe(force=True)
            return

    def run(self) -> None:
        curses.wrapper(self._curses_main)

    def _curses_main(self, stdscr) -> None:
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(C_GREEN, curses.COLOR_GREEN, -1)
            curses.init_pair(C_YELLOW, curses.COLOR_YELLOW, -1)
            curses.init_pair(C_RED, curses.COLOR_RED, -1)
            curses.init_pair(C_CYAN, curses.COLOR_CYAN, -1)
            curses.init_pair(C_SEL, curses.COLOR_BLACK, curses.COLOR_WHITE)
            curses.init_pair(C_FOCUS, curses.COLOR_BLACK, curses.COLOR_CYAN)
            curses.init_pair(C_BLUE, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.curs_set(0)
        stdscr.timeout(20)
        dt_target = 1.0 / self.rate_hz
        last_tick = 0.0
        while self._running:
            h, w = stdscr.getmaxyx()
            try:
                stdscr.erase()
                self.draw(stdscr, h, w)
                stdscr.refresh()
            except curses.error:
                pass
            key = stdscr.getch()
            if key != -1:
                self.handle_key(key)
            now = time.monotonic()
            if now - last_tick >= dt_target:
                self.tick()
                last_tick = now
        self._stop_loco()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Curses TUI for modules.sdk_client.Robot.")
    p.add_argument("--iface", default="eth0", help="Network interface for DDS")
    p.add_argument("--domain-id", type=int, default=0)
    p.add_argument("--rate-hz", type=float, default=30.0, help="UI/tick rate")
    p.add_argument("--send-hz", type=float, default=10.0, help="Continuous locomotion command rate")
    p.add_argument("--probe-interval", type=float, default=0.25, help="State refresh interval")
    p.add_argument("--wait-timeout", type=float, default=2.0, help="Initial sensor wait timeout")
    p.add_argument("--max-vx", type=float, default=0.5)
    p.add_argument("--max-vy", type=float, default=0.3)
    p.add_argument("--max-vyaw", type=float, default=0.8)
    p.add_argument("--step-v", type=float, default=0.05)
    p.add_argument("--step-yaw", type=float, default=0.10)
    return p.parse_args()


def main() -> None:
    app = RobotTUI(parse_args())
    app.run()


if __name__ == "__main__":
    main()
