#!/usr/bin/env python3
"""
USB gamepad + curses TUI controller for WBC on Unitree G1.

This follows the stick layout from usb_controller_scheme.txt:
  - Left stick  -> yaw rate (vyaw)
  - Right stick -> linear + lateral motion (vx, vy)

Implemented gamepad combos:
  - L2 + B          -> damp
  - L2 + A          -> zero torque
  - L2 + X          -> recapture WBC neutral pose
  - L2 + Y          -> FSM 3 if supported
  - L2 + D-pad Up   -> FSM 4 if supported
  - R1 + Y          -> walk mode
  - R1 + A          -> run mode
  - R1 + B          -> FSM 812 if supported
  - double tap R1   -> toggle gait type 0/1 if supported
  - Start/Menu      -> toggle help/menu pane

The D-pad mirrors the scheme's "select target / increment / decrement"
behavior, but targets WBC tuning fields instead of raw joint poses.
"""
from __future__ import annotations

import argparse
import curses
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODULES_DIR = os.path.join(ROOT_DIR, "modules")
for _p in (ROOT_DIR, MODULES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dds_env import ensure_cyclonedds_environment

ensure_cyclonedds_environment()

try:
    import pygame
except ModuleNotFoundError as exc:
    raise SystemExit(
        "The 'pygame' package is required for USB controller support.\n"
        "Install with: pip install pygame"
    ) from exc

from modules.sdk_client import Robot
from wbc import WBController, WBCConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOG = logging.getLogger("wbc_usb_controller")

# Xbox/Linux defaults used in usb_controller.py
BTN_A = 0
BTN_B = 1
BTN_X = 2
BTN_Y = 3
BTN_L1 = 4
BTN_R1 = 5
BTN_BACK = 6
BTN_START = 7
BTN_L2 = 8
BTN_R2 = 9

AXIS_LX = 0
AXIS_LY = 1
AXIS_RX = 3
AXIS_RY = 4

HAT_CENTER = (0, 0)

C_GREEN = 1
C_YELLOW = 2
C_RED = 3
C_CYAN = 4
C_SEL = 5


def apply_deadzone(value: float, dz: float) -> float:
    if abs(value) < dz:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - dz) / (1.0 - dz)


def clamp_abs(value: float, limit: float) -> float:
    return max(-limit, min(limit, float(value)))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.{digits}f}"
    if isinstance(value, tuple):
        return "(" + ", ".join(fmt(v, digits) for v in value) + ")"
    return str(value)


@dataclass
class TuneField:
    key: str
    label: str
    step: float
    minimum: float
    maximum: float


class WBCUsbControllerApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.iface = str(args.iface)
        self.domain_id = int(args.domain_id)
        self.deadzone = max(0.0, min(0.95, float(args.deadzone)))
        self.send_hz = max(1.0, float(args.send_hz))
        self.max_vx = abs(float(args.max_vx))
        self.max_vy = abs(float(args.max_vy))
        self.max_vyaw = abs(float(args.max_vyaw))
        self.probe_interval_s = max(0.05, float(args.probe_interval))
        self.r1_double_tap_s = max(0.1, float(args.r1_double_tap))

        self.robot = Robot(
            iface=self.iface,
            domain_id=self.domain_id,
            auto_start_sensors=True,
        )
        self.robot.wait_for_sport_state(timeout=float(args.wait_timeout))
        self.robot.wait_for_low_state(timeout=float(args.wait_timeout))
        self.robot.unrelease_arms()

        self.wbc = WBController(
            self.robot,
            WBCConfig(
                roll_kp=float(args.roll_kp),
                roll_kd=float(args.roll_kd),
                pitch_kp=float(args.pitch_kp),
                pitch_kd=float(args.pitch_kd),
                arm_compensation=float(args.arm_comp),
                rate_hz=float(args.rate_hz),
                pitch_offset=float(args.pitch_offset),
                roll_offset=float(args.roll_offset),
            ),
        )
        self.wbc.start()

        self.load_mass = float(args.load_mass)
        self.load_arm = float(args.load_arm)
        if self.load_mass > 0.0:
            self.wbc.set_load(self.load_mass, self.load_arm)

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= 0:
            raise SystemExit("No joystick detected. Connect a USB gamepad and retry.")
        if args.joy < 0 or args.joy >= pygame.joystick.get_count():
            raise SystemExit(f"Joystick index {args.joy} is out of range.")
        self.joy = pygame.joystick.Joystick(int(args.joy))
        self.joy.init()

        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.manual_hold = False
        self.show_help = True
        self.running = True
        self.status = f"Using {self.joy.get_name()}"
        self.last_error = False
        self.last_probe_s = 0.0
        self.last_r1_press_s = 0.0
        self.gait_toggle_state = 0
        self.prev_buttons: dict[int, bool] = {}
        self.prev_hat = HAT_CENTER
        self.last_snapshot: dict[str, Any] = {}

        self.tune_fields = [
            TuneField("pitch_offset", "Pitch offset", 0.01, -0.45, 0.45),
            TuneField("roll_offset", "Roll offset", 0.01, -0.35, 0.35),
            TuneField("roll_kp", "Roll kp", 0.05, 0.0, 2.0),
            TuneField("roll_kd", "Roll kd", 0.01, 0.0, 0.5),
            TuneField("pitch_kp", "Pitch kp", 0.05, 0.0, 2.0),
            TuneField("pitch_kd", "Pitch kd", 0.01, 0.0, 0.5),
            TuneField("arm_compensation", "Arm comp", 0.05, 0.0, 1.5),
            TuneField("load_mass", "Load mass", 0.25, 0.0, 20.0),
            TuneField("load_arm", "Load arm", 0.02, 0.1, 1.0),
        ]
        self.selected_field = 0

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

    def _set_status(self, text: str, *, error: bool = False) -> None:
        self.status = text
        self.last_error = error

    def _edge_pressed(self, button: int) -> bool:
        pressed = bool(self.joy.get_button(button))
        previous = self.prev_buttons.get(button, False)
        self.prev_buttons[button] = pressed
        return pressed and not previous

    def _button_down(self, button: int) -> bool:
        return button < self.joy.get_numbuttons() and bool(self.joy.get_button(button))

    def _hat(self) -> tuple[int, int]:
        return self.joy.get_hat(0) if self.joy.get_numhats() > 0 else HAT_CENTER

    def _call(self, label: str, func, *args) -> None:
        try:
            result = func(*args)
            suffix = "" if result is None else f" -> {result}"
            self._set_status(f"{label}{suffix}")
        except Exception as exc:
            self._set_status(f"{label} failed: {exc}", error=True)

    def _set_wbc_field(self, field: TuneField, delta: float) -> None:
        if field.key == "load_mass":
            self.load_mass = clamp_abs(self.load_mass + delta, field.maximum)
            self.load_mass = max(field.minimum, self.load_mass)
            self.wbc.set_load(self.load_mass, self.load_arm)
            self._set_status(f"load_mass={self.load_mass:.2f} kg")
            return
        if field.key == "load_arm":
            self.load_arm = max(field.minimum, min(field.maximum, self.load_arm + delta))
            self.wbc.set_load(self.load_mass, self.load_arm)
            self._set_status(f"load_arm={self.load_arm:.2f} m")
            return

        value = getattr(self.wbc.cfg, field.key)
        value = max(field.minimum, min(field.maximum, float(value) + delta))
        setattr(self.wbc.cfg, field.key, value)
        self._set_status(f"{field.key}={value:.3f}")

    def _field_value(self, field: TuneField) -> float:
        if field.key == "load_mass":
            return self.load_mass
        if field.key == "load_arm":
            return self.load_arm
        return float(getattr(self.wbc.cfg, field.key))

    def _try_set_fsm(self, fsm_id: int, label: str) -> None:
        client = getattr(self.robot, "_client", None)
        if client is None or not hasattr(client, "SetFsmId"):
            self._set_status(f"{label} unsupported by current locomotion client", error=True)
            return
        try:
            client.SetFsmId(int(fsm_id))
            self._set_status(f"{label} -> FSM {fsm_id}")
        except Exception as exc:
            self._set_status(f"{label} failed: {exc}", error=True)

    def _try_set_gait_type(self, gait_type: int) -> None:
        client = getattr(self.robot, "_client", None)
        if client is None or not hasattr(client, "SetGaitType"):
            self._set_status("Gait type toggle unsupported", error=True)
            return
        try:
            client.SetGaitType(int(gait_type))
            self._set_status(f"Gait type -> {gait_type}")
        except Exception as exc:
            self._set_status(f"SetGaitType failed: {exc}", error=True)

    def _poll_gamepad(self) -> None:
        pygame.event.pump()

        l2 = self._button_down(BTN_L2)
        r1 = self._button_down(BTN_R1)
        hat = self._hat()

        if self._edge_pressed(BTN_START):
            self.show_help = not self.show_help
            self._set_status(f"Menu {'shown' if self.show_help else 'hidden'}")

        if self._edge_pressed(BTN_BACK):
            self.manual_hold = not self.manual_hold
            self._set_status(f"Manual hold {'enabled' if self.manual_hold else 'disabled'}")

        if self._edge_pressed(BTN_R1):
            now = time.monotonic()
            if now - self.last_r1_press_s <= self.r1_double_tap_s:
                self.gait_toggle_state = 1 - self.gait_toggle_state
                self._try_set_gait_type(self.gait_toggle_state)
            self.last_r1_press_s = now

        if l2 and self._edge_pressed(BTN_A):
            self._call("Zero torque", self.robot.zero_torque)
            return
        if l2 and self._edge_pressed(BTN_B):
            self._call("Damp", self.robot.damp)
            return
        if l2 and self._edge_pressed(BTN_X):
            self.wbc.set_neutral_pose(None)
            self._set_status("WBC neutral pose recaptured")
            return
        if l2 and self._edge_pressed(BTN_Y):
            self._try_set_fsm(3, "Sit mode")
            return

        if r1 and self._edge_pressed(BTN_Y):
            self._call("Walk mode", self.robot.walk_mode)
            return
        if r1 and self._edge_pressed(BTN_A):
            self._call("Run mode", self.robot.run_mode)
            return
        if r1 and self._edge_pressed(BTN_B):
            self._try_set_fsm(812, "Climb mode")
            return

        if l2 and hat[1] > 0 and self.prev_hat[1] <= 0:
            self._try_set_fsm(4, "Preparation mode")
            self.prev_hat = hat
            return

        if hat != self.prev_hat:
            if hat[0] > 0 and self.prev_hat[0] <= 0:
                self.selected_field = (self.selected_field + 1) % len(self.tune_fields)
                self._set_status(f"Selected {self.tune_fields[self.selected_field].label}")
            elif hat[0] < 0 and self.prev_hat[0] >= 0:
                self.selected_field = (self.selected_field - 1) % len(self.tune_fields)
                self._set_status(f"Selected {self.tune_fields[self.selected_field].label}")
            elif hat[1] > 0 and self.prev_hat[1] <= 0 and not l2:
                field = self.tune_fields[self.selected_field]
                self._set_wbc_field(field, field.step)
            elif hat[1] < 0 and self.prev_hat[1] >= 0:
                field = self.tune_fields[self.selected_field]
                self._set_wbc_field(field, -field.step)
        self.prev_hat = hat

        lx = apply_deadzone(self.joy.get_axis(AXIS_LX), self.deadzone) if self.joy.get_numaxes() > AXIS_LX else 0.0
        rx = apply_deadzone(self.joy.get_axis(AXIS_RX), self.deadzone) if self.joy.get_numaxes() > AXIS_RX else 0.0
        ry = apply_deadzone(self.joy.get_axis(AXIS_RY), self.deadzone) if self.joy.get_numaxes() > AXIS_RY else 0.0

        self.vyaw = -lx * self.max_vyaw
        self.vx = -ry * self.max_vx
        self.vy = -rx * self.max_vy

        if self.manual_hold:
            self.vx = self.vy = self.vyaw = 0.0

        self.wbc.set_loco_cmd(self.vx, self.vy, self.vyaw)

    def refresh_probe(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_probe_s < self.probe_interval_s:
            return
        self.last_probe_s = now
        try:
            imu = self.robot.get_imu()
            self.last_snapshot = {
                "fsm": self.robot.get_fsm(),
                "mode": self.robot.get_mode(),
                "gait": self.robot.get_gait(),
                "position": self.robot.get_position(),
                "velocity": self.robot.get_velocity(),
                "imu_rpy": None if imu is None else imu.rpy,
                "imu_gyro": None if imu is None else imu.gyro,
                "is_moving": self.robot.is_moving(),
            }
        except Exception as exc:
            self._set_status(f"Probe refresh failed: {exc}", error=True)

    def tick(self) -> None:
        self._poll_gamepad()
        self.refresh_probe(force=False)

    def _draw_header(self, win, w: int) -> None:
        self._safe_addstr(win, 0, 0, "-" * w, self._cp(C_CYAN))
        title = "WBC USB Controller"
        self._safe_addstr(win, 0, max(0, (w - len(title)) // 2), title, self._cp(C_CYAN) | curses.A_BOLD)
        self._safe_addnstr(
            win,
            1,
            0,
            f" iface={self.iface} domain={self.domain_id} joy={self.joy.get_name()}",
            w,
            self._cp(C_CYAN) | curses.A_BOLD,
        )
        snap = self.last_snapshot
        self._safe_addnstr(
            win,
            2,
            0,
            (
                f" FSM={fmt((snap.get('fsm') or {}).get('id'))}"
                f" mode={fmt(snap.get('mode'))} gait={fmt(snap.get('gait'))}"
                f" moving={fmt(snap.get('is_moving'))}"
            ),
            w,
        )
        self._safe_addnstr(
            win,
            3,
            0,
            (
                f" pos={fmt(snap.get('position'))}"
                f" vel={fmt(snap.get('velocity'))}"
                f" imu.rpy={fmt(snap.get('imu_rpy'))}"
            ),
            w,
        )
        self._safe_addstr(win, 4, 0, "-" * w, self._cp(C_CYAN))

    def _draw_motion_panel(self, win, top: int, width: int) -> None:
        enabled_attr = self._cp(C_RED if self.manual_hold else C_GREEN) | curses.A_BOLD
        self._safe_addnstr(win, top, 0, " Motion from gamepad sticks ", width, self._cp(C_CYAN) | curses.A_BOLD)
        self._safe_addnstr(win, top + 1, 0, f" hold={('ON' if self.manual_hold else 'OFF')}", width, enabled_attr)
        self._safe_addnstr(win, top + 2, 0, f" vx   {self.vx:+.3f} / {self.max_vx:.2f} m/s", width)
        self._safe_addnstr(win, top + 3, 0, f" vy   {self.vy:+.3f} / {self.max_vy:.2f} m/s", width)
        self._safe_addnstr(win, top + 4, 0, f" vyaw {self.vyaw:+.3f} / {self.max_vyaw:.2f} rad/s", width)
        self._safe_addnstr(
            win,
            top + 5,
            0,
            f" imu roll={self.wbc.last_imu_roll:+.3f} pitch={self.wbc.last_imu_pitch:+.3f}",
            width,
        )
        self._safe_addnstr(
            win,
            top + 6,
            0,
            (
                f" waist cmd roll={self.wbc.last_waist_roll_cmd:+.3f}"
                f" pitch={self.wbc.last_waist_pitch_cmd:+.3f}"
            ),
            width,
        )
        self._safe_addnstr(win, top + 7, 0, f" odom={fmt(self.wbc.last_odom)}", width)

    def _draw_tuning_panel(self, win, top: int, left: int, width: int, height: int) -> None:
        self._safe_addnstr(win, top, left, " WBC tuning via D-pad ", width, self._cp(C_CYAN) | curses.A_BOLD)
        rows = min(len(self.tune_fields), max(0, height - 1))
        for idx in range(rows):
            field = self.tune_fields[idx]
            attr = self._cp(C_SEL) | curses.A_BOLD if idx == self.selected_field else 0
            text = f" {field.label:<14} {self._field_value(field):>7.3f}  step={field.step:.3f}"
            self._safe_addnstr(win, top + 1 + idx, left, text, width, attr)

    def _draw_help_panel(self, win, top: int, left: int, width: int, height: int) -> None:
        title = " Scheme / shortcuts " if self.show_help else " Shortcuts hidden (Start to show) "
        self._safe_addnstr(win, top, left, title, width, self._cp(C_CYAN) | curses.A_BOLD)
        if not self.show_help:
            return
        rows = [
            "Left stick = yaw, Right stick = vx/vy",
            "L2+B damp, L2+A zero torque",
            "L2+X recapture neutral, L2+Y FSM 3",
            "L2+Dpad Up FSM 4 if supported",
            "R1+Y walk, R1+A run, R1+B FSM 812",
            "Double tap R1 toggles gait type 0/1",
            "D-pad left/right select tuning field",
            "D-pad up/down adjust selected field",
            "Back toggles manual hold, Start toggles this menu",
            "q / Esc quits and stops locomotion",
        ]
        for idx, row in enumerate(rows[: max(0, height - 1)]):
            self._safe_addnstr(win, top + 1 + idx, left, row, width)

    def draw(self, win, h: int, w: int) -> None:
        if h < 22 or w < 90:
            self._safe_addstr(win, 0, 0, f"Terminal too small ({w}x{h}). Need at least 90x22.")
            return
        self._draw_header(win, w)
        self._draw_motion_panel(win, 6, w // 2 - 1)
        split = w // 2
        for y in range(5, h - 2):
            self._safe_addstr(win, y, split, "|", self._cp(C_CYAN))
        right_w = max(1, w - split - 1)
        self._draw_tuning_panel(win, 6, split + 1, right_w, 12)
        self._draw_help_panel(win, 18, split + 1, right_w, h - 20)
        hints = "q/Esc quit | Start menu | Back hold | D-pad tune | sticks drive via WBC"
        self._safe_addnstr(win, h - 2, 0, hints, w, self._cp(C_YELLOW))
        st_attr = self._cp(C_RED if self.last_error else C_GREEN)
        self._safe_addnstr(win, h - 1, 0, f" {self.status}"[:w], w, st_attr)

    def handle_key(self, key: int) -> None:
        if key in (ord("q"), 27):
            self.running = False
            return
        if key == ord("m"):
            self.manual_hold = not self.manual_hold
            self._set_status(f"Manual hold {'enabled' if self.manual_hold else 'disabled'}")
            return
        if key in (curses.KEY_RIGHT, ord("l")):
            self.selected_field = (self.selected_field + 1) % len(self.tune_fields)
            self._set_status(f"Selected {self.tune_fields[self.selected_field].label}")
            return
        if key in (curses.KEY_LEFT, ord("h")):
            self.selected_field = (self.selected_field - 1) % len(self.tune_fields)
            self._set_status(f"Selected {self.tune_fields[self.selected_field].label}")
            return
        if key in (curses.KEY_UP, ord("+"), ord("=")):
            field = self.tune_fields[self.selected_field]
            self._set_wbc_field(field, field.step)
            return
        if key in (curses.KEY_DOWN, ord("-")):
            field = self.tune_fields[self.selected_field]
            self._set_wbc_field(field, -field.step)
            return
        if key == ord("n"):
            self.wbc.set_neutral_pose(None)
            self._set_status("WBC neutral pose recaptured")
            return
        if key == ord("z"):
            self._call("Zero torque", self.robot.zero_torque)
            return
        if key == ord("d"):
            self._call("Damp", self.robot.damp)
            return
        if key == ord("w"):
            self._call("Walk mode", self.robot.walk_mode)
            return
        if key == ord("r"):
            self._call("Run mode", self.robot.run_mode)
            return
        if key == ord("g"):
            self.gait_toggle_state = 1 - self.gait_toggle_state
            self._try_set_gait_type(self.gait_toggle_state)
            return
        if key == ord("?"):
            self.show_help = not self.show_help
            self._set_status(f"Menu {'shown' if self.show_help else 'hidden'}")

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
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(int(1000 / self.send_hz))
        self.refresh_probe(force=True)

        while self.running:
            key = stdscr.getch()
            if key != -1:
                self.handle_key(key)
            self.tick()
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            self.draw(stdscr, h, w)
            try:
                stdscr.refresh()
            except curses.error:
                pass

    def shutdown(self) -> None:
        try:
            self.wbc.set_loco_cmd(0.0, 0.0, 0.0)
            time.sleep(0.2)
        except Exception:
            pass
        try:
            self.wbc.stop()
        except Exception:
            LOG.exception("Failed to stop WBC cleanly")
        try:
            pygame.quit()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="USB gamepad + TUI controller for WBC on G1.")
    p.add_argument("--iface", default="eth0", help="Network interface")
    p.add_argument("--domain-id", type=int, default=0)
    p.add_argument("--joy", type=int, default=0, help="Joystick index")
    p.add_argument("--wait-timeout", type=float, default=5.0)
    p.add_argument("--send-hz", type=float, default=20.0)
    p.add_argument("--probe-interval", type=float, default=0.1)
    p.add_argument("--deadzone", type=float, default=0.10)
    p.add_argument("--max-vx", type=float, default=0.50)
    p.add_argument("--max-vy", type=float, default=0.30)
    p.add_argument("--max-vyaw", type=float, default=0.80)
    p.add_argument("--rate-hz", type=float, default=100.0, help="WBC control rate")
    p.add_argument("--roll-kp", type=float, default=0.55)
    p.add_argument("--roll-kd", type=float, default=0.08)
    p.add_argument("--pitch-kp", type=float, default=0.45)
    p.add_argument("--pitch-kd", type=float, default=0.06)
    p.add_argument("--arm-comp", type=float, default=1.0)
    p.add_argument("--pitch-offset", type=float, default=0.0)
    p.add_argument("--roll-offset", type=float, default=0.0)
    p.add_argument("--load-mass", type=float, default=0.0)
    p.add_argument("--load-arm", type=float, default=0.4)
    p.add_argument("--r1-double-tap", type=float, default=0.35)
    return p.parse_args()


def main() -> None:
    app = WBCUsbControllerApp(parse_args())
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
