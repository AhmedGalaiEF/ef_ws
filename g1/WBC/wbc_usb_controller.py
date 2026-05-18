#!/usr/bin/env python3
"""
USB gamepad + curses TUI controller for Unitree G1.

This follows the stick layout from usb_controller_scheme.txt:
  - Left stick  -> linear + lateral motion (vx, vy)
  - Right stick -> yaw rate (vyaw)

Implemented gamepad combos:
  - L2 + B          -> damp
  - L2 + A          -> FSM 3 if supported
  - L2 + Y          -> zero torque
  - L2 + D-pad Up   -> FSM 4 if supported
  - R1 + Y          -> walk mode
  - R1 + A          -> run mode
  - R1 + B          -> FSM 812 if supported
  - double tap L2   -> toggle gait type 0/1 if supported
  - double tap Y/B/X/A -> shoulder / elbow / wrist / dex3 hand grip
  - L1 + Y/X/A/B    -> waist / knee / ankle / hip-thigh
  - L1 + L3         -> dex3 fingers
  - L1 + R2         -> release / reengage arms
  - R2 + left/right/up -> scope left / right / both
  - Start/Menu      -> toggle help/menu pane

The D-pad mirrors the scheme's "select target / increment / decrement"
behavior across the selected direct low-level body or Dex3 hand target family.
"""
from __future__ import annotations

import argparse
import curses
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WBC_DIR = SCRIPT_DIR if os.path.exists(os.path.join(SCRIPT_DIR, "dds_env.py")) else os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ROOT_DIR = WBC_DIR if os.path.isdir(os.path.join(WBC_DIR, "modules")) else os.path.abspath(os.path.join(WBC_DIR, ".."))
MODULES_DIR = os.path.join(ROOT_DIR, "modules")
for _p in (WBC_DIR, ROOT_DIR, MODULES_DIR):
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

from modules.sdk_client import HAND_JOINT_NAMES, Robot, WAIST_HOLD_KD, WAIST_HOLD_KP
from modules.sdk_hand import clamp_hand_targets, hand_grip_targets


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
LOG = logging.getLogger("wbc_usb_controller")

# Mapping verified from joystick_log_20260507_094223.txt for this controller:
#   A/B/X/Y       -> buttons 0/1/2/3
#   L1/R1         -> buttons 4/5
#   Select/Start  -> buttons 6/7
#   L3/R3         -> buttons 9/10
#   Left stick    -> axes 0/1
#   Right stick   -> axes 3/4
#   L2/R2         -> analog axes 2/5
#   D-pad         -> hat 0
BTN_A = 0
BTN_B = 1
BTN_X = 2
BTN_Y = 3
BTN_L1 = 4
BTN_R1 = 5
BTN_SELECT = 6
BTN_START = 7
BTN_L3 = 9
BTN_R3 = 10

AXIS_LX = 0
AXIS_LY = 1
AXIS_L2 = 2
AXIS_RX = 3
AXIS_RY = 4
AXIS_R2 = 5

HAT_CENTER = (0, 0)
TRIGGER_PRESSED_THRESHOLD = 0.5

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
class MenuItem:
    label: str
    action: str


@dataclass(frozen=True)
class BodySelection:
    label: str
    left_index: int
    right_index: int | None
    right_sign: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class BodyFamily:
    key: str
    label: str
    selections: tuple[BodySelection, ...]


BODY_FAMILIES: tuple[BodyFamily, ...] = (
    BodyFamily(
        "waist",
        "Waist",
        (
            BodySelection("waist yaw", 12, None, 1.0, -2.5, 2.5),
            BodySelection("waist roll", 13, None, 1.0, -2.5, 2.5),
            BodySelection("waist pitch", 14, None, 1.0, -2.5, 2.5),
        ),
    ),
    BodyFamily(
        "shoulder",
        "Shoulder",
        (
            BodySelection("shoulder pitch", 15, 22, 1.0, -3.0892, 2.6704),
            BodySelection("shoulder roll", 16, 23, -1.0, -1.5882, 2.2515),
            BodySelection("shoulder yaw", 17, 24, -1.0, -2.6180, 2.6180),
        ),
    ),
    BodyFamily(
        "elbow",
        "Elbow",
        (
            BodySelection("elbow", 18, 25, 1.0, -1.0472, 2.0944),
        ),
    ),
    BodyFamily(
        "wrist",
        "Wrist",
        (
            BodySelection("wrist pitch", 20, 27, 1.0, -1.6144, 1.6144),
            BodySelection("wrist roll", 19, 26, -1.0, -1.9722, 1.9722),
            BodySelection("wrist yaw", 21, 28, -1.0, -1.6144, 1.6144),
        ),
    ),
    BodyFamily(
        "hip",
        "Hip / Thigh",
        (
            BodySelection("hip pitch", 0, 6, 1.0, -2.2, 2.2),
            BodySelection("hip roll", 1, 7, 1.0, -1.2, 1.2),
            BodySelection("hip yaw", 2, 8, 1.0, -1.4, 1.4),
        ),
    ),
    BodyFamily(
        "knee",
        "Knee",
        (
            BodySelection("knee", 3, 9, 1.0, -0.2, 2.6),
        ),
    ),
    BodyFamily(
        "ankle",
        "Ankle",
        (
            BodySelection("ankle pitch", 4, 10, 1.0, -1.2, 1.2),
            BodySelection("ankle roll", 5, 11, -1.0, -0.8, 0.8),
        ),
    ),
)

BODY_FAMILY_BY_KEY = {family.key: family for family in BODY_FAMILIES}
CONTROLLED_BODY_JOINTS = sorted(
    {
        idx
        for family in BODY_FAMILIES
        for sel in family.selections
        for idx in (sel.left_index, sel.right_index)
        if idx is not None
    }
)

HAND_FINGER_LIMITS = {
    "thumb_0": (-1.4, 1.4),
    "thumb_1": (-1.4, 1.4),
    "thumb_2": (-1.4, 1.4),
    "middle_0": (-1.4, 1.4),
    "middle_1": (-1.4, 1.4),
    "index_0": (-1.4, 1.4),
    "index_1": (-1.4, 1.4),
}

BODY_ADJUST_STEP = 0.05
BODY_MAX_SPEED = 0.35
BODY_KP = 30.0
BODY_KD = 1.5
HAND_MAX_SPEED = 3.0
HAND_HOLD_S = 0.25
HAND_RATE_HZ = 50.0
HAND_KP = 1.2
HAND_KD = 0.05
HAND_TAU = 0.05
GRIP_ADJUST_STEP = 5.0
HELD_ADJUST_INITIAL_DELAY_S = 0.30
HELD_ADJUST_REPEAT_S = 0.08


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
        self.double_tap_s = max(0.1, float(args.double_tap_window))

        self.robot = Robot(
            iface=self.iface,
            domain_id=self.domain_id,
            auto_start_sensors=True,
        )
        self.robot.wait_for_sport_state(timeout=float(args.wait_timeout))
        self.robot.wait_for_low_state(timeout=float(args.wait_timeout))

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
        self.low_level_enabled = False
        self.show_help = True
        self.menu_open = False
        self.running = True
        self.status = f"Using {self.joy.get_name()}"
        self.last_error = False
        self.last_probe_s = 0.0
        self.last_l2_press_s = 0.0
        self.prev_l2_pressed = False
        self.prev_r2_pressed = False
        self.gait_toggle_state = 0
        self.prev_buttons: dict[int, bool] = {}
        self.button_edges: dict[int, bool] = {}
        self.prev_hat = HAT_CENTER
        self.prev_scope_hat = HAT_CENTER
        self.prev_target_hat = HAT_CENTER
        self.last_face_press_s = {BTN_A: 0.0, BTN_B: 0.0, BTN_X: 0.0, BTN_Y: 0.0}
        self.held_adjust_dir = 0
        self.held_adjust_started_s = 0.0
        self.held_adjust_last_step_s = 0.0
        self.last_snapshot: dict[str, Any] = {}
        self.last_inputs: dict[str, Any] = {
            "axes": {},
            "buttons": {},
            "hat": HAT_CENTER,
        }

        self.arms_released = True
        self.target_scope = "both"
        self.active_family_key = "shoulder"
        self.active_selection_index = 0
        self.ll_current_targets: dict[int, float] = {j: 0.0 for j in CONTROLLED_BODY_JOINTS}
        self.ll_desired_targets: dict[int, float] = {j: 0.0 for j in CONTROLLED_BODY_JOINTS}
        self.hand_current_targets: dict[str, list[float]] = {
            "left": hand_grip_targets("left", 0.0),
            "right": hand_grip_targets("right", 0.0),
        }
        self.hand_desired_targets: dict[str, list[float]] = {
            "left": list(self.hand_current_targets["left"]),
            "right": list(self.hand_current_targets["right"]),
        }
        self.hand_grip_percent = {"left": 0.0, "right": 0.0}
        self.last_hand_publish_s = 0.0
        self.saved_pose_path = os.path.join(SCRIPT_DIR, "saved_usb_joint_poses.json")
        self.menu_items = [
            MenuItem("Save current joint pose", "save_pose"),
            MenuItem("Sync live body and hand pose", "recapture_neutral"),
            MenuItem("Release / reengage arms", "toggle_arms"),
            MenuItem("Toggle manual hold", "toggle_hold"),
            MenuItem("Toggle low-level control", "toggle_low_level"),
            MenuItem("Exit controller", "exit"),
        ]
        self.menu_index = 0

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
        return bool(self.button_edges.get(int(button), False))

    def _button_down(self, button: int) -> bool:
        return button < self.joy.get_numbuttons() and bool(self.joy.get_button(button))

    def _axis_value(self, axis: int) -> float:
        return self.joy.get_axis(axis) if axis < self.joy.get_numaxes() else 0.0

    def _trigger_value(self, axis: int) -> float:
        raw = self._axis_value(axis)
        return 0.5 * (float(raw) + 1.0)

    def _trigger_down(self, axis: int) -> bool:
        return self._trigger_value(axis) >= TRIGGER_PRESSED_THRESHOLD

    def _trigger_edge_pressed(self, axis: int, previous: bool) -> tuple[bool, bool]:
        pressed = self._trigger_down(axis)
        return pressed and not previous, pressed

    def _hat(self) -> tuple[int, int]:
        return self.joy.get_hat(0) if self.joy.get_numhats() > 0 else HAT_CENTER

    def _capture_input_snapshot(self) -> None:
        self.button_edges = {}
        live_buttons: dict[int, bool] = {}
        for button in (
            BTN_A,
            BTN_B,
            BTN_X,
            BTN_Y,
            BTN_L1,
            BTN_R1,
            BTN_SELECT,
            BTN_START,
            BTN_L3,
            BTN_R3,
        ):
            pressed = self._button_down(button)
            live_buttons[int(button)] = pressed
            self.button_edges[int(button)] = pressed and not self.prev_buttons.get(int(button), False)
        self.prev_buttons = live_buttons
        axes = {
            "LX": self._axis_value(AXIS_LX),
            "LY": self._axis_value(AXIS_LY),
            "RX": self._axis_value(AXIS_RX),
            "RY": self._axis_value(AXIS_RY),
            "L2": self._trigger_value(AXIS_L2),
            "R2": self._trigger_value(AXIS_R2),
        }
        buttons = {
            "A": self._button_down(BTN_A),
            "B": self._button_down(BTN_B),
            "X": self._button_down(BTN_X),
            "Y": self._button_down(BTN_Y),
            "L1": self._button_down(BTN_L1),
            "R1": self._button_down(BTN_R1),
            "Select": self._button_down(BTN_SELECT),
            "Start": self._button_down(BTN_START),
            "L3": self._button_down(BTN_L3),
            "R3": self._button_down(BTN_R3),
        }
        self.last_inputs = {
            "axes": axes,
            "buttons": buttons,
            "hat": self._hat(),
        }

    def _call(self, label: str, func, *args) -> None:
        try:
            result = func(*args)
            suffix = "" if result is None else f" -> {result}"
            self._set_status(f"{label}{suffix}")
        except Exception as exc:
            self._set_status(f"{label} failed: {exc}", error=True)

    @property
    def _family(self) -> BodyFamily | None:
        return BODY_FAMILY_BY_KEY.get(self.active_family_key)

    @property
    def _selection(self) -> BodySelection | None:
        family = self._family
        if family is None or not family.selections:
            return None
        return family.selections[self.active_selection_index % len(family.selections)]

    def _sync_targets_to_live_pose(self) -> None:
        positions = self.robot._read_joint_positions_or_raise(CONTROLLED_BODY_JOINTS, timeout=3.0)
        self.ll_current_targets = {joint: float(positions[joint]) for joint in CONTROLLED_BODY_JOINTS}
        self.ll_desired_targets = dict(self.ll_current_targets)
        for side in ("left", "right"):
            snapshot = self.robot.get_hand_state_snapshot(side)
            if snapshot is not None:
                live_targets = [
                    float(snapshot["positions"].get(name, self.hand_desired_targets[side][idx]))
                    for idx, name in enumerate(HAND_JOINT_NAMES)
                ]
                self.hand_current_targets[side] = clamp_hand_targets(side, live_targets)
                self.hand_desired_targets[side] = list(self.hand_current_targets[side])
        self._set_status("Synced low-level targets to live body and hand state")

    def _toggle_arm_release(self) -> None:
        if self.arms_released:
            self.robot.unrelease_arms()
            self.arms_released = False
            self._sync_targets_to_live_pose()
            self._set_status("Arms reengaged and synced to live pose")
            return
        self.robot.release_arms()
        self.arms_released = True
        self._set_status("Arms released")

    def _set_active_family(self, family_key: str, *, source: str) -> None:
        family = BODY_FAMILY_BY_KEY.get(family_key)
        if family is None:
            return
        self.active_family_key = family_key
        self.active_selection_index = 0
        self._set_status(f"{source}: target {family.label.lower()}")

    def _cycle_active_selection(self, delta: int) -> None:
        family = self._family
        if family is None or not family.selections:
            return
        self.active_selection_index = (self.active_selection_index + delta) % len(family.selections)
        sel = self._selection
        if sel is not None:
            self._set_status(f"Target {sel.label}")

    def _set_scope(self, scope: str) -> None:
        if scope not in ("left", "right", "both"):
            return
        self.target_scope = scope
        target_group = "body or hand target"
        if self.active_family_key in ("hip", "knee", "ankle"):
            target_group = "leg target"
        elif self.active_family_key in ("shoulder", "elbow", "wrist", "dex3_fingers", "dex3_hand"):
            target_group = "arm or hand target"
        self._set_status(f"Control scope -> {scope} ({target_group})")

    def _set_body_target_value(self, selection: BodySelection, value: float) -> None:
        clamped = max(selection.minimum, min(selection.maximum, float(value)))
        if selection.right_index is None:
            self.ll_desired_targets[selection.left_index] = clamped
            return
        if self.target_scope in ("left", "both"):
            self.ll_desired_targets[selection.left_index] = clamped
        if self.target_scope in ("right", "both"):
            self.ll_desired_targets[selection.right_index] = clamped * selection.right_sign

    def _get_body_target_value(self, selection: BodySelection) -> float:
        if selection.right_index is None or self.target_scope == "left":
            return float(self.ll_desired_targets[selection.left_index])
        if self.target_scope == "right":
            return float(self.ll_desired_targets[selection.right_index]) / selection.right_sign
        left = float(self.ll_desired_targets[selection.left_index])
        right = float(self.ll_desired_targets[selection.right_index]) / selection.right_sign
        return 0.5 * (left + right)

    def _adjust_active_body_target(self, delta: float) -> None:
        selection = self._selection
        if selection is None:
            return
        current = self._get_body_target_value(selection)
        self._set_body_target_value(selection, current + delta)
        self._set_status(f"{selection.label}={self._get_body_target_value(selection):+.3f}")

    def _set_grip_percent(self, percent: float) -> None:
        clamped = max(0.0, min(100.0, float(percent)))
        sides = ("left", "right") if self.target_scope == "both" else (self.target_scope,)
        for side in sides:
            self.hand_grip_percent[side] = clamped
            self.hand_desired_targets[side] = clamp_hand_targets(side, hand_grip_targets(side, clamped))
        self._set_status(f"Dex3 grip={clamped:.0f}% ({self.target_scope})")

    def _adjust_active_finger(self, delta: float) -> None:
        finger_idx = self.active_selection_index % len(HAND_JOINT_NAMES)
        finger_name = HAND_JOINT_NAMES[finger_idx]
        lo, hi = HAND_FINGER_LIMITS[finger_name]
        sides = ("left", "right") if self.target_scope == "both" else (self.target_scope,)
        latest = None
        for side in sides:
            current = float(self.hand_desired_targets[side][finger_idx])
            updated = max(lo, min(hi, current + delta))
            self.hand_desired_targets[side][finger_idx] = updated
            self.hand_desired_targets[side] = clamp_hand_targets(side, self.hand_desired_targets[side])
            latest = updated
        if latest is not None:
            self._set_status(f"{finger_name}={latest:+.3f} ({self.target_scope})")

    def _apply_adjust_step(self, direction: int) -> None:
        if direction == 0 or not self.low_level_enabled:
            return
        if self.active_family_key == "dex3_fingers":
            self._adjust_active_finger(float(direction) * BODY_ADJUST_STEP)
            return
        if self.active_family_key == "dex3_hand":
            sides = ("left", "right") if self.target_scope == "both" else (self.target_scope,)
            current = sum(self.hand_grip_percent[side] for side in sides) / float(len(sides))
            self._set_grip_percent(current + float(direction) * GRIP_ADJUST_STEP)
            return
        self._adjust_active_body_target(float(direction) * BODY_ADJUST_STEP)

    def _update_held_adjust(self, direction: int, now: float) -> None:
        if direction == 0:
            self.held_adjust_dir = 0
            self.held_adjust_started_s = 0.0
            self.held_adjust_last_step_s = 0.0
            return
        if direction != self.held_adjust_dir:
            self.held_adjust_dir = direction
            self.held_adjust_started_s = now
            self.held_adjust_last_step_s = now
            self._apply_adjust_step(direction)
            return
        if now - self.held_adjust_started_s < HELD_ADJUST_INITIAL_DELAY_S:
            return
        if now - self.held_adjust_last_step_s < HELD_ADJUST_REPEAT_S:
            return
        self.held_adjust_last_step_s = now
        self._apply_adjust_step(direction)

    def _step_low_level_targets(self, dt: float) -> None:
        step = max(1e-6, BODY_MAX_SPEED * dt)
        for j in CONTROLLED_BODY_JOINTS:
            cur = float(self.ll_current_targets[j])
            des = float(self.ll_desired_targets[j])
            delta = des - cur
            if abs(delta) <= step:
                self.ll_current_targets[j] = des
            else:
                self.ll_current_targets[j] = cur + step * (1.0 if delta > 0.0 else -1.0)

    def _step_hand_targets(self, dt: float) -> None:
        step = max(1e-6, HAND_MAX_SPEED * dt)
        for side in ("left", "right"):
            current = self.hand_current_targets[side]
            desired = self.hand_desired_targets[side]
            for idx in range(len(current)):
                delta = float(desired[idx]) - float(current[idx])
                if abs(delta) <= step:
                    current[idx] = float(desired[idx])
                else:
                    current[idx] = float(current[idx]) + step * (1.0 if delta > 0.0 else -1.0)
            self.hand_current_targets[side] = clamp_hand_targets(side, current)

    def _publish_low_level_targets(self) -> None:
        waist_gains = {12: float(WAIST_HOLD_KP), 13: float(WAIST_HOLD_KP), 14: float(WAIST_HOLD_KP)}
        waist_damping = {12: float(WAIST_HOLD_KD), 13: float(WAIST_HOLD_KD), 14: float(WAIST_HOLD_KD)}
        self.robot._get_arm_sdk().publish_targets(
            self.ll_current_targets,
            kp=BODY_KP,
            kd=BODY_KD,
            kp_by_joint=waist_gains,
            kd_by_joint=waist_damping,
        )
        now = time.monotonic()
        if now - self.last_hand_publish_s >= 0.05:
            for side in ("left", "right"):
                self.robot.hand_pose(
                    list(self.hand_current_targets[side]),
                    hand=side,
                    hold_s=HAND_HOLD_S,
                    rate_hz=HAND_RATE_HZ,
                    kp=HAND_KP,
                    kd=HAND_KD,
                    tau=HAND_TAU,
                    ramp_s=0.05,
                )
            self.last_hand_publish_s = now

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

    def _enter_dev_mode(self) -> None:
        client = getattr(self.robot, "_client", None)
        if client is None:
            self._set_status("Dev mode unsupported by current locomotion client", error=True)
            return
        try:
            if hasattr(client, "Start"):
                client.Start()
                self._set_status("Dev mode entered via client.Start()")
                return
            if hasattr(client, "SetFsmId"):
                client.SetFsmId(500)
                self._set_status("Dev mode -> FSM 500")
                return
            self._set_status("Dev mode unsupported by current locomotion client", error=True)
        except Exception as exc:
            self._set_status(f"Dev mode failed: {exc}", error=True)

    def _try_set_gait_type(self, gait_type: int) -> None:
        client = getattr(self.robot, "_client", None)
        if client is None or not hasattr(client, "SetGaitType"):
            self._set_status("Gait type toggle unsupported", error=True)
            return
        try:
            client.SetGaitType(int(gait_type))
            self.gait_toggle_state = int(gait_type)
            self._set_status(f"Gait type -> {gait_type}")
        except Exception as exc:
            self._set_status(f"SetGaitType failed: {exc}", error=True)

    def _sync_gait_toggle_state(self) -> None:
        try:
            gait = self.robot.get_gait()
        except Exception:
            return
        if gait is None:
            return
        try:
            self.gait_toggle_state = int(gait)
        except Exception:
            pass

    def _set_walk_mode(self) -> None:
        self._call("Walk mode", self.robot.walk_mode)
        self._try_set_gait_type(0)

    def _menu_move(self, delta: int) -> None:
        if not self.menu_items:
            return
        self.menu_index = (self.menu_index + delta) % len(self.menu_items)
        self._set_status(f"Menu: {self.menu_items[self.menu_index].label}")

    def _set_low_level_enabled(self, enabled: bool) -> None:
        self.low_level_enabled = bool(enabled)
        if self.low_level_enabled:
            self.robot.unrelease_arms()
            self.arms_released = False
            self._sync_targets_to_live_pose()
            return
        self.robot.release_arms()
        self.arms_released = True
        self._set_status("Low-level control disabled")

    def _save_current_pose(self) -> None:
        snapshot = self.robot.get_low_state_snapshot()
        if snapshot is None or getattr(snapshot, "joint_positions", None) is None:
            self._set_status("Save pose failed: no low-state snapshot", error=True)
            return

        entry = {
            "name": f"pose_{time.strftime('%Y%m%d_%H%M%S')}",
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "joint_positions": [float(v) for v in snapshot.joint_positions],
        }
        payload = {"poses": []}
        try:
            if os.path.exists(self.saved_pose_path):
                with open(self.saved_pose_path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            poses = payload.get("poses")
            if not isinstance(poses, list):
                poses = []
            poses.append(entry)
            payload["poses"] = poses
            with open(self.saved_pose_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.write("\n")
            self._set_status(f"Saved pose to {os.path.basename(self.saved_pose_path)}")
        except Exception as exc:
            self._set_status(f"Save pose failed: {exc}", error=True)

    def _activate_menu_item(self) -> None:
        if not (0 <= self.menu_index < len(self.menu_items)):
            return
        action = self.menu_items[self.menu_index].action
        if action == "save_pose":
            self._save_current_pose()
        elif action == "recapture_neutral":
            self._sync_targets_to_live_pose()
        elif action == "toggle_arms":
            self._toggle_arm_release()
        elif action == "toggle_hold":
            self.manual_hold = not self.manual_hold
            self._set_status(f"Manual hold {'enabled' if self.manual_hold else 'disabled'}")
        elif action == "toggle_low_level":
            self._set_low_level_enabled(not self.low_level_enabled)
        elif action == "exit":
            self.running = False
            self._set_status("Exit requested from menu")

    def _handle_double_tap_face(self, button: int) -> bool:
        if not self._edge_pressed(button):
            return False
        now = time.monotonic()
        previous = self.last_face_press_s.get(button, 0.0)
        self.last_face_press_s[button] = now
        if now - previous > self.double_tap_s:
            return False
        if button == BTN_Y:
            self._set_active_family("shoulder", source="Double tap Y")
            return True
        if button == BTN_B:
            self._set_active_family("elbow", source="Double tap B")
            return True
        if button == BTN_X:
            self._set_active_family("wrist", source="Double tap X")
            return True
        if button == BTN_A:
            self.active_family_key = "dex3_hand"
            self.active_selection_index = 0
            self._set_status("Double tap A: target dex3 hand grip")
            return True
        return False

    def _poll_gamepad(self) -> None:
        pygame.event.pump()
        self._capture_input_snapshot()
        now = time.monotonic()

        l1 = self._button_down(BTN_L1)
        l2 = self._trigger_down(AXIS_L2)
        r1 = self._button_down(BTN_R1)
        r2 = self._trigger_down(AXIS_R2)
        hat = self.last_inputs["hat"]

        if self._edge_pressed(BTN_START):
            self.menu_open = not self.menu_open
            self._set_status(f"Menu {'opened' if self.menu_open else 'closed'}")

        if self._edge_pressed(BTN_SELECT):
            self._set_low_level_enabled(not self.low_level_enabled)

        if self.menu_open:
            if self._edge_pressed(BTN_B):
                self.menu_open = False
                self._set_status("Menu closed")
                self.prev_hat = hat
                return
            if self._edge_pressed(BTN_A):
                self._activate_menu_item()
                self.prev_hat = hat
                return
            if hat != self.prev_hat:
                if hat[1] > 0 and self.prev_hat[1] <= 0:
                    self._menu_move(-1)
                elif hat[1] < 0 and self.prev_hat[1] >= 0:
                    self._menu_move(1)
            self.prev_hat = hat
            return

        l2_edge, self.prev_l2_pressed = self._trigger_edge_pressed(AXIS_L2, self.prev_l2_pressed)
        if l2_edge:
            if now - self.last_l2_press_s <= self.double_tap_s:
                self._toggle_gait()
            self.last_l2_press_s = now

        r2_edge, self.prev_r2_pressed = self._trigger_edge_pressed(AXIS_R2, self.prev_r2_pressed)
        if l1 and r2_edge:
            self._toggle_arm_release()
            self.prev_hat = hat
            return

        if l2 and self._edge_pressed(BTN_A):
            self._try_set_fsm(3, "Sit mode")
            self.prev_hat = hat
            return
        if l2 and self._edge_pressed(BTN_B):
            self._call("Damp", self.robot.damp)
            self.prev_hat = hat
            return
        if l2 and self._edge_pressed(BTN_X):
            self._enter_dev_mode()
            self.prev_hat = hat
            return
        if l2 and self._edge_pressed(BTN_Y):
            self._call("Zero torque", self.robot.zero_torque)
            self.prev_hat = hat
            return

        if r1 and self._edge_pressed(BTN_Y):
            self._set_walk_mode()
            self.prev_hat = hat
            return
        if r1 and self._edge_pressed(BTN_A):
            self._call("Run mode", self.robot.run_mode)
            self.prev_hat = hat
            return
        if r1 and self._edge_pressed(BTN_B):
            self._try_set_fsm(812, "Climb mode")
            self.prev_hat = hat
            return

        if l2 and hat[1] > 0 and self.prev_hat[1] <= 0:
            self._try_set_fsm(4, "Preparation mode")
            self.prev_hat = hat
            return

        if l1 and self._edge_pressed(BTN_Y):
            self._set_active_family("waist", source="L1+Y")
            self.prev_hat = hat
            return
        if l1 and self._edge_pressed(BTN_X):
            self._set_active_family("knee", source="L1+X")
            self.prev_hat = hat
            return
        if l1 and self._edge_pressed(BTN_A):
            self._set_active_family("ankle", source="L1+A")
            self.prev_hat = hat
            return
        if l1 and self._edge_pressed(BTN_B):
            self._set_active_family("hip", source="L1+B")
            self.prev_hat = hat
            return
        if l1 and self._edge_pressed(BTN_L3):
            self.active_family_key = "dex3_fingers"
            self.active_selection_index = 0
            self._set_status("L1+L3: target dex3 fingers")
            self.prev_hat = hat
            return

        if not (l1 or l2 or r1):
            for button in (BTN_Y, BTN_B, BTN_X, BTN_A):
                if self._handle_double_tap_face(button):
                    self.prev_hat = hat
                    return

        if r2 and hat != self.prev_scope_hat:
            if hat[0] > 0 and self.prev_scope_hat[0] <= 0:
                self._set_scope("right")
                self.prev_scope_hat = hat
                return
            if hat[0] < 0 and self.prev_scope_hat[0] >= 0:
                self._set_scope("left")
                self.prev_scope_hat = hat
                return
            if hat[1] > 0 and self.prev_scope_hat[1] <= 0:
                self._set_scope("both")
                self.prev_scope_hat = hat
                return
        self.prev_scope_hat = hat if r2 else HAT_CENTER

        if self.low_level_enabled and hat != self.prev_target_hat and not r2:
            if hat[0] > 0 and self.prev_target_hat[0] <= 0:
                self._cycle_active_selection(+1)
            elif hat[0] < 0 and self.prev_target_hat[0] >= 0:
                self._cycle_active_selection(-1)
            elif hat[1] > 0 and self.prev_target_hat[1] <= 0:
                self._update_held_adjust(+1, now)
            elif hat[1] < 0 and self.prev_target_hat[1] >= 0:
                self._update_held_adjust(-1, now)
        if self.low_level_enabled and not r2 and hat[1] != 0:
            self._update_held_adjust(1 if hat[1] > 0 else -1, now)
        else:
            self._update_held_adjust(0, now)
        self.prev_target_hat = hat if (self.low_level_enabled and not r2) else HAT_CENTER
        self.prev_hat = hat

        lx = apply_deadzone(self._axis_value(AXIS_LX), self.deadzone)
        ly = apply_deadzone(self._axis_value(AXIS_LY), self.deadzone)
        rx = apply_deadzone(self._axis_value(AXIS_RX), self.deadzone)

        self.vyaw = -rx * self.max_vyaw
        self.vx = -ly * self.max_vx
        self.vy = -lx * self.max_vy

        if self.manual_hold:
            self.vx = self.vy = self.vyaw = 0.0

        self.robot.loco_move(self.vx, self.vy, self.vyaw)

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
            return
        self._sync_gait_toggle_state()

    def tick(self) -> None:
        self._poll_gamepad()
        if self.low_level_enabled:
            self._step_low_level_targets(1.0 / self.send_hz)
            self._step_hand_targets(1.0 / self.send_hz)
            self._publish_low_level_targets()
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

    def _draw_motion_panel(self, win, top: int, left: int, width: int, height: int) -> None:
        hold_attr = self._cp(C_RED if self.manual_hold else C_GREEN) | curses.A_BOLD
        low_level_attr = self._cp(C_GREEN if self.low_level_enabled else C_YELLOW) | curses.A_BOLD
        self._safe_addnstr(win, top, left, " Motion from gamepad sticks ", width, self._cp(C_CYAN) | curses.A_BOLD)
        rows = [
            (f" hold={('ON' if self.manual_hold else 'OFF')}", hold_attr),
            (f" low-level={('ON' if self.low_level_enabled else 'OFF')}", low_level_attr),
            (f" arms released={('YES' if self.arms_released else 'NO')}", self._cp(C_YELLOW if self.arms_released else C_GREEN)),
            (f" vx   {self.vx:+.3f} / {self.max_vx:.2f} m/s", 0),
            (f" vy   {self.vy:+.3f} / {self.max_vy:.2f} m/s", 0),
            (f" vyaw {self.vyaw:+.3f} / {self.max_vyaw:.2f} rad/s", 0),
            (f" imu rpy={fmt(self.last_snapshot.get('imu_rpy'))}", 0),
            (f" pos={fmt(self.last_snapshot.get('position'))}", 0),
            (f" vel={fmt(self.last_snapshot.get('velocity'))}", 0),
        ]
        for idx, (text, attr) in enumerate(rows[: max(0, height - 1)]):
            self._safe_addnstr(win, top + 1 + idx, left, text, width, attr)

    def _draw_input_panel(self, win, top: int, left: int, width: int, height: int) -> None:
        self._safe_addnstr(win, top, left, " Controller input ", width, self._cp(C_CYAN) | curses.A_BOLD)
        axes = self.last_inputs.get("axes", {})
        buttons = self.last_inputs.get("buttons", {})
        hat = self.last_inputs.get("hat", HAT_CENTER)

        rows = [
            (
                f" axes LX={float(axes.get('LX', 0.0)):+.3f}"
                f" LY={float(axes.get('LY', 0.0)):+.3f}"
            ),
            (
                f" axes RX={float(axes.get('RX', 0.0)):+.3f}"
                f" RY={float(axes.get('RY', 0.0)):+.3f}"
            ),
            (
                f" trig L2={float(axes.get('L2', 0.0)):.2f}"
                f" R2={float(axes.get('R2', 0.0)):.2f}"
            ),
            f" hat  x={hat[0]:+d} y={hat[1]:+d}",
            (
                " btn  "
                + " ".join(
                    f"{name}={'1' if state else '0'}"
                    for name, state in (
                        ("A", buttons.get("A", False)),
                        ("B", buttons.get("B", False)),
                        ("X", buttons.get("X", False)),
                        ("Y", buttons.get("Y", False)),
                    )
                )
            ),
            (
                " btn  "
                + " ".join(
                    f"{name}={'1' if state else '0'}"
                    for name, state in (
                        ("L1", buttons.get("L1", False)),
                        ("R1", buttons.get("R1", False)),
                        ("L3", buttons.get("L3", False)),
                        ("R3", buttons.get("R3", False)),
                    )
                )
            ),
            (
                " btn  "
                + " ".join(
                    f"{name}={'1' if state else '0'}"
                    for name, state in (
                        ("Select", buttons.get("Select", False)),
                        ("Start", buttons.get("Start", False)),
                    )
                )
            ),
        ]
        for idx, row in enumerate(rows[: max(0, height - 1)]):
            self._safe_addnstr(win, top + 1 + idx, left, row, width)

    def _draw_tuning_panel(self, win, top: int, left: int, width: int, height: int) -> None:
        title = " Low-level target via D-pad " if self.low_level_enabled else " Low-level target (locked) "
        self._safe_addnstr(win, top, left, title, width, self._cp(C_CYAN) | curses.A_BOLD)
        if not self.low_level_enabled:
            rows = [
                "Select enables low-level control",
                "Then D-pad selects and adjusts target",
            ]
            for idx, row in enumerate(rows[: max(0, height - 1)]):
                self._safe_addnstr(win, top + 1 + idx, left, row, width, self._cp(C_YELLOW))
            return
        rows: list[str] = [
            f"scope={self.target_scope}",
        ]
        if self.active_family_key == "dex3_fingers":
            finger_idx = self.active_selection_index % len(HAND_JOINT_NAMES)
            finger_name = HAND_JOINT_NAMES[finger_idx]
            if self.target_scope == "both":
                value = 0.5 * (
                    float(self.hand_desired_targets['left'][finger_idx])
                    + float(self.hand_desired_targets['right'][finger_idx])
                )
            else:
                value = float(self.hand_desired_targets[self.target_scope][finger_idx])
            rows.extend(
                [
                    "family=dex3 fingers",
                    f"target={finger_name}",
                    f"value={value:+.3f}",
                    f"step={BODY_ADJUST_STEP:.3f}",
                ]
            )
        elif self.active_family_key == "dex3_hand":
            if self.target_scope == "both":
                value = 0.5 * (self.hand_grip_percent["left"] + self.hand_grip_percent["right"])
            else:
                value = self.hand_grip_percent[self.target_scope]
            rows.extend(
                [
                    "family=dex3 hand grip",
                    "target=grip slider",
                    f"value={value:.1f}%",
                    f"step={GRIP_ADJUST_STEP:.1f}%",
                ]
            )
        else:
            selection = self._selection
            selection_value = "-" if selection is None else f"{self._get_body_target_value(selection):+.3f}"
            rows.append(f"family={(self._family.label if self._family else '-')}")
            rows.append(f"target={(selection.label if selection else '-')}")
            rows.append(f"value={selection_value}")
            rows.append(f"step={BODY_ADJUST_STEP:.3f}")
        for idx, row in enumerate(rows[: max(0, height - 1)]):
            self._safe_addnstr(win, top + 1 + idx, left, row, width)

    def _draw_fsm_panel(self, win, top: int, left: int, width: int, height: int) -> None:
        self._safe_addnstr(win, top, left, " FSM shortcuts ", width, self._cp(C_CYAN) | curses.A_BOLD)
        rows = [
            "L2+Y -> zero torque",
            "L2+Dpad Up -> FSM 4",
            "R1+Y -> walk",
            "R1+A -> run",
            "R1+B -> FSM 812",
            "L2 dbl tap -> gait 0/1",
            "L1+R2 -> release/reengage arms",
            "R2+left/right/up -> left/right/both limb scope",
        ]
        for idx, row in enumerate(rows[: max(0, height - 1)]):
            self._safe_addnstr(win, top + 1 + idx, left, row, width)

    def _draw_help_panel(self, win, top: int, left: int, width: int, height: int) -> None:
        title = " Controller shortcuts "
        self._safe_addnstr(win, top, left, title, width, self._cp(C_CYAN) | curses.A_BOLD)
        rows = [
            "Left stick = vx/vy, Right stick = yaw",
            "L2+A FSM 3, L2+B damp, L2+Y zero torque",
            "dbl Y/B/X/A = shoulder/elbow/wrist/grip",
            "L1+Y/X/A/B = waist/knee/ankle/hip-thigh",
            "L1+L3 = dex3 fingers",
            "Select toggles low-level control",
            "R2+left/right/up = left/right/both scope",
            "Scope applies to hips/knees/ankles too",
            "When enabled: D-pad selects/adjusts target",
            "USB controller is sole LL joint publisher",
            "Start opens menu, A confirms, B closes",
            "q / Esc quits and stops locomotion",
        ]
        for idx, row in enumerate(rows[: max(0, height - 1)]):
            self._safe_addnstr(win, top + 1 + idx, left, row, width)

    def _draw_menu_panel(self, win, top: int, left: int, width: int, height: int) -> None:
        title = " Start menu " if self.menu_open else " Start menu (press Start) "
        self._safe_addnstr(win, top, left, title, width, self._cp(C_CYAN) | curses.A_BOLD)
        rows = max(0, height - 1)
        for idx in range(min(rows, len(self.menu_items))):
            item = self.menu_items[idx]
            selected = self.menu_open and idx == self.menu_index
            prefix = ">" if selected else " "
            suffix = " <" if selected else ""
            attr = self._cp(C_SEL) | curses.A_BOLD if selected else 0
            self._safe_addnstr(win, top + 1 + idx, left, f"{prefix} {item.label}{suffix}", width, attr)

    def draw(self, win, h: int, w: int) -> None:
        if h < 14 or w < 60:
            self._safe_addstr(win, 0, 0, f"Terminal too small ({w}x{h}). Need at least 60x14.")
            return
        self._draw_header(win, w)
        split = w // 2
        for y in range(5, h - 2):
            self._safe_addstr(win, y, split, "|", self._cp(C_CYAN))
        left_w = max(1, split - 1)
        right_w = max(1, w - split - 1)
        top_left = 6
        left_remaining = max(0, h - top_left - 2)
        left_sections = [
            (self._draw_motion_panel, 10),
            (self._draw_input_panel, 8),
            (self._draw_tuning_panel, 8),
        ]
        for idx, (draw_fn, preferred) in enumerate(left_sections):
            if left_remaining <= 0:
                break
            later = len(left_sections) - idx - 1
            reserve = later if left_remaining > later else 0
            section_height = min(preferred, max(1, left_remaining - reserve))
            draw_fn(win, top_left, 0, left_w, section_height)
            top_left += section_height
            left_remaining = max(0, h - top_left - 2)
        top_right = 6
        right_remaining = max(0, h - top_right - 2)
        right_sections = [
            (self._draw_fsm_panel, 8),
            (self._draw_help_panel, 11),
            (self._draw_menu_panel, 6),
        ]
        for idx, (draw_fn, preferred) in enumerate(right_sections):
            if right_remaining <= 0:
                break
            later = len(right_sections) - idx - 1
            reserve = later if right_remaining > later else 0
            section_height = min(preferred, max(1, right_remaining - reserve))
            draw_fn(win, top_right, split + 1, right_w, section_height)
            top_right += section_height
            right_remaining = max(0, h - top_right - 2)
        hints = "q/Esc quit | Start menu | left pane=status | right pane=shortcuts"
        self._safe_addnstr(win, h - 2, 0, hints, w, self._cp(C_YELLOW))
        st_attr = self._cp(C_RED if self.last_error else C_GREEN)
        self._safe_addnstr(win, h - 1, 0, f" {self.status}"[:w], w, st_attr)

    def handle_key(self, key: int) -> None:
        if key in (ord("q"), 27):
            self.running = False
            return
        if key == ord("?"):
            self.menu_open = not self.menu_open
            self._set_status(f"Menu {'opened' if self.menu_open else 'closed'}")
            return
        if self.menu_open:
            if key in (curses.KEY_UP, ord("k")):
                self._menu_move(-1)
                return
            if key in (curses.KEY_DOWN, ord("j")):
                self._menu_move(1)
                return
            if key in (curses.KEY_ENTER, 10, 13, ord("a")):
                self._activate_menu_item()
                return
            if key in (ord("b"), curses.KEY_BACKSPACE, 127):
                self.menu_open = False
                self._set_status("Menu closed")
                return
        if key == ord("m"):
            self.manual_hold = not self.manual_hold
            self._set_status(f"Manual hold {'enabled' if self.manual_hold else 'disabled'}")
            return
        if key == ord("t"):
            self._set_low_level_enabled(not self.low_level_enabled)
            return
        if key == ord("u"):
            self._toggle_arm_release()
            return
        if key in (curses.KEY_RIGHT, ord("l")):
            if not self.low_level_enabled:
                self._set_status("Low-level control is disabled", error=True)
                return
            self._cycle_active_selection(+1)
            return
        if key in (curses.KEY_LEFT, ord("h")):
            if not self.low_level_enabled:
                self._set_status("Low-level control is disabled", error=True)
                return
            self._cycle_active_selection(-1)
            return
        if key in (curses.KEY_UP, ord("+"), ord("=")):
            if not self.low_level_enabled:
                self._set_status("Low-level control is disabled", error=True)
                return
            if self.active_family_key == "dex3_fingers":
                self._adjust_active_finger(BODY_ADJUST_STEP)
            elif self.active_family_key == "dex3_hand":
                sides = ("left", "right") if self.target_scope == "both" else (self.target_scope,)
                current = sum(self.hand_grip_percent[side] for side in sides) / float(len(sides))
                self._set_grip_percent(current + GRIP_ADJUST_STEP)
            else:
                self._adjust_active_body_target(BODY_ADJUST_STEP)
            return
        if key in (curses.KEY_DOWN, ord("-")):
            if not self.low_level_enabled:
                self._set_status("Low-level control is disabled", error=True)
                return
            if self.active_family_key == "dex3_fingers":
                self._adjust_active_finger(-BODY_ADJUST_STEP)
            elif self.active_family_key == "dex3_hand":
                sides = ("left", "right") if self.target_scope == "both" else (self.target_scope,)
                current = sum(self.hand_grip_percent[side] for side in sides) / float(len(sides))
                self._set_grip_percent(current - GRIP_ADJUST_STEP)
            else:
                self._adjust_active_body_target(-BODY_ADJUST_STEP)
            return
        if key == ord("n"):
            self._sync_targets_to_live_pose()
            return
        if key == ord("1"):
            self._set_scope("left")
            return
        if key == ord("2"):
            self._set_scope("right")
            return
        if key == ord("3"):
            self._set_scope("both")
            return
        if key == ord("y"):
            self._set_active_family("shoulder", source="Key y")
            return
        if key == ord("b"):
            self._set_active_family("elbow", source="Key b")
            return
        if key == ord("x"):
            self._set_active_family("wrist", source="Key x")
            return
        if key == ord("a"):
            self.active_family_key = "dex3_hand"
            self.active_selection_index = 0
            self._set_status("Key a: target dex3 hand grip")
            return
        if key == ord("f"):
            self.active_family_key = "dex3_fingers"
            self.active_selection_index = 0
            self._set_status("Key f: target dex3 fingers")
            return
        if key == ord("k"):
            self._set_active_family("knee", source="Key k")
            return
        if key == ord("p"):
            self._set_active_family("ankle", source="Key p")
            return
        if key == ord("i"):
            self._set_active_family("hip", source="Key i")
            return
        if key == ord("s"):
            self._set_active_family("waist", source="Key s")
            return
        if key == ord("z"):
            self._try_set_fsm(3, "Sit mode")
            return
        if key == ord("d"):
            self._call("Damp", self.robot.damp)
            return
        if key == ord("w"):
            self._set_walk_mode()
            return
        if key == ord("r"):
            self._call("Run mode", self.robot.run_mode)
            return
        if key == ord("g"):
            self._sync_gait_toggle_state()
            self.gait_toggle_state = 1 - int(self.gait_toggle_state)
            self._try_set_gait_type(self.gait_toggle_state)
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
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(20)
        self.refresh_probe(force=True)
        last_tick = 0.0
        dt_target = 1.0 / self.send_hz

        while self.running:
            key = stdscr.getch()
            if key != -1:
                self.handle_key(key)
            now = time.monotonic()
            if now - last_tick >= dt_target:
                self.tick()
                last_tick = now
            stdscr.erase()
            h, w = stdscr.getmaxyx()
            self.draw(stdscr, h, w)
            try:
                stdscr.refresh()
            except curses.error:
                pass

    def shutdown(self) -> None:
        try:
            self.robot.loco_move(0.0, 0.0, 0.0)
            time.sleep(0.2)
        except Exception:
            pass
        try:
            self.robot.release_arms()
        except Exception:
            pass
        try:
            self.robot.stop()
        except Exception:
            LOG.exception("Failed to stop locomotion cleanly")
        try:
            pygame.quit()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="USB gamepad + TUI controller for G1.")
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
    p.add_argument("--double-tap-window", type=float, default=0.35)
    p.add_argument("--r1-double-tap", type=float, dest="double_tap_window", help=argparse.SUPPRESS)
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
