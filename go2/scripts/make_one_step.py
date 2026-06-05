import argparse
import curses
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

import unitree_legged_const as go2


TOPIC_LOWSTATE = "rt/lowstate"

CONTROL_DT = 0.002
POSTURE_BLEND_DT = 0.02
NORMAL_HEIGHT_M = -0.04
HEIGHT_STEP_M = 0.01
MIN_HEIGHT_OFFSET_M = -0.10
MAX_HEIGHT_OFFSET_M = 0.06

DEFAULT_STEP_X = 0.030
DEFAULT_STEP_Y = 0.025
DEFAULT_TURN_STEP = 0.022
DEFAULT_LIFT_FOOT_Z = 0.065
DEFAULT_STANCE_FRACTION = 0.40
MIN_SWING_CLEARANCE = 0.060
MAX_FOOT_OFFSET_X = 0.090
MAX_FOOT_OFFSET_Y = 0.080

MAX_BODY_SHIFT_X = 0.025
MAX_BODY_SHIFT_Y = 0.025
BODY_SHIFT_GAIN = 0.55
IMU_PITCH_POS_GAIN = 0.035
IMU_ROLL_POS_GAIN = 0.03
IMU_GYRO_GAIN = 0.003
MAX_IMU_FOOT_Z = 0.025
MAX_TILT_FOR_STEP_RAD = 0.16

CONTACT_FORCE_MIN = 20.0
SHIFT_HOLD_SEC = 0.30
SHIFT_SEC = 0.60
SHIFT_TIMEOUT_SEC = 1.20
SWING_SEC = 1.10
SETTLE_SEC = 0.35
SETTLE_TIMEOUT_SEC = 1.20

IDLE_KP = 60.0
IDLE_KD = 5.0

LEG_INDEX = {
    "FR": (0, 1, 2),
    "FL": (3, 4, 5),
    "RR": (6, 7, 8),
    "RL": (9, 10, 11),
}
LEG_ORDER = ["FR", "FL", "RR", "RL"]
LEG_SIGNS = {
    "FL": {"left": 1.0, "front": 1.0},
    "FR": {"left": -1.0, "front": 1.0},
    "RL": {"left": 1.0, "front": -1.0},
    "RR": {"left": -1.0, "front": -1.0},
}
FOOT_FORCE_INDEX = {"FR": 0, "FL": 1, "RR": 2, "RL": 3}
DIAGONAL_LEG = {"FR": "RL", "FL": "RR", "RR": "FL", "RL": "FR"}

HIP_ORIGIN = {
    "FL": (0.1934, 0.0465, 0.0),
    "FR": (0.1934, -0.0465, 0.0),
    "RL": (-0.1934, 0.0465, 0.0),
    "RR": (-0.1934, -0.0465, 0.0),
}
HIP_LATERAL_OFFSET = 0.0955
THIGH_LENGTH = 0.213
CALF_LENGTH = 0.213

STEP_ACTIONS = [
    "forward",
    "backward",
    "right",
    "left",
    "turn_right_step",
    "turn_left_step",
]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_abs(value: float, limit: float) -> float:
    return clamp(value, -limit, limit)


def lerp_pose(src, dst, alpha: float):
    return [(1.0 - alpha) * a + alpha * b for a, b in zip(src, dst)]


def smoothstep(alpha: float) -> float:
    alpha = clamp(alpha, 0.0, 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def interp_profile(alpha: float, points):
    alpha = clamp(alpha, 0.0, 1.0)
    for idx in range(1, len(points)):
        x0, y0 = points[idx - 1]
        x1, y1 = points[idx]
        if alpha <= x1:
            span = max(x1 - x0, 1e-6)
            local = smoothstep((alpha - x0) / span)
            return (1.0 - local) * y0 + local * y1
    return points[-1][1]


def vec3(values):
    return f"{values[0]: .2f} {values[1]: .2f} {values[2]: .2f}"


def age_text(age: Optional[float]) -> str:
    if age is None:
        return "--"
    return f"{age:4.1f}s"


def status_text(age: Optional[float], timeout: float = 1.5) -> str:
    if age is None:
        return "waiting"
    if age <= timeout:
        return "live"
    return "stale"


def leg_forward_kinematics(leg_name: str, joint_triplet):
    q_hip, q_thigh, q_calf = joint_triplet
    side_sign = LEG_SIGNS[leg_name]["left"]
    px = -THIGH_LENGTH * math.sin(q_thigh) - CALF_LENGTH * math.sin(q_thigh + q_calf)
    pzr = -THIGH_LENGTH * math.cos(q_thigh) - CALF_LENGTH * math.cos(q_thigh + q_calf)
    py = side_sign * HIP_LATERAL_OFFSET * math.cos(q_hip) - pzr * math.sin(q_hip)
    pz = side_sign * HIP_LATERAL_OFFSET * math.sin(q_hip) + pzr * math.cos(q_hip)
    return [px, py, pz]


def leg_inverse_kinematics(leg_name: str, foot_pos_hip):
    x, y, z = foot_pos_hip
    side_sign = LEG_SIGNS[leg_name]["left"]
    radial_sq = y * y + z * z - HIP_LATERAL_OFFSET * HIP_LATERAL_OFFSET
    radial = math.sqrt(max(radial_sq, 1e-9))
    q_hip = math.atan2(y, -z) - math.atan2(side_sign * HIP_LATERAL_OFFSET, radial)

    knee_cos = (
        x * x + radial_sq - THIGH_LENGTH * THIGH_LENGTH - CALF_LENGTH * CALF_LENGTH
    ) / (2.0 * THIGH_LENGTH * CALF_LENGTH)
    knee_cos = clamp(knee_cos, -1.0, 1.0)
    q_calf = -math.acos(knee_cos)
    q_thigh = math.atan2(-x, radial) - math.atan2(
        CALF_LENGTH * math.sin(q_calf),
        THIGH_LENGTH + CALF_LENGTH * math.cos(q_calf),
    )
    return [q_hip, q_thigh, q_calf]


@dataclass
class TopicSnapshot:
    low_state: Optional[LowState_] = None
    low_state_time: float = 0.0
    motion_mode: str = "unknown"
    motion_code: int = 0
    motion_time: float = 0.0
    motion_error: str = ""
    service_status: str = "idle"
    service_time: float = 0.0


class OneStepController:
    def __init__(
        self,
        step_x: float,
        step_y: float,
        turn_step: float,
        lift_z: float,
        shift_scale: float,
        stance_fraction: float,
        require_contact: bool,
    ):
        self.Kp = IDLE_KP
        self.Kd = IDLE_KD
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self.lock = threading.Lock()
        self.snapshots = TopicSnapshot()
        self.first_run = True
        self.running = True

        self.stand_pose = [
            0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
            0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
        ]
        self.height_pose = [
            -0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
            -0.5, 1.36, -2.65, 0.5, 1.36, -2.65,
        ]
        self.sit_pose = [
            0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
            -0.2, 1.36, -2.65, 0.2, 1.36, -2.65,
        ]
        self.stand_foot_targets = self._build_foot_targets(self.stand_pose)
        self.lowered_foot_targets = self._build_foot_targets(self.height_pose)
        self.foot_offsets = {leg: [0.0, 0.0] for leg in LEG_ORDER}

        self.start_pose = [0.0] * 12
        self.current_pose = list(self.sit_pose)
        self.target_pose = list(self.sit_pose)
        self.height_offset_m = NORMAL_HEIGHT_M
        self.selected_leg = "FR"
        self.selected_action = "forward"
        self.step_x = clamp(step_x, 0.005, MAX_FOOT_OFFSET_X)
        self.step_y = clamp(step_y, 0.005, MAX_FOOT_OFFSET_Y)
        self.turn_step = clamp(turn_step, 0.005, max(MAX_FOOT_OFFSET_X, MAX_FOOT_OFFSET_Y))
        self.lift_z = clamp(lift_z, MIN_SWING_CLEARANCE, 0.10)
        self.shift_scale = clamp(shift_scale, 0.0, 1.0)
        self.stance_fraction = clamp(stance_fraction, 0.0, 0.8)
        self.require_contact = require_contact

        self.roll = 0.0
        self.pitch = 0.0
        self.gx = 0.0
        self.gy = 0.0
        self.foot_force = [0.0, 0.0, 0.0, 0.0]

        self.step_active = False
        self.step_state = "idle"
        self.step_state_started = 0.0
        self.support_started = 0.0
        self.step_leg = "FR"
        self.step_action = "forward"
        self.step_delta = (0.0, 0.0)
        self.step_start_offsets = {leg: [0.0, 0.0] for leg in LEG_ORDER}
        self.step_support_deltas = {leg: [0.0, 0.0] for leg in LEG_ORDER}

        self.transition_start_pose = list(self.sit_pose)
        self.transition_target_pose = list(self.sit_pose)
        self.transition_started = 0.0
        self.transition_duration = 1.0
        self.transition_active = False

        self.lowCmdWriteThreadPtr = None
        self.modePollThreadPtr = None

    def init(self):
        self._init_low_cmd()
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        self._release_motion_mode()

    def start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=CONTROL_DT, target=self._low_cmd_write, name="make_one_step_lowcmd"
        )
        self.lowCmdWriteThreadPtr.Start()
        self.modePollThreadPtr = RecurrentThread(
            interval=0.5, target=self._poll_motion_mode, name="make_one_step_mode_poll"
        )
        self.modePollThreadPtr.Start()

    def stop(self):
        self.running = False
        if self.lowCmdWriteThreadPtr is not None:
            self.lowCmdWriteThreadPtr.Wait(1.0)
        if self.modePollThreadPtr is not None:
            self.modePollThreadPtr.Wait(1.0)

    def increase_height(self):
        with self.lock:
            self.height_offset_m = clamp(
                self.height_offset_m + HEIGHT_STEP_M, MIN_HEIGHT_OFFSET_M, MAX_HEIGHT_OFFSET_M
            )
            self._begin_transition_locked(self._compose_pose_locked(), 0.35)

    def decrease_height(self):
        with self.lock:
            self.height_offset_m = clamp(
                self.height_offset_m - HEIGHT_STEP_M, MIN_HEIGHT_OFFSET_M, MAX_HEIGHT_OFFSET_M
            )
            self._begin_transition_locked(self._compose_pose_locked(), 0.35)

    def select_leg(self, leg_name: str):
        if leg_name not in LEG_ORDER:
            return
        with self.lock:
            if not self.step_active:
                self.selected_leg = leg_name

    def next_action(self, direction: int):
        with self.lock:
            if self.step_active:
                return
            idx = STEP_ACTIONS.index(self.selected_action)
            self.selected_action = STEP_ACTIONS[(idx + direction) % len(STEP_ACTIONS)]

    def set_action(self, action: str):
        if action not in STEP_ACTIONS:
            return
        with self.lock:
            if not self.step_active:
                self.selected_action = action

    def reset_offsets(self):
        with self.lock:
            if self.step_active:
                return
            self.foot_offsets = {leg: [0.0, 0.0] for leg in LEG_ORDER}
            self._begin_transition_locked(self._compose_pose_locked(), 0.45)

    def request_step(self):
        with self.lock:
            if self.step_active or self.low_state is None:
                return False
            tilt_mag = max(abs(self.roll), abs(self.pitch))
            if tilt_mag > MAX_TILT_FOR_STEP_RAD:
                self._set_status_locked("step refused: tilt limit")
                return False
            self.step_leg = self.selected_leg
            self.step_action = self.selected_action
            self.step_delta = self._action_delta(self.step_leg, self.step_action)
            self.step_start_offsets = {leg: list(value) for leg, value in self.foot_offsets.items()}
            self.step_support_deltas = self._support_stance_deltas(self.step_leg, self.step_delta)
            self.step_active = True
            self.step_state = "shift"
            self.step_state_started = time.time()
            self.support_started = 0.0
            self._set_status_locked(f"step shift:{self.step_leg}:{self.step_action}")
            return True

    def get_snapshot(self):
        with self.lock:
            return {
                "low_state": self.snapshots.low_state,
                "low_state_age": self._age(self.snapshots.low_state_time),
                "motion_mode": self.snapshots.motion_mode,
                "motion_code": self.snapshots.motion_code,
                "motion_age": self._age(self.snapshots.motion_time),
                "motion_error": self.snapshots.motion_error,
                "service_status": self.snapshots.service_status,
                "service_age": self._age(self.snapshots.service_time),
                "height_offset_m": self.height_offset_m,
                "selected_leg": self.selected_leg,
                "selected_action": self.selected_action,
                "step_active": self.step_active,
                "step_state": self.step_state,
                "step_leg": self.step_leg,
                "step_action": self.step_action,
                "roll": self.roll,
                "pitch": self.pitch,
                "foot_force": list(self.foot_force),
                "foot_offsets": {leg: list(offset) for leg, offset in self.foot_offsets.items()},
                "step_x": self.step_x,
                "step_y": self.step_y,
                "turn_step": self.turn_step,
                "lift_z": self.lift_z,
                "shift_scale": self.shift_scale,
                "stance_fraction": self.stance_fraction,
                "require_contact": self.require_contact,
            }

    def _age(self, ts: float) -> Optional[float]:
        if ts <= 0.0:
            return None
        return max(0.0, time.time() - ts)

    def _set_status_locked(self, text: str):
        self.snapshots.service_status = text
        self.snapshots.service_time = time.time()

    def _action_delta(self, leg_name: str, action: str):
        if action == "forward":
            return (self.step_x, 0.0)
        if action == "backward":
            return (-self.step_x, 0.0)
        if action == "left":
            return (0.0, self.step_y)
        if action == "right":
            return (0.0, -self.step_y)

        front_sign = LEG_SIGNS[leg_name]["front"]
        left_sign = LEG_SIGNS[leg_name]["left"]
        yaw_sign = -1.0 if action == "turn_right_step" else 1.0
        return (
            -yaw_sign * left_sign * self.turn_step,
            yaw_sign * front_sign * self.turn_step,
        )

    def _support_stance_deltas(self, swing_leg: str, swing_delta):
        deltas = {leg: [0.0, 0.0] for leg in LEG_ORDER}
        diagonal = DIAGONAL_LEG[swing_leg]
        for leg in LEG_ORDER:
            if leg == swing_leg:
                continue
            if leg == diagonal:
                weight = 1.00
            elif LEG_SIGNS[leg]["front"] == LEG_SIGNS[swing_leg]["front"]:
                weight = 0.45
            else:
                weight = 0.65
            deltas[leg][0] = -swing_delta[0] * self.stance_fraction * weight
            deltas[leg][1] = -swing_delta[1] * self.stance_fraction * weight
        return deltas

    def _build_foot_targets(self, pose):
        foot_targets = {}
        for leg_name, indices in LEG_INDEX.items():
            joints = pose[indices[0]: indices[2] + 1]
            hip_origin = HIP_ORIGIN[leg_name]
            foot_hip = leg_forward_kinematics(leg_name, joints)
            foot_targets[leg_name] = [
                hip_origin[0] + foot_hip[0],
                hip_origin[1] + foot_hip[1],
                hip_origin[2] + foot_hip[2],
            ]
        return foot_targets

    def _height_foot_targets_locked(self):
        alpha = clamp(
            (self.height_offset_m - MIN_HEIGHT_OFFSET_M) / (MAX_HEIGHT_OFFSET_M - MIN_HEIGHT_OFFSET_M),
            0.0,
            1.0,
        )
        targets = {}
        for leg_name in LEG_ORDER:
            stand_target = self.stand_foot_targets[leg_name]
            lowered_target = self.lowered_foot_targets[leg_name]
            targets[leg_name] = [
                (1.0 - alpha) * lowered_target[i] + alpha * stand_target[i] for i in range(3)
            ]
            targets[leg_name][0] += self.foot_offsets[leg_name][0]
            targets[leg_name][1] += self.foot_offsets[leg_name][1]
        return targets

    def _imu_balance_offsets_locked(self):
        if self.low_state is None:
            return {leg: 0.0 for leg in LEG_ORDER}
        pitch_term = clamp_abs(
            -(IMU_PITCH_POS_GAIN * self.pitch + IMU_GYRO_GAIN * self.gy),
            MAX_IMU_FOOT_Z,
        )
        roll_term = clamp_abs(
            -(IMU_ROLL_POS_GAIN * self.roll + IMU_GYRO_GAIN * self.gx),
            MAX_IMU_FOOT_Z,
        )
        return {
            leg_name: signs["front"] * pitch_term + signs["left"] * roll_term
            for leg_name, signs in LEG_SIGNS.items()
        }

    def _support_body_shift(self, support_legs, foot_targets):
        if not support_legs:
            return (0.0, 0.0)
        centroid_x = sum(foot_targets[leg][0] for leg in support_legs) / len(support_legs)
        centroid_y = sum(foot_targets[leg][1] for leg in support_legs) / len(support_legs)
        shift_x = clamp_abs(BODY_SHIFT_GAIN * centroid_x, MAX_BODY_SHIFT_X) * self.shift_scale
        shift_y = clamp_abs(BODY_SHIFT_GAIN * centroid_y, MAX_BODY_SHIFT_Y) * self.shift_scale
        return (shift_x, shift_y)

    def _foot_targets_to_pose(self, foot_targets):
        pose = [0.0] * 12
        for leg_name, indices in LEG_INDEX.items():
            hip_origin = HIP_ORIGIN[leg_name]
            target = foot_targets[leg_name]
            foot_pos_hip = [
                target[0] - hip_origin[0],
                target[1] - hip_origin[1],
                target[2] - hip_origin[2],
            ]
            q_hip, q_thigh, q_calf = leg_inverse_kinematics(leg_name, foot_pos_hip)
            pose[indices[0]] = q_hip
            pose[indices[1]] = q_thigh
            pose[indices[2]] = q_calf
        return pose

    def _compose_pose_locked(self):
        foot_targets = self._height_foot_targets_locked()
        imu_balance = self._imu_balance_offsets_locked()
        for leg_name in LEG_ORDER:
            foot_targets[leg_name][2] += imu_balance[leg_name]
        return self._foot_targets_to_pose(foot_targets)

    def _step_pose_locked(self):
        base_targets = self._height_foot_targets_locked()
        support_legs = [leg for leg in LEG_ORDER if leg != self.step_leg]
        shift_x, shift_y = self._support_body_shift(support_legs, base_targets)
        elapsed = time.time() - self.step_state_started
        tilt_mag = max(abs(self.roll), abs(self.pitch))
        force_ok = min(self.foot_force[FOOT_FORCE_INDEX[leg]] for leg in support_legs) >= CONTACT_FORCE_MIN

        if tilt_mag > MAX_TILT_FOR_STEP_RAD:
            self.step_active = False
            self.step_state = "idle"
            self._set_status_locked("step halted: tilt limit")
            return self._compose_pose_locked()

        shift_alpha = 1.0
        swing_alpha = 0.0
        lift_alpha = 0.0
        if self.step_state == "shift":
            shift_alpha = smoothstep(elapsed / SHIFT_SEC)
            contact_ready = force_ok or not self.require_contact
            if contact_ready and shift_alpha >= 1.0:
                if self.support_started == 0.0:
                    self.support_started = time.time()
                elif time.time() - self.support_started >= SHIFT_HOLD_SEC:
                    self._begin_step_state_locked("swing")
            elif self.require_contact and elapsed >= SHIFT_TIMEOUT_SEC:
                self.step_active = False
                self.step_state = "idle"
                self._set_status_locked("step refused: support contact")
                return self._compose_pose_locked()
            else:
                self.support_started = 0.0
        elif self.step_state == "swing":
            swing_alpha = clamp(elapsed / SWING_SEC, 0.0, 1.0)
            lift_alpha = interp_profile(
                swing_alpha,
                (
                    (0.00, 0.00),
                    (0.15, 0.75),
                    (0.50, 1.00),
                    (0.85, 0.75),
                    (1.00, 0.00),
                ),
            )
            if swing_alpha >= 1.0:
                self._commit_swing_offsets_locked()
                base_targets = self._height_foot_targets_locked()
                self._begin_step_state_locked("settle")
        elif self.step_state == "settle":
            swing_alpha = 1.0
            contact_ok = self.foot_force[FOOT_FORCE_INDEX[self.step_leg]] >= CONTACT_FORCE_MIN
            contact_ready = (force_ok and contact_ok) or not self.require_contact
            if contact_ready:
                if self.support_started == 0.0:
                    self.support_started = time.time()
                elif time.time() - self.support_started >= SETTLE_SEC:
                    self.step_active = False
                    self.step_state = "idle"
                    self.support_started = 0.0
                    self._set_status_locked("step complete")
                    neutral_pose = self._compose_pose_locked()
                    self._begin_transition_locked(neutral_pose, 0.35)
                    return neutral_pose
            elif self.require_contact and elapsed >= SETTLE_TIMEOUT_SEC:
                self.step_active = False
                self.step_state = "idle"
                self._set_status_locked("step halted: landing contact")
                return self._compose_pose_locked()
            else:
                self.support_started = 0.0

        if self.step_state == "settle":
            targets = self._height_foot_targets_locked()
        else:
            targets = self._height_foot_targets_for_offsets_locked(self.step_start_offsets)
        for leg in LEG_ORDER:
            targets[leg][0] -= shift_x * shift_alpha
            targets[leg][1] -= shift_y * shift_alpha

        stance_alpha = smoothstep(swing_alpha)
        if self.step_state != "settle":
            for leg in support_legs:
                targets[leg][0] += self.step_support_deltas[leg][0] * stance_alpha
                targets[leg][1] += self.step_support_deltas[leg][1] * stance_alpha
            targets[self.step_leg][0] += self.step_delta[0] * swing_alpha
            targets[self.step_leg][1] += self.step_delta[1] * swing_alpha

        clearance = max(self.lift_z, MIN_SWING_CLEARANCE)
        targets[self.step_leg][2] += clearance * lift_alpha

        imu_balance = self._imu_balance_offsets_locked()
        for leg in support_legs:
            targets[leg][2] += imu_balance[leg]

        self.target_pose = self._foot_targets_to_pose(targets)
        self._set_status_locked(f"step {self.step_state}:{self.step_leg}:{self.step_action}")
        return self.target_pose

    def _height_foot_targets_for_offsets_locked(self, offsets):
        alpha = clamp(
            (self.height_offset_m - MIN_HEIGHT_OFFSET_M) / (MAX_HEIGHT_OFFSET_M - MIN_HEIGHT_OFFSET_M),
            0.0,
            1.0,
        )
        targets = {}
        for leg_name in LEG_ORDER:
            stand_target = self.stand_foot_targets[leg_name]
            lowered_target = self.lowered_foot_targets[leg_name]
            targets[leg_name] = [
                (1.0 - alpha) * lowered_target[i] + alpha * stand_target[i] for i in range(3)
            ]
            targets[leg_name][0] += offsets[leg_name][0]
            targets[leg_name][1] += offsets[leg_name][1]
        return targets

    def _commit_swing_offsets_locked(self):
        x = self.step_start_offsets[self.step_leg][0] + self.step_delta[0]
        y = self.step_start_offsets[self.step_leg][1] + self.step_delta[1]
        self.foot_offsets[self.step_leg][0] = clamp_abs(x, MAX_FOOT_OFFSET_X)
        self.foot_offsets[self.step_leg][1] = clamp_abs(y, MAX_FOOT_OFFSET_Y)
        for leg in LEG_ORDER:
            if leg == self.step_leg:
                continue
            x = self.step_start_offsets[leg][0] + self.step_support_deltas[leg][0]
            y = self.step_start_offsets[leg][1] + self.step_support_deltas[leg][1]
            self.foot_offsets[leg][0] = clamp_abs(x, MAX_FOOT_OFFSET_X)
            self.foot_offsets[leg][1] = clamp_abs(y, MAX_FOOT_OFFSET_Y)

    def _begin_step_state_locked(self, state: str):
        self.step_state = state
        self.step_state_started = time.time()
        self.support_started = 0.0

    def _begin_transition_locked(self, pose, duration: float):
        self.transition_start_pose = list(self.current_pose)
        self.transition_target_pose = list(pose)
        self.transition_started = time.time()
        self.transition_duration = max(duration, POSTURE_BLEND_DT)
        self.transition_active = True
        self.target_pose = list(pose)

    def _apply_transition_locked(self):
        if not self.transition_active:
            return
        elapsed = time.time() - self.transition_started
        alpha = smoothstep(elapsed / self.transition_duration)
        self.current_pose = lerp_pose(self.transition_start_pose, self.transition_target_pose, alpha)
        if alpha >= 1.0:
            self.current_pose = list(self.transition_target_pose)
            self.transition_active = False
            self.target_pose = list(self.current_pose)

    def _init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def _release_motion_mode(self):
        while True:
            code, result = self.msc.CheckMode()
            name = (result or {}).get("name", "") if result is not None else ""
            if code == 0 and not name:
                return
            self.msc.ReleaseMode()
            time.sleep(1.0)

    def _low_state_handler(self, msg: LowState_):
        with self.lock:
            self.low_state = msg
            self.snapshots.low_state = msg
            self.snapshots.low_state_time = time.time()
            imu = msg.imu_state
            self.roll = float(imu.rpy[0])
            self.pitch = float(imu.rpy[1])
            self.gx = float(imu.gyroscope[0])
            self.gy = float(imu.gyroscope[1])
            self.foot_force = [float(v) for v in msg.foot_force[:4]]

    def _poll_motion_mode(self):
        try:
            code, result = self.msc.CheckMode()
            with self.lock:
                self.snapshots.motion_code = code
                self.snapshots.motion_time = time.time()
                self.snapshots.motion_mode = (result or {}).get("name", "") or "released"
                self.snapshots.motion_error = ""
        except Exception as exc:
            with self.lock:
                self.snapshots.motion_time = time.time()
                self.snapshots.motion_error = str(exc)

    def _low_cmd_write(self):
        with self.lock:
            if not self.running or self.low_state is None:
                return
            if self.first_run:
                for i in range(12):
                    self.start_pose[i] = self.low_state.motor_state[i].q
                self.current_pose = list(self.start_pose)
                self.transition_start_pose = list(self.start_pose)
                self.transition_target_pose = self._compose_pose_locked()
                self.transition_started = time.time()
                self.transition_duration = 1.5
                self.transition_active = True
                self.first_run = False

            desired_pose = self._step_pose_locked() if self.step_active else self._compose_pose_locked()
            if self.transition_active:
                self.transition_target_pose = list(desired_pose)
                self.target_pose = list(desired_pose)
                self._apply_transition_locked()
                pose = list(self.current_pose)
            else:
                self.current_pose = list(desired_pose)
                self.target_pose = list(desired_pose)
                pose = list(desired_pose)

        for i in range(12):
            self.low_cmd.motor_cmd[i].q = pose[i]
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = self.Kp
            self.low_cmd.motor_cmd[i].kd = self.Kd
            self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


def draw_panel(stdscr, controller: OneStepController):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    snapshot = controller.get_snapshot()
    low_state = snapshot["low_state"]

    lines = [
        "Go2 One-Step TUI",
        "Up/Down: height  Left/Right: action  1:FR 2:FL 3:RR 4:RL  Space/Enter: step  c: center feet  q: quit",
        "Direct actions: f forward  b backward  l left  r right  e turn_right_step  t turn_left_step",
        "",
        f"Height offset: {snapshot['height_offset_m']:+.3f} m",
        f"Selected: {snapshot['selected_leg']} / {snapshot['selected_action']}",
        f"Tuning: step_x={snapshot['step_x']:.3f} step_y={snapshot['step_y']:.3f} "
        f"turn={snapshot['turn_step']:.3f} lift={snapshot['lift_z']:.3f} shift={snapshot['shift_scale']:.2f} "
        f"stance={snapshot['stance_fraction']:.2f} contact={'on' if snapshot['require_contact'] else 'off'}",
        f"Step: {'active' if snapshot['step_active'] else 'ready'}  state={snapshot['step_state']}  "
        f"last={snapshot['step_leg']}:{snapshot['step_action']}",
        f"Status: {snapshot['service_status']}  age={age_text(snapshot['service_age'])}",
        "",
        f"LowState {status_text(snapshot['low_state_age'])} age={age_text(snapshot['low_state_age'])} topic={TOPIC_LOWSTATE}",
    ]

    if low_state is not None:
        imu = low_state.imu_state
        lines.extend(
            [
                f"  Power V/A: {float(low_state.power_v):.2f} / {float(low_state.power_a):.2f}",
                f"  IMU rpy:   {vec3([float(v) for v in imu.rpy])}",
                f"  IMU gyro:  {vec3([float(v) for v in imu.gyroscope])}",
                f"  Foot force:{' '.join(str(int(v)) for v in snapshot['foot_force'])}",
            ]
        )
    else:
        lines.append("  waiting for rt/lowstate")

    lines.extend(
        [
            "",
            f"Motion switcher age={age_text(snapshot['motion_age'])}",
            f"  mode={snapshot['motion_mode']} code={snapshot['motion_code']}",
            "",
            "Foot offsets x/y:",
        ]
    )
    for leg_name in LEG_ORDER:
        x, y = snapshot["foot_offsets"][leg_name]
        marker = "*" if leg_name == snapshot["selected_leg"] else " "
        lines.append(f" {marker} {leg_name}: {x:+.3f} {y:+.3f} m")
    if snapshot["motion_error"]:
        lines.append(f"  motion error={snapshot['motion_error']}")
    if h < 16 or w < 80:
        lines.append("")
        lines.append("Terminal is small; widen it for the full panel.")

    for idx, line in enumerate(lines[: max(0, h - 1)]):
        stdscr.addnstr(idx, 0, line, max(0, w - 1))
    stdscr.refresh()


def tui_main(stdscr, controller: OneStepController):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    action_keys = {
        ord("f"): "forward",
        ord("F"): "forward",
        ord("b"): "backward",
        ord("B"): "backward",
        ord("l"): "left",
        ord("L"): "left",
        ord("r"): "right",
        ord("R"): "right",
        ord("e"): "turn_right_step",
        ord("E"): "turn_right_step",
        ord("t"): "turn_left_step",
        ord("T"): "turn_left_step",
    }

    while True:
        draw_panel(stdscr, controller)
        key = stdscr.getch()
        if key == curses.KEY_UP:
            controller.increase_height()
        elif key == curses.KEY_DOWN:
            controller.decrease_height()
        elif key == curses.KEY_RIGHT:
            controller.next_action(1)
        elif key == curses.KEY_LEFT:
            controller.next_action(-1)
        elif key == ord("1"):
            controller.select_leg("FR")
        elif key == ord("2"):
            controller.select_leg("FL")
        elif key == ord("3"):
            controller.select_leg("RR")
        elif key == ord("4"):
            controller.select_leg("RL")
        elif key in action_keys:
            controller.set_action(action_keys[key])
        elif key in (ord(" "), ord("\n"), curses.KEY_ENTER, 10, 13):
            controller.request_step()
        elif key in (ord("c"), ord("C")):
            controller.reset_offsets()
        elif key in (ord("q"), ord("Q")):
            break


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Go2 low-level single-step controller."
    )
    parser.add_argument("iface", nargs="?", default=None, help="Robot network interface")
    parser.add_argument("--step-x", type=float, default=DEFAULT_STEP_X, help="Forward/back foot step in meters")
    parser.add_argument("--step-y", type=float, default=DEFAULT_STEP_Y, help="Left/right foot step in meters")
    parser.add_argument("--turn-step", type=float, default=DEFAULT_TURN_STEP, help="Per-foot turn step in meters")
    parser.add_argument("--lift-z", type=float, default=DEFAULT_LIFT_FOOT_Z, help="Swing foot lift in meters")
    parser.add_argument(
        "--shift-scale",
        type=float,
        default=1.0,
        help="Support-centroid body shift scale from 0.0 to 1.0",
    )
    parser.add_argument(
        "--stance-fraction",
        type=float,
        default=DEFAULT_STANCE_FRACTION,
        help="Fraction of selected-foot step applied oppositely to support feet",
    )
    parser.add_argument(
        "--ignore-contact",
        action="store_true",
        help="Do not require foot-force confirmation before lifting/finishing a step",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("WARNING: This script uses low-level motor commands and intentionally lifts one foot.")
    print("Use a spotter, keep the robot on a clear high-friction floor, and stay near e-stop.")
    input("Press Enter to continue...")

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    controller = OneStepController(
        step_x=args.step_x,
        step_y=args.step_y,
        turn_step=args.turn_step,
        lift_z=args.lift_z,
        shift_scale=args.shift_scale,
        stance_fraction=args.stance_fraction,
        require_contact=not args.ignore_contact,
    )
    controller.init()
    controller.start()
    try:
        curses.wrapper(tui_main, controller)
    finally:
        controller.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
