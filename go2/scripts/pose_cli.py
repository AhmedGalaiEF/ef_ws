import argparse
import curses
import glob
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

import unitree_legged_const as go2


TOPIC_LOWSTATE = "rt/lowstate"
CONTROL_DT = 0.002
HEIGHT_STEP_M = 0.01
MIN_HEIGHT_OFFSET_M = -0.10
MAX_HEIGHT_OFFSET_M = 0.06
COMMAND_RAMP_RATE = 1.8
KP = 60.0
KD = 5.0
ZMP_ROLL_P = 0.10
ZMP_ROLL_D = 0.02
ZMP_PITCH_P = 0.12
ZMP_PITCH_D = 0.02
ZMP_FORCE_GAIN = 0.0015
ZMP_EXT_CLAMP = 0.16
ZMP_THIGH_GAIN = 0.30
ZMP_CALF_GAIN = -0.45


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp_pose(src, dst, alpha: float):
    return [(1.0 - alpha) * a + alpha * b for a, b in zip(src, dst)]


def age_text(ts: float) -> str:
    if ts <= 0.0:
        return "--"
    return f"{max(0.0, time.time() - ts):4.1f}s"


def find_latest_recording():
    candidates = sorted(glob.glob("record_gait_*.jsonl"))
    return candidates[-1] if candidates else None


class PoseRegressor:
    def __init__(self, path: str):
        self.path = path
        self.valid = False
        self.error = ""
        self.reference_pose = None
        self.weights = None
        self.label_counts = {}
        self.label_means = {}
        self.label_amplitudes = {}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        except Exception as exc:
            self.error = str(exc)
            return

        x_rows = []
        y_rows = []
        pose_sum = np.zeros(12, dtype=np.float64)
        valid_count = 0
        label_counts = {}
        label_sum = {}
        label_min = {}
        label_max = {}
        idle_pose_sum = np.zeros(12, dtype=np.float64)
        idle_count = 0

        for row in rows:
            joints = row.get("joints", [])
            cmd = row.get("command", {})
            if len(joints) < 12:
                continue
            q = np.array([float(joints[i]["q"]) for i in range(12)], dtype=np.float64)
            x_rows.append(
                [
                    1.0,
                    float(cmd.get("forward", 0.0)),
                    float(cmd.get("lateral", 0.0)),
                    float(cmd.get("turn", 0.0)),
                ]
            )
            y_rows.append(q)
            pose_sum += q
            valid_count += 1
            label = cmd.get("label", "unknown")
            label_counts[label] = label_counts.get(label, 0) + 1
            label_sum[label] = label_sum.get(label, np.zeros(12, dtype=np.float64)) + q
            label_min[label] = np.minimum(label_min.get(label, q.copy()), q)
            label_max[label] = np.maximum(label_max.get(label, q.copy()), q)
            if label == "idle":
                idle_pose_sum += q
                idle_count += 1

        if valid_count < 20:
            self.error = "not enough valid samples for regression"
            return

        x = np.asarray(x_rows, dtype=np.float64)
        y = np.asarray(y_rows, dtype=np.float64)
        self.reference_pose = idle_pose_sum / idle_count if idle_count > 0 else pose_sum / valid_count
        delta = y - self.reference_pose
        self.weights, _, _, _ = np.linalg.lstsq(x, delta, rcond=None)
        self.label_counts = label_counts
        for label, count in label_counts.items():
            mean = label_sum[label] / count
            amp = 0.5 * (label_max[label] - label_min[label])
            self.label_means[label] = mean
            self.label_amplitudes[label] = amp
        self.valid = True

    def predict_delta(self, forward: float, lateral: float, turn: float):
        if not self.valid:
            return np.zeros(12, dtype=np.float64)
        x = np.array([1.0, forward, lateral, turn], dtype=np.float64)
        return x @ self.weights

    def amplitude_for_label(self, label: str):
        if not self.valid:
            return np.zeros(12, dtype=np.float64)
        return self.label_amplitudes.get(label, np.zeros(12, dtype=np.float64))


@dataclass
class PoseSnapshot:
    low_state_time: float = 0.0
    last_status: str = "idle"
    status_time: float = 0.0


class PoseController:
    def __init__(self, model: Optional[PoseRegressor]):
        self.lock = threading.Lock()
        self.snapshot = PoseSnapshot()
        self.model = model
        self.low_state = None
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.crc = CRC()
        self.low_level_enabled = True
        self.first_run = True

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
        self.height_offset_m = 0.0
        self.forward = 0.0
        self.lateral = 0.0
        self.turn = 0.0
        self.cmd_forward = 0.0
        self.cmd_lateral = 0.0
        self.cmd_turn = 0.0
        self.pose_gain = 1.6
        self.step_phase = 0.0
        self.sitting = False
        self.current_pose = list(self.sit_pose)
        self.target_pose = list(self.sit_pose)
        self.transition_active = False
        self.transition_started = 0.0
        self.transition_duration = 1.0
        self.transition_start_pose = list(self.sit_pose)
        self.transition_target_pose = list(self.sit_pose)
        self.control_thread = None
        self.roll = 0.0
        self.pitch = 0.0
        self.gx = 0.0
        self.gy = 0.0
        self.foot_force = [0.0, 0.0, 0.0, 0.0]

    def init(self):
        self._init_low_cmd()
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        self._release_motion_mode()
        self._set_status("initialized")

    def start(self):
        self.control_thread = RecurrentThread(interval=CONTROL_DT, target=self._control_loop, name="pose_cli_lowcmd")
        self.control_thread.Start()

    def stop(self):
        with self.lock:
            self.low_level_enabled = False
        if self.control_thread is not None:
            self.control_thread.Wait(1.0)
        try:
            self.sc.StandDown()
        except Exception:
            pass

    def _set_status(self, text: str):
        with self.lock:
            self.snapshot.last_status = text
            self.snapshot.status_time = time.time()

    def snapshot_data(self):
        with self.lock:
            return {
                "low_state_age": age_text(self.snapshot.low_state_time),
                "status": self.snapshot.last_status,
                "status_age": age_text(self.snapshot.status_time),
                "height_offset_m": self.height_offset_m,
                "forward": self.forward,
                "lateral": self.lateral,
                "turn": self.turn,
                "model_path": "" if self.model is None else self.model.path,
                "model_valid": self.model is not None and self.model.valid,
                "model_error": "" if self.model is None else self.model.error,
                "label_counts": {} if self.model is None else self.model.label_counts,
                "sitting": self.sitting,
                "roll": self.roll,
                "pitch": self.pitch,
                "foot_force": list(self.foot_force),
            }

    def increase_height(self):
        with self.lock:
            self.height_offset_m = clamp(self.height_offset_m + HEIGHT_STEP_M, MIN_HEIGHT_OFFSET_M, MAX_HEIGHT_OFFSET_M)
            self._begin_transition_locked(self._blend_height(self.height_offset_m), 0.30)

    def decrease_height(self):
        with self.lock:
            self.height_offset_m = clamp(self.height_offset_m - HEIGHT_STEP_M, MIN_HEIGHT_OFFSET_M, MAX_HEIGHT_OFFSET_M)
            self._begin_transition_locked(self._blend_height(self.height_offset_m), 0.30)

    def sit(self):
        with self.lock:
            self.sitting = True
            self.forward = 0.0
            self.lateral = 0.0
            self.turn = 0.0
            self.cmd_forward = 0.0
            self.cmd_lateral = 0.0
            self.cmd_turn = 0.0
            self._begin_transition_locked(list(self.sit_pose), 0.80)

    def stand(self):
        with self.lock:
            self.sitting = False
            self._begin_transition_locked(self._blend_height(self.height_offset_m), 0.50)

    def set_command(self, forward: float, lateral: float, turn: float):
        with self.lock:
            if self.sitting:
                self.forward = 0.0
                self.lateral = 0.0
                self.turn = 0.0
                return
            self.forward = clamp(forward, -1.0, 1.0)
            self.lateral = clamp(lateral, -1.0, 1.0)
            self.turn = clamp(turn, -1.0, 1.0)

    def _release_motion_mode(self):
        while True:
            code, result = self.msc.CheckMode()
            name = (result or {}).get("name", "") if result is not None else ""
            if code == 0 and not name:
                return
            try:
                self.sc.StandDown()
            except Exception:
                pass
            self.msc.ReleaseMode()
            time.sleep(1.0)

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

    def _low_state_handler(self, msg: LowState_):
        with self.lock:
            self.low_state = msg
            self.snapshot.low_state_time = time.time()
            imu = msg.imu_state
            self.roll = float(imu.rpy[0])
            self.pitch = float(imu.rpy[1])
            self.gx = float(imu.gyroscope[0])
            self.gy = float(imu.gyroscope[1])
            self.foot_force = [float(v) for v in msg.foot_force]

    def _blend_height(self, offset_m: float):
        alpha = clamp((offset_m - MIN_HEIGHT_OFFSET_M) / (MAX_HEIGHT_OFFSET_M - MIN_HEIGHT_OFFSET_M), 0.0, 1.0)
        return lerp_pose(self.height_pose, self.stand_pose, alpha)

    def _begin_transition_locked(self, pose, duration: float):
        self.transition_start_pose = list(self.current_pose)
        self.transition_target_pose = list(pose)
        self.transition_started = time.time()
        self.transition_duration = max(duration, 0.02)
        self.transition_active = True
        self.target_pose = list(pose)

    def _apply_transition_locked(self):
        if not self.transition_active:
            return
        alpha = clamp((time.time() - self.transition_started) / self.transition_duration, 0.0, 1.0)
        self.current_pose = lerp_pose(self.transition_start_pose, self.transition_target_pose, alpha)
        if alpha >= 1.0:
            self.current_pose = list(self.transition_target_pose)
            self.target_pose = list(self.current_pose)
            self.transition_active = False

    def _ramp_value(self, current: float, target: float) -> float:
        step = COMMAND_RAMP_RATE * CONTROL_DT
        if abs(target - current) <= step:
            return target
        return current + step if target > current else current - step

    def _regressed_pose_locked(self):
        base = np.asarray(self._blend_height(self.height_offset_m), dtype=np.float64)
        self.cmd_forward = self._ramp_value(self.cmd_forward, self.forward)
        self.cmd_lateral = self._ramp_value(self.cmd_lateral, self.lateral)
        self.cmd_turn = self._ramp_value(self.cmd_turn, self.turn)
        if self.model is None or not self.model.valid:
            self.target_pose = list(base)
            return list(base)

        delta = self.model.predict_delta(self.cmd_forward, self.cmd_lateral, self.cmd_turn) * self.pose_gain

        command_mag = max(abs(self.cmd_forward), abs(self.cmd_lateral), abs(self.cmd_turn))
        label = "idle"
        if abs(self.cmd_forward) >= abs(self.cmd_lateral) and abs(self.cmd_forward) >= abs(self.cmd_turn) and abs(self.cmd_forward) > 0.05:
            label = "forward" if self.cmd_forward > 0 else "backward"
        elif abs(self.cmd_lateral) >= abs(self.cmd_turn) and abs(self.cmd_lateral) > 0.05:
            label = "right" if self.cmd_lateral > 0 else "left"
        elif abs(self.cmd_turn) > 0.05:
            label = "turn_right" if self.cmd_turn > 0 else "turn_left"

        amp = self.model.amplitude_for_label(label) * min(1.0, 1.5 * command_mag)
        if command_mag > 0.05:
            self.step_phase = (self.step_phase + CONTROL_DT * (3.5 + 1.5 * command_mag)) % (2.0 * np.pi)
        else:
            self.step_phase = 0.0

        osc = np.zeros(12, dtype=np.float64)
        if command_mag > 0.05:
            front_group = np.sin(self.step_phase)
            rear_group = np.sin(self.step_phase + np.pi)
            side_group = np.sin(self.step_phase + np.pi / 2.0)
            for hip_idx, thigh_idx, calf_idx, group in (
                (0, 1, 2, front_group),
                (3, 4, 5, -front_group),
                (6, 7, 8, rear_group),
                (9, 10, 11, -rear_group),
            ):
                osc[hip_idx] += 0.7 * amp[hip_idx] * group * (1.0 if abs(self.cmd_forward) >= abs(self.cmd_lateral) else side_group)
                lift = max(0.0, group)
                osc[thigh_idx] += -0.8 * amp[thigh_idx] * lift
                osc[calf_idx] += 0.9 * amp[calf_idx] * lift

            if abs(self.cmd_lateral) > abs(self.cmd_forward):
                osc[0] += 0.5 * amp[0] * side_group
                osc[3] += 0.5 * amp[3] * side_group
                osc[6] -= 0.5 * amp[6] * side_group
                osc[9] -= 0.5 * amp[9] * side_group
            if abs(self.cmd_turn) > 0.05:
                turn_sign = np.sign(self.cmd_turn)
                osc[0] += 0.4 * amp[0] * turn_sign
                osc[3] -= 0.4 * amp[3] * turn_sign
                osc[6] += 0.2 * amp[6] * turn_sign
                osc[9] -= 0.2 * amp[9] * turn_sign

        pose = base + delta + osc
        self.target_pose = list(base)
        return self._apply_zmp_stabilizer_locked(pose, base)

    def _apply_zmp_stabilizer_locked(self, pose, base):
        q = np.asarray(pose, dtype=np.float64).copy()

        # Approximate ZMP/support correction from IMU tilt and foot-force imbalance.
        ff = self.foot_force
        front_force = ff[0] + ff[1]
        rear_force = ff[2] + ff[3]
        left_force = ff[1] + ff[3]
        right_force = ff[0] + ff[2]

        u_pitch = -(ZMP_PITCH_P * self.pitch + ZMP_PITCH_D * self.gy)
        u_roll = -(ZMP_ROLL_P * self.roll + ZMP_ROLL_D * self.gx)

        # Use force imbalance as a proxy for COM/ZMP distance from the support center.
        u_pitch += ZMP_FORCE_GAIN * (rear_force - front_force)
        u_roll += ZMP_FORCE_GAIN * (right_force - left_force)

        u_pitch = clamp(u_pitch, -ZMP_EXT_CLAMP, ZMP_EXT_CLAMP)
        u_roll = clamp(u_roll, -ZMP_EXT_CLAMP, ZMP_EXT_CLAMP)

        leg_groups = {
            "FR": (0, 1, 2),
            "FL": (3, 4, 5),
            "RR": (6, 7, 8),
            "RL": (9, 10, 11),
        }

        def apply_extension(leg, ext):
            hip, thigh, calf = leg_groups[leg]
            q[thigh] += ZMP_THIGH_GAIN * ext
            q[calf] += ZMP_CALF_GAIN * ext

        for leg in ("FR", "FL"):
            apply_extension(leg, +u_pitch)
        for leg in ("RR", "RL"):
            apply_extension(leg, -u_pitch)
        for leg in ("FL", "RL"):
            apply_extension(leg, +u_roll)
        for leg in ("FR", "RR"):
            apply_extension(leg, -u_roll)

        # Clamp the stabilizer around the commanded pose so it cannot run away.
        q = np.clip(q, base - 0.50, base + 0.50)
        return q.tolist()

    def _control_loop(self):
        with self.lock:
            if not self.low_level_enabled or self.low_state is None:
                return
            if self.first_run:
                self.current_pose = [self.low_state.motor_state[i].q for i in range(12)]
                self.target_pose = list(self.current_pose)
                self.transition_start_pose = list(self.current_pose)
                self.transition_target_pose = self._blend_height(self.height_offset_m)
                self.transition_started = time.time()
                self.transition_duration = 1.2
                self.transition_active = True
                self.first_run = False

            self._apply_transition_locked()
            if self.sitting and not self.transition_active:
                pose = list(self.sit_pose)
            elif self.transition_active:
                pose = list(self.current_pose)
            else:
                pose = self._regressed_pose_locked()

        for i in range(12):
            self.low_cmd.motor_cmd[i].q = pose[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = KP
            self.low_cmd.motor_cmd[i].kd = KD
            self.low_cmd.motor_cmd[i].tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


def draw(stdscr, controller: PoseController):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    snap = controller.snapshot_data()
    label_counts = snap["label_counts"]
    labels = " ".join(f"{k}:{v}" for k, v in sorted(label_counts.items())) if label_counts else "--"
    lines = [
        "Pose CLI",
        "Arrows: forward/back + turn  a/d: lateral left/right  +/-: height  s: sit  n: stand  q: quit",
        "",
        f"Sit hold: {snap['sitting']}",
        f"Height offset: {snap['height_offset_m']:+.3f} m",
        f"Command forward/lateral/turn: {snap['forward']:+.2f} {snap['lateral']:+.2f} {snap['turn']:+.2f}",
        f"IMU roll/pitch: {snap['roll']:+.3f} {snap['pitch']:+.3f}",
        f"Foot force: {' '.join(str(int(v)) for v in snap['foot_force'])}",
        f"LowState age: {snap['low_state_age']}",
        f"Status: {snap['status']} (age {snap['status_age']})",
        f"Model loaded: {snap['model_valid']}",
        f"Model path: {os.path.basename(snap['model_path']) if snap['model_path'] else '--'}",
        f"Model error: {snap['model_error'] or '--'}",
        f"Recorded labels: {labels}",
    ]
    for idx, line in enumerate(lines[: max(0, h - 1)]):
        stdscr.addnstr(idx, 0, line, max(0, w - 1))
    stdscr.refresh()


def tui_main(stdscr, controller: PoseController):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    while True:
        draw(stdscr, controller)
        key = stdscr.getch()
        forward = 0.0
        lateral = 0.0
        turn = 0.0
        if key == curses.KEY_UP:
            forward = 1.0
        elif key == curses.KEY_DOWN:
            forward = -1.0
        elif key == curses.KEY_LEFT:
            turn = 1.0
        elif key == curses.KEY_RIGHT:
            turn = -1.0
        elif key in (ord("a"), ord("A")):
            lateral = -1.0
        elif key in (ord("d"), ord("D")):
            lateral = 1.0
        elif key in (ord("+"), ord("=")):
            controller.increase_height()
        elif key in (ord("-"), ord("_")):
            controller.decrease_height()
        elif key in (ord("s"), ord("S")):
            controller.sit()
        elif key in (ord("n"), ord("N")):
            controller.stand()
        elif key in (ord("q"), ord("Q")):
            break
        controller.set_command(forward, lateral, turn)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Low-level joint pose controller learned from recorded joystick-command gait data."
    )
    parser.add_argument("--iface", required=True, help="Robot network interface")
    parser.add_argument("--recording", default=find_latest_recording(), help="record_gait JSONL to fit regression from")
    return parser.parse_args()


def main():
    args = parse_args()
    print("WARNING: Ensure the robot has clearance before running low-level pose control.")
    input("Press Enter to continue...")
    ChannelFactoryInitialize(0, args.iface)
    model = PoseRegressor(args.recording) if args.recording else None
    controller = PoseController(model)
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
