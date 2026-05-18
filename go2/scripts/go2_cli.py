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

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LidarState_, LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

import unitree_legged_const as go2


TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_ODOM = "rt/odom"
TOPIC_LIDAR_STATE = "rt/utlidar/map_state"
TOPIC_LIDAR_CLOUD = "rt/utlidar/cloud"

CONTROL_DT = 0.002
POSTURE_BLEND_DT = 0.02
NORMAL_HEIGHT_M = 0.0
HEIGHT_STEP_M = 0.01
LOWERED_OFFSET_M = -0.05
MIN_HEIGHT_OFFSET_M = -0.10
MAX_HEIGHT_OFFSET_M = 0.06
COMMAND_RAMP_RATE = 1.8

LOWLEVEL_STOP_WAIT = 3.0
SERVICE_RESTART_WAIT = 2.0
SERVICES_TO_ENABLE = ("mcf", "sport_mode")
MODE_ALIASES = ("normal", "mcf")

LEG_INDEX = {
    "FR": (0, 1, 2),
    "FL": (3, 4, 5),
    "RR": (6, 7, 8),
    "RL": (9, 10, 11),
}
LEG_ORDER = ["FR", "FL", "RR", "RL"]

GAITS = {
    "Walk": {
        "description": "Four-beat lateral walk: RL -> FL -> RR -> FR",
        "phase_offsets": {"RL": 0.00, "FL": 0.25, "RR": 0.50, "FR": 0.75},
        "duty": 0.74,
        "cycle_sec": 1.45,
        "step_height": 0.32,
        "step_length": 0.32,
        "body_roll": 0.05,
    },
    "Trot": {
        "description": "Diagonal trot",
        "phase_offsets": {"FR": 0.00, "RL": 0.00, "FL": 0.50, "RR": 0.50},
        "duty": 0.54,
        "cycle_sec": 1.05,
        "step_height": 0.28,
        "step_length": 0.28,
        "body_roll": 0.03,
    },
}


class RecordedGait:
    def __init__(self, path: str):
        self.path = path
        self.frames = []
        self.reference_pose = [0.0] * 12
        self.duration = 0.0
        self.valid = False
        self.error = ""
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw_frames = [json.loads(line) for line in f if line.strip()]
        except Exception as exc:
            self.error = str(exc)
            return

        if len(raw_frames) < 2:
            self.error = "recording has fewer than 2 frames"
            return

        t0 = float(raw_frames[0]["wall_time"])
        q_sum = [0.0] * 12
        parsed = []
        for frame in raw_frames:
            joints = frame.get("joints", [])
            if len(joints) < 12:
                continue
            q = [float(joints[i]["q"]) for i in range(12)]
            dq = [float(joints[i].get("dq", 0.0)) for i in range(12)]
            t = max(0.0, float(frame["wall_time"]) - t0)
            parsed.append({"t": t, "q": q, "dq": dq})
            for i in range(12):
                q_sum[i] += q[i]

        if len(parsed) < 2:
            self.error = "recording has too few valid joint frames"
            return

        count = float(len(parsed))
        self.reference_pose = [v / count for v in q_sum]
        self.frames = parsed
        self.duration = max(parsed[-1]["t"], CONTROL_DT)
        self.valid = True

    def sample(self, phase_time: float):
        if not self.valid:
            return None

        t = phase_time % self.duration
        frames = self.frames
        lo = frames[0]
        hi = frames[-1]
        for idx in range(1, len(frames)):
            if frames[idx]["t"] >= t:
                lo = frames[idx - 1]
                hi = frames[idx]
                break

        span = max(hi["t"] - lo["t"], 1e-6)
        alpha = clamp((t - lo["t"]) / span, 0.0, 1.0)
        q = [(1.0 - alpha) * lo["q"][i] + alpha * hi["q"][i] for i in range(12)]
        dq = [(1.0 - alpha) * lo["dq"][i] + alpha * hi["dq"][i] for i in range(12)]
        delta = [q[i] - self.reference_pose[i] for i in range(12)]
        return {"q": q, "dq": dq, "delta": delta}


def find_latest_recording():
    candidates = sorted(glob.glob("record_gait_*.jsonl"))
    if not candidates:
        return None
    return candidates[-1]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def lerp_pose(src, dst, alpha: float):
    return [(1.0 - alpha) * a + alpha * b for a, b in zip(src, dst)]


def vec3(values):
    return f"{values[0]: .2f} {values[1]: .2f} {values[2]: .2f}"


def quat_to_yaw(quat) -> float:
    x, y, z, w = [float(v) for v in quat]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def extract_point_count(msg: Optional[PointCloud2_]) -> int:
    if msg is None:
        return 0
    return int(msg.width) * int(msg.height)


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


@dataclass
class TopicSnapshot:
    low_state: Optional[LowState_] = None
    odom: Optional[Odometry_] = None
    lidar_state: Optional[LidarState_] = None
    lidar_cloud: Optional[PointCloud2_] = None
    low_state_time: float = 0.0
    odom_time: float = 0.0
    lidar_state_time: float = 0.0
    lidar_cloud_time: float = 0.0
    motion_mode: str = "unknown"
    motion_code: int = 0
    motion_time: float = 0.0
    motion_error: str = ""
    service_status: str = "idle"
    service_time: float = 0.0


class Go2CliController:
    def __init__(self, recorded_gait: Optional[RecordedGait] = None):
        self.Kp = 60.0
        self.Kd = 5.0
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self.lock = threading.Lock()
        self.snapshots = TopicSnapshot()
        self.first_run = True
        self.recover_in_progress = False

        self.stand_pose = [
            0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
            0.0, 0.67, -1.3, 0.0, 0.67, -1.3,
        ]
        self.sit_pose = [
            0.0, 1.36, -2.65, 0.0, 1.36, -2.65,
            -0.2, 1.36, -2.65, 0.2, 1.36, -2.65,
        ]
        self.height_pose = [
            -0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
            -0.5, 1.36, -2.65, 0.5, 1.36, -2.65,
        ]
        self.start_pose = [0.0] * 12
        self.current_pose = list(self.sit_pose)
        self.target_pose = list(self.sit_pose)
        self.height_offset_m = NORMAL_HEIGHT_M
        self.is_sitting = True
        self.manual_override = False
        self.walk_enabled = False
        self.gait_name = "Walk"
        self.phase = 0.0
        self.move_x = 0.0
        self.move_yaw = 0.0
        self.command_move_x = 0.0
        self.command_move_yaw = 0.0
        self.low_level_enabled = True
        self.recorded_gait = recorded_gait
        self.recorded_gait_speed = 1.0
        self.recorded_phase_time = 0.0

        self.sequence = [
            {"pose": list(self.stand_pose), "duration": 1.2, "hold": 0.5},
            {"pose": self._blend_height(LOWERED_OFFSET_M), "duration": 0.8, "hold": 0.5},
            {"pose": list(self.stand_pose), "duration": 0.8, "hold": 0.4},
        ]
        self.sequence_done = False
        self.sequence_index = 0
        self.sequence_hold_started = 0.0
        self.transition_start_pose = list(self.sit_pose)
        self.transition_target_pose = list(self.sit_pose)
        self.transition_started = 0.0
        self.transition_duration = 1.0
        self.transition_active = False

        self.lowCmdWriteThreadPtr = None
        self.modePollThreadPtr = None

    def init(self, odom_topic: str, lidar_state_topic: str, lidar_cloud_topic: str):
        self._init_low_cmd()

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

        self.odom_subscriber = ChannelSubscriber(odom_topic, Odometry_)
        self.odom_subscriber.Init(self._odom_handler, 10)

        self.lidar_state_subscriber = ChannelSubscriber(lidar_state_topic, LidarState_)
        self.lidar_state_subscriber.Init(self._lidar_state_handler, 10)

        self.lidar_cloud_subscriber = ChannelSubscriber(lidar_cloud_topic, PointCloud2_)
        self.lidar_cloud_subscriber.Init(self._lidar_cloud_handler, 10)

        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        self.robot_state = RobotStateClient()
        self.robot_state.SetTimeout(5.0)
        self.robot_state.Init()

        self._release_motion_mode()

    def start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=CONTROL_DT, target=self._low_cmd_write, name="go2_cli_lowcmd"
        )
        self.lowCmdWriteThreadPtr.Start()
        self.modePollThreadPtr = RecurrentThread(
            interval=0.5, target=self._poll_motion_mode, name="go2_cli_mode_poll"
        )
        self.modePollThreadPtr.Start()

    def stop(self):
        with self.lock:
            self.low_level_enabled = False
        if self.lowCmdWriteThreadPtr is not None:
            self.lowCmdWriteThreadPtr.Wait(1.0)
        if self.modePollThreadPtr is not None:
            self.modePollThreadPtr.Wait(1.0)
        try:
            self.sc.StandDown()
        except Exception:
            pass

    def increase_height(self):
        with self.lock:
            self._cancel_sequence_locked()
            self.is_sitting = False
            self.height_offset_m = clamp(
                self.height_offset_m + HEIGHT_STEP_M, MIN_HEIGHT_OFFSET_M, MAX_HEIGHT_OFFSET_M
            )
            self._begin_transition_locked(self._blend_height(self.height_offset_m), 0.35)

    def decrease_height(self):
        with self.lock:
            self._cancel_sequence_locked()
            self.is_sitting = False
            self.height_offset_m = clamp(
                self.height_offset_m - HEIGHT_STEP_M, MIN_HEIGHT_OFFSET_M, MAX_HEIGHT_OFFSET_M
            )
            self._begin_transition_locked(self._blend_height(self.height_offset_m), 0.35)

    def command_stand(self):
        with self.lock:
            self._cancel_sequence_locked()
            self.is_sitting = False
            self.walk_enabled = False
            self.move_x = 0.0
            self.move_yaw = 0.0
            self.height_offset_m = NORMAL_HEIGHT_M
            self._begin_transition_locked(self._blend_height(self.height_offset_m), 0.6)

    def command_sit(self):
        with self.lock:
            self._cancel_sequence_locked()
            self.is_sitting = True
            self.walk_enabled = False
            self.move_x = 0.0
            self.move_yaw = 0.0
            self._begin_transition_locked(list(self.sit_pose), 0.9)

    def toggle_walk(self):
        with self.lock:
            self._cancel_sequence_locked()
            self.is_sitting = False
            self.walk_enabled = not self.walk_enabled
            if not self.walk_enabled:
                self.move_x = 0.0
                self.move_yaw = 0.0
                self.command_move_x = 0.0
                self.command_move_yaw = 0.0
                self.phase = 0.0
                self.target_pose = list(self._blend_height(self.height_offset_m))
            elif self.transition_active:
                self.transition_target_pose = list(self._blend_height(self.height_offset_m))

    def cycle_gait(self):
        with self.lock:
            gait_names = list(GAITS.keys())
            if self.recorded_gait is not None and self.recorded_gait.valid:
                gait_names.append("Recorded")
            idx = gait_names.index(self.gait_name)
            self.gait_name = gait_names[(idx + 1) % len(gait_names)]
            self.phase = 0.0
            self.recorded_phase_time = 0.0

    def set_walk_command(self, move_x: float, move_yaw: float):
        with self.lock:
            if not self.walk_enabled:
                self.move_x = 0.0
                self.move_yaw = 0.0
                return
            self.move_x = clamp(move_x, -1.0, 1.0)
            self.move_yaw = clamp(move_yaw, -1.0, 1.0)

    def trigger_recover_mcf(self):
        with self.lock:
            if self.recover_in_progress:
                return
            self.recover_in_progress = True
            self.walk_enabled = False
            self.move_x = 0.0
            self.move_yaw = 0.0
            self.command_move_x = 0.0
            self.command_move_yaw = 0.0
            self.phase = 0.0
            self.is_sitting = True
            self._cancel_sequence_locked()
            self._begin_transition_locked(list(self.sit_pose), 0.9)
            self.snapshots.service_status = "recovering"
            self.snapshots.service_time = time.time()

        thread = threading.Thread(target=self._recover_mcf_worker, name="go2_cli_recover", daemon=True)
        thread.start()

    def get_snapshot(self):
        with self.lock:
            return {
                "low_state": self.snapshots.low_state,
                "odom": self.snapshots.odom,
                "lidar_state": self.snapshots.lidar_state,
                "lidar_cloud": self.snapshots.lidar_cloud,
                "low_state_age": self._age(self.snapshots.low_state_time),
                "odom_age": self._age(self.snapshots.odom_time),
                "lidar_state_age": self._age(self.snapshots.lidar_state_time),
                "lidar_cloud_age": self._age(self.snapshots.lidar_cloud_time),
                "motion_mode": self.snapshots.motion_mode,
                "motion_code": self.snapshots.motion_code,
                "motion_age": self._age(self.snapshots.motion_time),
                "motion_error": self.snapshots.motion_error,
                "height_offset_m": self.height_offset_m,
                "is_sitting": self.is_sitting,
                "sequence_done": self.sequence_done,
                "manual_override": self.manual_override,
                "service_status": self.snapshots.service_status,
                "service_age": self._age(self.snapshots.service_time),
                "recover_in_progress": self.recover_in_progress,
                "walk_enabled": self.walk_enabled,
                "gait_name": self.gait_name,
                "gait_description": self._gait_description(),
                "move_x": self.move_x,
                "move_yaw": self.move_yaw,
                "recorded_gait_loaded": self.recorded_gait is not None and self.recorded_gait.valid,
                "recorded_gait_path": self.recorded_gait.path if self.recorded_gait is not None else "",
                "recorded_gait_error": "" if self.recorded_gait is None else self.recorded_gait.error,
            }

    def _age(self, ts: float) -> Optional[float]:
        if ts <= 0.0:
            return None
        return max(0.0, time.time() - ts)

    def _blend_height(self, offset_m: float):
        alpha = clamp(
            (offset_m - MIN_HEIGHT_OFFSET_M) / (MAX_HEIGHT_OFFSET_M - MIN_HEIGHT_OFFSET_M), 0.0, 1.0
        )
        return lerp_pose(self.height_pose, self.stand_pose, alpha)

    def _begin_transition_locked(self, pose, duration: float):
        self.transition_start_pose = list(self.current_pose)
        self.transition_target_pose = list(pose)
        self.transition_started = time.time()
        self.transition_duration = max(duration, POSTURE_BLEND_DT)
        self.transition_active = True
        self.target_pose = list(pose)

    def _gait_description(self):
        if self.gait_name == "Recorded":
            if self.recorded_gait is None:
                return "Recorded gait unavailable"
            if not self.recorded_gait.valid:
                return f"Recorded gait error: {self.recorded_gait.error}"
            return f"Playback from {os.path.basename(self.recorded_gait.path)} with fixed user height"
        return GAITS[self.gait_name]["description"]

    def _cancel_sequence_locked(self):
        self.manual_override = True
        self.sequence_done = True
        self.sequence_index = len(self.sequence)
        self.sequence_hold_started = 0.0

    def _advance_sequence_locked(self):
        if self.manual_override or self.sequence_done:
            return
        if self.sequence_index >= len(self.sequence):
            self.sequence_done = True
            self.is_sitting = False
            self.height_offset_m = NORMAL_HEIGHT_M
            self.target_pose = list(self.stand_pose)
            return
        if not self.transition_active and self.sequence_hold_started == 0.0:
            step = self.sequence[self.sequence_index]
            self._begin_transition_locked(step["pose"], step["duration"])
            return
        if self.transition_active:
            return
        if self.sequence_hold_started == 0.0:
            self.sequence_hold_started = time.time()
            return
        step = self.sequence[self.sequence_index]
        if time.time() - self.sequence_hold_started >= step["hold"]:
            self.sequence_index += 1
            self.sequence_hold_started = 0.0

    def _apply_transition_locked(self):
        if not self.transition_active:
            return
        elapsed = time.time() - self.transition_started
        alpha = clamp(elapsed / self.transition_duration, 0.0, 1.0)
        self.current_pose = lerp_pose(self.transition_start_pose, self.transition_target_pose, alpha)
        if alpha >= 1.0:
            self.current_pose = list(self.transition_target_pose)
            self.transition_active = False
            self.target_pose = list(self.current_pose)

    def _ramp_value(self, current: float, target: float, rate: float) -> float:
        step = rate * CONTROL_DT
        if abs(target - current) <= step:
            return target
        return current + step if target > current else current - step

    def _stance_swing_value(self, leg_phase: float, duty: float):
        if leg_phase < duty:
            stance_phase = leg_phase / duty
            return 1.0 - 2.0 * stance_phase, 0.0
        swing_phase = (leg_phase - duty) / (1.0 - duty)
        sweep = -1.0 + 2.0 * swing_phase
        lift = math.sin(math.pi * swing_phase)
        return sweep, lift

    def _gait_target_locked(self):
        if self.gait_name == "Recorded":
            return self._recorded_gait_target_locked()

        gait = GAITS[self.gait_name]
        self.command_move_x = self._ramp_value(self.command_move_x, self.move_x, COMMAND_RAMP_RATE)
        self.command_move_yaw = self._ramp_value(
            self.command_move_yaw, self.move_yaw, COMMAND_RAMP_RATE * 1.4
        )

        base = list(self._blend_height(self.height_offset_m))
        move_mag = max(abs(self.command_move_x), abs(self.command_move_yaw))
        if not self.walk_enabled or move_mag < 0.03:
            self.phase = 0.0
            return base

        cycle_sec = gait["cycle_sec"]
        duty = gait["duty"]
        self.phase = (self.phase + CONTROL_DT / cycle_sec) % 1.0

        step_length = gait["step_length"] * abs(self.command_move_x)
        step_height = gait["step_height"] * max(abs(self.command_move_x), 0.45 * abs(self.command_move_yaw))
        turn_amount = 0.18 * self.command_move_yaw
        body_roll = gait["body_roll"] * abs(self.command_move_x)

        q = list(base)
        for leg in LEG_ORDER:
            hip_idx, thigh_idx, calf_idx = LEG_INDEX[leg]
            leg_phase = (self.phase + gait["phase_offsets"][leg]) % 1.0
            sweep, lift = self._stance_swing_value(leg_phase, duty)

            side_sign = 1.0 if leg in ("FL", "RL") else -1.0
            front_sign = 1.0 if leg in ("FR", "FL") else -1.0

            hip_delta = -sweep * step_length * self.command_move_x
            hip_delta += side_sign * turn_amount * (0.7 if front_sign > 0 else 1.0)

            # Lift the thigh and fold the calf enough to unload the foot.
            thigh_delta = -(0.70 * step_height) * lift
            calf_delta = (1.35 * step_height) * lift

            # Add a small body-roll bias during stance for load transfer.
            if lift < 0.05:
                thigh_delta += side_sign * body_roll
                calf_delta -= 0.5 * side_sign * body_roll

            q[hip_idx] += hip_delta
            q[thigh_idx] += thigh_delta
            q[calf_idx] += calf_delta

        self.target_pose = list(base)
        return q

    def _recorded_gait_target_locked(self):
        base = list(self._blend_height(self.height_offset_m))
        if self.recorded_gait is None or not self.recorded_gait.valid:
            self.target_pose = list(base)
            return base

        self.command_move_x = self._ramp_value(self.command_move_x, self.move_x, COMMAND_RAMP_RATE)
        self.command_move_yaw = self._ramp_value(
            self.command_move_yaw, self.move_yaw, COMMAND_RAMP_RATE * 1.4
        )
        drive = max(abs(self.command_move_x), abs(self.command_move_yaw))
        if not self.walk_enabled or drive < 0.03:
            self.recorded_phase_time = 0.0
            self.target_pose = list(base)
            return base

        self.recorded_phase_time += CONTROL_DT * max(0.35, abs(self.command_move_x)) * self.recorded_gait_speed
        sampled = self.recorded_gait.sample(self.recorded_phase_time)
        if sampled is None:
            self.target_pose = list(base)
            return base

        # Apply the recorded joint deltas around the user-selected stand height
        # so playback preserves the commanded body height.
        turn_mix = self.command_move_yaw
        q = list(base)
        for leg in LEG_ORDER:
            hip_idx, thigh_idx, calf_idx = LEG_INDEX[leg]
            front_sign = 1.0 if leg in ("FR", "FL") else -1.0
            side_sign = 1.0 if leg in ("FL", "RL") else -1.0

            hip_delta = sampled["delta"][hip_idx] * self.command_move_x
            hip_delta += 0.10 * turn_mix * side_sign * (0.7 if front_sign > 0 else 1.0)
            thigh_delta = sampled["delta"][thigh_idx] * max(abs(self.command_move_x), 0.6 * abs(turn_mix))
            calf_delta = sampled["delta"][calf_idx] * max(abs(self.command_move_x), 0.6 * abs(turn_mix))

            q[hip_idx] += hip_delta
            q[thigh_idx] += thigh_delta
            q[calf_idx] += calf_delta

        self.target_pose = list(base)
        return q

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
            try:
                self.sc.StandDown()
            except Exception:
                pass
            self.msc.ReleaseMode()
            time.sleep(1.0)

    def _low_state_handler(self, msg: LowState_):
        with self.lock:
            self.low_state = msg
            self.snapshots.low_state = msg
            self.snapshots.low_state_time = time.time()

    def _odom_handler(self, msg: Odometry_):
        with self.lock:
            self.snapshots.odom = msg
            self.snapshots.odom_time = time.time()

    def _lidar_state_handler(self, msg: LidarState_):
        with self.lock:
            self.snapshots.lidar_state = msg
            self.snapshots.lidar_state_time = time.time()

    def _lidar_cloud_handler(self, msg: PointCloud2_):
        with self.lock:
            self.snapshots.lidar_cloud = msg
            self.snapshots.lidar_cloud_time = time.time()

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
            if not self.low_level_enabled:
                return
            if self.low_state is None:
                return
            if self.first_run:
                for i in range(12):
                    self.start_pose[i] = self.low_state.motor_state[i].q
                self.current_pose = list(self.start_pose)
                self.transition_start_pose = list(self.start_pose)
                self.transition_target_pose = list(self.stand_pose)
                self.first_run = False
            self._advance_sequence_locked()
            self._apply_transition_locked()
            if self.transition_active:
                pose = list(self.current_pose)
            elif self.walk_enabled and not self.is_sitting:
                pose = self._gait_target_locked()
            else:
                self.command_move_x = self._ramp_value(self.command_move_x, 0.0, COMMAND_RAMP_RATE)
                self.command_move_yaw = self._ramp_value(self.command_move_yaw, 0.0, COMMAND_RAMP_RATE * 1.4)
                pose = list(self.target_pose)

        for i in range(12):
            self.low_cmd.motor_cmd[i].q = pose[i]
            self.low_cmd.motor_cmd[i].dq = 0
            self.low_cmd.motor_cmd[i].kp = self.Kp
            self.low_cmd.motor_cmd[i].kd = self.Kd
            self.low_cmd.motor_cmd[i].tau = 0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def _try_stop_lowlevel(self, duration_sec: float):
        pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        pub.Init()

        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0
        for i in range(20):
            cmd.motor_cmd[i].mode = 0x00
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0

        start = time.time()
        while time.time() - start < duration_sec:
            cmd.crc = self.crc.Crc(cmd)
            pub.Write(cmd)
            time.sleep(0.02)

    def _release_mode_for_recovery(self):
        self.msc.ReleaseMode()
        time.sleep(1.0)

    def _reacquire_motion_mode(self):
        code, data = self.msc.CheckMode()
        if code == 0 and data is not None and data.get("name"):
            return

        for alias in MODE_ALIASES:
            code, _ = self.msc.SelectMode(alias)
            if code != 0:
                continue
            time.sleep(1.0)
            check_code, check_data = self.msc.CheckMode()
            if check_code == 0 and check_data is not None and check_data.get("name"):
                return

    def _ensure_services_enabled(self):
        code, services = self.robot_state.ServiceList()
        if code != 0 or services is None:
            raise RuntimeError(f"ServiceList failed with code {code}")

        by_name = {service.name: service for service in services}
        for service_name in SERVICES_TO_ENABLE:
            service = by_name.get(service_name)
            if service is not None and service.status == 1:
                continue
            self.robot_state.ServiceSwitch(service_name, False)
            time.sleep(0.5)
            switch_code = self.robot_state.ServiceSwitch(service_name, True)
            if switch_code != 0:
                raise RuntimeError(f"ServiceSwitch({service_name}, on) failed with code {switch_code}")
            time.sleep(SERVICE_RESTART_WAIT)

    def _recover_mcf_worker(self):
        try:
            with self.lock:
                self.snapshots.service_status = "sitting before recovery"
                self.snapshots.service_time = time.time()
            time.sleep(1.2)

            with self.lock:
                self.snapshots.service_status = "stopping low-level publisher"
                self.snapshots.service_time = time.time()
                self.low_level_enabled = False
            time.sleep(0.1)

            with self.lock:
                self.snapshots.service_status = "neutralizing low-level control"
                self.snapshots.service_time = time.time()
            self._try_stop_lowlevel(LOWLEVEL_STOP_WAIT)

            with self.lock:
                self.snapshots.service_status = "releasing motion mode"
                self.snapshots.service_time = time.time()
            self._release_mode_for_recovery()

            with self.lock:
                self.snapshots.service_status = "enabling mcf/sport_mode"
                self.snapshots.service_time = time.time()
            self._ensure_services_enabled()

            with self.lock:
                self.snapshots.service_status = "reacquiring motion mode"
                self.snapshots.service_time = time.time()
            self._reacquire_motion_mode()

            with self.lock:
                self.snapshots.service_status = "recovery complete"
                self.snapshots.service_time = time.time()
        except Exception as exc:
            with self.lock:
                self.snapshots.service_status = f"recovery failed: {exc}"
                self.snapshots.service_time = time.time()
        finally:
            with self.lock:
                self.low_level_enabled = True
                self.is_sitting = True
                self.target_pose = list(self.sit_pose)
                self.current_pose = list(self.sit_pose)
                self.transition_active = False
                self.recover_in_progress = False


def draw_panel(stdscr, controller: Go2CliController, odom_topic: str, lidar_state_topic: str, lidar_cloud_topic: str):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    snapshot = controller.get_snapshot()
    low_state = snapshot["low_state"]
    odom = snapshot["odom"]
    lidar_state = snapshot["lidar_state"]
    lidar_cloud = snapshot["lidar_cloud"]

    lines = [
        "Go2 CLI",
        "Height: +/- or Shift+Up/Shift+Down  Walk: w toggle, arrows drive, g gait, s sit, n stand, r recover, q quit",
        "",
        f"Height offset: {snapshot['height_offset_m']:+.3f} m",
        f"Mode: {'sitting' if snapshot['is_sitting'] else 'standing'}",
        f"Startup sequence: {'manual override' if snapshot['manual_override'] else ('done' if snapshot['sequence_done'] else 'running')}",
        f"Walk mode: {'enabled' if snapshot['walk_enabled'] else 'disabled'}  gait={snapshot['gait_name']}",
        f"Walk cmd: x={snapshot['move_x']:+.1f}  yaw={snapshot['move_yaw']:+.1f}",
        f"Gait info: {snapshot['gait_description']}",
        f"Recorded gait: {os.path.basename(snapshot['recorded_gait_path']) if snapshot['recorded_gait_path'] else 'none'}",
        f"Service action: {snapshot['service_status']} (age {age_text(snapshot['service_age'])})",
        "",
        f"LowState  {status_text(snapshot['low_state_age'])}  age={age_text(snapshot['low_state_age'])}  topic={TOPIC_LOWSTATE}",
    ]
    if snapshot["recorded_gait_error"]:
        lines.insert(8, f"Recorded gait error: {snapshot['recorded_gait_error']}")

    if low_state is not None:
        imu = low_state.imu_state
        lines.extend(
            [
                f"  Power V/A: {float(low_state.power_v):.2f} / {float(low_state.power_a):.2f}",
                f"  IMU rpy:   {vec3([float(v) for v in imu.rpy])}",
                f"  IMU gyro:  {vec3([float(v) for v in imu.gyroscope])}",
                f"  IMU acc:   {vec3([float(v) for v in imu.accelerometer])}",
                f"  Foot force:{' '.join(str(int(v)) for v in low_state.foot_force)}",
            ]
        )
    else:
        lines.append("  waiting for rt/lowstate")

    lines.extend(
        [
            "",
            f"Odometry  {status_text(snapshot['odom_age'])}  age={age_text(snapshot['odom_age'])}  topic={odom_topic}",
        ]
    )
    if odom is not None:
        pos = odom.pose.pose.position
        quat = odom.pose.pose.orientation
        lin = odom.twist.twist.linear
        ang = odom.twist.twist.angular
        yaw = quat_to_yaw([quat.x, quat.y, quat.z, quat.w])
        lines.extend(
            [
                f"  Pos xyz:  {pos.x: .2f} {pos.y: .2f} {pos.z: .2f}",
                f"  Yaw deg:  {yaw: .1f}",
                f"  Lin vel:  {lin.x: .2f} {lin.y: .2f} {lin.z: .2f}",
                f"  Ang vel:  {ang.x: .2f} {ang.y: .2f} {ang.z: .2f}",
            ]
        )
    else:
        lines.append("  waiting for rt/odom")

    lines.extend(
        [
            "",
            f"UTLiDAR state  {status_text(snapshot['lidar_state_age'])}  age={age_text(snapshot['lidar_state_age'])}  topic={lidar_state_topic}",
        ]
    )
    if lidar_state is not None:
        lines.extend(
            [
                f"  Cloud size: {int(lidar_state.cloud_size)}",
                f"  Cloud freq: {float(lidar_state.cloud_frequency):.2f} Hz",
                f"  Cloud loss: {float(lidar_state.cloud_packet_loss_rate):.3f}",
                f"  IMU rpy:    {vec3([float(v) for v in lidar_state.imu_rpy])}",
            ]
        )
    else:
        lines.append("  waiting for utlidar state")

    lines.extend(
        [
            "",
            f"UTLiDAR cloud  {status_text(snapshot['lidar_cloud_age'])}  age={age_text(snapshot['lidar_cloud_age'])}  topic={lidar_cloud_topic}",
            f"  Points: {extract_point_count(lidar_cloud)}",
            "",
            f"Motion switcher  age={age_text(snapshot['motion_age'])}",
            f"  mode={snapshot['motion_mode']}  code={snapshot['motion_code']}",
        ]
    )
    if snapshot["motion_error"]:
        lines.append(f"  error={snapshot['motion_error']}")

    if h < 20 or w < 90:
        lines.append("")
        lines.append("Terminal is small; widen it for the full panel.")

    for idx, line in enumerate(lines[: max(0, h - 1)]):
        stdscr.addnstr(idx, 0, line, max(0, w - 1))
    stdscr.refresh()


def tui_main(stdscr, controller: Go2CliController, odom_topic: str, lidar_state_topic: str, lidar_cloud_topic: str):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    while True:
        draw_panel(stdscr, controller, odom_topic, lidar_state_topic, lidar_cloud_topic)
        key = stdscr.getch()
        move_x = 0.0
        move_yaw = 0.0

        if key == curses.KEY_SR:
            controller.increase_height()
        elif key == curses.KEY_SF:
            controller.decrease_height()
        elif key in (ord("+"), ord("=")):
            controller.increase_height()
        elif key in (ord("-"), ord("_")):
            controller.decrease_height()
        elif key == curses.KEY_UP:
            move_x = 1.0
        elif key == curses.KEY_DOWN:
            move_x = -1.0
        elif key == curses.KEY_LEFT:
            move_yaw = 1.0
        elif key == curses.KEY_RIGHT:
            move_yaw = -1.0
        elif key in (ord("s"), ord("S")):
            controller.command_sit()
        elif key in (ord("n"), ord("N")):
            controller.command_stand()
        elif key in (ord("w"), ord("W")):
            controller.toggle_walk()
        elif key in (ord("g"), ord("G")):
            controller.cycle_gait()
        elif key in (ord("r"), ord("R")):
            controller.trigger_recover_mcf()
        elif key in (ord("q"), ord("Q")):
            break

        controller.set_walk_command(move_x, move_yaw)


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Go2 CLI with stand-height control and telemetry.")
    parser.add_argument("--iface", required=True, help="Robot network interface")
    parser.add_argument("--odom-topic", default=TOPIC_ODOM)
    parser.add_argument("--lidar-state-topic", default=TOPIC_LIDAR_STATE)
    parser.add_argument("--lidar-cloud-topic", default=TOPIC_LIDAR_CLOUD)
    parser.add_argument(
        "--gait-recording",
        default=find_latest_recording(),
        help="Path to a record_gait JSONL file to use for recorded gait playback",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("WARNING: Ensure the robot has clearance before running low-level posture control.")
    input("Press Enter to continue...")

    ChannelFactoryInitialize(0, args.iface)

    recorded_gait = None
    if args.gait_recording:
        recorded_gait = RecordedGait(args.gait_recording)

    controller = Go2CliController(recorded_gait=recorded_gait)
    controller.init(args.odom_topic, args.lidar_state_topic, args.lidar_cloud_topic)
    controller.start()

    try:
        curses.wrapper(
            tui_main,
            controller,
            args.odom_topic,
            args.lidar_state_topic,
            args.lidar_cloud_topic,
        )
    finally:
        controller.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
