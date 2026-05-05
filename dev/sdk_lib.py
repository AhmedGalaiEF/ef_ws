from __future__ import annotations

import audioop
import csv
import json
import math
import os
import subprocess
import tempfile
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core import channel as channel_module
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
from unitree_sdk2py.g1.loco.g1_loco_api import (
    ROBOT_API_ID_LOCO_GET_FSM_ID,
    ROBOT_API_ID_LOCO_GET_FSM_MODE,
)
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__HandCmd_,
    unitree_hg_msg_dds__LowCmd_,
)
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, LowCmd_
from unitree_sdk2py.rpc.client import Client
from unitree_sdk2py.utils.crc import CRC

try:
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
except ImportError:
    AudioClient = None  # type: ignore[assignment]

try:
    from unitree_sdk2py.g1.video.video_client import VideoClient
except ImportError:
    try:
        from unitree_sdk2py.go2.video.video_client import VideoClient
    except ImportError:
        VideoClient = None  # type: ignore[assignment]

try:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_
except ImportError:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    HandState_ = None  # type: ignore[assignment]

try:
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import Imu_ as SensorImu_
except ImportError:
    SensorImu_ = None  # type: ignore[assignment]


def _is_valid_cyclonedds_home(path: str | None) -> bool:
    if not path:
        return False
    root = Path(path).expanduser()
    return (root / "lib" / "libddsc.so").is_file()


def _resolve_cyclonedds_home() -> str | None:
    current = os.environ.get("CYCLONEDDS_HOME")
    if _is_valid_cyclonedds_home(current):
        return str(Path(current).expanduser())
    home = Path.home()
    candidates = (
        home / "academy_prep" / "academy_content" / "docs" / "repos" / "cyclonedds_0_10" / "install_0_10",
        home / "cyclonedds" / "install",
        home / "Desktop" / "unitree" / "cyclonedds" / "install",
        home / "academy_prep" / "academy_content" / "docs" / "repos" / "cyclonedds" / "install",
    )
    for candidate in candidates:
        if _is_valid_cyclonedds_home(str(candidate)):
            return str(candidate)
    return None


def _resolve_cyclonedds_uri() -> str | None:
    current = os.environ.get("CYCLONEDDS_URI")
    if current and current.lstrip().startswith("<"):
        return current
    if current and Path(current).expanduser().is_file():
        return str(Path(current).expanduser())
    home = Path.home()
    candidates = (
        home / "Desktop" / "unitree" / "unitree_sdk2_python" / "cyclonedds.xml",
        home / "academy_prep" / "academy_content" / "docs" / "repos" / "unitree_sdk2_python" / "cyclonedds.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


_cyclonedds_home = _resolve_cyclonedds_home()
if _cyclonedds_home:
    os.environ["CYCLONEDDS_HOME"] = _cyclonedds_home

_cyclonedds_uri = _resolve_cyclonedds_uri()
if _cyclonedds_uri:
    os.environ["CYCLONEDDS_URI"] = _cyclonedds_uri


channel_module.ChannelConfigHasInterface = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>"""
channel_module.ChannelConfigAutoDetermine = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface autodetermine="true" priority="default" multicast="default" />
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>"""

os.environ.setdefault(
    "CYCLONEDDS_URI",
    "<CycloneDDS><Domain><Tracing><Category>none</Category></Tracing></Domain></CycloneDDS>",
)


FSM_ID_PREPARE = 4
FSM_ID_WALK = 501
FSM_ID_RUN = 802
FSM_ID_CLIMB = 812
DEFAULT_SPORT_TOPIC = "rt/odommodestate"
DEFAULT_LOWSTATE_TOPIC = "rt/lowstate"
DEFAULT_ODOM_TOPIC = "rt/odom"
DEFAULT_LIDAR_CLOUD_TOPIC = "rt/utlidar/cloud_deskewed"
DEFAULT_SECONDARY_IMU_TOPIC = "rt/secondary_imu"
DEFAULT_LIDAR_IMU_TOPIC = "rt/utlidar/imu_livox_mid360"

TOPIC_HAND_BY_SIDE = {
    "left": "rt/dex3/left/cmd",
    "right": "rt/dex3/right/cmd",
}
HAND_STATE_TOPIC_BY_SIDE = {
    "left": "rt/dex3/left/state",
    "right": "rt/dex3/right/state",
}
HAND_MAX_LIMITS = {
    "left": [1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0],
    "right": [1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75],
}
HAND_MIN_LIMITS = {
    "left": [-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75],
    "right": [-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0],
}
HAND_THUMB_0_HOLD_TARGETS = {
    "left": -0.09927542507648468,
    "right": -0.03510913997888565,
}
HL_ARM_ACTIONS = {
    "release arm": 99,
    "two-hand kiss": 11,
    "left kiss": 12,
    "right kiss": 13,
    "hands up": 15,
    "clap": 17,
    "high five": 18,
    "hug": 19,
    "heart": 20,
    "right heart": 21,
    "reject": 22,
    "right hand up": 23,
    "x-ray": 24,
    "face wave": 25,
    "high wave": 26,
    "shake hand": 27,
}
HL_ARM_ALIASES = {
    "release": "release arm",
    "two hand kiss": "two-hand kiss",
    "left hand kiss": "left kiss",
    "right hand kiss": "right kiss",
    "xray": "x-ray",
    "x ray": "x-ray",
}
NAMED_COLORS = {
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 105, 180),
}
ARM_RELEASE_DELAY_S = 2.0
GUARDED_FSM_IDS = frozenset((FSM_ID_WALK, FSM_ID_RUN, FSM_ID_CLIMB, FSM_ID_PREPARE))


def _require_dependency(name: str, module: Any) -> Any:
    if module is None:
        raise RuntimeError(f"{name} is not available in this environment.")
    return module


def _call_optional_init(client: Any, timeout: float) -> Any | None:
    if client is None:
        return None
    client.SetTimeout(float(timeout))
    client.Init()
    return client


def parse_color(value: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple) and len(value) == 3:
        return tuple(int(max(0, min(255, v))) for v in value)
    lowered = str(value).strip().lower()
    if lowered in NAMED_COLORS:
        return NAMED_COLORS[lowered]
    if lowered.startswith("#"):
        lowered = lowered[1:]
    if len(lowered) == 6:
        try:
            return (int(lowered[0:2], 16), int(lowered[2:4], 16), int(lowered[4:6], 16))
        except ValueError:
            pass
    parts = lowered.split(",")
    if len(parts) == 3:
        try:
            rgb = tuple(int(part) for part in parts)
        except ValueError:
            rgb = None
        if rgb is not None and all(0 <= value <= 255 for value in rgb):
            return rgb
    raise ValueError("color must be a name, #RRGGBB, or R,G,B")


def scale_color(rgb: tuple[int, int, int], intensity: int) -> tuple[int, int, int]:
    level = max(0, min(100, int(intensity)))
    factor = level / 100.0
    return (int(rgb[0] * factor), int(rgb[1] * factor), int(rgb[2] * factor))


def _quat_to_yaw(quat: tuple[float, float, float, float] | None) -> float | None:
    if quat is None:
        return None
    try:
        qx, qy, qz, qw = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    except Exception:
        return None
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _normalize_angle(angle: float) -> float:
    return float(math.atan2(math.sin(float(angle)), math.cos(float(angle))))


def _extract_imu_values(msg: Any) -> dict[str, Any]:
    if msg is None:
        return {"rpy": None, "gyro": None, "acc": None, "quat": None, "temp": None}

    def _vec3(obj: Any, *names: str) -> tuple[float, float, float] | None:
        for name in names:
            try:
                value = getattr(obj, name)
                if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
                    return (float(value.x), float(value.y), float(value.z))
                return (float(value[0]), float(value[1]), float(value[2]))
            except Exception:
                continue
        return None

    def _quat(obj: Any, *names: str) -> tuple[float, float, float, float] | None:
        for name in names:
            try:
                value = getattr(obj, name)
                if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z") and hasattr(value, "w"):
                    return (float(value.x), float(value.y), float(value.z), float(value.w))
                return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
            except Exception:
                continue
        return None

    quat = _quat(msg, "quaternion", "orientation")
    rpy = _vec3(msg, "rpy")
    if rpy is None:
        yaw = _quat_to_yaw(quat)
        if yaw is not None:
            rpy = (0.0, 0.0, yaw)
    temp = None
    for key in ("temperature", "temp"):
        try:
            temp = float(getattr(msg, key))
            break
        except Exception:
            continue
    return {
        "rpy": rpy,
        "gyro": _vec3(msg, "gyroscope", "gyro", "angular_velocity"),
        "acc": _vec3(msg, "accelerometer", "acc", "linear_acceleration"),
        "quat": quat,
        "temp": temp,
    }


def _convert_wav_for_robot(src_path: Path, dst_path: Path) -> Path:
    with wave.open(str(src_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_rate = wav_file.getframerate()
        pcm = wav_file.readframes(wav_file.getnframes())
    if channels == 2:
        pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
        channels = 1
    elif channels != 1:
        raise ValueError(f"WAV must be mono or stereo PCM, got {channels} channels")
    if sample_width != 2:
        pcm = audioop.lin2lin(pcm, sample_width, 2)
        sample_width = 2
    if frame_rate != 16000:
        pcm, _state = audioop.ratecv(pcm, sample_width, channels, frame_rate, 16000, None)
    with wave.open(str(dst_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(pcm)
    return dst_path


def clamp_hand_targets(hand: str, targets: list[float]) -> list[float]:
    side = str(hand).strip().lower()
    if len(targets) != 7:
        raise ValueError("Hand targets must contain 7 joint values.")
    return [
        max(float(lo), min(float(hi), float(value)))
        for value, lo, hi in zip(targets, HAND_MIN_LIMITS[side], HAND_MAX_LIMITS[side])
    ]


def hand_open_targets(hand: str) -> list[float]:
    side = str(hand).strip().lower()
    closed = hand_closed_targets(side)
    return [
        closed[0],
        *[
            hi if abs(closed_value - lo) < abs(closed_value - hi) else lo
            for closed_value, lo, hi in zip(
                closed[1:],
                HAND_MIN_LIMITS[side][1:],
                HAND_MAX_LIMITS[side][1:],
            )
        ],
    ]


def hand_closed_targets(hand: str) -> list[float]:
    side = str(hand).strip().lower()
    thumb = HAND_THUMB_0_HOLD_TARGETS[side]
    if side == "left":
        return [
            thumb,
            HAND_MAX_LIMITS[side][1],
            HAND_MAX_LIMITS[side][2],
            HAND_MIN_LIMITS[side][3],
            HAND_MIN_LIMITS[side][4],
            HAND_MIN_LIMITS[side][5],
            HAND_MIN_LIMITS[side][6],
        ]
    return [
        thumb,
        HAND_MIN_LIMITS[side][1],
        HAND_MIN_LIMITS[side][2],
        HAND_MAX_LIMITS[side][3],
        HAND_MAX_LIMITS[side][4],
        HAND_MAX_LIMITS[side][5],
        HAND_MAX_LIMITS[side][6],
    ]


def pack_ris_mode(motor_id: int, status: int = 1, timeout: int = 0) -> int:
    return (int(motor_id) & 0x0F) | ((int(status) & 0x07) << 4) | ((int(timeout) & 0x01) << 7)


def build_hand_msg(targets: list[float], kp: float, kd: float, tau: float, timeout: int = 0) -> HandCmd_:
    if len(targets) != 7:
        raise ValueError("Hand targets must contain 7 joint values.")
    msg = unitree_hg_msg_dds__HandCmd_()
    for idx in range(7):
        cmd = msg.motor_cmd[idx]
        cmd.mode = pack_ris_mode(idx, timeout=timeout)
        cmd.q = float(targets[idx])
        cmd.dq = 0.0
        cmd.kp = float(kp)
        cmd.kd = float(kd)
        cmd.tau = float(tau)
    return msg


class Dex3HandController:
    def __init__(self, hand: str) -> None:
        side = str(hand).strip().lower()
        if side not in TOPIC_HAND_BY_SIDE:
            raise ValueError(f"Invalid hand '{hand}'.")
        self.hand = side
        self._pub = ChannelPublisher(TOPIC_HAND_BY_SIDE[side], HandCmd_)
        self._pub.Init()
        self._last_targets: list[float] | None = None
        self._state_positions: list[float] | None = None
        self._state_ts = 0.0
        self._lock = threading.Lock()
        self._state_sub: ChannelSubscriber | None = None
        self._release_stop: threading.Event | None = None
        self._release_thread: threading.Thread | None = None
        if HandState_ is not None:
            self._state_sub = ChannelSubscriber(HAND_STATE_TOPIC_BY_SIDE[side], HandState_)
            self._state_sub.Init(self._state_cb, 20)

    def _state_cb(self, msg: Any) -> None:
        positions: list[float] = []
        for idx, motor in enumerate(list(getattr(msg, "motor_state", []) or [])[:7]):
            try:
                positions.append(float(motor.q))
            except Exception:
                return
        if len(positions) == 7:
            with self._lock:
                self._state_positions = positions
                self._state_ts = time.time()

    def _latest_positions(self, max_age: float = 1.0) -> list[float] | None:
        with self._lock:
            if self._state_positions is None or (time.time() - self._state_ts) > max_age:
                return None
            return list(self._state_positions)

    def _publish_targets(self, targets: list[float], seconds: float, rate_hz: float, kp: float, kd: float, tau: float) -> None:
        msg = build_hand_msg(clamp_hand_targets(self.hand, targets), kp=kp, kd=kd, tau=tau)
        steps = max(1, int(max(0.01, float(seconds)) * max(1.0, float(rate_hz))))
        delay = 1.0 / max(1.0, float(rate_hz))
        for _ in range(steps):
            self._pub.Write(msg)
            time.sleep(delay)
        self._last_targets = list(targets)

    def write_targets_once(
        self,
        targets: list[float],
        *,
        kp: float = 0.8,
        kd: float = 0.05,
        tau: float = 0.02,
        timeout: int = 0,
    ) -> bool:
        target_list = clamp_hand_targets(self.hand, [float(value) for value in targets])
        ok = self._pub.Write(build_hand_msg(target_list, kp=kp, kd=kd, tau=tau, timeout=timeout))
        self._last_targets = list(target_list)
        return ok is not False

    def _stop_release_thread(self) -> None:
        if self._release_stop is not None:
            self._release_stop.set()
        if self._release_thread is not None and self._release_thread.is_alive():
            self._release_thread.join(timeout=1.0)
        self._release_stop = None
        self._release_thread = None

    def set_targets(
        self,
        targets: list[float],
        hold_s: float = 0.6,
        rate_hz: float = 50.0,
        kp: float = 1.2,
        kd: float = 0.05,
        tau: float = 0.05,
    ) -> None:
        self._stop_release_thread()
        target_list = clamp_hand_targets(self.hand, [float(value) for value in targets])
        start = self._latest_positions() or self._last_targets or hand_open_targets(self.hand)
        start = clamp_hand_targets(self.hand, list(start))
        ramp_steps = max(2, int(max(0.1, min(float(hold_s), 0.25)) * max(1.0, float(rate_hz))))
        if any(abs(dst - src) > 1e-6 for src, dst in zip(start, target_list)):
            for step_idx in range(1, ramp_steps + 1):
                alpha = float(step_idx) / float(ramp_steps)
                interpolated = [src + (dst - src) * alpha for src, dst in zip(start, target_list)]
                self._publish_targets(interpolated, seconds=1.0 / max(1.0, float(rate_hz)), rate_hz=rate_hz, kp=kp, kd=kd, tau=tau)
        remaining = max(1.0 / max(1.0, float(rate_hz)), float(hold_s))
        self._publish_targets(target_list, seconds=remaining, rate_hz=rate_hz, kp=kp, kd=kd, tau=tau)

    def open(self, hold_s: float = 0.6, rate_hz: float = 50.0) -> None:
        self.set_targets(hand_open_targets(self.hand), hold_s=hold_s, rate_hz=rate_hz, kp=1.5, kd=0.1, tau=0.03)

    def close(self, hold_s: float = 0.6, rate_hz: float = 50.0) -> None:
        self.set_targets(hand_closed_targets(self.hand), hold_s=hold_s, rate_hz=rate_hz, kp=1.5, kd=0.1, tau=0.03)

    def release_fingers(
        self,
        hold_s: float = 0.5,
        rate_hz: float = 50.0,
        *,
        persistent: bool = True,
    ) -> None:
        self._stop_release_thread()
        targets = list(self._last_targets) if self._last_targets is not None else hand_open_targets(self.hand)
        msg = build_hand_msg(clamp_hand_targets(self.hand, targets), kp=0.0, kd=0.0, tau=0.0, timeout=1)
        if persistent:
            stop_event = threading.Event()

            def _loop() -> None:
                delay = 1.0 / max(1.0, float(rate_hz))
                while not stop_event.is_set():
                    self._pub.Write(msg)
                    time.sleep(delay)

            self._release_stop = stop_event
            self._release_thread = threading.Thread(target=_loop, name=f"dex3-{self.hand}-release", daemon=True)
            self._release_thread.start()
        else:
            steps = max(1, int(max(0.01, float(hold_s)) * max(1.0, float(rate_hz))))
            delay = 1.0 / max(1.0, float(rate_hz))
            for _ in range(steps):
                self._pub.Write(msg)
                time.sleep(delay)
        self._last_targets = None

    def stop_release_fingers(self) -> None:
        self._stop_release_thread()


@dataclass
class SlamResponse:
    code: int
    raw: Any


class SlamOperateClient(Client):
    def __init__(self, enable_lease: bool = False) -> None:
        super().__init__("slam_operate", enable_lease)

    def Init(self) -> None:
        for api_id in (1801, 1802, 1804, 1102, 1201, 1202, 1901):
            self._RegistApi(api_id, 0)
        self._SetApiVerson("1.0.0.1")

    def _call(self, api_id: int, payload: dict[str, Any]) -> SlamResponse:
        code, data = self._Call(api_id, json.dumps(payload, ensure_ascii=True))
        return SlamResponse(code=int(code), raw=data)

    def start_mapping(self, slam_type: str = "indoor") -> SlamResponse:
        return self._call(1801, {"data": {"slam_type": slam_type}})

    def end_mapping(self, address: str) -> SlamResponse:
        return self._call(1802, {"data": {"address": address}})

    def pose_nav(self, x: float, y: float, z: float, q_x: float, q_y: float, q_z: float, q_w: float, mode: int = 1) -> SlamResponse:
        return self._call(
            1102,
            {
                "data": {
                    "targetPose": {"x": x, "y": y, "z": z, "q_x": q_x, "q_y": q_y, "q_z": q_z, "q_w": q_w},
                    "mode": mode,
                }
            },
        )

    def close_slam(self) -> SlamResponse:
        return self._call(1901, {"data": {}})


class SlamInfoSubscriber:
    def __init__(self, info_topic: str = "rt/slam_info", key_topic: str = "rt/slam_key_info") -> None:
        self._lock = threading.Lock()
        self._info: str | None = None
        self._key: str | None = None
        self._info_sub = ChannelSubscriber(info_topic, String_)
        self._key_sub = ChannelSubscriber(key_topic, String_)
        self._info_sub.Init(self._info_cb, 10)
        self._key_sub.Init(self._key_cb, 10)

    def _info_cb(self, msg: String_) -> None:
        with self._lock:
            self._info = str(msg.data)

    def _key_cb(self, msg: String_) -> None:
        with self._lock:
            self._key = str(msg.data)

    def get_pose(self) -> tuple[float, float, float] | None:
        with self._lock:
            payloads = (self._info, self._key)
        for payload_raw in payloads:
            if not payload_raw:
                continue
            try:
                payload = json.loads(payload_raw)
                current_pose = payload["data"]["currentPose"]
                x = float(current_pose["x"])
                y = float(current_pose["y"])
                qx = float(current_pose.get("q_x", 0.0))
                qy = float(current_pose.get("q_y", 0.0))
                qz = float(current_pose.get("q_z", 0.0))
                qw = float(current_pose.get("q_w", 1.0))
                yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
                return (x, y, yaw)
            except Exception:
                continue
        return None


class G1:
    def __init__(self, iface: str = "eth0", domain_id: int = 0, timeout: float = 10.0) -> None:
        self.iface = str(iface)
        self.domain_id = int(domain_id)
        self.timeout = float(timeout)

        ChannelFactoryInitialize(self.domain_id, self.iface)

        self._lock = threading.Lock()
        self._sport: SportModeState_ | None = None
        self._lowstate: LowState_ | None = None
        self._lidar_cloud: PointCloud2_ | None = None
        self._odom: Odometry_ | None = None
        self._secondary_imu: Any | None = None
        self._lidar_imu: Any | None = None
        self._sensor_timestamps: dict[str, float] = {}
        self._path_points: list[tuple[float, float, float]] = []
        self.slam_is_running = False

        self.loco_client = LocoClient()
        self.loco_client.SetTimeout(self.timeout)
        self.loco_client.Init()

        self.ms_client = MotionSwitcherClient()
        self.ms_client.SetTimeout(self.timeout)
        self.ms_client.Init()

        self.arm_action_client = G1ArmActionClient()
        self.arm_action_client.SetTimeout(self.timeout)
        self.arm_action_client.Init()

        self.video_client = None if VideoClient is None else _call_optional_init(VideoClient(), 2.0)
        self.audio_client = None if AudioClient is None else _call_optional_init(AudioClient(), 5.0)

        self.arm_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_pub.Init()
        self.arm_cmd = unitree_hg_msg_dds__LowCmd_()
        self.arm_crc = CRC()

        self.slam_client = SlamOperateClient()
        self.slam_client.Init()
        self.slam_client.SetTimeout(self.timeout)
        self.slam_info_sub = SlamInfoSubscriber()

        self.hands = {side: Dex3HandController(side) for side in ("left", "right")}

        self.sport_sub = ChannelSubscriber(DEFAULT_SPORT_TOPIC, SportModeState_)
        self.sport_sub.Init(self._on_sport, 10)
        self.lowstate_sub = ChannelSubscriber(DEFAULT_LOWSTATE_TOPIC, LowState_)
        self.lowstate_sub.Init(self._on_lowstate, 10)
        self.lidar_sub = ChannelSubscriber(DEFAULT_LIDAR_CLOUD_TOPIC, PointCloud2_)
        self.lidar_sub.Init(self._on_lidar, 10)
        self.odom_sub = ChannelSubscriber(DEFAULT_ODOM_TOPIC, Odometry_)
        self.odom_sub.Init(self._on_odom, 10)
        self.secondary_imu_sub = None
        self.lidar_imu_sub = None
        if SensorImu_ is not None:
            self.secondary_imu_sub = ChannelSubscriber(DEFAULT_SECONDARY_IMU_TOPIC, SensorImu_)
            self.secondary_imu_sub.Init(self._on_secondary_imu, 20)
            self.lidar_imu_sub = ChannelSubscriber(DEFAULT_LIDAR_IMU_TOPIC, SensorImu_)
            self.lidar_imu_sub.Init(self._on_lidar_imu, 20)

    def _on_sport(self, msg: SportModeState_) -> None:
        with self._lock:
            self._sport = msg
            self._sensor_timestamps["sport"] = time.time()

    def _on_lowstate(self, msg: LowState_) -> None:
        with self._lock:
            self._lowstate = msg
            self._sensor_timestamps["lowstate"] = time.time()

    def _on_lidar(self, msg: PointCloud2_) -> None:
        with self._lock:
            self._lidar_cloud = msg
            self._sensor_timestamps["lidar"] = time.time()

    def _on_odom(self, msg: Odometry_) -> None:
        with self._lock:
            self._odom = msg
            self._sensor_timestamps["odom"] = time.time()

    def _on_secondary_imu(self, msg: Any) -> None:
        with self._lock:
            self._secondary_imu = msg
            self._sensor_timestamps["secondary_imu"] = time.time()

    def _on_lidar_imu(self, msg: Any) -> None:
        with self._lock:
            self._lidar_imu = msg
            self._sensor_timestamps["lidar_imu"] = time.time()

    def _rpc(self, api_id: int) -> int | None:
        try:
            code, data = self.loco_client._Call(api_id, "{}")
            return int(json.loads(data).get("data")) if code == 0 and data else None
        except Exception:
            return None

    def _run_pose_nav(self, x: float, y: float, yaw: float = 0.0) -> int:
        qz = math.sin(float(yaw) * 0.5)
        qw = math.cos(float(yaw) * 0.5)
        response = self.slam_client.pose_nav(float(x), float(y), 0.0, 0.0, 0.0, qz, qw, mode=1)
        return int(response.code)

    def toggle_service(self, name: str, on_off: bool) -> int:
        name = str(name)
        if on_off:
            code, _data = self.ms_client.SelectMode(name)
        else:
            code, active = self.ms_client.CheckMode()
            if int(code) != 0:
                return int(code)
            if not active or str(active.get("name", "")) in ("", name):
                code, _data = self.ms_client.ReleaseMode()
            else:
                return 0
        return int(code)

    def _service_is_active(self, name: str) -> bool:
        code, active = self.ms_client.CheckMode()
        if int(code) != 0 or not active:
            return False
        return str(active.get("name", "")) == str(name)

    def _normalized_hand_selection(self, hand: str) -> tuple[str, ...]:
        side = str(hand).strip().lower()
        if side == "both":
            return ("left", "right")
        if side in self.hands:
            return (side,)
        raise ValueError(f"Invalid hand '{hand}'.")

    def get_state(self, df: bool = True) -> Any:
        fsm = self.get_fsm()
        imu = self.get_sensors_imu()
        odom = self.get_odomstate()
        joints = self.get_joint_states()
        row = {"fsm_id": fsm.get("id"), "fsm_mode": fsm.get("mode")}
        if odom:
            row.update({"odom_x": odom[0], "odom_y": odom[1], "odom_yaw": odom[2]})
        if imu:
            row.update(
                {
                    "imu_roll": imu.get("roll"),
                    "imu_pitch": imu.get("pitch"),
                    "imu_yaw": imu.get("yaw"),
                    "gyro_x": imu["gyro"][0] if imu.get("gyro") else None,
                    "gyro_y": imu["gyro"][1] if imu.get("gyro") else None,
                    "gyro_z": imu["gyro"][2] if imu.get("gyro") else None,
                    "acc_x": imu["acc"][0] if imu.get("acc") else None,
                    "acc_y": imu["acc"][1] if imu.get("acc") else None,
                    "acc_z": imu["acc"][2] if imu.get("acc") else None,
                }
            )
        if joints:
            for idx, values in joints.items():
                row.update({f"j{idx}_q": values["q"], f"j{idx}_dq": values["dq"], f"j{idx}_tau": values["tau"]})
        if not df:
            return row
        _require_dependency("pandas", pd)
        return pd.DataFrame([row])

    def get_robot_state(self) -> dict[str, Any]:
        return {
            "fsm": self.get_fsm(),
            "mode": self.get_mode(),
            "gait": self.get_gait(),
            "body_height": self.get_body_height(),
            "position": self.get_position(),
            "velocity": self.get_velocity(),
            "yaw": self.get_yaw(),
            "is_moving": self.is_moving(),
            "imu": self.get_imu(),
            "secondary_imu": self.get_secondary_imu(),
            "lidar_imu": self.get_lidar_imu(),
            "odom_pose": self.get_odomstate(),
            "sensor_timestamps": self.get_sensor_timestamps(),
            "slam_is_running": bool(self.slam_is_running),
            "queued_path_points": len(self._path_points),
        }

    def get_fsm(self) -> dict[str, int | None]:
        return {
            "id": self._rpc(ROBOT_API_ID_LOCO_GET_FSM_ID),
            "mode": self._rpc(ROBOT_API_ID_LOCO_GET_FSM_MODE),
        }

    @staticmethod
    def _status_code(value: Any) -> int:
        return 0 if value is None else int(value)

    def switch_fsm(self, fsm_id: int) -> int:
        return self._status_code(self.loco_client.SetFsmId(int(fsm_id)))

    def _hanger_loaded_standup_state(self, fsm_id: int | None, fsm_mode: int | None, current_height: float) -> bool:
        return fsm_mode == 0 and (fsm_id == FSM_ID_PREPARE or fsm_id is None) and float(current_height) > 0.2

    def _run_hanged_boot_guard(
        self,
        *,
        target_height: float = 0.22,
        step: float = 0.02,
        max_attempts: int = 3,
        retry_delay: float = 0.08,
    ) -> dict[str, Any]:
        target_height = max(0.0, float(target_height))
        step = max(0.005, float(step))
        attempts_limit = max(1, int(max_attempts))
        self.switch_fsm(FSM_ID_PREPARE)

        attempt = 0
        accepted_height: float | None = None
        while attempt < attempts_limit:
            attempt += 1
            height = 0.0
            loaded_height: float | None = None
            while height < target_height:
                height = min(target_height, height + step)
                self.loco_client.SetStandHeight(height)
                time.sleep(0.05)
                fsm = self.get_fsm()
                if self._hanger_loaded_standup_state(fsm.get("id"), fsm.get("mode"), height):
                    loaded_height = height
                    break

            time.sleep(max(0.0, float(retry_delay)))
            fsm = self.get_fsm()
            if loaded_height is not None and self._hanger_loaded_standup_state(fsm.get("id"), fsm.get("mode"), loaded_height):
                accepted_height = loaded_height
                break
            if loaded_height is not None:
                accepted_height = loaded_height
                break

            self.loco_client.SetStandHeight(0.0)
            time.sleep(0.05)

        if accepted_height is None:
            raise TimeoutError(
                "Hanger boot guard did not reach a loaded stand state. "
                "Adjust the hanger/support and retry."
            )

        if hasattr(self.loco_client, "BalanceStand"):
            self.loco_client.BalanceStand(0)
        self.loco_client.SetStandHeight(float(accepted_height))
        if hasattr(self.loco_client, "Start"):
            self.loco_client.Start()
        for _ in range(3):
            self.switch_fsm(FSM_ID_WALK)
            fsm = self.get_fsm()
            if fsm.get("id") == FSM_ID_WALK:
                break
            time.sleep(0.1)
        return {
            "accepted_height": float(accepted_height),
            "fsm": self.get_fsm(),
            "attempts": attempt,
        }

    def _guard_motion_fsm(
        self,
        *,
        target_height: float = 0.22,
        step: float = 0.02,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        fsm = self.get_fsm()
        current_fsm_id = fsm.get("id")
        for _ in range(5):
            if current_fsm_id is not None:
                break
            time.sleep(0.05)
            fsm = self.get_fsm()
            current_fsm_id = fsm.get("id")
        if current_fsm_id in (FSM_ID_WALK, FSM_ID_RUN, FSM_ID_CLIMB):
            return {"guarded": True, "boot": False, "fsm": fsm}
        if current_fsm_id == FSM_ID_PREPARE:
            boot = self._run_hanged_boot_guard(
                target_height=target_height,
                step=step,
                max_attempts=max_attempts,
            )
            return {"guarded": True, "boot": True, "boot_result": boot, "fsm": self.get_fsm()}
        raise RuntimeError(
            f"Motion FSM guard rejected current FSM {current_fsm_id}. "
            f"Expected one of {sorted(GUARDED_FSM_IDS)}."
        )

    def fsm_zt(self) -> int:
        if not self._service_is_active("ai"):
            toggle_code = self.toggle_service("ai", True)
            if int(toggle_code) != 0:
                return int(toggle_code)
        return self._status_code(self.loco_client.ZeroTorque()) if hasattr(self.loco_client, "ZeroTorque") else self.switch_fsm(0)

    def fsm_damp(self) -> int:
        return self._status_code(self.loco_client.Damp()) if hasattr(self.loco_client, "Damp") else self.switch_fsm(1)

    def fsm_dev(self, on_off: bool) -> int:
        return self.toggle_service("ai", not bool(on_off))

    def fsm_run(self, *, target_height: float = 0.22, step: float = 0.02, max_attempts: int = 3) -> int:
        self._guard_motion_fsm(target_height=target_height, step=step, max_attempts=max_attempts)
        return self.switch_fsm(FSM_ID_RUN)

    def fsm_walk(self, *, target_height: float = 0.22, step: float = 0.02, max_attempts: int = 3) -> int:
        self._guard_motion_fsm(target_height=target_height, step=step, max_attempts=max_attempts)
        return self.switch_fsm(FSM_ID_WALK)

    def fsm_prepare(self) -> int:
        return self.switch_fsm(FSM_ID_PREPARE)

    def fsm_climb(self, *, target_height: float = 0.22, step: float = 0.02, max_attempts: int = 3) -> int:
        self._guard_motion_fsm(target_height=target_height, step=step, max_attempts=max_attempts)
        return self.switch_fsm(FSM_ID_CLIMB)

    def fsm_dance(self, *, target_height: float = 0.22, step: float = 0.02, max_attempts: int = 3) -> int:
        self._guard_motion_fsm(target_height=target_height, step=step, max_attempts=max_attempts)
        raise NotImplementedError("fsm_dance is a placeholder.")

    def fsm_sit(self) -> int:
        return self._status_code(self.loco_client.Sit()) if hasattr(self.loco_client, "Sit") else self.switch_fsm(2)

    def loco_move(self, vx: float, vy: float, vyaw: float) -> Any:
        return self.loco_client.Move(float(vx), float(vy), float(vyaw), continous_move=True)

    def move_with_odom(
        self,
        x: float = 0.0,
        y: float = 0.0,
        yaw: float = 0.0,
        *,
        kp_xy: float = 0.8,
        kp_yaw: float = 1.0,
        max_vx: float = 0.35,
        max_vy: float = 0.25,
        max_vyaw: float = 0.8,
        pos_tolerance: float = 0.03,
        yaw_tolerance: float = 0.05,
        timeout_s: float = 10.0,
        settle_s: float = 0.1,
    ) -> dict[str, Any]:
        start_pose = self.get_odomstate()
        if start_pose is None:
            raise RuntimeError("odommodestate/odom pose is unavailable.")

        target_x = float(x)
        target_y = float(y)
        target_yaw = float(yaw)
        start_yaw = float(start_pose[2])
        start_cos = math.cos(start_yaw)
        start_sin = math.sin(start_yaw)
        target_world_x = float(start_pose[0]) + start_cos * target_x - start_sin * target_y
        target_world_y = float(start_pose[1]) + start_sin * target_x + start_cos * target_y
        target_world_yaw = _normalize_angle(start_yaw + target_yaw)
        deadline = time.time() + max(0.1, float(timeout_s))
        final_error = {"x": None, "y": None, "yaw": None}

        try:
            while time.time() < deadline:
                pose = self.get_odomstate()
                if pose is None:
                    time.sleep(0.02)
                    continue

                dx_world = float(pose[0] - start_pose[0])
                dy_world = float(pose[1] - start_pose[1])
                dx_body = start_cos * dx_world + start_sin * dy_world
                dy_body = -start_sin * dx_world + start_cos * dy_world
                dyaw = _normalize_angle(float(pose[2] - start_pose[2]))

                err_x = target_x - dx_body
                err_y = target_y - dy_body
                err_yaw = _normalize_angle(target_yaw - dyaw)
                final_error = {"x": err_x, "y": err_y, "yaw": err_yaw}

                if math.hypot(err_x, err_y) <= float(pos_tolerance) and abs(err_yaw) <= float(yaw_tolerance):
                    break

                world_err_x = target_world_x - float(pose[0])
                world_err_y = target_world_y - float(pose[1])
                current_yaw = float(pose[2])
                current_cos = math.cos(current_yaw)
                current_sin = math.sin(current_yaw)
                cmd_body_x = current_cos * world_err_x + current_sin * world_err_y
                cmd_body_y = -current_sin * world_err_x + current_cos * world_err_y
                cmd_yaw_err = _normalize_angle(target_world_yaw - current_yaw)
                cmd_vx = max(-float(max_vx), min(float(max_vx), float(kp_xy) * cmd_body_x))
                cmd_vy = max(-float(max_vy), min(float(max_vy), float(kp_xy) * cmd_body_y))
                cmd_vyaw = max(-float(max_vyaw), min(float(max_vyaw), float(kp_yaw) * cmd_yaw_err))
                self.loco_move(cmd_vx, cmd_vy, cmd_vyaw)
                time.sleep(0.05)
        finally:
            self.stop()

        if settle_s > 0.0:
            time.sleep(float(settle_s))
            pose = self.get_odomstate()
            if pose is not None:
                dx_world = float(pose[0] - start_pose[0])
                dy_world = float(pose[1] - start_pose[1])
                dx_body = start_cos * dx_world + start_sin * dy_world
                dy_body = -start_sin * dx_world + start_cos * dy_world
                dyaw = _normalize_angle(float(pose[2] - start_pose[2]))
                final_error = {
                    "x": target_x - dx_body,
                    "y": target_y - dy_body,
                    "yaw": _normalize_angle(target_yaw - dyaw),
                }

        timed_out = time.time() >= deadline and (
            final_error["x"] is None
            or math.hypot(float(final_error["x"]), float(final_error["y"])) > float(pos_tolerance)
            or abs(float(final_error["yaw"])) > float(yaw_tolerance)
        )
        return {
            "target": {"x": target_x, "y": target_y, "yaw": target_yaw},
            "error": final_error,
            "timed_out": timed_out,
            "start_pose": start_pose,
            "final_pose": self.get_odomstate(),
        }

    def move_for(self, duration: float, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> Any:
        result = self.loco_move(vx, vy, vyaw)
        try:
            time.sleep(float(duration))
        finally:
            self.stop()
        return result

    def stop(self) -> None:
        if hasattr(self.loco_client, "StopMove"):
            self.loco_client.StopMove()
        else:
            self.loco_client.Move(0.0, 0.0, 0.0, continous_move=False)

    def get_mode(self) -> int | None:
        with self._lock:
            msg = self._sport
        if msg is None:
            return None
        try:
            return int(msg.mode)
        except Exception:
            return None

    def get_gait(self) -> int | None:
        with self._lock:
            msg = self._sport
        if msg is None:
            return None
        for key in ("gait_type", "gaitType", "gait"):
            try:
                return int(getattr(msg, key))
            except Exception:
                continue
        return None

    def get_body_height(self) -> float | None:
        with self._lock:
            msg = self._sport
        if msg is None:
            return None
        for key in ("body_height", "bodyHeight", "stand_height", "standHeight"):
            try:
                return float(getattr(msg, key))
            except Exception:
                continue
        return None

    def get_position(self) -> tuple[float, float, float] | None:
        with self._lock:
            msg = self._sport
        if msg is None:
            return None
        for key in ("position", "pos", "position_w"):
            try:
                vector = getattr(msg, key)
                return (float(vector[0]), float(vector[1]), float(vector[2]))
            except Exception:
                continue
        return None

    def get_velocity(self) -> tuple[float, float, float] | None:
        with self._lock:
            msg = self._sport
        if msg is None:
            return None
        for key in ("velocity", "vel", "velocity_w"):
            try:
                vector = getattr(msg, key)
                return (float(vector[0]), float(vector[1]), float(vector[2]))
            except Exception:
                continue
        return None

    def get_yaw(self) -> float | None:
        imu = self.get_imu()
        return float(imu["rpy"][2]) if imu else None

    def is_moving(self, linear_eps: float = 0.03, yaw_eps: float = 0.08) -> bool:
        velocity = self.get_velocity()
        if velocity is None:
            return False
        return math.hypot(velocity[0], velocity[1]) > linear_eps or abs(velocity[2]) > yaw_eps

    def get_imu(self) -> dict[str, Any] | None:
        with self._lock:
            msg = self._sport
        if msg is None:
            return None
        imu = getattr(msg, "imu_state", None)
        if imu is None:
            return None
        try:
            rpy = tuple(float(imu.rpy[i]) for i in range(3))
        except Exception:
            rpy = (0.0, 0.0, 0.0)
        try:
            gyro = tuple(float(imu.gyroscope[i]) for i in range(3))
        except Exception:
            gyro = None
        try:
            acc = tuple(float(imu.accelerometer[i]) for i in range(3))
        except Exception:
            acc = None
        try:
            quat = tuple(float(imu.quaternion[i]) for i in range(4))
        except Exception:
            quat = None
        try:
            temp = float(imu.temperature)
        except Exception:
            temp = None
        return {"rpy": rpy, "gyro": gyro, "acc": acc, "quat": quat, "temp": temp}

    def get_secondary_imu(self) -> dict[str, Any] | None:
        with self._lock:
            msg = self._secondary_imu
        if msg is None:
            return None
        return _extract_imu_values(msg)

    def get_lidar_imu(self) -> dict[str, Any] | None:
        with self._lock:
            msg = self._lidar_imu
        if msg is None:
            return None
        return _extract_imu_values(msg)

    def get_sensor_timestamps(self) -> dict[str, float]:
        with self._lock:
            return dict(self._sensor_timestamps)

    def get_sensors_imu(self) -> dict[str, Any] | None:
        imu = self.get_imu()
        if imu is None:
            return None
        return {
            "roll": imu["rpy"][0],
            "pitch": imu["rpy"][1],
            "yaw": imu["rpy"][2],
            "gyro": imu["gyro"],
            "acc": imu["acc"],
        }

    def get_sensors_secondary_imu(self) -> dict[str, Any] | None:
        imu = self.get_secondary_imu()
        if imu is None or imu.get("rpy") is None:
            return None
        return {
            "roll": imu["rpy"][0],
            "pitch": imu["rpy"][1],
            "yaw": imu["rpy"][2],
            "gyro": imu["gyro"],
            "acc": imu["acc"],
        }

    def get_sensors_lidar_imu(self) -> dict[str, Any] | None:
        imu = self.get_lidar_imu()
        if imu is None:
            return None
        rpy = imu.get("rpy")
        return {
            "roll": None if rpy is None else rpy[0],
            "pitch": None if rpy is None else rpy[1],
            "yaw": None if rpy is None else rpy[2],
            "gyro": imu.get("gyro"),
            "acc": imu.get("acc"),
            "quat": imu.get("quat"),
            "temp": imu.get("temp"),
        }

    def get_joint_states(self) -> dict[int, dict[str, float | None]] | None:
        with self._lock:
            msg = self._lowstate
        if msg is None:
            return None
        joints: dict[int, dict[str, float | None]] = {}
        for idx, motor in enumerate(msg.motor_state or []):
            try:
                joints[idx] = {"q": float(motor.q), "dq": float(motor.dq), "tau": float(motor.tau_est)}
            except Exception:
                joints[idx] = {"q": None, "dq": None, "tau": None}
        return joints

    def _read_joint_positions_or_raise(self, timeout: float = 3.0) -> dict[int, float]:
        deadline = time.time() + max(0.05, float(timeout))
        while time.time() < deadline:
            joints = self.get_joint_states()
            if joints:
                positions = {
                    int(idx): float(values["q"])
                    for idx, values in joints.items()
                    if values.get("q") is not None
                }
                if positions:
                    return positions
            time.sleep(0.02)
        raise TimeoutError("Timed out waiting for joint positions from rt/lowstate.")

    def _get_hand_positions_or_raise(self, hand: str, timeout: float = 1.0) -> list[float]:
        controller = self.hands[str(hand).strip().lower()]
        deadline = time.time() + max(0.05, float(timeout))
        while time.time() < deadline:
            positions = controller._latest_positions(max_age=2.0)
            if positions is not None:
                return [float(value) for value in positions]
            time.sleep(0.02)
        raise TimeoutError(f"Timed out waiting for {hand} hand positions.")

    def get_odomstate(self) -> tuple[float, float, float] | None:
        with self._lock:
            msg = self._odom or self._sport
        if msg is None:
            return None
        try:
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            x, y = float(pos.x), float(pos.y)
            qx, qy, qz, qw = float(ori.x), float(ori.y), float(ori.z), float(ori.w)
        except Exception:
            try:
                pos = msg.position
                quat = msg.imu_state.quaternion
                x, y = float(pos[0]), float(pos[1])
                qw, qx, qy, qz = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            except Exception:
                return None
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return (x, y, float(yaw))

    def get_odom_pose(self) -> tuple[float, float, float] | None:
        return self.get_odomstate()

    def get_camera_image_jpeg(self) -> bytes:
        client = _require_dependency("VideoClient", self.video_client)
        code, data = client.GetImageSample()
        if int(code) != 0:
            raise RuntimeError(f"GetImageSample failed: {code}")
        return bytes(data)

    def get_sensors_rgbd(self) -> bytes:
        return self.get_camera_image_jpeg()

    def get_camera_frame_bgr(self) -> Any:
        _require_dependency("opencv-python", cv2)
        _require_dependency("numpy", np)
        return cv2.imdecode(np.frombuffer(self.get_camera_image_jpeg(), dtype=np.uint8), cv2.IMREAD_COLOR)

    def get_camera_frame_rgb(self) -> Any:
        _require_dependency("opencv-python", cv2)
        return cv2.cvtColor(self.get_camera_frame_bgr(), cv2.COLOR_BGR2RGB)

    def get_rgbd(self) -> dict[str, Any]:
        _require_dependency("opencv-python", cv2)
        bgr = self.get_camera_frame_bgr()
        return {"rgb_bgr": bgr, "rgb_rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "jpeg": self.get_camera_image_jpeg()}

    def get_sensors_lidar(self) -> Any:
        with self._lock:
            msg = self._lidar_cloud
        if msg is None:
            return None
        _require_dependency("numpy", np)
        try:
            width = int(msg.width)
            height = int(msg.height)
            step = int(msg.point_step)
            raw = bytes(msg.data)
            fields = {field.name.lower(): field for field in msg.fields}
            dtype = np.dtype(
                {
                    "names": ["x", "y", "z"],
                    "formats": ["<f4", "<f4", "<f4"],
                    "offsets": [fields["x"].offset, fields["y"].offset, fields["z"].offset],
                    "itemsize": step,
                }
            )
            arr = np.frombuffer(raw, dtype=dtype, count=min(width * height, len(raw) // step))
            pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype("float32")
            return pts[np.isfinite(pts).all(axis=1)]
        except Exception:
            return None

    def get_lidar_points(self, max_points: int = 20000) -> list[dict[str, float]]:
        pts = self.get_sensors_lidar()
        if pts is None:
            return []
        if max_points and len(pts) > max_points:
            _require_dependency("numpy", np)
            pts = pts[np.linspace(0, len(pts) - 1, max_points, dtype=np.int64)]
        return [{"x": float(point[0]), "y": float(point[1]), "z": float(point[2])} for point in pts]

    def move_joint(self, q: Any, dq: Any, kp: Any, kd: Any, tau: Any) -> None:
        targets = q if isinstance(q, dict) else {idx: float(value) for idx, value in enumerate(q)}
        for idx, pos in targets.items():
            motor_cmd = self.arm_cmd.motor_cmd[int(idx)]
            motor_cmd.mode = 1
            motor_cmd.q = float(pos)
            motor_cmd.dq = float(dq[idx] if isinstance(dq, dict) else dq)
            motor_cmd.kp = float(kp[idx] if isinstance(kp, dict) else kp)
            motor_cmd.kd = float(kd[idx] if isinstance(kd, dict) else kd)
            motor_cmd.tau = float(tau[idx] if isinstance(tau, dict) else tau)
        self.arm_cmd.crc = self.arm_crc.Crc(self.arm_cmd)
        self.arm_pub.Write(self.arm_cmd)

    def execute_arm_action(self, action: str | int, release_after_s: float | None = None) -> int:
        if isinstance(action, str):
            key = " ".join(str(action).strip().lower().replace("_", " ").split())
            key = HL_ARM_ALIASES.get(key, key)
            action_id = HL_ARM_ACTIONS[key]
        else:
            action_id = int(action)
        code = int(self.arm_action_client.ExecuteAction(action_id))
        if release_after_s is not None:
            time.sleep(max(0.0, float(release_after_s)))
            self.arm_action_client.ExecuteAction(HL_ARM_ACTIONS["release arm"])
        return code

    def release_arm(self) -> int:
        return self.execute_arm_action("release arm")

    def shake_hand(self, release_after_s: float = ARM_RELEASE_DELAY_S) -> int:
        return self.execute_arm_action("shake hand", release_after_s=release_after_s)

    def high_five(self, release_after_s: float = ARM_RELEASE_DELAY_S) -> int:
        return self.execute_arm_action("high five", release_after_s=release_after_s)

    def hug(self, release_after_s: float = ARM_RELEASE_DELAY_S) -> int:
        return self.execute_arm_action("hug", release_after_s=release_after_s)

    def high_wave(self) -> int:
        return self.execute_arm_action("high wave")

    def clap(self) -> int:
        return self.execute_arm_action("clap")

    def face_wave(self) -> int:
        return self.execute_arm_action("face wave")

    def left_kiss(self) -> int:
        return self.execute_arm_action("left kiss")

    def right_kiss(self) -> int:
        return self.execute_arm_action("right kiss")

    def two_hand_kiss(self) -> int:
        return self.execute_arm_action("two-hand kiss")

    def heart(self, release_after_s: float = ARM_RELEASE_DELAY_S) -> int:
        return self.execute_arm_action("heart", release_after_s=release_after_s)

    def right_heart(self, release_after_s: float = ARM_RELEASE_DELAY_S) -> int:
        return self.execute_arm_action("right heart", release_after_s=release_after_s)

    def hands_up(self, release_after_s: float = ARM_RELEASE_DELAY_S) -> int:
        return self.execute_arm_action("hands up", release_after_s=release_after_s)

    def x_ray(self, release_after_s: float = ARM_RELEASE_DELAY_S) -> int:
        return self.execute_arm_action("x-ray", release_after_s=release_after_s)

    def right_hand_up(self, release_after_s: float = ARM_RELEASE_DELAY_S) -> int:
        return self.execute_arm_action("right hand up", release_after_s=release_after_s)

    def reject(self, release_after_s: float = ARM_RELEASE_DELAY_S) -> int:
        return self.execute_arm_action("reject", release_after_s=release_after_s)

    def say(self, text: str, volume: int | None = None) -> int:
        client = _require_dependency("AudioClient", self.audio_client)
        if subprocess.call(["/usr/bin/env", "which", "espeak"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
            raise RuntimeError("espeak is required for say().")
        if volume is not None:
            code = int(client.SetVolume(int(volume)))
            if code != 0:
                return code
        with tempfile.TemporaryDirectory(prefix="g1_say_") as temp_dir:
            wav_path = Path(temp_dir) / "speech.wav"
            robot_wav_path = Path(temp_dir) / "speech_robot.wav"
            subprocess.run(["espeak", "-w", str(wav_path), text], check=True)
            return self.play_wav(_convert_wav_for_robot(wav_path, robot_wav_path), volume=None)

    def play_wav(self, wav_path: str | os.PathLike[str], volume: int | None = None) -> int:
        client = _require_dependency("AudioClient", self.audio_client)
        if volume is not None:
            code = int(client.SetVolume(int(volume)))
            if code != 0:
                return code
        with wave.open(str(wav_path), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getframerate() != 16000 or wav_file.getsampwidth() != 2:
                raise ValueError("WAV must be mono 16-bit PCM at 16kHz for robot playback")
            pcm = wav_file.readframes(wav_file.getnframes())
        code, _data = client.PlayStream("sdk_lib", "sdk-lib-1", pcm)
        return int(code)

    def headlight(self, color: str = "white", intensity: int = 100, duration: float | None = None) -> int:
        client = _require_dependency("AudioClient", self.audio_client)
        rgb = scale_color(parse_color(color), intensity)
        code = int(client.LedControl(*rgb))
        if code != 0:
            return code
        if duration is not None:
            time.sleep(max(0.0, float(duration)))
            return int(client.LedControl(0, 0, 0))
        return 0

    def hand_open(self, hand: str = "right", hold_s: float = 0.6, rate_hz: float = 50.0) -> None:
        self.hands[str(hand).strip().lower()].open(hold_s=hold_s, rate_hz=rate_hz)

    def hand_close(self, hand: str = "right", hold_s: float = 0.6, rate_hz: float = 50.0) -> None:
        self.hands[str(hand).strip().lower()].close(hold_s=hold_s, rate_hz=rate_hz)

    def hand_pose(
        self,
        targets: list[float],
        hand: str = "right",
        hold_s: float = 0.6,
        rate_hz: float = 50.0,
        kp: float = 1.2,
        kd: float = 0.05,
        tau: float = 0.05,
    ) -> None:
        self.hands[str(hand).strip().lower()].set_targets(targets, hold_s=hold_s, rate_hz=rate_hz, kp=kp, kd=kd, tau=tau)

    def release_fingers(
        self,
        hand: str = "right",
        hold_s: float = 0.5,
        rate_hz: float = 50.0,
        persistent: bool = True,
    ) -> None:
        for side in self._normalized_hand_selection(hand):
            self.hands[side].release_fingers(hold_s=hold_s, rate_hz=rate_hz, persistent=persistent)

    def stop_release_fingers(self, hand: str = "both") -> None:
        for side in self._normalized_hand_selection(hand):
            self.hands[side].stop_release_fingers()

    def teach(
        self,
        *,
        out: str = "/tmp/pbd_motion.npz",
        log_path: str | None = None,
        duration_s: float = 0.0,
        poll_s: float = 0.01,
        timeout: float = 3.0,
    ) -> dict[str, Any]:
        _require_dependency("numpy", np)
        joint_positions = self._read_joint_positions_or_raise(timeout=timeout)
        joint_indices = sorted(joint_positions)
        left_hand = self._get_hand_positions_or_raise("left", timeout=timeout)
        right_hand = self._get_hand_positions_or_raise("right", timeout=timeout)

        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        resolved_log_path = log_path or f"{os.path.splitext(out)[0]}.csv"
        os.makedirs(os.path.dirname(os.path.abspath(resolved_log_path)), exist_ok=True)

        done_event = threading.Event()

        def _wait_for_enter() -> None:
            try:
                input("Press Enter when the teach motion is complete...")
            except EOFError:
                return
            done_event.set()

        threading.Thread(target=_wait_for_enter, name="sdk-lib-teach-enter", daemon=True).start()

        sample_period = max(1e-3, float(poll_s))
        duration_limit = max(0.0, float(duration_s))
        timestamps: list[float] = []
        samples: list[list[float]] = []
        left_hand_samples: list[list[float]] = []
        right_hand_samples: list[list[float]] = []
        start = time.time()
        next_tick = start
        duration_notice_sent = False

        with open(resolved_log_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["phase", "t_s", "joint_indices", "target_positions", "actual_positions"])
            try:
                while True:
                    now = time.time()
                    if now < next_tick:
                        time.sleep(min(0.02, next_tick - now))
                        continue
                    next_tick += sample_period
                    if done_event.is_set():
                        break
                    if duration_limit > 0.0 and (now - start) >= duration_limit and not duration_notice_sent:
                        print("Teach duration limit reached. Press Enter to finish recording.")
                        duration_notice_sent = True
                    joints = self._read_joint_positions_or_raise(timeout=timeout)
                    left_hand = self._get_hand_positions_or_raise("left", timeout=timeout)
                    right_hand = self._get_hand_positions_or_raise("right", timeout=timeout)
                    row = [float(joints[joint_index]) for joint_index in joint_indices]
                    t_rel = now - start
                    timestamps.append(t_rel)
                    samples.append(row)
                    left_hand_samples.append(left_hand)
                    right_hand_samples.append(right_hand)
                    writer.writerow(
                        [
                            "teach",
                            f"{t_rel:.6f}",
                            " ".join([str(joint_index) for joint_index in joint_indices] + [f"left_{idx}" for idx in range(7)] + [f"right_{idx}" for idx in range(7)]),
                            " ".join([f"{value:.6f}" for value in row + left_hand + right_hand]),
                            " ".join([f"{value:.6f}" for value in row + left_hand + right_hand]),
                        ]
                    )
                    handle.flush()
            except KeyboardInterrupt:
                pass

        if not timestamps:
            raise RuntimeError("No samples recorded. Is rt/lowstate publishing?")

        np.savez(
            out,
            joints=np.asarray(joint_indices, dtype=np.int32),
            ts=np.asarray(timestamps, dtype=np.float32),
            qs=np.asarray(samples, dtype=np.float32),
            left_hand_qs=np.asarray(left_hand_samples, dtype=np.float32),
            right_hand_qs=np.asarray(right_hand_samples, dtype=np.float32),
            poll_s=np.asarray([sample_period], dtype=np.float32),
            representation=np.asarray(["joint_space"], dtype="<U16"),
        )
        return {
            "joint_count": len(joint_indices),
            "sample_count": len(timestamps),
            "duration_s": float(timestamps[-1]) if timestamps else 0.0,
            "poll_s": sample_period,
            "out": os.path.abspath(out),
            "log_path": os.path.abspath(resolved_log_path),
        }

    def repeat(
        self,
        *,
        motion_file: str = "/tmp/pbd_motion.npz",
        log_path: str | None = None,
        speed: float = 1.0,
        command_rate_hz: float = 50.0,
        start_ramp_s: float = 0.8,
        final_hold_s: float = 0.8,
        kp: float = 40.0,
        kd: float = 1.0,
        timeout: float = 3.0,
    ) -> dict[str, Any]:
        _require_dependency("numpy", np)
        data = np.load(motion_file, allow_pickle=False)
        if "joints" not in data or "ts" not in data or "qs" not in data:
            raise ValueError("Motion file must contain 'joints', 'ts', and 'qs'.")
        joint_indices = [int(joint_index) for joint_index in np.asarray(data["joints"]).astype(int).tolist()]
        ts = np.asarray(data["ts"], dtype=float)
        qs = np.asarray(data["qs"], dtype=float)
        if ts.size == 0 or qs.size == 0:
            raise ValueError("No samples in motion file.")
        if qs.shape[0] != ts.shape[0] or qs.shape[1] != len(joint_indices):
            raise ValueError("Invalid motion file: joint trajectory shape mismatch.")
        left_hand_qs = np.asarray(data.get("left_hand_qs", np.empty((0, 7))), dtype=float)
        right_hand_qs = np.asarray(data.get("right_hand_qs", np.empty((0, 7))), dtype=float)
        replay_hands = left_hand_qs.shape == (ts.shape[0], 7) and right_hand_qs.shape == (ts.shape[0], 7)
        resolved_log_path = log_path or f"{os.path.splitext(motion_file)[0]}_repeat.csv"
        os.makedirs(os.path.dirname(os.path.abspath(resolved_log_path)), exist_ok=True)

        current_positions = self._read_joint_positions_or_raise(timeout=timeout)
        first_targets = {joint_index: float(qs[0, idx]) for idx, joint_index in enumerate(joint_indices)}
        ramp_steps = max(1, int(max(0.0, float(start_ramp_s)) * max(1.0, float(command_rate_hz))))
        dt = 1.0 / max(1.0, float(command_rate_hz))
        for step_idx in range(1, ramp_steps + 1):
            alpha = float(step_idx) / float(ramp_steps)
            blended = {
                joint_index: float(current_positions[joint_index]) + (float(first_targets[joint_index]) - float(current_positions[joint_index])) * alpha
                for joint_index in joint_indices
            }
            self.move_joint(blended, dq=0.0, kp=kp, kd=kd, tau=0.0)
            if replay_hands:
                self.hands["left"].write_targets_once(left_hand_qs[0].tolist(), kp=0.8, kd=0.05, tau=0.02)
                self.hands["right"].write_targets_once(right_hand_qs[0].tolist(), kp=0.8, kd=0.05, tau=0.02)
            time.sleep(dt)

        replay_ts = ts / max(1e-6, float(speed))
        duration_total = float(replay_ts[-1])
        start = time.time()
        with open(resolved_log_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["phase", "t_s", "joint_indices", "target_positions", "actual_positions"])
            while True:
                elapsed = time.time() - start
                if elapsed > duration_total:
                    break
                desired_row = np.asarray(
                    [np.interp(elapsed, replay_ts, qs[:, idx]) for idx in range(len(joint_indices))],
                    dtype=float,
                )
                targets = {joint_index: float(desired_row[idx]) for idx, joint_index in enumerate(joint_indices)}
                self.move_joint(targets, dq=0.0, kp=kp, kd=kd, tau=0.0)
                if replay_hands:
                    left_desired = [float(np.interp(elapsed, replay_ts, left_hand_qs[:, idx])) for idx in range(7)]
                    right_desired = [float(np.interp(elapsed, replay_ts, right_hand_qs[:, idx])) for idx in range(7)]
                    self.hands["left"].write_targets_once(left_desired, kp=0.8, kd=0.05, tau=0.02)
                    self.hands["right"].write_targets_once(right_desired, kp=0.8, kd=0.05, tau=0.02)
                else:
                    left_desired = []
                    right_desired = []
                actual_positions = self._read_joint_positions_or_raise(timeout=timeout)
                actual_row = [float(actual_positions.get(joint_index, 0.0)) for joint_index in joint_indices]
                writer.writerow(
                    [
                        "repeat",
                        f"{elapsed:.6f}",
                        " ".join([str(joint_index) for joint_index in joint_indices] + ([f"left_{idx}" for idx in range(7)] + [f"right_{idx}" for idx in range(7)] if replay_hands else [])),
                        " ".join([f"{value:.6f}" for value in list(desired_row) + left_desired + right_desired]),
                        " ".join([f"{value:.6f}" for value in actual_row]),
                    ]
                )
                handle.flush()
                time.sleep(dt)

            final_targets = {joint_index: float(qs[-1, idx]) for idx, joint_index in enumerate(joint_indices)}
            hold_deadline = time.time() + max(0.0, float(final_hold_s))
            while time.time() < hold_deadline:
                self.move_joint(final_targets, dq=0.0, kp=kp, kd=kd, tau=0.0)
                if replay_hands:
                    self.hands["left"].write_targets_once(left_hand_qs[-1].tolist(), kp=0.8, kd=0.05, tau=0.02)
                    self.hands["right"].write_targets_once(right_hand_qs[-1].tolist(), kp=0.8, kd=0.05, tau=0.02)
                time.sleep(dt)

        return {
            "motion_file": os.path.abspath(motion_file),
            "joint_count": len(joint_indices),
            "sample_count": int(ts.shape[0]),
            "command_rate_hz": float(command_rate_hz),
            "speed": max(1e-6, float(speed)),
            "duration_s": duration_total,
            "final_hold_s": max(0.0, float(final_hold_s)),
            "log_path": os.path.abspath(resolved_log_path),
        }

    def start_slam(self, slam_type: str = "indoor") -> int:
        response = self.slam_client.start_mapping(slam_type=slam_type)
        self.slam_is_running = response.code == 0
        return int(response.code)

    def stop_slam(self, save_path: str | None = None) -> int:
        response = self.slam_client.end_mapping(save_path) if save_path else self.slam_client.close_slam()
        self.slam_is_running = False
        return int(response.code)

    def get_slam_pose(self, timeout_s: float = 0.4) -> tuple[float, float, float] | None:
        deadline = time.time() + max(0.05, float(timeout_s))
        while time.time() < deadline:
            pose = self.slam_info_sub.get_pose()
            if pose is not None:
                return pose
            time.sleep(0.03)
        return None

    def set_path_point(self, x: float, y: float, yaw: float = 0.0) -> None:
        self._path_points.append((float(x), float(y), float(yaw)))

    def get_path_points(self) -> list[tuple[float, float, float]]:
        return list(self._path_points)

    def clear_path_points(self) -> None:
        self._path_points.clear()

    def navigate_path(self, clear_on_finish: bool = True) -> bool:
        if not self._path_points:
            raise RuntimeError("No path points. Call set_path_point() first.")
        if not self.slam_is_running:
            print("[navigate_path] SLAM not running.")
            return False
        try:
            self.fsm_walk()
        except Exception:
            pass
        ok = True
        try:
            for idx, (x, y, yaw) in enumerate(self._path_points, 1):
                pos = self.get_position()
                if pos is not None and math.hypot(x - pos[0], y - pos[1]) <= 0.20:
                    print(f"[navigate_path] step={idx} skipped: already within 0.20m of target.")
                    continue
                rc = self._run_pose_nav(x, y, yaw)
                print(f"[navigate_path] step={idx} pose_nav rc={rc}")
                if rc != 0:
                    print(f"[navigate_path] failed at point {idx}: ({x:.3f},{y:.3f},{yaw:.3f})")
                    ok = False
                    break
        finally:
            if clear_on_finish:
                self._path_points.clear()
        return ok
