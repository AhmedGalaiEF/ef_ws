from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any

from dds_env import ensure_cyclonedds_environment

ensure_cyclonedds_environment()

from unitree_sdk2py.core.channel import ChannelSubscriber

try:
    from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
except Exception:
    Odometry_ = None  # type: ignore[assignment]

try:
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import Imu_ as LidarImu_
except Exception:
    LidarImu_ = None  # type: ignore[assignment]


def _try_import(module_path: str) -> Any:
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None


def resolve_lowstate_type() -> type | None:
    for module_path in (
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "unitree_sdk2py.idl.unitree_go.msg.dds_",
    ):
        module = _try_import(module_path)
        if module is not None and hasattr(module, "LowState_"):
            return getattr(module, "LowState_")
    return None


def load_video_client_type():
    last_exc: Exception | None = None
    for path in (
        "unitree_sdk2py.g1.video.video_client",
        "unitree_sdk2py.go2.video.video_client",
    ):
        try:
            module = importlib.import_module(path)
            return module.VideoClient
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Could not import VideoClient from known paths. Last error: {last_exc}")


@dataclass
class LowStateSnapshot:
    stamp: float
    joint_positions: list[float]
    joint_velocities: list[float]
    joint_torques: list[float]
    imu_rpy: tuple[float, float, float] | None
    imu_gyro: tuple[float, float, float] | None
    imu_acc: tuple[float, float, float] | None


class LatestSubscriber:
    def __init__(self, topic: str, msg_type: type) -> None:
        self.topic = topic
        self.msg_type = msg_type
        self._msg: Any = None
        self._last_ts: float = 0.0
        self._sub: ChannelSubscriber | None = None

    def start(self, queue_len: int = 10) -> None:
        if self._sub is None:
            self._sub = ChannelSubscriber(self.topic, self.msg_type)
            self._sub.Init(self._callback, int(queue_len))

    def _callback(self, msg: Any) -> None:
        self._msg = msg
        self._last_ts = time.time()

    def get_latest(self) -> tuple[Any, float]:
        return self._msg, self._last_ts

    def is_stale(self, max_age: float = 1.0) -> bool:
        if self._last_ts <= 0.0:
            return True
        return (time.time() - self._last_ts) > float(max_age)


def decode_video_frame_bgr(payload: Any):
    import cv2
    import numpy as np

    jpg = np.frombuffer(bytes(payload), dtype=np.uint8)
    if jpg.size == 0:
        raise RuntimeError("Received empty image payload from GetImageSample().")
    frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode JPEG payload into BGR frame.")
    return frame


def lowstate_snapshot_from_msg(msg: Any) -> LowStateSnapshot:
    positions: list[float] = []
    velocities: list[float] = []
    torques: list[float] = []
    motor_state = list(getattr(msg, "motor_state", []) or [])
    for motor in motor_state:
        try:
            positions.append(float(getattr(motor, "q")))
        except Exception:
            positions.append(0.0)
        try:
            velocities.append(float(getattr(motor, "dq")))
        except Exception:
            velocities.append(0.0)
        try:
            torques.append(float(getattr(motor, "tau_est")))
        except Exception:
            torques.append(0.0)

    imu = getattr(msg, "imu_state", None)
    imu_rpy = imu_gyro = imu_acc = None
    if imu is not None:
        try:
            imu_rpy = tuple(float(imu.rpy[i]) for i in range(3))  # type: ignore[assignment]
        except Exception:
            imu_rpy = None
        try:
            imu_gyro = tuple(float(imu.gyroscope[i]) for i in range(3))  # type: ignore[assignment]
        except Exception:
            imu_gyro = None
        try:
            imu_acc = tuple(float(imu.accelerometer[i]) for i in range(3))  # type: ignore[assignment]
        except Exception:
            imu_acc = None

    return LowStateSnapshot(
        stamp=time.time(),
        joint_positions=positions,
        joint_velocities=velocities,
        joint_torques=torques,
        imu_rpy=imu_rpy,
        imu_gyro=imu_gyro,
        imu_acc=imu_acc,
    )


def odom_pose_from_msg(msg: Any) -> tuple[float, float, float] | None:
    try:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        x = float(pos.x)
        y = float(pos.y)
        qx = float(ori.x)
        qy = float(ori.y)
        qz = float(ori.z)
        qw = float(ori.w)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = float(__import__("math").atan2(siny_cosp, cosy_cosp))
        return (x, y, yaw)
    except Exception:
        return None


__all__ = [
    "LatestSubscriber",
    "LidarImu_",
    "LowStateSnapshot",
    "Odometry_",
    "decode_video_frame_bgr",
    "load_video_client_type",
    "lowstate_snapshot_from_msg",
    "odom_pose_from_msg",
    "resolve_lowstate_type",
]
