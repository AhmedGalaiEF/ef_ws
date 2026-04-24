"""
sdk_client.py
=============

SDK-native Robot wrapper for Unitree G1.

The implementation is intentionally local to `modules/` and avoids imports
from `../scripts`. Script-backed workflows were replaced with direct SDK
helpers or removed from the core wrapper.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

from sdk_audio import RobotAudio
from sdk_boot import create_loco_client, hanger_boot_sequence, rpc_get_int
from sdk_hand import Dex3HandController
from sdk_sensors import (
    LatestSubscriber,
    LidarImu_,
    LowStateSnapshot,
    Odometry_,
    decode_video_frame_bgr,
    load_video_client_type,
    lowstate_snapshot_from_msg,
    odom_pose_from_msg,
    resolve_lowstate_type,
)
from sdk_slam import SlamInfoSubscriber, SlamOdomSubscriber, SlamOperateClient, SlamResponse

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import HeightMap_, SportModeState_
    from unitree_sdk2py.g1.loco.g1_loco_api import (
        ROBOT_API_ID_LOCO_GET_FSM_ID,
        ROBOT_API_ID_LOCO_GET_FSM_MODE,
    )
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


DEFAULT_SPORT_TOPIC = "rt/odommodestate"
DEFAULT_LIDAR_MAP_TOPIC = "rt/utlidar/map_state"
DEFAULT_LIDAR_CLOUD_TOPIC = "rt/utlidar/cloud_deskewed"


@dataclass
class ImuData:
    rpy: tuple[float, float, float]
    gyro: tuple[float, float, float] | None
    acc: tuple[float, float, float] | None
    quat: tuple[float, float, float, float] | None
    temp: float | None


class Robot:
    """End-user wrapper around common G1 SDK workflows."""

    def __init__(
        self,
        iface: str = "eth0",
        domain_id: int = 0,
        safety_boot: bool = False,
        auto_start_sensors: bool = True,
        sport_topic: str = DEFAULT_SPORT_TOPIC,
        lidar_map_topic: str = DEFAULT_LIDAR_MAP_TOPIC,
        lidar_cloud_topic: str = DEFAULT_LIDAR_CLOUD_TOPIC,
        slam_info_topic: str = "rt/slam_info",
        slam_key_topic: str = "rt/slam_key_info",
    ) -> None:
        self.iface = iface
        self.domain_id = int(domain_id)
        self.sport_topic = sport_topic
        self.lidar_map_topic = lidar_map_topic
        self.lidar_cloud_topic = lidar_cloud_topic
        self.slam_info_topic = slam_info_topic
        self.slam_key_topic = slam_key_topic

        self._lock = threading.Lock()
        self._sport: SportModeState_ | None = None
        self._lidar_map: HeightMap_ | None = None
        self._lidar_cloud: PointCloud2_ | None = None
        self._last_sport_ts = 0.0
        self._last_lidar_map_ts = 0.0
        self._last_lidar_cloud_ts = 0.0

        self._sport_sub: ChannelSubscriber | None = None
        self._lidar_map_sub: ChannelSubscriber | None = None
        self._lidar_cloud_sub: ChannelSubscriber | None = None
        self._lowstate_sub: LatestSubscriber | None = None
        self._odom_sub: LatestSubscriber | None = None
        self._lidar_imu_sub: LatestSubscriber | None = None
        self._slam_info_sub: SlamInfoSubscriber | None = None
        self._slam_odom_sub: SlamOdomSubscriber | None = None

        self._path_points: list[tuple[float, float, float]] = []
        self._slam_client: SlamOperateClient | None = None
        self._audio: RobotAudio | None = None
        self._video_client: Any = None
        self._hands: dict[str, Dex3HandController] = {}
        self._usb_controller_thread: threading.Thread | None = None
        self._usb_controller_stop = threading.Event()
        self.slam_is_running = False

        if safety_boot:
            self._client = hanger_boot_sequence(iface=self.iface, domain_id=self.domain_id)
        else:
            self._client = create_loco_client(domain_id=self.domain_id, iface=self.iface)

        if auto_start_sensors:
            self.start_sensors()

    def _get_slam_client(self) -> SlamOperateClient:
        if self._slam_client is None:
            self._slam_client = SlamOperateClient()
            self._slam_client.Init()
            self._slam_client.SetTimeout(10.0)
        return self._slam_client

    def _get_audio(self) -> RobotAudio:
        if self._audio is None:
            self._audio = RobotAudio()
        return self._audio

    def _get_video_client(self) -> Any:
        if self._video_client is None:
            video_client_cls = load_video_client_type()
            self._video_client = video_client_cls()
            self._video_client.SetTimeout(2.0)
            self._video_client.Init()
        return self._video_client

    def _get_hand(self, hand: str = "right") -> Dex3HandController:
        side = str(hand).strip().lower()
        if side not in self._hands:
            self._hands[side] = Dex3HandController(side)
        return self._hands[side]

    def _ensure_slam_info_subscriber(self) -> SlamInfoSubscriber:
        if self._slam_info_sub is None:
            self._slam_info_sub = SlamInfoSubscriber(self.slam_info_topic, self.slam_key_topic)
            self._slam_info_sub.start()
        return self._slam_info_sub

    def _ensure_slam_odom_subscriber(self) -> SlamOdomSubscriber:
        if self._slam_odom_sub is None:
            self._slam_odom_sub = SlamOdomSubscriber()
            self._slam_odom_sub.start()
        return self._slam_odom_sub

    # ------------------------------------------------------------------
    # Sensor subscriptions
    # ------------------------------------------------------------------

    def start_sensors(self) -> None:
        if self._sport_sub is None:
            self._sport_sub = ChannelSubscriber(self.sport_topic, SportModeState_)
            self._sport_sub.Init(self._sport_cb, 10)
        if self._lidar_map_sub is None:
            self._lidar_map_sub = ChannelSubscriber(self.lidar_map_topic, HeightMap_)
            self._lidar_map_sub.Init(self._lidar_map_cb, 10)
        if self._lidar_cloud_sub is None:
            self._lidar_cloud_sub = ChannelSubscriber(self.lidar_cloud_topic, PointCloud2_)
            self._lidar_cloud_sub.Init(self._lidar_cloud_cb, 10)
        lowstate_type = resolve_lowstate_type()
        if lowstate_type is not None and self._lowstate_sub is None:
            self._lowstate_sub = LatestSubscriber("rt/lowstate", lowstate_type)
            self._lowstate_sub.start()
        if Odometry_ is not None and self._odom_sub is None:
            self._odom_sub = LatestSubscriber("rt/odom", Odometry_)
            self._odom_sub.start()
        if LidarImu_ is not None and self._lidar_imu_sub is None:
            self._lidar_imu_sub = LatestSubscriber("rt/utlidar/imu_livox_mid360", LidarImu_)
            self._lidar_imu_sub.start()

    def _sport_cb(self, msg: SportModeState_) -> None:
        with self._lock:
            self._sport = msg
            self._last_sport_ts = time.time()

    def _lidar_map_cb(self, msg: HeightMap_) -> None:
        with self._lock:
            self._lidar_map = msg
            self._last_lidar_map_ts = time.time()

    def _lidar_cloud_cb(self, msg: PointCloud2_) -> None:
        with self._lock:
            self._lidar_cloud = msg
            self._last_lidar_cloud_ts = time.time()

    # ------------------------------------------------------------------
    # Generic state helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_attr(obj: Any, *path: str) -> Any:
        cur = obj
        for name in path:
            if cur is None or not hasattr(cur, name):
                return None
            cur = getattr(cur, name)
        return cur

    @staticmethod
    def _vector3_from(value: Any) -> tuple[float, float, float] | None:
        try:
            if value is None:
                return None
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return None
        return None

    def get_sport_state(self) -> SportModeState_ | None:
        with self._lock:
            return self._sport

    def get_lidar_map(self) -> HeightMap_ | None:
        with self._lock:
            return self._lidar_map

    def get_lidar_cloud(self) -> PointCloud2_ | None:
        with self._lock:
            return self._lidar_cloud

    def get_sensor_timestamps(self) -> dict[str, float]:
        with self._lock:
            timestamps = {
                "sport": float(self._last_sport_ts),
                "lidar_map": float(self._last_lidar_map_ts),
                "lidar_cloud": float(self._last_lidar_cloud_ts),
            }
        if self._lowstate_sub is not None:
            timestamps["lowstate"] = float(self._lowstate_sub.get_latest()[1])
        if self._odom_sub is not None:
            timestamps["odom"] = float(self._odom_sub.get_latest()[1])
        if self._lidar_imu_sub is not None:
            timestamps["lidar_imu"] = float(self._lidar_imu_sub.get_latest()[1])
        if self._slam_odom_sub is not None:
            timestamps["slam_odom"] = float(self._slam_odom_sub.get_latest()[1])
        return timestamps

    def sensors_stale(self, max_age: float = 1.0) -> dict[str, bool]:
        now = time.time()
        return {
            name: (ts <= 0.0) or ((now - ts) > max_age)
            for name, ts in self.get_sensor_timestamps().items()
        }

    def wait_for_sport_state(self, timeout: float = 2.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < max(0.0, timeout):
            if self.get_sport_state() is not None:
                return True
            time.sleep(0.05)
        return self.get_sport_state() is not None

    def wait_for_low_state(self, timeout: float = 2.0) -> bool:
        if self._lowstate_sub is None:
            return False
        t0 = time.time()
        while time.time() - t0 < max(0.0, timeout):
            if self._lowstate_sub.get_latest()[0] is not None:
                return True
            time.sleep(0.05)
        return self._lowstate_sub.get_latest()[0] is not None

    def get_mode(self) -> int | None:
        msg = self.get_sport_state()
        if msg is None:
            return None
        value = self._read_attr(msg, "mode")
        try:
            return int(value)
        except Exception:
            return None

    def get_gait(self) -> int | None:
        msg = self.get_sport_state()
        if msg is None:
            return None
        for key in ("gait_type", "gaitType", "gait"):
            value = self._read_attr(msg, key)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return None

    def get_body_height(self) -> float | None:
        msg = self.get_sport_state()
        if msg is None:
            return None
        for key in ("body_height", "bodyHeight", "stand_height", "standHeight"):
            value = self._read_attr(msg, key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                continue
        return None

    def get_position(self) -> tuple[float, float, float] | None:
        msg = self.get_sport_state()
        if msg is None:
            return None
        for key in ("position", "pos", "position_w"):
            vec = self._vector3_from(self._read_attr(msg, key))
            if vec is not None:
                return vec
        return None

    def get_velocity(self) -> tuple[float, float, float] | None:
        msg = self.get_sport_state()
        if msg is None:
            return None
        for key in ("velocity", "vel", "velocity_w"):
            vec = self._vector3_from(self._read_attr(msg, key))
            if vec is not None:
                return vec
        return None

    def get_low_state(self) -> Any | None:
        if self._lowstate_sub is None:
            return None
        return self._lowstate_sub.get_latest()[0]

    def get_low_state_snapshot(self) -> LowStateSnapshot | None:
        msg = self.get_low_state()
        if msg is None:
            return None
        return lowstate_snapshot_from_msg(msg)

    def get_joint_positions(self) -> list[float]:
        snap = self.get_low_state_snapshot()
        return [] if snap is None else list(snap.joint_positions)

    def get_joint_velocities(self) -> list[float]:
        snap = self.get_low_state_snapshot()
        return [] if snap is None else list(snap.joint_velocities)

    def get_joint_torques(self) -> list[float]:
        snap = self.get_low_state_snapshot()
        return [] if snap is None else list(snap.joint_torques)

    def get_joint_position(self, joint_index: int) -> float | None:
        positions = self.get_joint_positions()
        idx = int(joint_index)
        if idx < 0 or idx >= len(positions):
            return None
        return float(positions[idx])

    def get_odom(self) -> Any | None:
        if self._odom_sub is None:
            return None
        return self._odom_sub.get_latest()[0]

    def get_odom_pose(self) -> tuple[float, float, float] | None:
        msg = self.get_odom()
        if msg is None:
            return None
        return odom_pose_from_msg(msg)

    def get_lidar_imu(self) -> Any | None:
        if self._lidar_imu_sub is None:
            return None
        return self._lidar_imu_sub.get_latest()[0]

    def get_yaw(self) -> float | None:
        imu = self.get_imu()
        if imu is None:
            return None
        return float(imu.rpy[2])

    def is_moving(self, linear_eps: float = 0.03, yaw_eps: float = 0.08) -> bool:
        velocity = self.get_velocity()
        if velocity is None:
            return False
        vx, vy, vz = velocity
        return math.hypot(vx, vy) > linear_eps or abs(vz) > yaw_eps

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
            "odom_pose": self.get_odom_pose(),
            "slam_pose": self.get_slam_pose(),
            "joint_count": len(self.get_joint_positions()),
            "sensor_timestamps": self.get_sensor_timestamps(),
            "sensor_stale": self.sensors_stale(),
            "slam_is_running": bool(self.slam_is_running),
            "queued_path_points": len(self._path_points),
        }

    # ------------------------------------------------------------------
    # Locomotion + FSM
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_sdk_status(result: Any) -> int:
        # Some SDK bindings return None for successful non-blocking motion calls.
        if result is None:
            return 0
        return int(result)

    def loco_move(self, vx: float, vy: float, vyaw: float) -> int:
        result = self._client.Move(float(vx), float(vy), float(vyaw), continous_move=True)
        return self._normalize_sdk_status(result)

    def walk(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> int:
        self.set_gait_type(0)
        return self.loco_move(vx, vy, vyaw)

    def run(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> int:
        self.set_gait_type(1)
        return self.loco_move(vx, vy, vyaw)

    def stop_moving(self) -> None:
        if hasattr(self._client, "StopMove"):
            self._client.StopMove()
            return
        self._client.Move(0.0, 0.0, 0.0, continous_move=False)

    def stop(self) -> None:
        self.stop_moving()

    @staticmethod
    def _apply_deadzone(value: float, deadzone: float) -> float:
        dz = min(0.99, max(0.0, float(deadzone)))
        sample = float(value)
        if abs(sample) < dz:
            return 0.0
        sign = 1.0 if sample > 0.0 else -1.0
        return sign * (abs(sample) - dz) / (1.0 - dz)

    def zero_torque(self) -> None:
        self.fsm_0_zt()

    def damp(self) -> None:
        self.fsm_1_damp()

    def enter_walking_ready_mode(self) -> None:
        if hasattr(self._client, "Damp"):
            self._client.Damp()
            time.sleep(0.5)
        if hasattr(self._client, "Squat2StandUp"):
            self._client.Squat2StandUp()
            time.sleep(1.0)
        if hasattr(self._client, "Start"):
            self._client.Start()
            return
        self.balanced_stand()

    def _usb_controller_loop(
        self,
        *,
        joy_index: int,
        send_hz: float,
        max_vx: float,
        max_vy: float,
        max_vyaw: float,
        deadzone: float,
    ) -> None:
        try:
            import pygame
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "The 'pygame' package is required for USB controller support.\n"
                "Install with: pip install pygame"
            ) from exc

        btn_a = 0
        btn_b = 1
        btn_x = 2
        btn_y = 3
        btn_start = 7
        axis_lx = 0
        axis_ly = 1
        axis_rx = 3

        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            pygame.quit()
            raise RuntimeError("No joystick detected. Connect a USB gamepad and retry.")
        if joy_index < 0 or joy_index >= pygame.joystick.get_count():
            pygame.quit()
            raise IndexError(f"Joystick index {joy_index} is out of range.")

        joy = pygame.joystick.Joystick(int(joy_index))
        joy.init()

        active = True
        dt = 1.0 / max(1.0, float(send_hz))
        try:
            while not self._usb_controller_stop.is_set():
                pygame.event.pump()

                if joy.get_numbuttons() > btn_y and joy.get_button(btn_y):
                    self.zero_torque()
                    active = False
                    time.sleep(0.5)
                    continue

                if joy.get_numbuttons() > btn_a and joy.get_button(btn_a):
                    self.damp()
                    active = False
                    time.sleep(0.5)
                    continue

                if joy.get_numbuttons() > btn_x and joy.get_button(btn_x):
                    self.enter_walking_ready_mode()
                    active = True
                    time.sleep(0.5)
                    continue

                if joy.get_numbuttons() > btn_b and joy.get_button(btn_b):
                    self.balanced_stand()
                    active = True
                    time.sleep(0.5)
                    continue

                if joy.get_numbuttons() > btn_start and joy.get_button(btn_start):
                    self.stop()
                    time.sleep(0.2)
                    continue

                if active:
                    lx = self._apply_deadzone(joy.get_axis(axis_lx), deadzone)
                    ly = self._apply_deadzone(joy.get_axis(axis_ly), deadzone)
                    rx = self._apply_deadzone(joy.get_axis(axis_rx), deadzone)

                    vx = -ly * float(max_vx)
                    vy = -lx * float(max_vy)
                    vyaw = -rx * float(max_vyaw)
                    self.walk(vx=vx, vy=vy, vyaw=vyaw)

                time.sleep(dt)
        finally:
            self.stop()
            joy.quit()
            pygame.joystick.quit()
            pygame.quit()

    def start_usb_controller(
        self,
        joy_index: int = 0,
        send_hz: float = 10.0,
        max_vx: float = 0.5,
        max_vy: float = 0.3,
        max_vyaw: float = 0.8,
        deadzone: float = 0.1,
    ) -> threading.Thread:
        thread = self._usb_controller_thread
        if thread is not None and thread.is_alive():
            raise RuntimeError("USB controller loop is already running.")

        try:
            import pygame
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The 'pygame' package is required for USB controller support. "
                "Install it with: pip install pygame"
            ) from exc

        pygame.init()
        pygame.joystick.init()
        try:
            if pygame.joystick.get_count() == 0:
                raise RuntimeError("No joystick detected. Connect a USB gamepad and retry.")
            if joy_index < 0 or joy_index >= pygame.joystick.get_count():
                raise IndexError(f"Joystick index {joy_index} is out of range.")
        finally:
            pygame.joystick.quit()
            pygame.quit()

        self._usb_controller_stop = threading.Event()
        self._usb_controller_thread = threading.Thread(
            target=self._usb_controller_loop,
            kwargs={
                "joy_index": int(joy_index),
                "send_hz": float(send_hz),
                "max_vx": float(max_vx),
                "max_vy": float(max_vy),
                "max_vyaw": float(max_vyaw),
                "deadzone": float(deadzone),
            },
            name=f"usb-controller-{int(joy_index)}",
            daemon=True,
        )
        self._usb_controller_thread.start()
        return self._usb_controller_thread

    def stop_usb_controller(self, join_timeout: float = 1.0) -> None:
        self._usb_controller_stop.set()
        thread = self._usb_controller_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(join_timeout)))
        self._usb_controller_thread = None

    @staticmethod
    def _wrap_angle(value: float) -> float:
        while value > math.pi:
            value -= 2.0 * math.pi
        while value < -math.pi:
            value += 2.0 * math.pi
        return value

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _normalize_gait_type(gait_type: int | str) -> int:
        if isinstance(gait_type, str):
            key = gait_type.strip().lower().replace("-", "_").replace(" ", "_")
            alias = {
                "normal": 0,
                "balanced": 0,
                "balance": 0,
                "static": 0,
                "stand": 0,
                "continuous": 1,
                "walk": 1,
                "walking": 1,
                "dynamic": 1,
                "run": 1,
            }
            if key not in alias:
                raise ValueError(f"Unknown gait_type '{gait_type}'.")
            return int(alias[key])
        return int(gait_type)

    def set_gait_type(self, gait_type: int | str = 0) -> int:
        mode = self._normalize_gait_type(gait_type)
        if hasattr(self._client, "SetGaitType"):
            return int(self._client.SetGaitType(mode))
        if hasattr(self._client, "SetBalanceMode"):
            return int(self._client.SetBalanceMode(mode))
        raise AttributeError("Current locomotion client does not support gait mode setting API.")

    def balanced_stand(self, mode: int = 0) -> None:
        if hasattr(self._client, "BalanceStand"):
            self._client.BalanceStand(int(mode))
        else:
            self.set_gait_type(int(mode))

    def _move_for_feedback(
        self,
        distance: float,
        gait_type: int,
        max_vx: float,
        max_vyaw: float,
        pos_tolerance: float,
        yaw_tolerance: float,
        timeout: float,
        tick: float,
        kp_lin: float,
        kp_yaw: float,
    ) -> bool:
        pos0 = self.get_position()
        yaw0 = self.get_yaw()
        if pos0 is None or yaw0 is None:
            raise RuntimeError("walk_for/run_for requires live position and IMU yaw.")

        if abs(float(distance)) <= float(pos_tolerance):
            self.stop()
            return True

        sign = 1.0 if float(distance) >= 0.0 else -1.0
        target_x = float(pos0[0]) + float(distance) * math.cos(float(yaw0))
        target_y = float(pos0[1]) + float(distance) * math.sin(float(yaw0))

        self.set_gait_type(int(gait_type))
        t0 = time.time()
        ok = False
        try:
            while (time.time() - t0) <= max(0.1, float(timeout)):
                pos = self.get_position()
                yaw = self.get_yaw()
                if pos is None or yaw is None:
                    time.sleep(max(0.01, float(tick)))
                    continue
                dx = target_x - float(pos[0])
                dy = target_y - float(pos[1])
                dist = math.hypot(dx, dy)
                if dist <= float(pos_tolerance):
                    ok = True
                    break
                target_heading = math.atan2(dy, dx)
                heading_err = self._wrap_angle(target_heading - float(yaw))
                if abs(heading_err) > float(yaw_tolerance):
                    vx_cmd = 0.0
                else:
                    vx_cmd = sign * self._clamp(float(kp_lin) * dist, 0.0, max(0.0, float(max_vx)))
                vyaw_cmd = self._clamp(float(kp_yaw) * heading_err, -max_vyaw, max_vyaw)
                self.loco_move(vx_cmd, 0.0, vyaw_cmd)
                time.sleep(max(0.01, float(tick)))
        finally:
            self.stop()
        return ok

    def walk_for(
        self,
        distance: float,
        max_vx: float = 0.25,
        max_vyaw: float = 0.5,
        pos_tolerance: float = 0.05,
        yaw_tolerance: float = 0.20,
        timeout: float = 20.0,
        tick: float = 0.05,
        kp_lin: float = 0.9,
        kp_yaw: float = 1.6,
    ) -> bool:
        return self._move_for_feedback(
            distance=distance,
            gait_type=0,
            max_vx=max_vx,
            max_vyaw=max_vyaw,
            pos_tolerance=pos_tolerance,
            yaw_tolerance=yaw_tolerance,
            timeout=timeout,
            tick=tick,
            kp_lin=kp_lin,
            kp_yaw=kp_yaw,
        )

    def run_for(
        self,
        distance: float,
        max_vx: float = 0.45,
        max_vyaw: float = 0.8,
        pos_tolerance: float = 0.07,
        yaw_tolerance: float = 0.25,
        timeout: float = 15.0,
        tick: float = 0.05,
        kp_lin: float = 1.0,
        kp_yaw: float = 1.8,
    ) -> bool:
        return self._move_for_feedback(
            distance=distance,
            gait_type=1,
            max_vx=max_vx,
            max_vyaw=max_vyaw,
            pos_tolerance=pos_tolerance,
            yaw_tolerance=yaw_tolerance,
            timeout=timeout,
            tick=tick,
            kp_lin=kp_lin,
            kp_yaw=kp_yaw,
        )

    def turn_for(
        self,
        angle_deg: float,
        max_vyaw: float = 0.8,
        yaw_tolerance_deg: float = 2.5,
        timeout: float = 10.0,
        tick: float = 0.05,
        kp_yaw: float = 1.8,
        gait_type: int = 0,
    ) -> bool:
        yaw0 = self.get_yaw()
        if yaw0 is None:
            raise RuntimeError("turn_for requires live IMU yaw.")
        delta = math.radians(float(angle_deg))
        tol = math.radians(max(0.1, float(yaw_tolerance_deg)))
        if abs(delta) <= tol:
            self.stop()
            return True
        target = self._wrap_angle(float(yaw0) + delta)
        self.set_gait_type(int(gait_type))
        t0 = time.time()
        ok = False
        try:
            while (time.time() - t0) <= max(0.1, float(timeout)):
                yaw = self.get_yaw()
                if yaw is None:
                    time.sleep(max(0.01, float(tick)))
                    continue
                err = self._wrap_angle(target - float(yaw))
                if abs(err) <= tol:
                    ok = True
                    break
                vyaw_cmd = self._clamp(float(kp_yaw) * err, -max_vyaw, max_vyaw)
                self.loco_move(0.0, 0.0, vyaw_cmd)
                time.sleep(max(0.01, float(tick)))
        finally:
            self.stop()
        return ok

    def _rpc_get_int(self, api_id: int) -> Optional[int]:
        return rpc_get_int(self._client, api_id)

    def get_fsm(self) -> dict[str, Optional[int]]:
        return {
            "id": self._rpc_get_int(ROBOT_API_ID_LOCO_GET_FSM_ID),
            "mode": self._rpc_get_int(ROBOT_API_ID_LOCO_GET_FSM_MODE),
        }

    def fsm_0_zt(self) -> None:
        if hasattr(self._client, "ZeroTorque"):
            self._client.ZeroTorque()
        elif hasattr(self._client, "SetFsmId"):
            self._client.SetFsmId(0)

    def fsm_1_damp(self) -> None:
        if hasattr(self._client, "Damp"):
            self._client.Damp()
        elif hasattr(self._client, "SetFsmId"):
            self._client.SetFsmId(1)

    def fsm_2_airborne(self) -> None:
        if hasattr(self._client, "SetFsmId"):
            self._client.SetFsmId(2)

    def fsm_2_squat(self) -> None:
        self.fsm_2_airborne()

    # ------------------------------------------------------------------
    # IMU + lidar getters
    # ------------------------------------------------------------------

    def get_imu(self) -> ImuData | None:
        with self._lock:
            msg = self._sport
        if msg is None:
            return None

        rpy = (0.0, 0.0, 0.0)
        gyro = acc = quat = None
        temp = None

        try:
            rpy = tuple(float(msg.imu_state.rpy[i]) for i in range(3))  # type: ignore[assignment]
        except Exception:
            pass
        try:
            gyro = tuple(float(msg.imu_state.gyroscope[i]) for i in range(3))  # type: ignore[assignment]
        except Exception:
            pass
        try:
            acc = tuple(float(msg.imu_state.accelerometer[i]) for i in range(3))  # type: ignore[assignment]
        except Exception:
            pass
        try:
            quat = tuple(float(msg.imu_state.quaternion[i]) for i in range(4))  # type: ignore[assignment]
        except Exception:
            pass
        try:
            temp = float(msg.imu_state.temperature)
        except Exception:
            pass

        return ImuData(rpy=rpy, gyro=gyro, acc=acc, quat=quat, temp=temp)

    @staticmethod
    def _extract_xyz_from_cloud(msg: PointCloud2_, max_points: int | None = None) -> list[tuple[float, float, float]]:
        try:
            width = int(msg.width)
            height = int(msg.height)
            point_step = int(msg.point_step)
            raw = bytes(msg.data)
        except Exception:
            return []

        if point_step <= 0:
            return []

        x_off, y_off, z_off = 0, 4, 8
        try:
            fields = list(msg.fields)
            name_to_off = {str(field.name).lower(): int(field.offset) for field in fields}
            x_off = name_to_off.get("x", x_off)
            y_off = name_to_off.get("y", y_off)
            z_off = name_to_off.get("z", z_off)
        except Exception:
            pass

        total = max(0, width * height)
        if max_points is not None:
            total = min(total, max_points)

        import struct

        points: list[tuple[float, float, float]] = []
        for idx in range(total):
            base = idx * point_step
            try:
                x = struct.unpack_from("<f", raw, base + x_off)[0]
                y = struct.unpack_from("<f", raw, base + y_off)[0]
                z = struct.unpack_from("<f", raw, base + z_off)[0]
            except Exception:
                break
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                points.append((float(x), float(y), float(z)))
        return points

    def get_lidar_points(self, max_points: int | None = 20000) -> list[tuple[float, float, float]]:
        with self._lock:
            msg = self._lidar_cloud
        if msg is None:
            return []
        return self._extract_xyz_from_cloud(msg, max_points=max_points)

    def get_camera_image_jpeg(self) -> bytes:
        code, data = self._get_video_client().GetImageSample()
        if int(code) != 0:
            raise RuntimeError(f"GetImageSample failed with code={code}")
        return bytes(data)

    def get_camera_frame_bgr(self):
        return decode_video_frame_bgr(self.get_camera_image_jpeg())

    def get_camera_frame_rgb(self):
        import cv2

        frame = self.get_camera_frame_bgr()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # SLAM + navigation
    # ------------------------------------------------------------------

    def start_slam(self, slam_type: str = "indoor") -> int:
        response = self._get_slam_client().start_mapping(slam_type=slam_type)
        self.slam_is_running = response.code == 0
        return int(response.code)

    def stop_slam(self, save_path: str | None = None) -> int:
        client = self._get_slam_client()
        response = client.end_mapping(save_path) if save_path else client.close_slam()
        self.slam_is_running = False
        return int(response.code)

    def set_path_point(self, x: float, y: float, yaw: float = 0.0) -> None:
        self._path_points.append((float(x), float(y), float(yaw)))

    def get_path_points(self) -> list[tuple[float, float, float]]:
        return list(self._path_points)

    def clear_path_points(self) -> None:
        self._path_points.clear()

    def _run_pose_nav(self, x: float, y: float, yaw: float = 0.0) -> int:
        client = self._get_slam_client()
        qz = math.sin(float(yaw) * 0.5)
        qw = math.cos(float(yaw) * 0.5)
        response = client.pose_nav(float(x), float(y), 0.0, 0.0, 0.0, qz, qw, mode=1)
        return int(response.code)

    @staticmethod
    def _format_pose_debug(pose: tuple[float, float, float] | None) -> str:
        if pose is None:
            return "None"
        return f"({float(pose[0]):.3f}, {float(pose[1]):.3f}, {float(pose[2]):.3f})"

    def _trace_nav_result(
        self,
        *,
        step_idx: int,
        target: tuple[float, float, float],
        before_slam: tuple[float, float, float] | None,
        trace_duration_s: float = 2.0,
        sample_period_s: float = 0.5,
    ) -> None:
        target_x, target_y, target_yaw = target
        t0 = time.time()
        deadline = t0 + max(0.0, float(trace_duration_s))
        sample_idx = 0
        while True:
            now = time.time()
            slam_pose = self.get_slam_pose(timeout_s=0.15)
            odom_pose = self.get_odom_pose()
            dist = None
            if slam_pose is not None:
                dist = math.hypot(float(target_x) - float(slam_pose[0]), float(target_y) - float(slam_pose[1]))
            moved = None
            if before_slam is not None and slam_pose is not None:
                moved = math.hypot(float(slam_pose[0]) - float(before_slam[0]), float(slam_pose[1]) - float(before_slam[1]))
            extra = []
            if dist is not None:
                extra.append(f"dist_to_target={dist:.3f}m")
            if moved is not None:
                extra.append(f"slam_delta={moved:.3f}m")
            print(
                f"[navigate_path] trace step={step_idx} sample={sample_idx} "
                f"target={self._format_pose_debug((target_x, target_y, target_yaw))} "
                f"slam_pose={self._format_pose_debug(slam_pose)} "
                f"odom_pose={self._format_pose_debug(odom_pose)}"
                + (f" {' '.join(extra)}" if extra else "")
            )
            if now >= deadline:
                break
            sample_idx += 1
            time.sleep(max(0.05, float(sample_period_s)))

    def navigate_path(self, clear_on_finish: bool = True) -> bool:
        if not self._path_points:
            raise RuntimeError("No path points queued. Call set_path_point(...) first.")

        if not self.slam_is_running:
            print("[navigate_path] SLAM is not running; pose_nav requests are expected to fail.")
            return False

        slam_status = self.get_slam_pose_status(timeout_s=0.40)
        if not bool(slam_status.get("usable")):
            print(
                "[navigate_path] SLAM pose is not usable for navigation: "
                f"reason={slam_status.get('reason')} "
                f"slam_pose={slam_status.get('pose')} "
                f"sport_pose={slam_status.get('sport_pose')} "
                f"sport_vs_slam_xy_gap_m={slam_status.get('sport_vs_slam_xy_gap_m')}"
            )
            return False

        try:
            self.set_gait_type(0)
        except Exception as exc:
            print(f"[navigate_path] warning: failed to set gait_type=0 ({exc})")

        ok = True
        try:
            for idx, (x, y, yaw) in enumerate(self._path_points, start=1):
                pos = self.get_position()
                slam_pos = self.get_slam_pose(timeout_s=0.20)
                odom_pose = self.get_odom_pose()
                target_pose = (float(x), float(y), float(yaw))
                frame_gap = None
                if pos is not None and slam_pos is not None:
                    frame_gap = math.hypot(float(pos[0]) - float(slam_pos[0]), float(pos[1]) - float(slam_pos[1]))
                print(
                    f"[navigate_path] step={idx} target={self._format_pose_debug(target_pose)} "
                    f"sport_pose={self._format_pose_debug(pos)} "
                    f"slam_pose={self._format_pose_debug(slam_pos)} "
                    f"odom_pose={self._format_pose_debug(odom_pose)}"
                    + (f" sport_vs_slam_xy_gap={frame_gap:.3f}m" if frame_gap is not None else "")
                )
                if pos is not None:
                    dxy = math.hypot(float(x) - float(pos[0]), float(y) - float(pos[1]))
                    # pose_nav commonly rejects goals that are already effectively reached.
                    if dxy <= 0.20:
                        print(f"[navigate_path] step={idx} skipped: sport_pose already within {dxy:.3f}m of target.")
                        continue
                rc = self._run_pose_nav(x, y, yaw)
                print(f"[navigate_path] step={idx} pose_nav rc={rc}")
                ref = slam_pos if slam_pos is not None else pos
                if rc == 4 and ref is not None:
                    dxy = math.hypot(float(x) - float(ref[0]), float(y) - float(ref[1]))
                    print(
                        "[navigate_path] pose_nav rc=4 likely frame/relocalization mismatch or planner rejection; "
                        f"reference_dist={dxy:.3f}m slam_pose={slam_pos} odom_pose={pos} goal=({x:.3f},{y:.3f})"
                    )
                self._trace_nav_result(step_idx=idx, target=target_pose, before_slam=slam_pos)
                if rc != 0:
                    print(f"[navigate_path] failed at point {idx}: ({x:.3f},{y:.3f},{yaw:.3f}) rc={rc}")
                    ok = False
                    break
        finally:
            if clear_on_finish:
                self._path_points.clear()
        return ok

    def get_slam_info(self) -> str | None:
        return self._ensure_slam_info_subscriber().get_info()

    def get_slam_key(self) -> str | None:
        return self._ensure_slam_info_subscriber().get_key()

    def get_slam_pose(self, timeout_s: float = 0.4) -> tuple[float, float, float] | None:
        sub = self._ensure_slam_info_subscriber()
        t0 = time.time()
        while time.time() - t0 < max(0.05, float(timeout_s)):
            pose = sub.get_pose()
            if pose is not None:
                return pose
            time.sleep(0.03)
        return None

    @staticmethod
    def _is_origin_like_pose(
        pose: tuple[float, float, float] | None,
        *,
        xy_eps: float = 0.05,
        yaw_eps: float = 0.15,
    ) -> bool:
        if pose is None:
            return False
        return math.hypot(float(pose[0]), float(pose[1])) <= float(xy_eps) and abs(float(pose[2])) <= float(yaw_eps)

    def get_slam_pose_status(self, timeout_s: float = 0.4) -> dict[str, Any]:
        pose = self.get_slam_pose(timeout_s=timeout_s)
        sport_pose = self.get_position()
        status: dict[str, Any] = {
            "pose": pose,
            "sport_pose": sport_pose,
            "slam_running": bool(self.slam_is_running),
            "usable": pose is not None,
            "reason": "ok" if pose is not None else "no_pose",
            "sport_vs_slam_xy_gap_m": None,
        }

        if pose is not None and sport_pose is not None:
            gap = math.hypot(float(sport_pose[0]) - float(pose[0]), float(sport_pose[1]) - float(pose[1]))
            status["sport_vs_slam_xy_gap_m"] = float(gap)
            sport_radius = math.hypot(float(sport_pose[0]), float(sport_pose[1]))
            if self._is_origin_like_pose(pose) and sport_radius > 0.50 and gap > 0.50:
                status["usable"] = False
                status["reason"] = "origin_like_pose_but_robot_not_near_origin"
        return status

    def get_slam_odom_pose(self) -> tuple[float, float, float] | None:
        return self._ensure_slam_odom_subscriber().get_pose()

    def debug_api(
        self,
        save_path: str = "/home/unitree/test1.pcd",
        load_path: str = "/home/unitree/test1.pcd",
        goal_x: float = 1.0,
        goal_y: float = 0.0,
        goal_yaw: float = 0.0,
        pause: bool = False,
        resume: bool = False,
        wait_task_result: bool = False,
    ) -> None:
        def print_resp(label: str, req: dict[str, Any], resp: SlamResponse) -> None:
            print(f"\n[{label}]")
            print("request:", json.dumps(req, indent=2))
            print(f"response: code={resp.code} raw={resp.raw}")

        def wait_task(sub: SlamInfoSubscriber, timeout: float = 10.0) -> None:
            t0 = time.time()
            while time.time() - t0 < timeout:
                key = sub.get_key()
                if key:
                    try:
                        payload = json.loads(key)
                        if payload.get("type") == "task_result":
                            print("task_result:", json.dumps(payload, indent=2))
                            return
                    except Exception:
                        pass
                time.sleep(0.05)
            print("task_result: timeout")

        info_sub = SlamInfoSubscriber(self.slam_info_topic, self.slam_key_topic)
        info_sub.start()

        client = self._get_slam_client()

        req = {"data": {"slam_type": "indoor"}}
        print_resp("start_mapping (1801)", req, client.start_mapping("indoor"))

        req = {"data": {"address": save_path}}
        print_resp("end_mapping (1802)", req, client.end_mapping(save_path))

        req = {
            "data": {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "q_x": 0.0,
                "q_y": 0.0,
                "q_z": 0.0,
                "q_w": 1.0,
                "address": load_path,
            }
        }
        print_resp("init_pose (1804)", req, client.init_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, load_path))

        qz = math.sin(float(goal_yaw) * 0.5)
        qw = math.cos(float(goal_yaw) * 0.5)
        req = {
            "data": {
                "targetPose": {
                    "x": float(goal_x),
                    "y": float(goal_y),
                    "z": 0.0,
                    "q_x": 0.0,
                    "q_y": 0.0,
                    "q_z": qz,
                    "q_w": qw,
                },
                "mode": 1,
            }
        }
        print_resp("pose_nav (1102)", req, client.pose_nav(float(goal_x), float(goal_y), 0.0, 0.0, 0.0, qz, qw, mode=1))

        if pause:
            print_resp("pause_nav (1201)", {"data": {}}, client.pause_nav())
        if resume:
            print_resp("resume_nav (1202)", {"data": {}}, client.resume_nav())
        if wait_task_result:
            wait_task(info_sub)

        print_resp("close_slam (1901)", {"data": {}}, client.close_slam())

    # ------------------------------------------------------------------
    # Safety + audio
    # ------------------------------------------------------------------

    def hanged_boot(self) -> None:
        self._client = hanger_boot_sequence(iface=self.iface, domain_id=self.domain_id)

    def hanging_boot(self) -> None:
        self.balanced_stand(0)

    def say(self, text: str = "what would you like me to say?", volume: int | None = None) -> int:
        return self._get_audio().speak(text, volume=volume)

    def play_wav(self, wav_path: str, volume: int | None = None) -> int:
        return self._get_audio().play_wav(wav_path, volume=volume)

    def headlight(
        self,
        color: str = "white",
        intensity: int = 100,
        duration: float | None = None,
    ) -> int:
        return self._get_audio().set_headlight(color=color, intensity=intensity, duration=duration)

    def hand_open(
        self,
        hand: str = "right",
        hold_s: float = 0.6,
        rate_hz: float = 50.0,
        ramp_s: float | None = None,
    ) -> None:
        self._get_hand(hand).open(hold_s=hold_s, rate_hz=rate_hz, ramp_s=ramp_s)

    def hand_close(
        self,
        hand: str = "right",
        hold_s: float = 0.6,
        rate_hz: float = 50.0,
        ramp_s: float | None = None,
    ) -> None:
        self._get_hand(hand).close(hold_s=hold_s, rate_hz=rate_hz, ramp_s=ramp_s)

    def hand_pose(
        self,
        targets: list[float],
        hand: str = "right",
        hold_s: float = 0.6,
        rate_hz: float = 50.0,
        kp: float = 1.2,
        kd: float = 0.05,
        tau: float = 0.05,
        ramp_s: float | None = None,
    ) -> None:
        self._get_hand(hand).set_targets(
            targets,
            hold_s=hold_s,
            rate_hz=rate_hz,
            kp=kp,
            kd=kd,
            tau=tau,
            ramp_s=ramp_s,
        )

    def hand_move_finger(
        self,
        finger_name: str,
        hand: str = "right",
        hold_s: float = 1.0,
        settle_s: float = 0.6,
        rate_hz: float = 50.0,
    ) -> None:
        self._get_hand(hand).move_finger(finger_name, hold_s=hold_s, settle_s=settle_s, rate_hz=rate_hz)


__all__ = ["Robot", "ImuData"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for sdk_client Robot wrapper")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument(
        "--safety-boot",
        action="store_true",
        help="Run the hanged safety boot sequence during initialization.",
    )
    args = parser.parse_args()

    bot = Robot(iface=args.iface, safety_boot=args.safety_boot)
    time.sleep(0.6)
    print("FSM:", bot.get_fsm())
    print("IMU:", bot.get_imu())
