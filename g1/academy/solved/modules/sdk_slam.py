from __future__ import annotations
from unitree_sdk2py.rpc.client import Client
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.core.channel import ChannelSubscriber

import json
import threading
import time
from dataclasses import dataclass
from typing import Any

import math

from dds_env import ensure_cyclonedds_environment

ensure_cyclonedds_environment()


SERVICE_NAME = "slam_operate"
SERVICE_VERSION = "1.0.0.1"

API_START_MAPPING = 1801
API_END_MAPPING = 1802
API_INIT_POSE = 1804
API_POSE_NAV = 1102
API_PAUSE_NAV = 1201
API_RESUME_NAV = 1202
API_CLOSE_SLAM = 1901


@dataclass
class SlamResponse:
    code: int
    raw: Any


class SlamOperateClient(Client):
    def __init__(self, enable_lease: bool = False) -> None:
        super().__init__(SERVICE_NAME, enable_lease)

    def Init(self) -> None:
        self._RegistApi(API_START_MAPPING, 0)
        self._RegistApi(API_END_MAPPING, 0)
        self._RegistApi(API_INIT_POSE, 0)
        self._RegistApi(API_POSE_NAV, 0)
        self._RegistApi(API_PAUSE_NAV, 0)
        self._RegistApi(API_RESUME_NAV, 0)
        self._RegistApi(API_CLOSE_SLAM, 0)
        self._SetApiVerson(SERVICE_VERSION)

    def _call(self, api_id: int, payload: dict[str, Any]) -> SlamResponse:
        code, data = self._Call(api_id, json.dumps(payload, ensure_ascii=True))
        return SlamResponse(code=int(code), raw=data)

    def start_mapping(self, slam_type: str = "indoor") -> SlamResponse:
        return self._call(API_START_MAPPING, {"data": {"slam_type": slam_type}})

    def end_mapping(self, address: str) -> SlamResponse:
        return self._call(API_END_MAPPING, {"data": {"address": address}})

    def init_pose(
        self,
        x: float,
        y: float,
        z: float,
        q_x: float,
        q_y: float,
        q_z: float,
        q_w: float,
        address: str,
    ) -> SlamResponse:
        return self._call(
            API_INIT_POSE,
            {
                "data": {
                    "x": x,
                    "y": y,
                    "z": z,
                    "q_x": q_x,
                    "q_y": q_y,
                    "q_z": q_z,
                    "q_w": q_w,
                    "address": address,
                }
            },
        )

    def pose_nav(
        self,
        x: float,
        y: float,
        z: float,
        q_x: float,
        q_y: float,
        q_z: float,
        q_w: float,
        mode: int = 1,
    ) -> SlamResponse:
        return self._call(
            API_POSE_NAV,
            {
                "data": {
                    "targetPose": {
                        "x": x,
                        "y": y,
                        "z": z,
                        "q_x": q_x,
                        "q_y": q_y,
                        "q_z": q_z,
                        "q_w": q_w,
                    },
                    "mode": mode,
                }
            },
        )

    def pause_nav(self) -> SlamResponse:
        return self._call(API_PAUSE_NAV, {"data": {}})

    def resume_nav(self) -> SlamResponse:
        return self._call(API_RESUME_NAV, {"data": {}})

    def close_slam(self) -> SlamResponse:
        return self._call(API_CLOSE_SLAM, {"data": {}})


class SlamInfoSubscriber:
    def __init__(self, info_topic: str = "rt/slam_info", key_topic: str = "rt/slam_key_info") -> None:
        self.info_topic = info_topic
        self.key_topic = key_topic
        self._lock = threading.Lock()
        self._info: str | None = None
        self._key: str | None = None
        self._last_info: float = 0.0
        self._last_key: float = 0.0
        self._info_sub: ChannelSubscriber | None = None
        self._key_sub: ChannelSubscriber | None = None

    def start(self) -> None:
        if self._info_sub is None:
            self._info_sub = ChannelSubscriber(self.info_topic, String_)
            self._info_sub.Init(self._info_cb, 10)
        if self._key_sub is None:
            self._key_sub = ChannelSubscriber(self.key_topic, String_)
            self._key_sub.Init(self._key_cb, 10)

    def _info_cb(self, msg: String_) -> None:
        with self._lock:
            self._info = str(msg.data)
            self._last_info = time.time()

    def _key_cb(self, msg: String_) -> None:
        with self._lock:
            self._key = str(msg.data)
            self._last_key = time.time()

    def get_info(self) -> str | None:
        with self._lock:
            return self._info

    def get_key(self) -> str | None:
        with self._lock:
            return self._key

    def get_info_with_ts(self) -> tuple[str | None, float]:
        with self._lock:
            return self._info, self._last_info

    def get_key_with_ts(self) -> tuple[str | None, float]:
        with self._lock:
            return self._key, self._last_key

    @staticmethod
    def parse_status(payload_raw: str | None) -> dict[str, Any] | None:
        """Parse the fuller envelope some rt/slam_info / rt/slam_key_info
        messages carry beyond bare pose data.

        `errorCode`/`info` are present on every message (this is how e.g. the
        Unitree SLAM demo, keyDemo.cpp, surfaces navigation problems: it just
        prints `info` whenever `errorCode != 0`, with no separate "obstacle"
        field). Message `type` seen in practice: "pos_info" (pose only),
        "ctrl_info" (the richer status blob below, streamed continuously
        while navigating), "task_result" (terminal per-target result, the one
        keyDemo.cpp's slamKeyInfoHandler waits on). Every field is optional;
        this never raises, it returns None fields for whatever is absent.

        `data.obsInfo.state` is the actual obstacle-blocked flag: true while
        the nav stack currently has the path blocked. Source: a live
        rt/slam_info capture (dev/slam_viz_in_jupyter.ipynb) and the DDS
        probe notes in Inspire_hands/topics.md.
        """
        if not payload_raw:
            return None
        try:
            payload = json.loads(payload_raw)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        data = data if isinstance(data, dict) else {}
        obs_info = data.get("obsInfo")
        obs_info = obs_info if isinstance(obs_info, dict) else {}
        progress = data.get("progress")
        progress = progress if isinstance(progress, dict) else {}
        state_machine = data.get("stateMachine")
        state_machine = state_machine if isinstance(state_machine, dict) else {}
        return {
            "type": payload.get("type"),
            "error_code": int(payload.get("errorCode", 0) or 0),
            "info": payload.get("info"),
            "is_arrived": data.get("is_arrived"),
            "target_node_name": data.get("targetNodeName"),
            "obstacle_blocked": bool(obs_info["state"]) if "state" in obs_info else None,
            "obstacle_time": obs_info.get("time"),
            "completion_pct": progress.get("completion_percentage"),
            "nav_state": state_machine.get("state"),
            "nav_paused": state_machine.get("isPause"),
        }

    @staticmethod
    def parse_pose(payload_raw: str | None) -> tuple[float, float, float] | None:
        if not payload_raw:
            return None
        try:
            payload = json.loads(payload_raw)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        current_pose = data.get("currentPose")
        if not isinstance(current_pose, dict):
            return None
        try:
            x = float(current_pose.get("x"))
            y = float(current_pose.get("y"))
        except Exception:
            return None
        yaw = 0.0
        try:
            qx = float(current_pose.get("q_x", 0.0))
            qy = float(current_pose.get("q_y", 0.0))
            qz = float(current_pose.get("q_z", 0.0))
            qw = float(current_pose.get("q_w", 1.0))
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
        except Exception:
            try:
                yaw = float(current_pose.get("yaw", 0.0))
            except Exception:
                yaw = 0.0
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
            return None
        return (x, y, yaw)

    def get_pose(self) -> tuple[float, float, float] | None:
        for payload in (self.get_info(), self.get_key()):
            pose = self.parse_pose(payload)
            if pose is not None:
                return pose
        return None


class SlamOdomSubscriber:
    def __init__(self, topic: str = "rt/unitree/slam_mapping/odom") -> None:
        self.topic = topic
        self._lock = threading.Lock()
        self._odom: Odometry_ | None = None
        self._last_ts: float = 0.0
        self._sub: ChannelSubscriber | None = None

    def start(self) -> None:
        if self._sub is None:
            self._sub = ChannelSubscriber(self.topic, Odometry_)
            self._sub.Init(self._callback, 10)

    def _callback(self, msg: Odometry_) -> None:
        with self._lock:
            self._odom = msg
            self._last_ts = time.time()

    def get_latest(self) -> tuple[Odometry_ | None, float]:
        with self._lock:
            return self._odom, self._last_ts

    def get_pose(self) -> tuple[float, float, float] | None:
        with self._lock:
            msg = self._odom
        if msg is None:
            return None
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
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return (x, y, yaw)
        except Exception:
            return None


__all__ = ["SlamInfoSubscriber", "SlamOdomSubscriber", "SlamOperateClient", "SlamResponse"]
