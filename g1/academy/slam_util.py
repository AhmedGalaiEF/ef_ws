"""Standalone G1 SLAM support for Academy notebook 4.

No sdk_wrapper and no sdk_slam dependency.  This module contains the deployed
SLAM RPC identifiers, DDS status subscription, pose parsing, named-point
persistence, arrival observation, and external-map-visualization references.
Participants use the high-level operations; they do not re-create this RPC
boilerplate in a notebook.

The available deployed RPC operations are mapping start/stop, relocate, and
single-pose navigation.  Native complete-path navigation is version-dependent:
provide a verified native_path_callback to enable navigate_path; otherwise the
method refuses rather than silently navigating one point at a time.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.rpc.client import Client

_factory_config: tuple[int, str] | None = None


def ensure_channel_factory(domain_id: int, interface: str) -> tuple[int, str]:
    global _factory_config
    config = (int(domain_id), str(interface))
    if _factory_config is None:
        ChannelFactoryInitialize(*config)
        _factory_config = config
    elif _factory_config != config:
        raise RuntimeError(f"ChannelFactory is already { _factory_config }; restart process for {config}.")
    return config


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float = 0.0
    z: float = 0.0

    def quaternion(self) -> tuple[float, float, float, float]:
        return 0.0, 0.0, math.sin(self.yaw / 2), math.cos(self.yaw / 2)

    def distance_xy(self, other: "Pose") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


def _find_pose(value: Any) -> Pose | None:
    if isinstance(value, str):
        try:
            return _find_pose(json.loads(value))
        except Exception:
            return None
    if isinstance(value, dict):
        candidate = value.get("data", value)
        if all(key in candidate for key in ("x", "y")):
            try:
                return Pose(float(candidate["x"]), float(candidate["y"]), float(candidate.get("yaw", 0.0)), float(candidate.get("z", 0.0)))
            except (TypeError, ValueError):
                return None
        for child in value.values():
            found = _find_pose(child)
            if found is not None:
                return found
    return None


class SlamRpc(Client):
    """Native Unitree SLAM RPC client for the API IDs used by the working nav flow."""

    def __init__(self) -> None:
        super().__init__("slam_operate", False)
        for api_id in (1801, 1802, 1804, 1102, 1901):
            self._RegistApi(api_id, 0)
        self._SetApiVerson("1.0.0.1")

    def _call_json(self, api_id: int, payload: dict[str, Any]) -> tuple[int, Any]:
        code, data = self._Call(api_id, json.dumps(payload))
        return int(code), data

    def start_mapping(self, slam_type: str = "indoor") -> tuple[int, Any]:
        return self._call_json(1801, {"data": {"slam_type": slam_type}})

    def stop_mapping(self, save_path: str | None = None) -> tuple[int, Any]:
        return self._call_json(1802, {"data": {"address": save_path}}) if save_path else self._call_json(1901, {"data": {}})

    def relocate(self, pose: Pose, map_path: str) -> tuple[int, Any]:
        qx, qy, qz, qw = pose.quaternion()
        return self._call_json(1804, {"data": {"x": pose.x, "y": pose.y, "z": pose.z, "q_x": qx, "q_y": qy, "q_z": qz, "q_w": qw, "address": map_path}})

    def pose_nav(self, pose: Pose) -> tuple[int, Any]:
        qx, qy, qz, qw = pose.quaternion()
        return self._call_json(1102, {"data": {"targetPose": {"x": pose.x, "y": pose.y, "z": pose.z, "q_x": qx, "q_y": qy, "q_z": qz, "q_w": qw}, "mode": 1}})


class SlamUtility:
    """Operational SLAM helper: status, mapping, relocation, named poses, navigation."""

    def __init__(self, interface: str = "eth0", domain_id: int = 0, map_path: str = "/home/unitree/test.pcd", points_path: str = "slam_points.json") -> None:
        ensure_channel_factory(domain_id, interface)
        self.map_path, self.points_path = str(map_path), Path(points_path)
        self.latest: dict[str, tuple[str, float]] = {}
        self.points = self._load_points()
        self.visualization_path: str | None = None
        self.relocated = False
        self.client = SlamRpc()
        self.client.SetTimeout(10.0)
        self.client.Init()
        self._subscribe("rt/slam_info")
        self._subscribe("rt/slam_key_info")

    def _subscribe(self, topic: str) -> None:
        subscriber = ChannelSubscriber(topic, String_)
        subscriber.Init(lambda message, topic=topic: self.latest.__setitem__(topic, (str(message.data), time.time())), 10)
        setattr(self, "_" + topic.replace("/", "_"), subscriber)

    def _load_points(self) -> dict[str, Pose]:
        if not self.points_path.exists():
            return {}
        try:
            return {name: Pose(**data) for name, data in json.loads(self.points_path.read_text()).items()}
        except Exception:
            return {}

    def _save_points(self) -> None:
        temporary = self.points_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({name: asdict(pose) for name, pose in self.points.items()}, indent=2))
        os.replace(temporary, self.points_path)

    def current_pose(self, max_age_s: float = 2.0) -> Pose | None:
        for topic in ("rt/slam_info", "rt/slam_key_info"):
            raw = self.latest.get(topic)
            if raw and time.time() - raw[1] <= max_age_s:
                pose = _find_pose(raw[0])
                if pose is not None:
                    return pose
        return None

    def status(self) -> dict[str, Any]:
        return {"pose": None if self.current_pose() is None else asdict(self.current_pose()), "relocated": self.relocated, "map_path": self.map_path, "points": sorted(self.points), "visualization_path": self.visualization_path}

    def _result(self, result: tuple[int, Any]) -> dict[str, Any]:
        code, raw = result
        return {"ok": code == 0, "code": code, "raw": raw}

    def start_mapping(self, slam_type: str = "indoor") -> dict[str, Any]:
        self.relocated = False
        return self._result(self.client.start_mapping(slam_type))

    def stop_mapping(self, save_path: str | None = None) -> dict[str, Any]:
        return self._result(self.client.stop_mapping(save_path or self.map_path))

    def relocate(self, map_path: str | None = None) -> dict[str, Any]:
        pose = self.current_pose()
        if pose is None:
            return {"ok": False, "code": None, "raw": "No fresh SLAM pose is available."}
        result = self._result(self.client.relocate(pose, map_path or self.map_path))
        self.relocated = result["ok"]
        return result

    def save_named_pose(self, name: str) -> Pose:
        pose = self.current_pose()
        if pose is None:
            raise RuntimeError("Cannot save point without a fresh SLAM pose.")
        self.points[str(name)] = pose
        self._save_points()
        return pose

    def remove_named_pose(self, name: str) -> None:
        self.points.pop(str(name), None)
        self._save_points()

    def navigate_to(self, name: str) -> dict[str, Any]:
        if not self.relocated:
            raise RuntimeError("Relocate successfully before navigation.")
        return self._result(self.client.pose_nav(self.points[str(name)]))

    def wait_for_arrival(self, name: str, tolerance_m: float = 0.35, timeout_s: float = 120.0) -> dict[str, Any]:
        target, deadline = self.points[str(name)], time.time() + timeout_s
        while time.time() < deadline:
            pose = self.current_pose()
            if pose and pose.distance_xy(target) <= tolerance_m:
                return {"arrived": True, "pose": asdict(pose)}
            time.sleep(0.5)
        return {"arrived": False, "pose": None if self.current_pose() is None else asdict(self.current_pose())}

    def set_visualization_path(self, path: str) -> None:
        self.visualization_path = str(path)

    def view_map(self) -> Path | None:
        return Path(self.visualization_path) if self.visualization_path and Path(self.visualization_path).exists() else None

    def navigate_path(self, names: list[str], native_path_callback: Callable[[list[Pose]], Any] | None = None) -> Any:
        if native_path_callback is None:
            raise RuntimeError("No verified native complete-path callback is configured; refusing sequential fallback.")
        return native_path_callback([self.points[name] for name in names])
