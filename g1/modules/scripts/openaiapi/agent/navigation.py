"""Canonical navigation/SLAM snapshot adapter for CLI and cognition."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict


EXPECTED_ROS_TOPICS: Dict[str, str] = {
    "/lowstate": "unitree_hg/msg/LowState",
    "/odommodestate": "unitree_go/msg/SportModeState",
    "/slam_info": "std_msgs/msg/String",
    "/slam_key_info": "std_msgs/msg/String",
    "/utlidar/cloud_livox_mid360": "sensor_msgs/msg/PointCloud2",
    "/utlidar/imu_livox_mid360": "sensor_msgs/msg/Imu",
    "/unitree/slam_mapping/odom": "nav_msgs/msg/Odometry",
    "/unitree/slam_mapping/points": "sensor_msgs/msg/PointCloud2",
    "/unitree/slam_relocation/points": "sensor_msgs/msg/PointCloud2",
    "/api/slam_operate/request": "unitree_api/msg/Request",
    "/api/slam_operate/response": "unitree_api/msg/Response",
}


@dataclass
class NavigationSnapshot:
    timestamp: float = field(default_factory=time.time)
    slam: str = "unknown"
    navigation: str = "unknown"
    localization: str = "unknown"
    map_name: str = ""
    current_pose: str = "unknown"
    goal: str = ""
    goal_status: str = "idle"
    velocity_command: str = ""
    planner_status: str = ""
    recovery_status: str = ""
    last_error: str = ""
    topics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "slam": self.slam,
            "navigation": self.navigation,
            "localization": self.localization,
            "map": self.map_name,
            "current_pose": self.current_pose,
            "goal": self.goal,
            "goal_status": self.goal_status,
            "velocity_command": self.velocity_command,
            "planner_status": self.planner_status,
            "recovery_status": self.recovery_status,
            "last_error": self.last_error,
            "topics": self.topics,
        }


class NavigationAdapter:
    def __init__(self, *, slam_backend: Any = None, robot: Any = None) -> None:
        self.slam_backend = slam_backend
        self.robot = robot
        self._last_error = ""

    def snapshot(self) -> NavigationSnapshot:
        snap = NavigationSnapshot()
        snap.topics = {
            name: {"type": msg_type, "expected": True, "alive": None, "age_s": None}
            for name, msg_type in EXPECTED_ROS_TOPICS.items()
        }
        if self.robot is not None:
            try:
                stale = self.robot.sensors_stale(max_age=2.0) if hasattr(self.robot, "sensors_stale") else {}
                timestamps = getattr(self.robot, "_sensor_timestamps", {}) or {}
                topic_by_key = {
                    "lowstate": "/lowstate",
                    "sport": "/odommodestate",
                    "odommodestate": "/odommodestate",
                    "slam_info": "/slam_info",
                    "lidar_cloud": "/utlidar/cloud_livox_mid360",
                    "lidar_imu": "/utlidar/imu_livox_mid360",
                    "slam_odom": "/unitree/slam_mapping/odom",
                }
                now = time.time()
                for key, topic in topic_by_key.items():
                    if topic not in snap.topics:
                        continue
                    if key in stale:
                        snap.topics[topic]["alive"] = not bool(stale[key])
                    if key in timestamps:
                        snap.topics[topic]["age_s"] = max(0.0, now - float(timestamps[key]))
            except Exception as exc:
                self._last_error = str(exc)
        if self.slam_backend is not None:
            try:
                status = json.loads(self.slam_backend.status())
                snap.slam = "running" if status.get("slam_running") else "stopped"
                snap.localization = "valid" if status.get("relocation_ready") else "invalid"
                snap.navigation = "active" if status.get("last_action") in {"go_to", "execute_tasks"} else "inactive"
                snap.current_pose = self._format_pose(status.get("pose"))
                snap.goal_status = "active" if snap.navigation == "active" else "idle"
                snap.planner_status = str(status.get("last_action") or "")
                for topic in status.get("fresh_topics") or []:
                    stripped = str(topic).lstrip("/")
                    if stripped.startswith("rt/"):
                        stripped = stripped[3:]
                    canonical = "/" + stripped
                    if canonical in snap.topics:
                        snap.topics[canonical]["alive"] = True
            except Exception as exc:
                self._last_error = str(exc)
        snap.last_error = self._last_error
        return snap

    def action(self, name: str, **kwargs: Any) -> str:
        if self.slam_backend is None:
            return "navigation backend unavailable"
        try:
            if name == "start_mapping":
                return self.slam_backend.start_mapping(str(kwargs.get("slam_type") or "indoor"))
            if name == "stop_slam":
                return self.slam_backend.stop_slam()
            if name == "save_map":
                return self.slam_backend.save_map(kwargs.get("path"))
            if name == "start_relocation":
                return self.slam_backend.start_relocation(kwargs.get("path"))
            if name == "status":
                return self.slam_backend.status()
            if name == "preflight":
                return json.dumps(self.slam_backend.preflight(), ensure_ascii=False, sort_keys=True)
        except Exception as exc:
            self._last_error = str(exc)
            return f"{name} failed: {exc}"
        return f"unsupported navigation action: {name}"

    @staticmethod
    def _format_pose(pose: Any) -> str:
        if not pose:
            return "unknown"
        if isinstance(pose, dict):
            try:
                return "x={x:.2f} y={y:.2f} yaw={yaw:.2f}".format(
                    x=float(pose.get("x", 0.0)),
                    y=float(pose.get("y", 0.0)),
                    yaw=float(pose.get("yaw", 0.0)),
                )
            except Exception:
                return str(pose)
        return str(pose)
