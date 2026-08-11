"""Lazy SLAM skill backend reusing modules/scripts/slam_web_app.py."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


class SlamBackendError(RuntimeError):
    pass


class SlamBackend:
    def __init__(
        self,
        *,
        iface: str = "eth0",
        domain_id: int = 0,
        map_path: str = "/home/unitree/test.pcd",
        points_file: str | None = None,
        robot: Any | None = None,
    ) -> None:
        self.iface = iface
        self.domain_id = int(domain_id)
        self.map_path = map_path
        self.points_file = Path(
            points_file
            or os.environ.get("G1_SLAM_POINTS_FILE", "~/.g1_agent/slam_points.json")
        ).expanduser()
        self.robot = robot
        self._state: Any = None
        self._pose_cls: Any = None

    def _ensure_state(self) -> Any:
        if self._state is not None:
            return self._state
        scripts_dir = Path(__file__).resolve().parents[2]
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        try:
            from slam_web_app import DEFAULT_TOPICS, PoseTarget, SlamWebState
        except Exception as exc:
            raise SlamBackendError(f"slam_web_app backend unavailable: {exc}") from exc
        self._pose_cls = PoseTarget
        self._state = SlamWebState(self.iface, self.domain_id, DEFAULT_TOPICS, self.map_path)
        return self._state

    def start_mapping(self, slam_type: str = "indoor") -> str:
        before = self.preflight()
        result = self._ensure_state().start_mapping(slam_type)
        message = self._format(result)
        if not bool(result.get("ok")):
            message += f"; preflight={json.dumps(before, ensure_ascii=False, sort_keys=True)}"
        return message

    def save_map(self, path: str | None = None) -> str:
        return self._format(self._ensure_state().save_map(path or self.map_path))

    def start_relocation(self, path: str | None = None) -> str:
        return self._format(self._ensure_state().relocate(path or self.map_path))

    def save_current_point(self, name: str) -> str:
        clean = self._clean_point_name(name)
        if not clean:
            return "add_current_point failed: point name was empty"
        pose = self._ensure_state().current_pose()
        if pose is None:
            return "add_current_point failed: no current non-zero SLAM pose is available"
        points = self._load_points()
        points[clean] = self._pose_to_dict(pose)
        self._save_points(points)
        return f"point saved: {json.dumps({'name': clean, **points[clean], 'point_count': len(points)}, ensure_ascii=False, sort_keys=True)}"

    def list_points(self) -> str:
        points = self._load_points()
        if not points:
            return "no saved SLAM points"
        return json.dumps(
            {
                "map_path": self.map_path,
                "points_file": str(self.points_file),
                "points": points,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def clear_points(self) -> str:
        count = len(self._load_points())
        self._save_points({})
        return f"cleared {count} saved SLAM points"

    def go_to_point(self, name: str, *, auto_relocate: bool = True) -> str:
        point_name, target, score = self._find_point(name)
        if target is None or point_name is None:
            return f"navigate_named_point failed: no saved point named {name!r}; match_score={score:.2f}"
        state = self._ensure_state()
        state.selected_pose = self._dict_to_pose(target)
        if auto_relocate and not state.relocation_ready:
            relocation = state.relocate(self.map_path)
            if not bool(relocation.get("ok")):
                return f"navigate_named_point failed: relocation failed before navigation: {relocation}"
        result = state.go_to_selected_pose()
        result["point"] = point_name
        result["match_score"] = round(score, 3)
        result["target"] = target
        return self._format(result)

    def add_current_pose(self) -> str:
        return self._format(self._ensure_state().add_current_pose())

    def go_to_selected_pose(self) -> str:
        return self._format(self._ensure_state().go_to_selected_pose())

    def execute_tasks(self) -> str:
        return self._format(self._ensure_state().execute_tasks())

    def pause(self) -> str:
        return self._format(self._ensure_state().pause())

    def resume(self) -> str:
        return self._format(self._ensure_state().resume())

    def stop_slam(self) -> str:
        return self._format(self._ensure_state().stop_slam())

    def status(self) -> str:
        state = self._ensure_state()
        status = state.status()
        compact = {
            "slam_running": status.get("slam_running"),
            "relocation_ready": status.get("relocation_ready"),
            "pose": status.get("pose"),
            "last_action": status.get("last_action"),
            "fresh_topics": [
                row["name"]
                for row in status.get("topics", [])
                if row.get("fresh") and int(row.get("count") or 0) > 0
            ],
        }
        return json.dumps(compact, ensure_ascii=False, sort_keys=True)

    def _load_points(self) -> dict[str, dict[str, float]]:
        if not self.points_file.exists():
            return {}
        try:
            data = json.loads(self.points_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw_points = data.get("points", data) if isinstance(data, dict) else {}
        points: dict[str, dict[str, float]] = {}
        if isinstance(raw_points, dict):
            for name, raw in raw_points.items():
                if not isinstance(raw, dict):
                    continue
                try:
                    clean = self._clean_point_name(str(name))
                    if clean:
                        points[clean] = {
                            "x": float(raw["x"]),
                            "y": float(raw["y"]),
                            "z": float(raw.get("z", 0.0)),
                            "yaw": float(raw.get("yaw", 0.0)),
                        }
                except Exception:
                    continue
        return points

    def _save_points(self, points: dict[str, dict[str, float]]) -> None:
        payload = {
            "map_path": self.map_path,
            "updated": time.time(),
            "points": dict(sorted(points.items())),
        }
        self.points_file.parent.mkdir(parents=True, exist_ok=True)
        self.points_file.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _find_point(self, requested_name: str) -> tuple[str | None, dict[str, float] | None, float]:
        wanted = self._clean_point_name(requested_name)
        if not wanted:
            return None, None, 0.0
        points = self._load_points()
        if wanted in points:
            return wanted, points[wanted], 1.0
        best_name: str | None = None
        best_score = 0.0
        wanted_compact = wanted.replace(" ", "")
        wanted_tokens = set(wanted.split()) - {"a", "an", "and", "at", "go", "navigate", "point", "to"}
        for name in points:
            name_compact = name.replace(" ", "")
            score = max(
                SequenceMatcher(None, wanted, name).ratio(),
                SequenceMatcher(None, wanted_compact, name_compact).ratio(),
            )
            name_tokens = set(name.split())
            if wanted_tokens and wanted_tokens <= name_tokens:
                score = max(score, 0.92)
            if score > best_score:
                best_name = name
                best_score = score
        if best_name is None or best_score < 0.62:
            return None, None, best_score
        return best_name, points[best_name], best_score

    def _dict_to_pose(self, data: dict[str, float]) -> Any:
        self._ensure_state()
        if self._pose_cls is None:
            raise SlamBackendError("PoseTarget class is unavailable")
        return self._pose_cls(
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data.get("z", 0.0)),
            yaw=float(data.get("yaw", 0.0)),
        )

    @staticmethod
    def _pose_to_dict(pose: Any) -> dict[str, float]:
        return {
            "x": float(getattr(pose, "x")),
            "y": float(getattr(pose, "y")),
            "z": float(getattr(pose, "z", 0.0)),
            "yaw": float(getattr(pose, "yaw", 0.0)),
        }

    @staticmethod
    def _clean_point_name(text: str) -> str:
        text = str(text).strip().lower()
        text = re.sub(r"^(call it|name it|save it as|save as|called|named)\s+", "", text).strip()
        text = re.sub(r"[^a-z0-9 _-]+", "", text)
        text = re.sub(r"\s+", " ", text).strip(" -_")
        return text

    def preflight(self) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        robot = self.robot
        if robot is not None:
            try:
                if hasattr(robot, "start_sensors"):
                    robot.start_sensors()
            except Exception as exc:
                diagnostics["start_sensors_error"] = str(exc)
            try:
                stale = robot.sensors_stale(max_age=2.0) if hasattr(robot, "sensors_stale") else {}
                diagnostics["sensor_stale"] = stale
                diagnostics["fresh_lidar_cloud"] = any(
                    str(name).startswith("lidar_cloud") and not bool(is_stale)
                    for name, is_stale in stale.items()
                )
                diagnostics["fresh_lidar_imu"] = not bool(stale.get("lidar_imu", True))
                diagnostics["fresh_lowstate"] = not bool(stale.get("lowstate", True))
                diagnostics["fresh_slam_odom"] = not bool(stale.get("slam_odom", True))
            except Exception as exc:
                diagnostics["sensor_error"] = str(exc)
            try:
                if hasattr(robot, "_service_status"):
                    diagnostics["unitree_slam_service_status"] = robot._service_status("unitree_slam", timeout=1.0)
            except Exception as exc:
                diagnostics["unitree_slam_service_error"] = str(exc)
            try:
                if hasattr(robot, "get_slam_info"):
                    diagnostics["slam_info_available"] = bool(robot.get_slam_info())
            except Exception as exc:
                diagnostics["slam_info_error"] = str(exc)
        try:
            status = self._ensure_state().status()
            diagnostics["web_fresh_topics"] = [
                row["name"]
                for row in status.get("topics", [])
                if row.get("fresh") and int(row.get("count") or 0) > 0
            ]
        except Exception as exc:
            diagnostics["web_status_error"] = str(exc)
        diagnostics["hint"] = (
            "SLAM API 1801 needs the robot-side unitree_slam service plus lidar cloud and lidar IMU. "
            "A ROS2 /utlidar/cloud_livox_mid360 echo alone does not prove the DDS rt/utlidar/imu_livox_mid360 "
            "input required by the SLAM service is fresh."
        )
        return diagnostics

    @staticmethod
    def _format(result: dict[str, Any]) -> str:
        label = result.get("label", "slam")
        ok = bool(result.get("ok"))
        raw = result.get("raw")
        return f"{label} {'ok' if ok else 'failed'}: {raw}"
