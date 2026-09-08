#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make `from sdk_wrapper import G1` resolve regardless of cwd. This file
# lives in academy/visualizations/; sdk_wrapper.py sits one level up in
# academy/, and viz_util.py sits alongside this file.
_SCRIPT_DIR = Path(__file__).resolve().parent      # academy/visualizations
_ACADEMY_DIR = _SCRIPT_DIR.parent                    # academy
for _p in (_SCRIPT_DIR, _ACADEMY_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2py.core.channel import ChannelSubscriber

from sdk_wrapper import G1
from viz_util import normalize_rpc

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

try:
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP

    KISS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on optional wheel
    KISS_AVAILABLE = False
    KISS_IMPORT_ERROR = str(exc)


DEFAULT_TOPICS = {
    "slam_mapping": "rt/unitree/slam_mapping/points",
    "slam_relocation": "rt/unitree/slam_relocation/points",
    "slam_global_map": "rt/unitree/slam_relocation/global_map",
    "slam_web_points": "rt/unitree/slam_relocation/web_points",
    "deskewed": "rt/utlidar/cloud_deskewed",
    "livox": "rt/utlidar/cloud_livox_mid360",
    "collision": "rt/collision_clouds",
    "pre_collision": "rt/pre_collision_clouds",
    "safe": "rt/safe_clouds",
    "pre_safe": "rt/pre_safe_clouds",
    "warning": "rt/warning_clouds",
    "no_warning": "rt/no_warning_clouds",
    "grid": "rt/grid_clouds",
}

LAYER_STYLE = {
    "slam_mapping": ("SLAM mapping", "#55c7ff", 3, 0.95),
    "slam_relocation": ("SLAM relocation", "#37e05f", 3, 0.90),
    "slam_global_map": ("SLAM global map", "#f5c84b", 3, 0.95),
    "slam_web_points": ("SLAM web points", "#9f8cff", 3, 0.90),
    "deskewed": ("Deskewed cloud", "#aeb7c2", 2, 0.45),
    "livox": ("Raw Livox", "#697381", 2, 0.25),
    "collision": ("Collision cloud", "#ff365d", 4, 0.90),
    "pre_collision": ("Pre-collision cloud", "#ff8a3d", 4, 0.75),
    "safe": ("Safe cloud", "#2ed47a", 3, 0.55),
    "pre_safe": ("Pre-safe cloud", "#8fe3a8", 3, 0.45),
    "warning": ("Warning cloud", "#ffd166", 4, 0.85),
    "no_warning": ("No-warning cloud", "#ffe9a8", 3, 0.40),
    "grid": ("Grid cloud", "#f0f0f0", 2, 0.50),
    "kiss_map": ("KISS-ICP voxel map", "#ffffff", 2, 0.80),
    "occupancy": ("Derived occupied cells", "#d9d9d9", 4, 0.70),
}

DEFAULT_SELECTED_LAYERS = [
    "slam_mapping",
    "slam_relocation",
    "slam_global_map",
    "slam_web_points",
]
DEFAULT_MAX_POINTS_PER_LAYER = 500
DEFAULT_MAP_REFRESH_INTERVAL_MS = 2000
NAV_REACHED_DISTANCE_M = 0.35
NAV_TARGET_TIMEOUT_S = 120.0
NAV_POLL_INTERVAL_S = 0.5


@dataclass
class PoseTarget:
    x: float
    y: float
    yaw: float = 0.0
    z: float = 0.0

    def xy_distance_to(self, other: "PoseTarget") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


def is_default_zero_pose(pose: PoseTarget) -> bool:
    return (
        abs(pose.x) < 1e-5
        and abs(pose.y) < 1e-5
        and abs(pose.z) < 1e-5
        and abs(pose.yaw) < 1e-5
    )


class LatestCloudSubscriber:
    def __init__(self, name: str, topic: str) -> None:
        self.name = name
        self.topic = topic
        self.msg: Optional[PointCloud2_] = None
        self.ts = 0.0
        self.count = 0
        self._lock = threading.Lock()
        self._sub: Optional[ChannelSubscriber] = None

    def start(self) -> None:
        if self._sub is None:
            self._sub = ChannelSubscriber(self.topic, PointCloud2_)
            self._sub.Init(self._callback, 10)

    def _callback(self, msg: PointCloud2_) -> None:
        with self._lock:
            self.msg = msg
            self.ts = time.time()
            self.count += 1

    def latest(self) -> tuple[Optional[PointCloud2_], float, int]:
        with self._lock:
            return self.msg, self.ts, self.count


def parse_pose(payload_raw: Optional[str]) -> Optional[PoseTarget]:
    if not payload_raw:
        return None
    try:
        payload = json.loads(payload_raw)
        if int(payload.get("errorCode", 0)) != 0:
            return None
        cur = payload.get("data", {}).get("currentPose", {})
        x = float(cur.get("x", 0.0))
        y = float(cur.get("y", 0.0))
        z = float(cur.get("z", 0.0))
        if {"q_x", "q_y", "q_z", "q_w"} <= set(cur):
            qx = float(cur.get("q_x", 0.0))
            qy = float(cur.get("q_y", 0.0))
            qz = float(cur.get("q_z", 0.0))
            qw = float(cur.get("q_w", 1.0))
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
        else:
            yaw = float(cur.get("yaw", 0.0))
        pose = PoseTarget(x=x, y=y, z=z, yaw=yaw)
        # /slam_info can contain interleaved zero-pose messages from other
        # publishers. Treat those as missing data so relocation never starts
        # from the default origin by accident.
        if is_default_zero_pose(pose):
            return None
        return pose
    except Exception:
        return None


def decode_xyz(
    msg: PointCloud2_,
    *,
    stride: int = 1,
    max_points: int = 30000,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> np.ndarray:
    try:
        fields = {field.name: field for field in msg.fields}
        if not {"x", "y", "z"} <= set(fields):
            return np.empty((0, 3), dtype=np.float32)
        point_step = int(msg.point_step)
        if point_step <= 0:
            return np.empty((0, 3), dtype=np.float32)
        data = bytes(msg.data)
        if not data:
            return np.empty((0, 3), dtype=np.float32)
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": ["<f4", "<f4", "<f4"],
                "offsets": [
                    int(fields["x"].offset),
                    int(fields["y"].offset),
                    int(fields["z"].offset),
                ],
                "itemsize": point_step,
            }
        )
        raw = np.frombuffer(data, dtype=dtype, count=len(data) // point_step)
        step = max(1, int(stride))
        pts = np.stack([raw["x"][::step], raw["y"][::step], raw["z"]
                       [::step]], axis=1).astype(np.float32)
        mask = np.isfinite(pts).all(axis=1)
        if z_min is not None:
            mask &= pts[:, 2] >= float(z_min)
        if z_max is not None:
            mask &= pts[:, 2] <= float(z_max)
        pts = pts[mask]
        if max_points > 0 and pts.shape[0] > int(max_points):
            idx = np.linspace(0, pts.shape[0] - 1, int(max_points), dtype=np.int64)
            pts = pts[idx]
        return pts
    except Exception:
        return np.empty((0, 3), dtype=np.float32)


def transform_xy(points_xyz: np.ndarray, pose: Optional[PoseTarget]) -> np.ndarray:
    if points_xyz.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    xy = points_xyz[:, :2].astype(np.float32)
    if pose is None:
        return xy
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    out = np.empty_like(xy)
    out[:, 0] = c * xy[:, 0] - s * xy[:, 1] + pose.x
    out[:, 1] = s * xy[:, 0] + c * xy[:, 1] + pose.y
    return out


class OccupancyAccumulator:
    def __init__(self, resolution: float = 0.08, max_cells: int = 45000) -> None:
        self.resolution = float(resolution)
        self.max_cells = int(max_cells)
        self._cells: dict[tuple[int, int], float] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._cells.clear()

    def insert(self, xy: np.ndarray) -> None:
        if xy.size == 0:
            return
        now = time.time()
        cells = np.floor(xy / self.resolution).astype(np.int32)
        with self._lock:
            for gx, gy in cells:
                self._cells[(int(gx), int(gy))] = now
            if len(self._cells) > self.max_cells:
                keep = sorted(self._cells.items(), key=lambda item: item[1], reverse=True)[
                    : self.max_cells]
                self._cells = dict(keep)

    def points(self, max_points: int = 12000) -> np.ndarray:
        with self._lock:
            keys = list(self._cells.keys())
        if not keys:
            return np.empty((0, 2), dtype=np.float32)
        if len(keys) > max_points:
            step = int(len(keys) / max_points) + 1
            keys = keys[::step]
        arr = np.asarray(keys, dtype=np.float32)
        return (arr + 0.5) * self.resolution


class KissIcpLayer:
    def __init__(self) -> None:
        self.enabled = KISS_AVAILABLE
        self.error = "" if KISS_AVAILABLE else globals().get("KISS_IMPORT_ERROR", "kiss-icp unavailable")
        self.frames = 0
        self.last_update = 0.0
        self._lock = threading.Lock()
        self._map = np.empty((0, 3), dtype=np.float32)
        self._pose = np.eye(4, dtype=np.float64)
        self._icp = None
        if self.enabled:
            try:
                config = KISSConfig()
                config.data.max_range = 25.0
                config.data.min_range = 0.2
                config.mapping.max_points_per_voxel = 12
                self._icp = KissICP(config)
            except Exception as exc:
                self.enabled = False
                self.error = str(exc)

    def reset(self) -> None:
        if not KISS_AVAILABLE:
            return
        with self._lock:
            try:
                config = KISSConfig()
                config.data.max_range = 25.0
                config.data.min_range = 0.2
                config.mapping.max_points_per_voxel = 12
                self._icp = KissICP(config)
                self._map = np.empty((0, 3), dtype=np.float32)
                self._pose = np.eye(4, dtype=np.float64)
                self.frames = 0
                self.last_update = 0.0
                self.enabled = True
                self.error = ""
            except Exception as exc:
                self.enabled = False
                self.error = str(exc)

    def process(self, points_xyz: np.ndarray) -> None:
        if not self.enabled or self._icp is None or points_xyz.shape[0] < 80:
            return
        if points_xyz.shape[0] > 9000:
            idx = np.linspace(0, points_xyz.shape[0] - 1, 9000, dtype=np.int64)
            frame = points_xyz[idx].astype(np.float64)
        else:
            frame = points_xyz.astype(np.float64)
        timestamps = np.zeros((frame.shape[0],), dtype=np.float64)
        try:
            self._icp.register_frame(frame, timestamps)
            cloud = np.asarray(self._icp.local_map.point_cloud(), dtype=np.float32)
            if cloud.shape[0] > 45000:
                idx = np.linspace(0, cloud.shape[0] - 1, 45000, dtype=np.int64)
                cloud = cloud[idx]
            with self._lock:
                self._map = cloud
                self._pose = np.asarray(self._icp.last_pose, dtype=np.float64)
                self.frames += 1
                self.last_update = time.time()
                self.error = ""
        except Exception as exc:
            with self._lock:
                self.error = str(exc)

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        with self._lock:
            cloud = self._map.copy()
            pose = self._pose.copy()
            meta = {
                "enabled": self.enabled,
                "frames": self.frames,
                "last_update": self.last_update,
                "error": self.error,
            }
        return cloud, pose, meta


class SlamWebState:
    def __init__(self, iface: str, domain_id: int, topics: dict[str, str], map_path: str) -> None:
        # G1() brings up the DDS channel factory itself and eagerly
        # subscribes to rt/slam_info, rt/slam_key_info, and
        # rt/unitree/slam_mapping/odom -- do this before creating the
        # LatestCloudSubscriber layers below, which reuse that same
        # process-wide factory for their own direct ChannelSubscriber calls.
        self.g1 = G1(iface=iface, domain_id=domain_id)
        self.iface = iface
        self.domain_id = domain_id
        self.map_path = map_path
        self.subs = {name: LatestCloudSubscriber(name, topic) for name, topic in topics.items()}
        for sub in self.subs.values():
            sub.start()
        self.tasks: list[PoseTarget] = []
        self.initial_pose: Optional[PoseTarget] = None
        self.last_valid_pose: Optional[PoseTarget] = None
        self.selected_pose: Optional[PoseTarget] = None
        self.slam_running = False
        self.relocation_ready = False
        self.last_action: dict[str, Any] = {"label": "startup", "ok": True, "raw": "ready"}
        self.last_notice: Optional[dict[str, Any]] = None
        self.occupancy = OccupancyAccumulator()
        self.kiss = KissIcpLayer()
        self._seen_counts: dict[str, int] = {}
        # Guards every mutable field above (self.tasks, last_action,
        # last_notice, selected_pose, initial_pose, slam_running,
        # relocation_ready, task_progress): both the Dash callback thread(s)
        # and the background task-sequence thread below touch these.
        self._state_lock = threading.RLock()
        self.task_progress: dict[str, Any] = {"running": False}
        self._task_thread: Optional[threading.Thread] = None
        self._task_cancel = threading.Event()
        self._worker_stop = threading.Event()
        self._worker = threading.Thread(target=self._mapping_worker,
                                        name="slam-web-map-worker", daemon=True)
        self._worker.start()

    def current_pose(self) -> Optional[PoseTarget]:
        pose = parse_pose(self.g1.get_slam_info()) or parse_pose(self.g1.get_slam_key_info())
        if pose is None:
            # Neither status string carried a non-zero pose (or none has
            # arrived yet) -- fall back to raw SLAM odometry (a different
            # frame from G1.get_odom()'s sport/body odometry -- see
            # G1.get_slam_odom_pose()'s docstring).
            odom_pose = self.g1.get_slam_odom_pose()
            if odom_pose is not None and not is_default_zero_pose(PoseTarget(*odom_pose)):
                x, y, yaw = odom_pose
                pose = PoseTarget(x=x, y=y, yaw=yaw)
        # G1.get_slam_notice() already picks whichever of rt/slam_key_info /
        # rt/slam_info is the relevant one and timestamps it.
        notice = self.g1.get_slam_notice()
        if notice is not None:
            with self._state_lock:
                self.last_notice = notice
        if pose is not None:
            self.last_valid_pose = pose
        return pose

    def _record(self, label: str, result: dict[str, Any]) -> dict[str, Any]:
        with self._state_lock:
            self.last_action = {"label": label, **result, "stamp": time.time()}
            return self.last_action

    def start_mapping(self, slam_type: str) -> dict[str, Any]:
        with self._state_lock:
            self.tasks.clear()
        self.occupancy.reset()
        self.kiss.reset()
        initial_pose = self.current_pose()
        result = normalize_rpc(self.g1.start_mapping(slam_type))
        with self._state_lock:
            self.initial_pose = initial_pose
            self.slam_running = bool(result["ok"])
            self.relocation_ready = False
        return self._record("start_mapping", result)

    def save_map(self, path: str) -> dict[str, Any]:
        self.map_path = path
        # G1.stop_mapping(save_path=...) -> end_mapping (1802); see stop_slam
        # below for the no-save close_slam (1901) case.
        return self._record("end_mapping", normalize_rpc(self.g1.stop_mapping(save_path=path)))

    def relocate(self, path: str) -> dict[str, Any]:
        self.map_path = path
        pose = self.current_pose() or self.last_valid_pose or self.initial_pose
        if pose is None:
            return self._record("init_pose", {
                "code": 1,
                "ok": False,
                "raw": "Cannot start relocation: no valid non-zero SLAM pose has been received yet.",
            })
        # Pass our own already-validated pose explicitly -- G1.relocate()'s
        # own fallback would otherwise silently relocate to (0, 0, 0) if it
        # ever ran out of poses, which is exactly what the check above
        # refuses to do.
        result = normalize_rpc(self.g1.relocate(map_path=path, pose=(pose.x, pose.y, pose.yaw)))
        with self._state_lock:
            self.relocation_ready = bool(result["ok"])
            if self.relocation_ready:
                self.initial_pose = pose
        return self._record("init_pose", result)

    def stop_slam(self) -> dict[str, Any]:
        self._task_cancel.set()  # abort any in-flight task sequence, mirrors
        # keyDemo.cpp's taskThreadStop() being called before stopNodeFun()
        result = normalize_rpc(self.g1.stop_mapping(save=False))  # no save -> close_slam (1901)
        with self._state_lock:
            self.slam_running = False
            self.relocation_ready = False
        return self._record("close_slam", result)

    def pause(self) -> dict[str, Any]:
        return self._record("pause_nav", normalize_rpc(self.g1.pause_nav()))

    def resume(self) -> dict[str, Any]:
        return self._record("resume_nav", normalize_rpc(self.g1.resume_nav()))

    def add_task(self, x: float, y: float, yaw: Optional[float] = None) -> PoseTarget:
        pose = self.current_pose()
        target = PoseTarget(float(x), float(y), float(
            pose.yaw if yaw is None and pose else yaw or 0.0))
        with self._state_lock:
            self.tasks.append(target)
            self.selected_pose = target
            self.last_action = {"label": "add_task", "ok": True, "code": 0, "raw": {
                "x": target.x, "y": target.y, "yaw": target.yaw}, "stamp": time.time()}
        return target

    def add_current_pose(self) -> dict[str, Any]:
        pose = self.current_pose()
        if pose is None:
            return self._record("add_current_pose", {"code": 1, "ok": False, "raw": "No current SLAM pose available."})
        with self._state_lock:
            self.tasks.append(pose)
            self.selected_pose = pose
            task_count = len(self.tasks)
        return self._record("add_current_pose", {"code": 0, "ok": True, "raw": {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.yaw, "task_count": task_count}})

    def go_to_selected_pose(self) -> dict[str, Any]:
        with self._state_lock:
            target = self.selected_pose
            relocation_ready = self.relocation_ready
        if target is None:
            return self._record("go_to_selected_pose", {"code": 1, "ok": False, "raw": "No selected pose. Click the map or add a task point first."})
        if not relocation_ready:
            return self._record("go_to_selected_pose", {"code": 1, "ok": False, "raw": "Relocation is not active."})
        result = normalize_rpc(self.g1.pose_nav(target.x, target.y, target.yaw))
        result["target"] = {"x": target.x, "y": target.y, "z": target.z, "yaw": target.yaw}
        return self._record("go_to_selected_pose", result)

    def clear_tasks(self) -> dict[str, Any]:
        self._task_cancel.set()  # stop an in-flight sequence, if any
        with self._state_lock:
            self.tasks.clear()
            self.selected_pose = None
        return self._record("clear_tasks", {"code": 0, "ok": True, "raw": {}})

    def execute_tasks(self) -> dict[str, Any]:
        """Kicks off the queued task points on a background thread and
        returns immediately.

        Blocking here (as the previous implementation did, waiting up to
        NAV_TARGET_TIMEOUT_S per point inside the Dash button callback) froze
        the whole single-threaded dev server -- including the periodic map
        refresh -- for the entire multi-point run. keyDemo.cpp avoids exactly
        this by running its task loop on a detached thread
        (taskThreadRun/taskLoopFun); this mirrors that.
        """
        with self._state_lock:
            if not self.tasks:
                return self._record("execute_tasks", {"code": 1, "ok": False, "raw": "No task points queued."})
            if not self.relocation_ready:
                return self._record("execute_tasks", {"code": 1, "ok": False, "raw": "Relocation is not active."})
            if self._task_thread is not None and self._task_thread.is_alive():
                return self._record("execute_tasks", {"code": 1, "ok": False, "raw": "A task sequence is already running."})
            snapshot = list(self.tasks)
            self._task_cancel.clear()
            self.task_progress = {"running": True, "index": 0, "total": len(snapshot)}
            thread = threading.Thread(
                target=self._execute_tasks_worker, args=(snapshot,),
                name="slam-web-task-exec", daemon=True,
            )
            self._task_thread = thread
        thread.start()
        return self._record("execute_tasks", {
            "code": 0, "ok": True,
            "raw": f"Started navigating {len(snapshot)} queued point(s) in the background.",
        })

    def _execute_tasks_worker(self, snapshot: list[PoseTarget]) -> None:
        results: list[dict[str, Any]] = []
        ok = True
        for idx, target in enumerate(snapshot, start=1):
            if self._task_cancel.is_set():
                ok = False
                break
            with self._state_lock:
                self.task_progress = {
                    "running": True, "index": idx, "total": len(snapshot),
                    "target": {"x": target.x, "y": target.y},
                }
            result = normalize_rpc(self.g1.pose_nav(target.x, target.y, target.yaw))
            result["target_index"] = idx
            result["target"] = {"x": target.x, "y": target.y, "yaw": target.yaw}
            if result["ok"]:
                reached, final_pose, elapsed, notice = self._wait_for_target(target)
                result["reached"] = reached
                result["elapsed_s"] = round(elapsed, 2)
                if notice:
                    result["notice"] = notice
                if final_pose is not None:
                    result["final_pose"] = {"x": final_pose.x, "y": final_pose.y, "yaw": final_pose.yaw}
                    result["final_distance_m"] = round(final_pose.xy_distance_to(target), 3)
                if not reached:
                    result["ok"] = False
                    result["code"] = 1
                    blocked = bool(notice and notice.get("obstacle_blocked"))
                    result["raw"] = (
                        f"Task {idx} looks blocked by an obstacle (obsInfo.state=true)."
                        if blocked else f"Timed out waiting for task {idx} to be reached."
                    )
            results.append(result)
            with self._state_lock:
                self.task_progress = {"running": True, "index": idx, "total": len(snapshot), "done": idx}
                self._record("execute_tasks", {"code": 0 if result["ok"] else 1, "ok": result["ok"], "raw": list(results)})
            if not result["ok"]:
                ok = False
                break
        with self._state_lock:
            self.task_progress = {"running": False, "index": len(results), "total": len(snapshot)}
            self._record("execute_tasks", {"code": 0 if ok else 1, "ok": ok, "raw": results})

    def _wait_for_target(self, target: PoseTarget) -> tuple[bool, Optional[PoseTarget], float, Optional[dict[str, Any]]]:
        """Waits for `target` to be reached.

        Primarily waits for the robot's own arrival confirmation
        (`data.is_arrived` on rt/slam_info / rt/slam_key_info -- see
        G1.get_slam_notice()) rather than only a client-computed xy-distance
        threshold, mirroring how keyDemo.cpp's taskLoopFun waits
        on the `is_arrived` flag its slamKeyInfoHandler sets from the
        robot's own task_result messages. The distance check remains as a
        fallback for firmware that never publishes that field. Whatever the
        latest relevant status notice was (including an obstacle-blocked
        flag) is returned alongside the result so callers can report it.
        """
        start = time.time()
        last_pose: Optional[PoseTarget] = None
        last_notice: Optional[dict[str, Any]] = None
        while time.time() - start < NAV_TARGET_TIMEOUT_S:
            if self._task_cancel.is_set():
                break
            pose = self.current_pose()
            if pose is not None:
                last_pose = pose
            with self._state_lock:
                notice = self.last_notice
            if notice is not None and notice.get("stamp", 0.0) >= start:
                last_notice = notice
                if notice.get("is_arrived") is True:
                    return True, pose or last_pose, time.time() - start, last_notice
            if pose is not None and pose.xy_distance_to(target) <= NAV_REACHED_DISTANCE_M:
                return True, pose, time.time() - start, last_notice
            time.sleep(NAV_POLL_INTERVAL_S)
        return False, last_pose, time.time() - start, last_notice

    def _mapping_worker(self) -> None:
        while not self._worker_stop.is_set():
            pose = self.current_pose()
            for name in ("slam_mapping", "slam_relocation", "slam_global_map", "slam_web_points", "deskewed", "livox"):
                sub = self.subs.get(name)
                if sub is None:
                    continue
                msg, _ts, count = sub.latest()
                if msg is None or self._seen_counts.get(name) == count:
                    continue
                self._seen_counts[name] = count
                pts = decode_xyz(msg, stride=3 if name in ("deskewed", "livox")
                                 else 1, max_points=18000, z_min=-0.35, z_max=1.7)
                if pts.size == 0:
                    continue
                if name in ("slam_mapping", "slam_relocation", "slam_global_map", "slam_web_points"):
                    self.occupancy.insert(pts[:, :2])
                elif pose is not None:
                    self.occupancy.insert(transform_xy(pts, pose))
                if name in ("deskewed", "livox"):
                    self.kiss.process(pts)
            time.sleep(0.04)

    def status(self) -> dict[str, Any]:
        now = time.time()
        pose = self.current_pose()
        topic_rows = []
        for name, sub in self.subs.items():
            msg, ts, count = sub.latest()
            topic_rows.append(
                {
                    "name": name,
                    "topic": sub.topic,
                    "age_s": None if ts <= 0 else round(now - ts, 2),
                    "messages": count,
                    "width": None if msg is None else int(getattr(msg, "width", 0) or 0),
                    "bytes": None if msg is None else len(bytes(getattr(msg, "data", b""))),
                }
            )
        _cloud, _pose, kiss_meta = self.kiss.snapshot()
        with self._state_lock:
            return {
                "iface": self.iface,
                "domain_id": self.domain_id,
                "slam_running": self.slam_running,
                "relocation_ready": self.relocation_ready,
                "pose": None if pose is None else {"x": pose.x, "y": pose.y, "yaw": pose.yaw},
                "initial_pose": None if self.initial_pose is None else {"x": self.initial_pose.x, "y": self.initial_pose.y, "yaw": self.initial_pose.yaw},
                "selected_pose": None if self.selected_pose is None else {"x": self.selected_pose.x, "y": self.selected_pose.y, "yaw": self.selected_pose.yaw},
                "task_count": len(self.tasks),
                "last_action": self.last_action,
                "last_notice": self.last_notice,
                "task_progress": dict(self.task_progress),
                "slam_info": self.g1.get_slam_info(),
                "slam_key": self.g1.get_slam_key_info(),
                "topics": topic_rows,
                "kiss": kiss_meta,
            }


def make_figure(
    state: SlamWebState,
    selected_layers: list[str],
    max_points: int,
    view_mode: str,
    stored_view: Optional[dict[str, Any]] = None,
) -> go.Figure:
    fig = go.Figure()
    pose = state.current_pose()
    extent_points: list[tuple[float, float]] = []

    for name, sub in state.subs.items():
        if name not in selected_layers:
            continue
        msg, ts, _count = sub.latest()
        if msg is None:
            continue
        stride = 1 if name.startswith("slam") or name in ("collision", "warning") else 5
        pts = decode_xyz(msg, stride=stride, max_points=max_points, z_min=-1.0, z_max=2.4)
        if pts.size == 0:
            continue
        if view_mode == "world" and not name.startswith("slam") and pose is not None:
            xy = transform_xy(pts, pose)
        else:
            xy = pts[:, :2]
        if xy.size:
            sample = xy[:: max(1, int(xy.shape[0] / 1000) + 1)]
            extent_points.extend((float(x), float(y)) for x, y in sample[:, :2])
        label, color, size, opacity = LAYER_STYLE.get(name, (name, "#ffffff", 2, 0.6))
        fig.add_trace(
            go.Scattergl(
                x=xy[:, 0],
                y=xy[:, 1],
                mode="markers",
                name=f"{label} ({time.time() - ts:.1f}s)",
                marker={"size": size, "color": color, "opacity": opacity},
                hovertemplate=f"{label}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<extra></extra>",
            )
        )

    if "occupancy" in selected_layers:
        occ = state.occupancy.points(max_points=max_points)
        if occ.size:
            sample = occ[:: max(1, int(occ.shape[0] / 1000) + 1)]
            extent_points.extend((float(x), float(y)) for x, y in sample[:, :2])
            label, color, size, opacity = LAYER_STYLE["occupancy"]
            fig.add_trace(go.Scattergl(x=occ[:, 0], y=occ[:, 1], mode="markers", name=label, marker={
                          "size": size, "color": color, "opacity": opacity}))

    if "kiss_map" in selected_layers:
        kiss_cloud, kiss_pose, kiss_meta = state.kiss.snapshot()
        if kiss_cloud.size:
            label, color, size, opacity = LAYER_STYLE["kiss_map"]
            xy = kiss_cloud[:, :2]
            if xy.shape[0] > max_points:
                idx = np.linspace(0, xy.shape[0] - 1, max_points, dtype=np.int64)
                xy = xy[idx]
            if xy.size:
                sample = xy[:: max(1, int(xy.shape[0] / 1000) + 1)]
                extent_points.extend((float(x), float(y)) for x, y in sample[:, :2])
            fig.add_trace(go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode="markers", name=f"{label} ({kiss_meta['frames']} frames)", marker={
                          "size": size, "color": color, "opacity": opacity}))
            fig.add_trace(
                go.Scattergl(
                    x=[float(kiss_pose[0, 3])],
                    y=[float(kiss_pose[1, 3])],
                    mode="markers",
                    name="KISS-ICP pose",
                    marker={"size": 12, "color": "#ff4fd8", "symbol": "diamond"},
                )
            )

    if state.initial_pose is not None:
        extent_points.append((state.initial_pose.x, state.initial_pose.y))
        fig.add_trace(
            go.Scattergl(
                x=[state.initial_pose.x],
                y=[state.initial_pose.y],
                mode="markers",
                name="Initial pose",
                marker={"size": 14, "color": "#ffae00", "symbol": "circle"},
            )
        )
    if pose is not None:
        extent_points.append((pose.x, pose.y))
        fig.add_trace(
            go.Scattergl(
                x=[pose.x],
                y=[pose.y],
                mode="markers",
                name="Current SLAM pose",
                marker={"size": 15, "color": "#ff3154", "symbol": "diamond"},
            )
        )
    with state._state_lock:
        tasks_snapshot = list(state.tasks)
    if tasks_snapshot:
        extent_points.extend((p.x, p.y) for p in tasks_snapshot)
        fig.add_trace(
            go.Scattergl(
                x=[p.x for p in tasks_snapshot],
                y=[p.y for p in tasks_snapshot],
                mode="markers+lines",
                name="Task points",
                marker={"size": 13, "color": "#111111", "symbol": "circle"},
                line={"color": "#111111", "width": 3},
                text=[str(i) for i in range(1, len(tasks_snapshot) + 1)],
                hovertemplate="task %{text}<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )
    if state.selected_pose is not None:
        extent_points.append((state.selected_pose.x, state.selected_pose.y))
        fig.add_trace(
            go.Scattergl(
                x=[state.selected_pose.x],
                y=[state.selected_pose.y],
                mode="markers",
                name="Selected pose",
                marker={"size": 17, "color": "#00e5ff", "symbol": "x"},
                hovertemplate="selected<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
            )
        )

    if extent_points:
        arr = np.asarray(extent_points, dtype=np.float32)
        xmin, ymin = np.nanmin(arr, axis=0)
        xmax, ymax = np.nanmax(arr, axis=0)
        pad = max(1.5, 0.10 * float(max(xmax - xmin, ymax - ymin, 1.0)))
        xmin -= pad
        xmax += pad
        ymin -= pad
        ymax += pad
    else:
        xmin, xmax, ymin, ymax = -5.0, 5.0, -5.0, 5.0
    grid_n = 90
    gx = np.linspace(float(xmin), float(xmax), grid_n)
    gy = np.linspace(float(ymin), float(ymax), grid_n)
    mx, my = np.meshgrid(gx, gy)
    fig.add_trace(
        go.Scatter(
            x=mx.ravel(),
            y=my.ravel(),
            mode="markers",
            name="Click target grid",
            showlegend=False,
            marker={"size": 8, "color": "rgba(80,180,255,0.035)"},
            hovertemplate="add task here<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        )
    )

    xaxis = {"title": "x (m)", "scaleanchor": "y", "scaleratio": 1, "gridcolor": "#303642"}
    yaxis = {"title": "y (m)", "gridcolor": "#303642"}
    if stored_view and stored_view.get("xrange") and stored_view.get("yrange"):
        xaxis["range"] = stored_view["xrange"]
        xaxis["autorange"] = False
        yaxis["range"] = stored_view["yrange"]
        yaxis["autorange"] = False

    fig.update_layout(
        template="plotly_dark",
        uirevision="slam-map",
        clickmode="event+select",
        margin={"l": 0, "r": 0, "t": 28, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0},
        paper_bgcolor="#111318",
        plot_bgcolor="#171a21",
        xaxis=xaxis,
        yaxis=yaxis,
        height=760,
    )
    return fig


def app_layout(state: SlamWebState, refresh_interval_ms: int = DEFAULT_MAP_REFRESH_INTERVAL_MS) -> html.Div:
    layer_options = [
        {"label": LAYER_STYLE.get(name, (name,))[0], "value": name}
        for name in [
            "slam_mapping",
            "slam_relocation",
            "slam_global_map",
            "slam_web_points",
            "deskewed",
            "livox",
            "kiss_map",
            "occupancy",
            "collision",
            "pre_collision",
            "safe",
            "pre_safe",
            "warning",
            "no_warning",
            "grid",
        ]
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("G1 SLAM Web Map"),
                            html.Div(id="status-line", className="status-line"),
                        ],
                        className="title-block",
                    ),
                    html.Div(
                        [
                            html.Button("Start Mapping", id="btn-start", n_clicks=0),
                            html.Button("Save Map", id="btn-save", n_clicks=0),
                            html.Button("Relocate", id="btn-relocate", n_clicks=0),
                            html.Button("Execute Tasks", id="btn-execute", n_clicks=0),
                            html.Button("Go To Selected", id="btn-go-selected", n_clicks=0),
                            html.Button("Add Current Pose", id="btn-add-current", n_clicks=0),
                            html.Button("Pause", id="btn-pause", n_clicks=0),
                            html.Button("Resume", id="btn-resume", n_clicks=0),
                            html.Button("Stop SLAM", id="btn-stop", n_clicks=0),
                            html.Button("Clear Tasks", id="btn-clear", n_clicks=0),
                        ],
                        className="toolbar",
                    ),
                ],
                className="topbar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Map path"),
                            dcc.Input(id="map-path", value=state.map_path,
                                      type="text", debounce=True),
                            html.Label("SLAM type"),
                            dcc.Dropdown(id="slam-type", options=[{"label": "indoor", "value": "indoor"}, {
                                         "label": "outdoor", "value": "outdoor"}], value="indoor", clearable=False),
                            html.Label("View"),
                            dcc.RadioItems(
                                id="view-mode",
                                options=[{"label": "World/map frame", "value": "world"},
                                         {"label": "Sensor frame", "value": "sensor"}],
                                value="world",
                                inline=False,
                            ),
                            html.Label("Layers"),
                            dcc.Checklist(id="layers", options=layer_options,
                                          value=DEFAULT_SELECTED_LAYERS, className="layers"),
                            html.Label("Max points per layer"),
                            dcc.Slider(id="max-points", min=100, max=60000, step=100, value=DEFAULT_MAX_POINTS_PER_LAYER,
                                       marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                            html.Div(id="action-output", className="action-output"),
                            html.Pre(id="topic-output", className="topic-output"),
                        ],
                        className="side",
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="map-graph", config={"displayModeBar": True, "scrollZoom": True}, clear_on_unhover=True),
                            html.Div(
                                "Click on any visible map/cloud feature to append a task point at that map coordinate.", className="hint"),
                        ],
                        className="map-pane",
                    ),
                ],
                className="main",
            ),
            dcc.Interval(id="map-interval", interval=max(250, int(refresh_interval_ms)), n_intervals=0),
            dcc.Store(id="map-view-store"),
        ],
        className="app",
    )


CSS = """
body { margin: 0; background: #111318; color: #e8ecf3; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.app { min-height: 100vh; }
.topbar { display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 12px 18px; background: #171a21; border-bottom: 1px solid #2b303b; }
h1 { font-size: 20px; line-height: 1.1; margin: 0 0 5px; font-weight: 650; }
.status-line { font-size: 12px; color: #aeb7c2; min-height: 16px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
button { background: #252b36; color: #f6f8fb; border: 1px solid #3b4352; border-radius: 6px; padding: 8px 10px; font-size: 13px; cursor: pointer; }
button:hover { background: #303849; }
.main { display: grid; grid-template-columns: 310px minmax(0, 1fr); min-height: calc(100vh - 72px); }
.side { padding: 14px; border-right: 1px solid #2b303b; background: #14171d; overflow: auto; }
.map-pane { min-width: 0; padding: 10px 12px 0; }
label { display: block; color: #e9eef7; font-size: 12px; margin: 13px 0 6px; }
input { width: 100%; box-sizing: border-box; background: #f7f9fc; color: #111827; border: 1px solid #596579; border-radius: 6px; padding: 8px; }
.side, .side * { color: #edf2fb; }
.side input, .side textarea { color: #111827; background: #f7f9fc; }
.layers label, .dash-radioitems label, .dash-checklist label { margin: 7px 0; color: #edf2fb !important; }
.Select-control, .Select-menu-outer, .Select-menu, .Select-option { background: #f7f9fc !important; color: #111827 !important; border-color: #596579 !important; }
.Select-value-label, .Select-placeholder, .Select-input, .Select-input input { color: #111827 !important; }
.Select-option.is-focused { background: #dbeafe !important; color: #111827 !important; }
.rc-slider-mark-text, .rc-slider-tooltip-inner { color: #f8fafc !important; }
.rc-slider-track { background-color: #55c7ff; }
.rc-slider-handle { border-color: #55c7ff; background: #f8fafc; }
.action-output { margin-top: 14px; padding: 10px; background: #0f1218; border: 1px solid #2b303b; border-radius: 6px; font-size: 12px; white-space: pre-wrap; }
.topic-output { margin-top: 10px; max-height: 280px; overflow: auto; background: #0f1218; border: 1px solid #2b303b; border-radius: 6px; padding: 10px; font-size: 11px; color: #cfd7e3; }
.hint { color: #aeb7c2; font-size: 12px; padding: 5px 0 10px; }
"""


def create_dash_app(
    state: SlamWebState,
    refresh_interval_ms: int = DEFAULT_MAP_REFRESH_INTERVAL_MS,
) -> dash.Dash:
    app = dash.Dash(__name__)
    app.index_string = f"""<!DOCTYPE html>
<html>
  <head>{{%metas%}}<title>G1 SLAM Web Map</title>{{%favicon%}}{{%css%}}<style>{CSS}</style></head>
  <body>{{%app_entry%}}<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer></body>
</html>"""
    app.layout = app_layout(state, refresh_interval_ms=refresh_interval_ms)

    @app.callback(
        Output("map-graph", "figure"),
        Output("status-line", "children"),
        Output("topic-output", "children"),
        Input("map-interval", "n_intervals"),
        Input("layers", "value"),
        Input("max-points", "value"),
        Input("view-mode", "value"),
        State("map-view-store", "data"),
    )
    def update_map(_n: int, layers: list[str], max_points: int, view_mode: str, stored_view: Optional[dict[str, Any]]):
        status = state.status()
        fig = make_figure(
            state,
            layers or [],
            int(max_points or DEFAULT_MAX_POINTS_PER_LAYER),
            str(view_mode or "world"),
            stored_view,
        )
        pose = status["pose"]
        pose_text = "pose=<none>" if pose is None else f"pose x={pose['x']:.2f} y={pose['y']:.2f} yaw={pose['yaw']:.2f}"
        kiss = status["kiss"]
        progress = status["task_progress"]
        if progress.get("running"):
            progress_text = f"task {progress.get('index', 0)}/{progress.get('total', 0)} running"
        else:
            progress_text = "no task sequence running"
        notice = status["last_notice"]
        notice_text = ""
        if notice:
            if notice.get("obstacle_blocked"):
                notice_text = " ⚠ OBSTACLE: path blocked"
            elif notice.get("error_code"):
                notice_text = f" ⚠ {notice.get('info') or ('errorCode=' + str(notice['error_code']))}"
            elif notice.get("nav_state") and notice.get("nav_state") != "ready":
                notice_text = f" nav_state={notice['nav_state']}"
        line = (
            f"iface={status['iface']} domain={status['domain_id']} "
            f"slam={'RUNNING' if status['slam_running'] else 'idle'} "
            f"relocation={'ready' if status['relocation_ready'] else 'not ready'} "
            f"tasks={status['task_count']} ({progress_text}) {pose_text} "
            f"KISS frames={kiss['frames']} {'err=' + kiss['error'] if kiss.get('error') else ''}"
            f"{notice_text}"
        )
        topics = json.dumps(
            {"topics": status["topics"], "last_action": status["last_action"], "last_notice": notice},
            indent=2, sort_keys=True, default=str,
        )
        return fig, line, topics

    @app.callback(
        Output("map-view-store", "data"),
        Input("map-graph", "relayoutData"),
        prevent_initial_call=True,
    )
    def remember_map_view(relayout_data: Optional[dict[str, Any]]):
        if not relayout_data:
            return dash.no_update
        if relayout_data.get("xaxis.autorange") or relayout_data.get("yaxis.autorange"):
            return None
        xrange = relayout_data.get("xaxis.range")
        yrange = relayout_data.get("yaxis.range")
        if xrange is None:
            xrange = [relayout_data.get("xaxis.range[0]"), relayout_data.get("xaxis.range[1]")]
        if yrange is None:
            yrange = [relayout_data.get("yaxis.range[0]"), relayout_data.get("yaxis.range[1]")]
        if any(value is None for value in [*xrange, *yrange]):
            return dash.no_update
        return {"xrange": [float(xrange[0]), float(xrange[1])], "yrange": [float(yrange[0]), float(yrange[1])]}

    @app.callback(
        Output("action-output", "children"),
        Input("btn-start", "n_clicks"),
        Input("btn-save", "n_clicks"),
        Input("btn-relocate", "n_clicks"),
        Input("btn-execute", "n_clicks"),
        Input("btn-go-selected", "n_clicks"),
        Input("btn-add-current", "n_clicks"),
        Input("btn-pause", "n_clicks"),
        Input("btn-resume", "n_clicks"),
        Input("btn-stop", "n_clicks"),
        Input("btn-clear", "n_clicks"),
        State("map-path", "value"),
        State("slam-type", "value"),
        prevent_initial_call=True,
    )
    def run_action(_start, _save, _reloc, _exec, _go_selected, _add_current, _pause, _resume, _stop, _clear, map_path: str, slam_type: str):
        trigger = dash.ctx.triggered_id
        if trigger == "btn-start":
            result = state.start_mapping(str(slam_type or "indoor"))
        elif trigger == "btn-save":
            result = state.save_map(str(map_path or state.map_path))
        elif trigger == "btn-relocate":
            result = state.relocate(str(map_path or state.map_path))
        elif trigger == "btn-execute":
            result = state.execute_tasks()
        elif trigger == "btn-go-selected":
            result = state.go_to_selected_pose()
        elif trigger == "btn-add-current":
            result = state.add_current_pose()
        elif trigger == "btn-pause":
            result = state.pause()
        elif trigger == "btn-resume":
            result = state.resume()
        elif trigger == "btn-stop":
            result = state.stop_slam()
        elif trigger == "btn-clear":
            result = state.clear_tasks()
        else:
            result = state.last_action
        return json.dumps(result, indent=2, sort_keys=True, default=str)

    @app.callback(
        Output("action-output", "children", allow_duplicate=True),
        Input("map-graph", "clickData"),
        prevent_initial_call=True,
    )
    def add_task_from_click(click_data: Optional[dict[str, Any]]):
        if not click_data or not click_data.get("points"):
            return dash.no_update
        point = click_data["points"][0]
        if "x" not in point or "y" not in point:
            return "Click did not include map coordinates."
        task = state.add_task(float(point["x"]), float(point["y"]))
        return json.dumps({"label": "add_task_from_click", "ok": True, "selected_pose": {"x": task.x, "y": task.y, "yaw": task.yaw}, "task_count": len(state.tasks)}, indent=2)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G1 SLAM mapping web app with live map layers and task-point clicks.")
    parser.add_argument("--iface", default=os.environ.get("G1_IFACE", "eth0"))
    parser.add_argument("--domain-id", type=int, default=int(os.environ.get("G1_DOMAIN_ID", "0")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--map-path", default="/home/unitree/test.pcd")
    parser.add_argument("--refresh-ms", type=int, default=int(os.environ.get("G1_SLAM_REFRESH_MS", str(DEFAULT_MAP_REFRESH_INTERVAL_MS))))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = SlamWebState(args.iface, args.domain_id, DEFAULT_TOPICS, args.map_path)
    app = create_dash_app(state, refresh_interval_ms=args.refresh_ms)
    print(
        f"SLAM web app: http://{args.host}:{args.port} iface={args.iface} domain={args.domain_id}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
