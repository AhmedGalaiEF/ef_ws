#!/usr/bin/env python3
"""
SLAM visualization runner with robot-nav overlay.

Runs the existing navigation/obstacle_avoidance/live_slam_save.py pipeline, but
injects a viewer that overlays robot SLAM navigation info (goal + trajectory)
inside the same Open3D map window.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


def _load_live_slam_save_module():
    script_dir = Path(__file__).resolve().parent
    nav_dir = script_dir.parent.parent / "navigation" / "obstacle_avoidance"
    if not nav_dir.exists():
        raise SystemExit(f"Missing navigation path: {nav_dir}")
    # live_slam_save expects relative assets (e.g. mid360_config.json) in CWD.
    os.chdir(str(nav_dir))
    if str(nav_dir) not in sys.path:
        sys.path.insert(0, str(nav_dir))
    import live_slam_save as lss  # type: ignore

    return lss


class _NavOverlay:
    def __init__(
        self,
        iface: str,
        domain_id: int,
        info_topic: str,
        key_topic: str,
        plan_file: str | None = None,
    ) -> None:
        self._ok = False
        self._sub = None
        self._lock = threading.Lock()
        self._cur_xy: tuple[float, float] | None = None
        self._goal_xy: tuple[float, float] | None = None
        self._polyline_xy: list[tuple[float, float]] = []
        self._trail_xy: list[tuple[float, float]] = []
        self._last_info_ts = 0.0
        self._plan_file = Path(plan_file).expanduser() if plan_file else None
        self._last_plan_mtime = 0.0

        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from slam_map import SlamInfoSubscriber  # type: ignore

            ChannelFactoryInitialize(int(domain_id), iface)
            sub = SlamInfoSubscriber(info_topic, key_topic)
            sub.start()
            self._sub = sub
            self._ok = True
        except Exception:
            self._ok = False

    @property
    def ready(self) -> bool:
        return bool(self._ok)

    @staticmethod
    def _extract_xy_points(data: dict[str, Any]) -> list[tuple[float, float]]:
        candidates = [
            data.get("path"),
            data.get("paths"),
            data.get("route"),
            data.get("waypoints"),
            data.get("pathPoints"),
            data.get("globalPath"),
        ]
        for obj in candidates:
            if not isinstance(obj, list):
                continue
            pts: list[tuple[float, float]] = []
            for p in obj:
                if not isinstance(p, dict):
                    continue
                try:
                    x = float(p.get("x"))
                    y = float(p.get("y"))
                except Exception:
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    pts.append((x, y))
            if len(pts) >= 2:
                return pts
        return []

    def update(self) -> None:
        if self._plan_file is not None:
            try:
                if self._plan_file.exists():
                    mtime = float(self._plan_file.stat().st_mtime)
                    if mtime > self._last_plan_mtime:
                        self._last_plan_mtime = mtime
                        payload = json.loads(self._plan_file.read_text(encoding="utf-8"))
                        if isinstance(payload, dict):
                            goal = payload.get("goal")
                            if isinstance(goal, dict):
                                gx = float(goal.get("x"))
                                gy = float(goal.get("y"))
                                if math.isfinite(gx) and math.isfinite(gy):
                                    with self._lock:
                                        self._goal_xy = (gx, gy)
                            path = payload.get("path")
                            if isinstance(path, list):
                                pts: list[tuple[float, float]] = []
                                for p in path:
                                    if not isinstance(p, dict):
                                        continue
                                    try:
                                        px = float(p.get("x"))
                                        py = float(p.get("y"))
                                    except Exception:
                                        continue
                                    if math.isfinite(px) and math.isfinite(py):
                                        pts.append((px, py))
                                if len(pts) >= 2:
                                    with self._lock:
                                        self._polyline_xy = pts
            except Exception:
                pass

        if not self._ok or self._sub is None:
            return

        for payload_raw in (self._sub.get_info(), self._sub.get_key()):
            if not payload_raw:
                continue
            try:
                payload = json.loads(payload_raw)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue

            ptype = payload.get("type")
            data = payload.get("data", {})
            if not isinstance(data, dict):
                continue

            if ptype == "pos_info":
                cur = data.get("currentPose", {})
                if isinstance(cur, dict):
                    try:
                        cx = float(cur.get("x"))
                        cy = float(cur.get("y"))
                    except Exception:
                        cx = cy = None  # type: ignore[assignment]
                    if cx is not None and cy is not None and math.isfinite(cx) and math.isfinite(cy):
                        with self._lock:
                            self._cur_xy = (cx, cy)
                            if not self._trail_xy or math.hypot(self._trail_xy[-1][0] - cx, self._trail_xy[-1][1] - cy) > 0.05:
                                self._trail_xy.append((cx, cy))
                                if len(self._trail_xy) > 1200:
                                    self._trail_xy = self._trail_xy[-1200:]
                        self._last_info_ts = time.time()
                continue

            if ptype == "ctrl_info":
                target = data.get("targetPose", {})
                if isinstance(target, dict):
                    try:
                        gx = float(target.get("x"))
                        gy = float(target.get("y"))
                    except Exception:
                        gx = gy = None  # type: ignore[assignment]
                    if gx is not None and gy is not None and math.isfinite(gx) and math.isfinite(gy):
                        with self._lock:
                            self._goal_xy = (gx, gy)
                pts = self._extract_xy_points(data)
                if pts:
                    with self._lock:
                        self._polyline_xy = pts
                continue

            if ptype == "task_result":
                arrived = bool(data.get("is_arrived", False))
                if arrived:
                    with self._lock:
                        self._polyline_xy = []

    def snapshot(self) -> tuple[list[tuple[float, float]], tuple[float, float] | None]:
        with self._lock:
            cur = self._cur_xy
            goal = self._goal_xy
            path = list(self._polyline_xy) if len(self._polyline_xy) >= 2 else []
            trail = list(self._trail_xy)
        if not path:
            if cur is not None and goal is not None:
                path = [cur, goal]
            elif len(trail) >= 2:
                path = trail
        return path, goal


def _xy_to_xyz(points_xy: list[tuple[float, float]], z: float) -> np.ndarray:
    if not points_xy:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.zeros((len(points_xy), 3), dtype=np.float64)
    arr[:, 0] = [p[0] for p in points_xy]
    arr[:, 1] = [p[1] for p in points_xy]
    arr[:, 2] = float(z)
    return arr


def _make_overlay_viewer_class(lss, overlay: _NavOverlay):
    o3d = lss.o3d

    class _OverlayViewer(lss._Viewer):  # type: ignore[attr-defined]
        def __init__(self):
            super().__init__()
            self._overlay = overlay
            self._path_ls = o3d.geometry.LineSet()
            self._goal_pcd = o3d.geometry.PointCloud()
            self._added_path = False
            self._added_goal = False

        def _update_overlay(self) -> None:
            self._overlay.update()
            path_xy, goal_xy = self._overlay.snapshot()

            if path_xy:
                path_xyz = _xy_to_xyz(path_xy, z=0.05)
                lines = np.array([[i, i + 1] for i in range(len(path_xyz) - 1)], dtype=np.int32)
                colors = np.tile(np.array([[1.0, 0.2, 0.2]], dtype=np.float64), (max(0, len(lines)), 1))
                self._path_ls.points = o3d.utility.Vector3dVector(path_xyz)
                self._path_ls.lines = o3d.utility.Vector2iVector(lines)
                self._path_ls.colors = o3d.utility.Vector3dVector(colors)
                if not self._added_path:
                    self._vis.add_geometry(self._path_ls, reset_bounding_box=False)
                    self._added_path = True
                else:
                    self._vis.update_geometry(self._path_ls)

            if goal_xy is not None:
                goal_xyz = _xy_to_xyz([goal_xy], z=0.12)
                self._goal_pcd.points = o3d.utility.Vector3dVector(goal_xyz)
                self._goal_pcd.colors = o3d.utility.Vector3dVector(np.array([[0.0, 1.0, 1.0]], dtype=np.float64))
                if not self._added_goal:
                    self._vis.add_geometry(self._goal_pcd, reset_bounding_box=False)
                    self._added_goal = True
                else:
                    self._vis.update_geometry(self._goal_pcd)

        def tick(self) -> bool:
            alive = super().tick()
            if not alive:
                return False
            try:
                self._update_overlay()
            except Exception:
                pass
            return True

    return _OverlayViewer


def main() -> None:
    ap = argparse.ArgumentParser(description="SLAM visualization with nav overlay")
    ap.add_argument("--save-dir", default="./maps", help="Directory to save maps (use empty to disable)")
    ap.add_argument("--save-every", type=int, default=1, help="Save every N updates")
    ap.add_argument("--save-latest", action="store_true", help="Overwrite a single latest file")
    ap.add_argument("--save-prefix", default="live_slam", help="Filename prefix for saved maps")
    ap.add_argument("--iface", default="eth0", help="network interface for DDS slam info overlay")
    ap.add_argument("--domain-id", type=int, default=0, help="DDS domain id")
    ap.add_argument("--slam-info-topic", default="rt/slam_info", help="SLAM info topic")
    ap.add_argument("--slam-key-topic", default="rt/slam_key_info", help="SLAM key topic")
    ap.add_argument("--overlay-plan-file", default="/tmp/g1_nav_overlay_plan.json", help="JSON file for external planned path overlay")
    args = ap.parse_args()

    lss = _load_live_slam_save_module()
    overlay = _NavOverlay(
        args.iface,
        int(args.domain_id),
        args.slam_info_topic,
        args.slam_key_topic,
        plan_file=args.overlay_plan_file,
    )
    lss._Viewer = _make_overlay_viewer_class(lss, overlay)  # type: ignore[attr-defined]

    save_dir = Path(args.save_dir) if args.save_dir else None
    demo = lss.LiveSLAMDemo(save_dir, args.save_every, args.save_latest, args.save_prefix)

    stop = False

    def _sigint(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop and demo._viewer.tick():
            time.sleep(0.01)
    finally:
        demo.shutdown()


if __name__ == "__main__":
    main()
