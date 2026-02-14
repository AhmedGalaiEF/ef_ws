#!/usr/bin/env python3
"""
view_map.py
===========

View a saved point cloud map (PCD/PLY) and highlight poses.

Features:
- Shows the saved map in Open3D.
- Highlights the initial robot pose (from sidecar JSON if available).
- Allows adding start/goal poses via console input while the viewer runs.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d


def _quat_to_mat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    xx, yy, zz = qx * x2, qy * y2, qz * z2
    xy, xz, yz = qx * y2, qx * z2, qy * z2
    wx, wy, wz = qw * x2, qw * y2, qw * z2
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=float,
    )


def _pose_to_mat(pose: dict) -> np.ndarray:
    R = _quat_to_mat(
        float(pose.get("q_x", 0.0)),
        float(pose.get("q_y", 0.0)),
        float(pose.get("q_z", 0.0)),
        float(pose.get("q_w", 1.0)),
    )
    x = float(pose.get("x", 0.0))
    y = float(pose.get("y", 0.0))
    z = float(pose.get("z", 0.0))
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = R
    mat[:3, 3] = [x, y, z]
    return mat


def _read_pose_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        if "pose" in payload:
            return payload["pose"]
    except Exception:
        pass
    return None


class MapViewer:
    def __init__(self, cloud: o3d.geometry.PointCloud, initial_pose: Optional[dict]):
        self.cloud = cloud
        self.initial_pose = initial_pose
        self.start_pose: Optional[dict] = None
        self.goal_pose: Optional[dict] = None

        self.vis = o3d.visualization.Visualizer()
        ok = self.vis.create_window(window_name="Saved SLAM Map", width=1280, height=720)
        if not ok:
            raise RuntimeError("Open3D failed to create window")

        self.vis.add_geometry(self.cloud)

        self._init_frame: Optional[o3d.geometry.TriangleMesh] = None
        self._start_frame: Optional[o3d.geometry.TriangleMesh] = None
        self._goal_frame: Optional[o3d.geometry.TriangleMesh] = None

        self._frame_size = self._estimate_frame_size()

        if self.initial_pose is not None:
            self._init_frame = self._make_frame(self.initial_pose, self._frame_size)
            self.vis.add_geometry(self._init_frame)

    def _estimate_frame_size(self) -> float:
        size = 0.5
        if len(self.cloud.points) > 0:
            bbox = self.cloud.get_axis_aligned_bounding_box()
            extent = bbox.get_max_bound() - bbox.get_min_bound()
            size = float(np.linalg.norm(extent)) * 0.03
            size = max(0.2, min(size, 2.0))
        return size

    def _make_frame(self, pose: dict, size: float) -> o3d.geometry.TriangleMesh:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        frame.transform(_pose_to_mat(pose))
        return frame

    def set_start(self, pose: dict) -> None:
        if self._start_frame is not None:
            self.vis.remove_geometry(self._start_frame, reset_bounding_box=False)
        self.start_pose = pose
        self._start_frame = self._make_frame(pose, self._frame_size)
        self.vis.add_geometry(self._start_frame, reset_bounding_box=False)

    def set_goal(self, pose: dict) -> None:
        if self._goal_frame is not None:
            self.vis.remove_geometry(self._goal_frame, reset_bounding_box=False)
        self.goal_pose = pose
        self._goal_frame = self._make_frame(pose, self._frame_size)
        self.vis.add_geometry(self._goal_frame, reset_bounding_box=False)

    def loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            alive = self.vis.poll_events()
            self.vis.update_renderer()
            if not alive:
                stop_event.set()
                break
            time.sleep(0.01)

    def close(self) -> None:
        self.vis.destroy_window()


def _parse_pose_input(prompt: str) -> Optional[dict]:
    try:
        raw = input(prompt).strip()
    except Exception:
        return None
    if not raw:
        return None
    parts = raw.split()
    if len(parts) < 2:
        return None
    try:
        x = float(parts[0])
        y = float(parts[1])
        yaw_deg = float(parts[2]) if len(parts) >= 3 else 0.0
    except Exception:
        return None
    yaw = math.radians(yaw_deg)
    half = yaw / 2.0
    return {
        "x": x,
        "y": y,
        "z": 0.0,
        "q_x": 0.0,
        "q_y": 0.0,
        "q_z": math.sin(half),
        "q_w": math.cos(half),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="View a saved map and add start/goal poses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("map_path", help="Path to saved map (.pcd/.ply)")
    ap.add_argument("--pose-json", default="", help="Path to pose JSON (sidecar)")
    args = ap.parse_args()

    map_path = Path(args.map_path)
    if not map_path.exists():
        raise SystemExit(f"Map not found: {map_path}")

    cloud = o3d.io.read_point_cloud(str(map_path))
    if len(cloud.points) == 0:
        raise SystemExit("Loaded point cloud is empty.")

    pose_json = Path(args.pose_json) if args.pose_json else map_path.with_suffix(".json")
    initial_pose = _read_pose_json(pose_json)

    viewer = MapViewer(cloud, initial_pose)
    stop_event = threading.Event()

    def _input_loop() -> None:
        print("Commands:")
        print("  start  -> set start pose (x y yaw_deg)")
        print("  goal   -> set goal pose (x y yaw_deg)")
        print("  quit   -> exit")
        while not stop_event.is_set():
            try:
                cmd = input("> ").strip().lower()
            except Exception:
                stop_event.set()
                break
            if cmd == "start":
                pose = _parse_pose_input("start x y yaw_deg: ")
                if pose:
                    viewer.set_start(pose)
                    print("start pose set")
                else:
                    print("invalid start pose")
            elif cmd == "goal":
                pose = _parse_pose_input("goal x y yaw_deg: ")
                if pose:
                    viewer.set_goal(pose)
                    print("goal pose set")
                else:
                    print("invalid goal pose")
            elif cmd in ("quit", "exit"):
                stop_event.set()
                break
            else:
                print("unknown command")

    threading.Thread(target=_input_loop, daemon=True).start()

    try:
        viewer.loop(stop_event)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
