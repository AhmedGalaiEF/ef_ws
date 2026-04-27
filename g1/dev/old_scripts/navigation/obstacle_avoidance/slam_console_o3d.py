#!/usr/bin/env python3
"""
slam_console_o3d.py
===================

Console SLAM controller + Open3D map visualizer.

- Open3D window shows the latest HeightMap as a point cloud.
- Terminal controls mapping, map loading, and start/goal definition.

Only the Open3D window is a GUI; all controls are in the console.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Force CPU-only Open3D to avoid CUDA driver/runtime mismatch crashes.
os.environ.setdefault("OPEN3D_CPU_ONLY", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import open3d as o3d  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Open3D not available: {exc}") from exc

try:
    import curses
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"curses not available: {exc}") from exc

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc

from slam_map import SlamMapSubscriber, SlamInfoSubscriber, SlamPointCloudSubscriber, SlamOdomSubscriber
from slam_service import SlamOperateClient


class _Open3DViewer:
    def __init__(self) -> None:
        self._vis = o3d.visualization.Visualizer()
        ok = self._vis.create_window(window_name="SLAM Map (HeightMap)", width=1280, height=720)
        if not ok:
            raise RuntimeError("Open3D failed to create window")
        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)
        self._pose_frame: Optional[o3d.geometry.TriangleMesh] = None
        self._first = True

    def update(self, pts: np.ndarray, pose: Optional[np.ndarray]) -> bool:
        self._pcd.points = o3d.utility.Vector3dVector(pts)
        self._vis.update_geometry(self._pcd)

        if pose is not None:
            if self._pose_frame is not None:
                self._vis.remove_geometry(self._pose_frame, reset_bounding_box=False)
            size = 0.5
            if len(self._pcd.points) > 0:
                bbox = self._pcd.get_axis_aligned_bounding_box()
                extent = bbox.get_max_bound() - bbox.get_min_bound()
                size = float(np.linalg.norm(extent)) * 0.03
                size = max(0.2, min(size, 2.0))
            self._pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
            self._pose_frame.transform(pose)
            self._vis.add_geometry(self._pose_frame, reset_bounding_box=False)
            self._vis.update_geometry(self._pose_frame)

        if self._first and len(self._pcd.points) > 0:
            self._vis.reset_view_point(True)
            self._first = False

        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def close(self) -> None:
        self._vis.destroy_window()


def _quat_to_mat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # Convert quaternion to rotation matrix (right-handed).
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


def _heightmap_to_points(
    msg,
    height_threshold: float,
    max_height: Optional[float],
    origin_centered: bool,
    stride: int,
    max_points: int,
) -> Optional[np.ndarray]:
    try:
        width = int(msg.width)
        height = int(msg.height)
        resolution = float(msg.resolution)
        data = np.array(list(msg.data), dtype=np.float32)
    except Exception:
        return None

    if width <= 0 or height <= 0 or resolution <= 0:
        return None
    if data.size != width * height:
        return None

    grid = data.reshape((height, width))
    if max_height is not None:
        grid = np.minimum(grid, max_height)

    mask = grid >= height_threshold
    if not np.any(mask):
        return None

    rows, cols = np.where(mask)
    if stride > 1:
        rows = rows[::stride]
        cols = cols[::stride]

    if origin_centered:
        origin_x = -width * resolution / 2.0
        origin_y = -height * resolution / 2.0
    else:
        origin_x = 0.0
        origin_y = 0.0

    xs = origin_x + cols.astype(np.float32) * resolution
    ys = origin_y + rows.astype(np.float32) * resolution
    zs = grid[rows, cols]
    pts = np.stack([xs, ys, zs], axis=1).astype(np.float64)

    if max_points > 0 and pts.shape[0] > max_points:
        step = int(pts.shape[0] / max_points) + 1
        pts = pts[::step]

    return pts


def _decode_points_xyz(
    msg,
    stride: int,
    zmin: float,
    zmax: float,
    max_points: int,
) -> Optional[np.ndarray]:
    try:
        fields = {f.name: f for f in msg.fields}
        if "x" not in fields or "y" not in fields or "z" not in fields:
            return None
        point_step = int(msg.point_step)
        if point_step <= 0:
            return None
        data = bytes(msg.data)
        if not data:
            return None
        xoff = int(fields["x"].offset)
        yoff = int(fields["y"].offset)
        zoff = int(fields["z"].offset)
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": ["<f4", "<f4", "<f4"],
                "offsets": [xoff, yoff, zoff],
                "itemsize": point_step,
            }
        )
        arr = np.frombuffer(data, dtype=dtype, count=len(data) // point_step)
        xs = arr["x"][:: max(1, stride)]
        ys = arr["y"][:: max(1, stride)]
        zs = arr["z"][:: max(1, stride)]
        mask = (zs >= zmin) & (zs <= zmax)
        if not np.any(mask):
            return None
        pts = np.stack([xs[mask], ys[mask], zs[mask]], axis=1).astype(np.float64)
        if max_points > 0 and pts.shape[0] > max_points:
            step = int(pts.shape[0] / max_points) + 1
            pts = pts[::step]
        return pts
    except Exception:
        return None


class SlamConsoleApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.map_sub = SlamMapSubscriber(args.slam_map_topic)
        self.info_sub = SlamInfoSubscriber(args.slam_info_topic, args.slam_key_topic)
        self.slam_pts_sub = SlamPointCloudSubscriber(args.slam_points_topic)
        self.lidar_pts_sub = SlamPointCloudSubscriber(args.lidar_points_topic)
        self.odom_sub = SlamOdomSubscriber(args.slam_odom_topic)
        self.map_sub.start()
        self.info_sub.start()
        self.slam_pts_sub.start()
        self.lidar_pts_sub.start()
        self.odom_sub.start()

        self.slam_client = SlamOperateClient()
        self.slam_client.Init()
        self.slam_client.SetTimeout(10.0)

        self.viewer = _Open3DViewer()
        self.last_map_ts = 0.0
        self.last_cloud_ts = 0.0

        self.cur_pose: Optional[dict] = None
        self.odom_pose: Optional[dict] = None
        self.task_stop = threading.Event()
        self.task_thread: Optional[threading.Thread] = None
        self.task_arrived = False
        self.task_last_ts = 0.0

        self.start_pose: Optional[dict] = None
        self.goal_pose: Optional[dict] = None

    def _log(self, stdscr, msg: str) -> None:
        stdscr.addstr(0, 0, msg.ljust(80))

    def _update_pose_from_info(self) -> None:
        info = self.info_sub.get_info()
        if not info:
            return
        try:
            payload = json.loads(info)
            if payload.get("type") == "pos_info":
                cur = payload.get("data", {}).get("currentPose", {})
                self.cur_pose = {
                    "x": float(cur.get("x", 0.0)),
                    "y": float(cur.get("y", 0.0)),
                    "z": float(cur.get("z", 0.0)),
                    "q_x": float(cur.get("q_x", 0.0)),
                    "q_y": float(cur.get("q_y", 0.0)),
                    "q_z": float(cur.get("q_z", 0.0)),
                    "q_w": float(cur.get("q_w", 1.0)),
                }
        except Exception:
            pass

    def _update_pose_from_odom(self) -> None:
        pose = self.odom_sub.get_pose_full()
        if pose is None:
            return
        x, y, z, qx, qy, qz, qw = pose
        self.odom_pose = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "q_x": float(qx),
            "q_y": float(qy),
            "q_z": float(qz),
            "q_w": float(qw),
        }

    def _update_task_result(self) -> None:
        key = self.info_sub.get_key()
        if not key:
            return
        try:
            payload = json.loads(key)
            if payload.get("type") == "task_result":
                arrived = bool(payload.get("data", {}).get("is_arrived", False))
                if arrived and (time.time() - self.task_last_ts) < self.args.nav_timeout:
                    self.task_arrived = True
        except Exception:
            pass

    def _start_mapping(self) -> None:
        resp = self.slam_client.start_mapping("indoor")
        print(f"start_mapping: code={resp.code} raw={resp.raw}")

    def _end_mapping(self) -> None:
        resp = self.slam_client.end_mapping(self.args.save_path)
        print(f"end_mapping: code={resp.code} raw={resp.raw}")

    def _reset_mapping(self) -> None:
        resp = self.slam_client.close_slam()
        print(f"close_slam: code={resp.code} raw={resp.raw}")
        time.sleep(0.5)
        self._start_mapping()

    def _add_current_pose(self) -> None:
        pose = self.cur_pose or self.odom_pose
        if pose is None:
            print("No current pose available (slam_info/odom).")
            return
        self.task_list.append(dict(pose))
        print(f"Added pose to task list (len={len(self.task_list)})")

    def _set_start_from_current(self) -> None:
        pose = self.cur_pose or self.odom_pose
        if pose is None:
            print("No current pose available (slam_info/odom).")
            return
        self.start_pose = dict(pose)
        print(f"Start pose set: x={pose['x']:+.2f}, y={pose['y']:+.2f}")

    def _set_goal_from_input(self) -> None:
        try:
            raw = input("Enter goal as: x y yaw_deg (e.g. 1.0 2.0 0): ").strip()
            parts = raw.split()
            if len(parts) < 2:
                print("Invalid input.")
                return
            x = float(parts[0])
            y = float(parts[1])
            yaw_deg = float(parts[2]) if len(parts) >= 3 else 0.0
        except Exception:
            print("Invalid input.")
            return
        yaw = math.radians(yaw_deg)
        half = yaw / 2.0
        pose = {
            "x": x,
            "y": y,
            "z": 0.0,
            "q_x": 0.0,
            "q_y": 0.0,
            "q_z": math.sin(half),
            "q_w": math.cos(half),
        }
        self.goal_pose = pose
        print(f"Goal pose set: x={x:+.2f}, y={y:+.2f}, yaw={yaw_deg:+.1f}deg")

    def _load_map(self) -> None:
        try:
            path = input(f"Map PCD path [{self.args.load_path}]: ").strip()
        except Exception:
            return
        if not path:
            path = self.args.load_path
        if not path:
            print("No map path provided.")
            return
        self.args.load_path = path
        resp = self.slam_client.init_pose(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            path,
        )
        print(f"load map (init_pose): code={resp.code} raw={resp.raw}")

    def _build_pose_mat(self) -> Optional[np.ndarray]:
        pose = self.cur_pose or self.odom_pose
        if pose is None:
            return None
        r = _quat_to_mat(
            pose["q_x"],
            pose["q_y"],
            pose["q_z"],
            pose["q_w"],
        )
        x, y, z = pose["x"], pose["y"], pose["z"]
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = r
        mat[:3, 3] = [x, y, z]
        return mat

    def run(self) -> None:
        def _loop(stdscr) -> None:
            curses.cbreak()
            stdscr.nodelay(True)
            stdscr.clear()
            stdscr.addstr(1, 0, "Keys: q=start  w=save/stop  r=reset  l=load map  s=set start  g=set goal  x=exit")
            stdscr.addstr(2, 0, f"save_path: {self.args.save_path}")
            stdscr.refresh()

            while True:
                ch = stdscr.getch()
                if ch != -1:
                    if ch in (ord("q"), ord("Q")):
                        self._start_mapping()
                    elif ch in (ord("w"), ord("W")):
                        self._end_mapping()
                    elif ch in (ord("r"), ord("R")):
                        self._reset_mapping()
                    elif ch in (ord("l"), ord("L")):
                        self._load_map()
                    elif ch in (ord("s"), ord("S")):
                        self._set_start_from_current()
                    elif ch in (ord("g"), ord("G")):
                        self._set_goal_from_input()
                    elif ch in (ord("x"), ord("X"), 27):
                        return

                slam_msg, slam_ts = self.slam_pts_sub.get_latest()
                lidar_msg, lidar_ts = self.lidar_pts_sub.get_latest()
                use_msg = slam_msg
                use_ts = slam_ts
                if use_msg is None or use_ts == 0.0:
                    use_msg = lidar_msg
                    use_ts = lidar_ts

                if use_msg is not None and use_ts > self.last_cloud_ts:
                    pts = _decode_points_xyz(
                        use_msg,
                        stride=self.args.points_stride,
                        zmin=self.args.points_z_min,
                        zmax=self.args.points_z_max,
                        max_points=self.args.max_points,
                    )
                    self.last_cloud_ts = use_ts
                    if pts is not None:
                        self._update_pose_from_info()
                        self._update_pose_from_odom()
                        pose_mat = self._build_pose_mat()
                        if not self.viewer.update(pts, pose_mat):
                            return

                self._update_task_result()

                sp = "set" if self.start_pose is not None else "unset"
                gp = "set" if self.goal_pose is not None else "unset"
                stdscr.addstr(4, 0, f"start: {sp}  goal: {gp}  cloud_ts: {self.last_cloud_ts:.1f}".ljust(80))
                stdscr.refresh()
                time.sleep(0.02)

        curses.wrapper(_loop)
        self.viewer.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Console SLAM controller with Open3D map viewer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iface", default="eth0", help="Network interface for DDS")
    p.add_argument("--domain-id", type=int, default=0, help="DDS domain id")
    p.add_argument("--slam-map-topic", default="rt/utlidar/map_state", help="SLAM map topic")
    p.add_argument("--slam-info-topic", default="rt/slam_info", help="SLAM info topic")
    p.add_argument("--slam-key-topic", default="rt/slam_key_info", help="SLAM key info topic")
    p.add_argument("--slam-odom-topic", default="rt/unitree/slam_mapping/odom", help="SLAM odom topic")
    p.add_argument("--save-path", default="/home/unitree/test1.pcd", help="PCD save path on robot")
    p.add_argument("--load-path", default="/home/unitree/test1.pcd", help="PCD load path on robot")
    p.add_argument("--slam-points-topic", default="rt/unitree/slam_mapping/points", help="SLAM points topic")
    p.add_argument("--lidar-points-topic", default="rt/utlidar/cloud_livox_mid360", help="Lidar points topic")
    p.add_argument("--points-stride", type=int, default=6, help="Stride for point clouds")
    p.add_argument("--points-z-min", type=float, default=-0.5, help="Min z for points")
    p.add_argument("--points-z-max", type=float, default=1.5, help="Max z for points")
    p.add_argument("--max-points", type=int, default=300000, help="Max points to render (0=all)")
    p.add_argument("--nav-timeout", type=float, default=60.0, help="SLAM pose-nav timeout (s)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ChannelFactoryInitialize(args.domain_id, args.iface)
    app = SlamConsoleApp(args)
    app.run()


if __name__ == "__main__":
    main()
