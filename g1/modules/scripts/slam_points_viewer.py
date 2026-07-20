#!/usr/bin/env python3
from __future__ import annotations
from sdk_slam import SlamInfoSubscriber, SlamOdomSubscriber

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("OPEN3D_CPU_ONLY", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import open3d as o3d
except Exception as exc:
    raise SystemExit(f"Open3D not available: {exc}") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))


try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


class PointCloudSubscriber:
    def __init__(self, topic: str) -> None:
        self.topic = topic
        self._msg: PointCloud2_ | None = None
        self._ts = 0.0
        self._sub: ChannelSubscriber | None = None

    def start(self) -> None:
        if self._sub is None:
            self._sub = ChannelSubscriber(self.topic, PointCloud2_)
            self._sub.Init(self._callback, 10)

    def _callback(self, msg: PointCloud2_) -> None:
        self._msg = msg
        self._ts = time.time()

    def get_latest(self) -> tuple[PointCloud2_ | None, float]:
        return self._msg, self._ts


def _decode_points_xyz(msg: PointCloud2_, stride: int, zmin: float, zmax: float, max_points: int) -> np.ndarray | None:
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
        mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs) & (zs >= zmin) & (zs <= zmax)
        if not np.any(mask):
            return None
        pts = np.stack([xs[mask], ys[mask], zs[mask]], axis=1).astype(np.float64)
        if max_points > 0 and pts.shape[0] > max_points:
            step = int(pts.shape[0] / max_points) + 1
            pts = pts[::step]
        return pts
    except Exception:
        return None


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


def _pose_from_info_payload(payload_raw: str | None) -> np.ndarray | None:
    if not payload_raw:
        return None
    try:
        payload = json.loads(payload_raw)
        if payload.get("type") != "pos_info":
            return None
        cur = payload.get("data", {}).get("currentPose", {})
        qx = float(cur.get("q_x", 0.0))
        qy = float(cur.get("q_y", 0.0))
        qz = float(cur.get("q_z", 0.0))
        qw = float(cur.get("q_w", 1.0))
        x = float(cur.get("x", 0.0))
        y = float(cur.get("y", 0.0))
        z = float(cur.get("z", 0.0))
    except Exception:
        return None
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = _quat_to_mat(qx, qy, qz, qw)
    pose[:3, 3] = [x, y, z]
    return pose


class Viewer:
    def __init__(self) -> None:
        self._vis = o3d.visualization.Visualizer()
        ok = self._vis.create_window(window_name="SLAM Points Viewer", width=1280, height=720)
        if not ok:
            raise RuntimeError("Open3D failed to create window")

        render = self._vis.get_render_option()
        render.background_color = np.array([0.94, 0.94, 0.96], dtype=float)
        render.point_size = 2.0

        self._pcd = o3d.geometry.PointCloud()
        self._origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self._pose_frame: o3d.geometry.TriangleMesh | None = None
        self._vis.add_geometry(self._pcd)
        self._vis.add_geometry(self._origin)
        self._first = True

    def update(self, pts: np.ndarray | None, pose: np.ndarray | None) -> bool:
        if pts is not None:
            self._pcd.points = o3d.utility.Vector3dVector(pts)
            self._vis.update_geometry(self._pcd)

        if pose is not None:
            if self._pose_frame is not None:
                self._vis.remove_geometry(self._pose_frame, reset_bounding_box=False)
            self._pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            self._pose_frame.transform(pose)
            self._vis.add_geometry(self._pose_frame, reset_bounding_box=False)
            self._vis.update_geometry(self._pose_frame)

        if self._first and (pts is not None and len(pts) > 0):
            self._vis.reset_view_point(True)
            self._first = False

        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def close(self) -> None:
        self._vis.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View built-in SLAM DDS point clouds in Open3D.")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--slam-points-topic", default="rt/unitree/slam_mapping/points")
    parser.add_argument("--lidar-points-topic", default="rt/utlidar/cloud_livox_mid360")
    parser.add_argument("--slam-info-topic", default="rt/slam_info")
    parser.add_argument("--slam-key-topic", default="rt/slam_key_info")
    parser.add_argument("--points-stride", type=int, default=4)
    parser.add_argument("--points-z-min", type=float, default=-0.5)
    parser.add_argument("--points-z-max", type=float, default=1.5)
    parser.add_argument("--max-points", type=int, default=250000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ChannelFactoryInitialize(args.domain_id, args.iface)

    slam_pts_sub = PointCloudSubscriber(args.slam_points_topic)
    lidar_pts_sub = PointCloudSubscriber(args.lidar_points_topic)
    info_sub = SlamInfoSubscriber(args.slam_info_topic, args.slam_key_topic)
    odom_sub = SlamOdomSubscriber()

    slam_pts_sub.start()
    lidar_pts_sub.start()
    info_sub.start()
    odom_sub.start()

    viewer = Viewer()
    last_cloud_ts = 0.0
    latest_pts: np.ndarray | None = None

    try:
        while True:
            slam_msg, slam_ts = slam_pts_sub.get_latest()
            lidar_msg, lidar_ts = lidar_pts_sub.get_latest()

            use_msg = slam_msg
            use_ts = slam_ts
            source = "slam_points"
            if use_msg is None or use_ts <= 0.0:
                use_msg = lidar_msg
                use_ts = lidar_ts
                source = "lidar_points"

            if use_msg is not None and use_ts > last_cloud_ts:
                pts = _decode_points_xyz(
                    use_msg,
                    stride=args.points_stride,
                    zmin=args.points_z_min,
                    zmax=args.points_z_max,
                    max_points=args.max_points,
                )
                if pts is not None:
                    latest_pts = pts
                    last_cloud_ts = use_ts
                    print(f"[slam_points_viewer] source={source} points={len(pts)}")

            pose = _pose_from_info_payload(info_sub.get_info())
            if pose is None:
                odom_pose = odom_sub.get_pose()
                if odom_pose is not None:
                    x, y, yaw = odom_pose
                    pose = np.eye(4, dtype=float)
                    pose[:3, :3] = _quat_to_mat(0.0, 0.0, math.sin(
                        float(yaw) * 0.5), math.cos(float(yaw) * 0.5))
                    pose[:3, 3] = [float(x), float(y), 0.0]

            if not viewer.update(latest_pts, pose):
                break
            time.sleep(0.03)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
