#!/usr/bin/env python3
"""
live_slam_save.py
=================

Copy of ../slam/live_slam.py with auto-save of the displayed point cloud.
The saved map matches the Open3D visualization (after downsampling).
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any
import os

# Force CPU-only Open3D to avoid CUDA driver/runtime mismatch crashes.
os.environ.setdefault("OPEN3D_CPU_ONLY", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import open3d as o3d

# ---------------------------------------------------------------------------
# Mount orientation correction
# ---------------------------------------------------------------------------

_VALID_MOUNTS = {"normal", "upside_down"}

MOUNT = os.environ.get("LIVOX_MOUNT", "upside_down").lower()
if MOUNT not in _VALID_MOUNTS:
    raise SystemExit(f"LIVOX_MOUNT must be one of {_VALID_MOUNTS}")

import math  # standard

_TILT_AXIS = os.environ.get("LIDAR_TILT_AXIS", "y").lower()
if _TILT_AXIS not in {"x", "y", "z"}:
    raise SystemExit("LIDAR_TILT_AXIS must be one of 'x', 'y', 'z'")

try:
    _TILT_DEG = float(os.environ.get("LIDAR_TILT_DEG", "0"))
except ValueError:
    _TILT_DEG = 0.0

_R_MOUNT = None

_R_FLIP = np.diag([1.0, -1.0, -1.0, 1.0]) if MOUNT == "upside_down" else np.eye(4)

if abs(_TILT_DEG) > 1e-3:
    _rad = math.radians(-_TILT_DEG)
    c, s = math.cos(_rad), math.sin(_rad)

    if _TILT_AXIS == "x":
        _R_TILT = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, -s, 0.0],
                [0.0, s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    elif _TILT_AXIS == "y":
        _R_TILT = np.array(
            [
                [c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    else:
        _R_TILT = np.array(
            [
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
else:
    _R_TILT = np.eye(4)

_R_TOTAL = _R_TILT @ _R_FLIP
if not np.allclose(_R_TOTAL, np.eye(4)):
    _R_MOUNT = _R_TOTAL

# ---------------------------------------------------------------------------
# KISS-ICP import logic
# ---------------------------------------------------------------------------

KissICP = None  # type: ignore
_IMPORT_ERRORS = []
try:
    from kiss_icp.pipeline import KissICP  # type: ignore
except Exception as e:
    _IMPORT_ERRORS.append(e)

if KissICP is None:
    try:
        from kiss_icp.pybind import KissICP  # type: ignore
    except Exception as e:
        _IMPORT_ERRORS.append(e)

if KissICP is None:
    _msgs = " | ".join(str(e) for e in _IMPORT_ERRORS)
    raise SystemExit(
        "Could not import KISS-ICP (tried kiss_icp.pipeline & kiss_icp.pybind).\n"
        "Package is missing or broken.  Install/upgrade with:\n"
        "    pip install --upgrade 'kiss-icp'\n\nDetails: "
        + _msgs
    )

try:
    from livox2_python import Livox2 as _Livox
except Exception as e:
    raise SystemExit(
        "livox2_python unavailable. Install and build Livox-SDK2 first.\n"
        f"Details: {e}"
    )

# ---------------------------------------------------------------------------
# User-selectable presets (INDOOR / OUTDOOR)
# ---------------------------------------------------------------------------

PRESET = os.environ.get("LIVOX_PRESET", "indoor").lower()

_PRESETS: Dict[str, Dict[str, Any]] = {
    "indoor": {
        "frame_time": 0.35,
        "frame_packets": 200,
        "voxel_size": 0.4,
        "max_range": 30.0,
        "downsample_limit": 5_000_000,
        "min_motion": 0.03,
        "conv_criterion": 5e-5,
        "max_iters": 800,
    },
    "outdoor": {
        "frame_time": 0.20,
        "frame_packets": 120,
        "voxel_size": 1.0,
        "max_range": 120.0,
        "downsample_limit": 3_000_000,
        "min_motion": 0.10,
        "conv_criterion": 1e-4,
        "max_iters": 500,
    },
}

if PRESET not in _PRESETS:
    raise SystemExit(f"Unknown PRESET '{PRESET}'. Choose one of {_PRESETS.keys()}.")

_P = _PRESETS[PRESET]


def _mat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    t = np.trace(R)
    if t > 0.0:
        S = math.sqrt(t + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return (float(qx), float(qy), float(qz), float(qw))


class _Viewer:
    def __init__(self):
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Livox SLAM", width=1280, height=720)

        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)

        self._cam_frame: Optional[o3d.geometry.TriangleMesh] = None
        self._latest_pts: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None
        self._first = True

    def push(self, xyz: np.ndarray, pose: np.ndarray):
        self._latest_pts = xyz
        self._latest_pose = pose

    def tick(self) -> bool:
        updated = False

        if self._latest_pts is not None:
            self._pcd.points = o3d.utility.Vector3dVector(self._latest_pts)
            self._vis.update_geometry(self._pcd)
            self._latest_pts = None
            updated = True

        if self._latest_pose is not None:
            self._update_pose_vis(self._latest_pose)
            self._latest_pose = None
            updated = True

        if self._first and updated:
            self._vis.reset_view_point(True)
            self._first = False

        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def _update_pose_vis(self, pose: np.ndarray):
        if self._cam_frame is not None:
            self._vis.remove_geometry(self._cam_frame, reset_bounding_box=False)

        size = 0.5
        if len(self._pcd.points) > 0:
            bbox = self._pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_max_bound() - bbox.get_min_bound()
            size = float(np.linalg.norm(extent)) * 0.03
            size = max(0.2, min(size, 2.0))

        self._cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        self._cam_frame.transform(pose)
        self._vis.add_geometry(self._cam_frame, reset_bounding_box=False)
        self._vis.update_geometry(self._cam_frame)

    def close(self):
        self._vis.destroy_window()


class LiveSLAMDemo(_Livox):
    def __init__(self, save_dir: Optional[Path], save_every: int, save_latest: bool, save_prefix: str):
        _sdk_kwargs = {}
        if _Livox.__name__ == "Livox2":  # type: ignore[attr-defined]
            _sdk_kwargs.update(frame_time=_P["frame_time"], frame_packets=_P["frame_packets"])

        try:
            super().__init__("mid360_config.json", host_ip="192.168.123.222", **_sdk_kwargs)  # type: ignore[arg-type]
        except TypeError:
            super().__init__()

        try:
            from kiss_icp.config import load_config  # type: ignore

            cfg = load_config(config_file=None, max_range=_P["max_range"])
        except Exception as e:
            print("[KISS-ICP] Could not create config via load_config:", e)
            raise SystemExit(
                "Your installed kiss-icp wheel is too old – please upgrade: `pip install -U kiss-icp`. "
            ) from e

        try:
            cfg.mapping.voxel_size = _P["voxel_size"]
            cfg.mapping.max_points_per_voxel = 30
        except AttributeError:
            pass

        cfg.adaptive_threshold.min_motion_th = _P["min_motion"]
        cfg.registration.convergence_criterion = _P["conv_criterion"]
        cfg.registration.max_num_iterations = _P["max_iters"]

        self._slam = KissICP(cfg)
        self._viewer = _Viewer()
        self._vis_max_points = _P["downsample_limit"]

        self._save_dir = save_dir
        self._save_every = max(1, int(save_every))
        self._save_latest = bool(save_latest)
        self._save_prefix = save_prefix
        self._save_counter = 0

        if self._save_dir is not None:
            self._save_dir.mkdir(parents=True, exist_ok=True)

    def _save_cloud(self, cloud: np.ndarray, pose: np.ndarray) -> None:
        if self._save_dir is None:
            return
        self._save_counter += 1
        if self._save_counter % self._save_every != 0:
            return

        if self._save_latest:
            stem = f"{self._save_prefix}_latest"
        else:
            stem = f"{self._save_prefix}_{self._save_counter:06d}"

        pcd_path = self._save_dir / f"{stem}.pcd"
        json_path = self._save_dir / f"{stem}.json"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        try:
            o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False, compressed=False)
            R = pose[:3, :3]
            x, y, z = pose[:3, 3].tolist()
            qx, qy, qz, qw = _mat_to_quat(R)
            payload = {
                "pose": {
                    "x": x,
                    "y": y,
                    "z": z,
                    "q_x": qx,
                    "q_y": qy,
                    "q_z": qz,
                    "q_w": qw,
                },
                "matrix": pose.tolist(),
                "count": int(self._save_counter),
                "timestamp": time.time(),
            }
            json_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            print(f"[save] failed: {exc}")

    def handle_points(self, xyz: np.ndarray):
        import numpy as _np

        try:
            r_xy = float(os.environ.get("LIDAR_SELF_FILTER_RADIUS", 0.30))
            dz = float(os.environ.get("LIDAR_SELF_FILTER_Z", 0.24))
        except ValueError:
            r_xy, dz = 0.08, 0.05

        if xyz.size > 0:
            dist_xy = _np.linalg.norm(xyz[:, :2], axis=1)
            close = dist_xy < r_xy
            near_plane = _np.abs(xyz[:, 2]) < dz
            mask = ~(close & near_plane)
            if mask.sum() != xyz.shape[0]:
                xyz = xyz[mask]

        try:
            self._slam.register_frame(xyz)
        except TypeError:
            period = 1.0 / 20.0
            ts = _np.linspace(0.0, period, num=xyz.shape[0], dtype=_np.float64)
            self._slam.register_frame(xyz, ts)
        try:
            cloud = self._slam.get_map()
        except AttributeError:
            cloud = self._slam.local_map.point_cloud()

        if _R_MOUNT is not None:
            cloud = (cloud @ _R_MOUNT[:3, :3].T).astype(cloud.dtype, copy=False)

        if cloud.shape[0] > self._vis_max_points:
            step = int(cloud.shape[0] / self._vis_max_points) + 1
            cloud = cloud[::step]

        pose = self._slam.last_pose.copy()  # type: ignore[attr-defined]
        if _R_MOUNT is not None:
            pose = _R_MOUNT @ pose

        self._save_cloud(cloud, pose)
        self._viewer.push(cloud, pose)

    def shutdown(self):
        super().shutdown()
        self._viewer.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Live SLAM with auto-saving point cloud map")
    ap.add_argument("--save-dir", default="./maps", help="Directory to save maps (use empty to disable)")
    ap.add_argument("--save-every", type=int, default=1, help="Save every N updates")
    ap.add_argument("--save-latest", action="store_true", help="Overwrite a single latest file")
    ap.add_argument("--save-prefix", default="live_slam", help="Filename prefix for saved maps")
    args = ap.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else None
    demo = LiveSLAMDemo(save_dir, args.save_every, args.save_latest, args.save_prefix)

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
