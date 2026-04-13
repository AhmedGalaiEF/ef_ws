#!/usr/bin/env python3
"""
slam_all.py
===========

All-in-one SLAM utility UI for Unitree G1.

Features:
- Live SLAM occupancy map preview + dynamic lidar/slam points overlay.
- Start mapping / end mapping (save PCD on robot) / reset SLAM.
- Export current occupancy map to local .npz.
- Load and view a local .npz occupancy map.
- Click to define start/goal points on the map canvas.
- Navigate start -> goal using either:
  - SLAM pose navigation RPC (simple), or
  - Local A* waypoints + locomotion with optional obstacle avoidance.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
import tkinter as tk
import os
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Optional

import numpy as np

from create_map import OccupancyGrid, create_empty_map
from locomotion import Locomotion
from obstacle_detection import ObstacleDetector
from path_planner import astar, smooth_path, grid_path_to_world_waypoints
from slam_map import (
    SlamMapSubscriber,
    SlamOdomSubscriber,
    SlamPointCloudSubscriber,
    SlamInfoSubscriber,
    LidarSwitch,
)
from slam_service import SlamOperateClient, SlamResponse
from safety.hanger_boot_sequence import hanger_boot_sequence

# Force CPU-only Open3D to avoid CUDA driver/runtime mismatch crashes.
os.environ.setdefault("OPEN3D_CPU_ONLY", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import open3d as o3d  # type: ignore
except Exception:
    o3d = None  # type: ignore

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


@dataclass
class OverlayStats:
    slam_pts: int = 0
    lidar_pts: int = 0


class _Open3DViewer:
    """Open3D visualiser for live point-cloud + pose frame."""

    def __init__(self) -> None:
        if o3d is None:
            raise RuntimeError("open3d is not available")
        self._vis = o3d.visualization.Visualizer()
        ok = self._vis.create_window(window_name="SLAM Point Cloud", width=1280, height=720)
        if not ok:
            raise RuntimeError("Open3D failed to create window")

        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)

        self._pose_frame: Optional[o3d.geometry.TriangleMesh] = None
        self._latest_pts: Optional[np.ndarray] = None
        self._latest_pose: Optional[np.ndarray] = None
        self._first = True

    def push(self, xyz: np.ndarray, pose: Optional[np.ndarray]) -> None:
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

    def _update_pose_vis(self, pose: np.ndarray) -> None:
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

    def close(self) -> None:
        self._vis.destroy_window()


class SlamAllApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        self.root = tk.Tk()
        self.root.title("G1 SLAM All-In-One")
        self.root.geometry("1300x860")

        self.log_text: tk.Text | None = None

        self.status_var = tk.StringVar(value="Initializing...")
        self.map_var = tk.StringVar(value="Map: waiting")
        self.pose_var = tk.StringVar(value="Pose: waiting")
        self.sel_var = tk.StringVar(value="Start/Goal: not set")
        self.health_var = tk.StringVar(value="DDS: waiting")
        self.topic_var = tk.StringVar(value="Topics: waiting")
        self.overlay_var = tk.StringVar(value="Overlay: slam=0 lidar=0")
        self.task_var = tk.StringVar(value="SLAM tasks: 0")
        self.save_path_var = tk.StringVar(value=args.save_path)
        self.export_var = tk.StringVar(value=args.export_npz)
        self.load_var = tk.StringVar(value="")

        self.canvas_w = 980
        self.canvas_h = 620

        self.mapping_active = False
        self.latest_grid: OccupancyGrid | None = None
        self.latest_pose: tuple[float, float, float] | None = None
        self.start_xy: tuple[float, float] | None = None
        self.goal_xy: tuple[float, float] | None = None

        self.slam_sub: SlamMapSubscriber | None = None
        self.odom_sub: SlamOdomSubscriber | None = None
        self.slam_pts_sub: SlamPointCloudSubscriber | None = None
        self.lidar_pts_sub: SlamPointCloudSubscriber | None = None
        self.slam_info_sub: SlamInfoSubscriber | None = None
        self.slam_client: SlamOperateClient | None = None

        self.detector: ObstacleDetector | None = None
        self.walker: Locomotion | None = None
        self.nav_thread: threading.Thread | None = None
        self.nav_stop = threading.Event()
        self.slam_task_thread: threading.Thread | None = None
        self.slam_task_stop = threading.Event()
        self.slam_task_list: list[dict] = []
        self.slam_pose: dict | None = None
        self.slam_task_arrived = False
        self.slam_task_last_ts = 0.0

        self._build_ui()
        self._init_backend()

        self.o3d_viewer: _Open3DViewer | None = None
        if not self.args.no_open3d and o3d is not None:
            try:
                self.o3d_viewer = _Open3DViewer()
            except Exception:
                self.o3d_viewer = None
        if self.args.no_open3d:
            self._log("Open3D viewer disabled via --no-open3d")
        elif o3d is None:
            self._log("Open3D not available; point cloud viewer disabled")
        elif self.o3d_viewer is None:
            self._log("Open3D failed to initialize; point cloud viewer disabled")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(self.args.update_ms, self._periodic_update)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Status:").grid(row=0, column=0, sticky="w")
        ttk.Label(top, textvariable=self.status_var, width=115).grid(row=0, column=1, columnspan=7, sticky="w")

        ttk.Label(top, textvariable=self.map_var).grid(row=1, column=0, sticky="w")
        ttk.Label(top, textvariable=self.pose_var).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(top, textvariable=self.sel_var).grid(row=1, column=2, sticky="w", padx=6)
        ttk.Label(top, textvariable=self.overlay_var).grid(row=1, column=3, sticky="w", padx=6)
        ttk.Label(top, textvariable=self.health_var).grid(row=1, column=4, sticky="w", padx=6)
        ttk.Label(top, textvariable=self.task_var).grid(row=1, column=5, sticky="w", padx=6)
        ttk.Label(top, textvariable=self.topic_var).grid(row=2, column=0, columnspan=8, sticky="w")

        c1 = ttk.Frame(self.root, padding=8)
        c1.pack(fill=tk.X)
        ttk.Button(c1, text="Start Mapping", command=self._start_mapping).pack(side=tk.LEFT, padx=4)
        ttk.Button(c1, text="End Mapping + Save PCD", command=self._end_mapping).pack(side=tk.LEFT, padx=4)
        ttk.Button(c1, text="Reset SLAM", command=self._reset_slam).pack(side=tk.LEFT, padx=4)

        ttk.Label(c1, text="PCD path (robot):").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Entry(c1, textvariable=self.save_path_var, width=44).pack(side=tk.LEFT, padx=4)

        c2 = ttk.Frame(self.root, padding=8)
        c2.pack(fill=tk.X)
        ttk.Label(c2, text="Export map (.npz local):").pack(side=tk.LEFT)
        ttk.Entry(c2, textvariable=self.export_var, width=44).pack(side=tk.LEFT, padx=4)
        ttk.Button(c2, text="Export NPZ", command=self._export_map).pack(side=tk.LEFT, padx=4)
        ttk.Button(c2, text="Load NPZ", command=self._load_map).pack(side=tk.LEFT, padx=4)
        ttk.Entry(c2, textvariable=self.load_var, width=44).pack(side=tk.LEFT, padx=4)

        c3 = ttk.Frame(self.root, padding=8)
        c3.pack(fill=tk.X)
        self.use_local_nav = tk.BooleanVar(value=False)
        self.use_obstacle_avoid = tk.BooleanVar(value=True)
        self.goal_yaw_var = tk.StringVar(value="0.0")
        ttk.Checkbutton(c3, text="Use local navigation (A* + locomotion)", variable=self.use_local_nav).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(c3, text="Obstacle avoidance", variable=self.use_obstacle_avoid).pack(side=tk.LEFT, padx=4)
        ttk.Label(c3, text="Goal yaw (rad):").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(c3, textvariable=self.goal_yaw_var, width=8).pack(side=tk.LEFT)
        ttk.Button(c3, text="Navigate A -> B", command=self._navigate).pack(side=tk.LEFT, padx=10)
        ttk.Button(c3, text="Stop Navigation", command=self._stop_navigation).pack(side=tk.LEFT, padx=4)
        ttk.Button(c3, text="Add SLAM Goal", command=self._add_slam_goal).pack(side=tk.LEFT, padx=4)
        ttk.Button(c3, text="Run SLAM Tasks", command=self._run_slam_tasks).pack(side=tk.LEFT, padx=4)
        ttk.Button(c3, text="Clear SLAM Tasks", command=self._clear_slam_tasks).pack(side=tk.LEFT, padx=4)
        ttk.Button(c3, text="Clear Start/Goal", command=self._clear_selection).pack(side=tk.LEFT, padx=4)

        self.map_canvas = tk.Canvas(self.root, width=self.canvas_w, height=self.canvas_h, bg="#0f1116", highlightthickness=0)
        self.map_canvas.pack(padx=8, pady=(0, 8), fill=tk.BOTH, expand=False)
        self.map_canvas.bind("<Button-1>", self._on_canvas_click)
        self.map_canvas.bind("<Button-3>", lambda _e: self._clear_selection())

        logs = ttk.Frame(self.root, padding=8)
        logs.pack(fill=tk.BOTH, expand=True)
        ttk.Label(logs, text="Logs").pack(anchor="w")
        self.log_text = tk.Text(logs, height=12, wrap="word", bg="#0c0f13", fg="#d7e3ef")
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def _init_backend(self) -> None:
        self._init_channel_factory()

        self.slam_sub = SlamMapSubscriber(self.args.slam_map_topic)
        self.slam_sub.start()
        self.odom_sub = SlamOdomSubscriber(self.args.slam_odom_topic)
        self.odom_sub.start()
        self.slam_info_sub = SlamInfoSubscriber(self.args.slam_info_topic, self.args.slam_key_topic)
        self.slam_info_sub.start()

        if self.args.slam_points_topic:
            self.slam_pts_sub = SlamPointCloudSubscriber(self.args.slam_points_topic)
            self.slam_pts_sub.start()

        if self.args.lidar_topic:
            self.lidar_pts_sub = SlamPointCloudSubscriber(self.args.lidar_topic)
            self.lidar_pts_sub.start()

        if not self.args.no_lidar_switch:
            try:
                LidarSwitch().set("ON")
                self._log("Published rt/utlidar/switch = ON")
            except Exception as exc:
                self._log(f"WARN: failed to publish lidar switch ON: {exc}")

        self.slam_client = SlamOperateClient()
        self.slam_client.Init()
        self.slam_client.SetTimeout(5.0)
        self._log("SLAM service client initialized: slam_operate v1.0.0.1")
        self.status_var.set("Ready")

    def _init_channel_factory(self) -> None:
        iface = (self.args.iface or "").strip()
        if iface.lower() in {"", "auto", "autodetect", "none"}:
            ChannelFactoryInitialize(self.args.domain_id, None)
            self._log(f"DDS initialized (domain={self.args.domain_id}, iface=autodetect)")
            return
        try:
            ChannelFactoryInitialize(self.args.domain_id, iface)
            self._log(f"DDS initialized (domain={self.args.domain_id}, iface={iface})")
        except Exception as exc:
            self._log(f"WARN: DDS init with iface '{iface}' failed ({exc}); retry autodetect")
            ChannelFactoryInitialize(self.args.domain_id, None)
            self._log(f"DDS initialized (domain={self.args.domain_id}, iface=autodetect)")

    def _decode_resp(self, resp: SlamResponse) -> str:
        payload = self._parse_resp_payload(resp)
        if payload is None:
            return f"code={resp.code}, raw={resp.raw}"
        return (
            f"code={resp.code}, succeed={payload.get('succeed')}, "
            f"errorCode={payload.get('errorCode')}, info={payload.get('info', '')}"
        )

    def _parse_resp_payload(self, resp: SlamResponse) -> dict | None:
        raw = resp.raw
        if isinstance(raw, bytes):
            raw = raw.decode(errors="ignore")
        if not isinstance(raw, str):
            return None
        try:
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _resp_ok(self, resp: SlamResponse) -> bool:
        if resp.code != 0:
            return False
        payload = self._parse_resp_payload(resp)
        if payload is None:
            return True
        if payload.get("succeed") is False:
            return False
        err = payload.get("errorCode")
        return not isinstance(err, int) or err == 0

    def _start_mapping(self) -> None:
        if self.slam_client is None:
            return
        resp = self.slam_client.start_mapping("indoor")
        self._log(f"Start mapping (api 1801): {self._decode_resp(resp)}")
        if self._resp_ok(resp):
            self.mapping_active = True
            self.status_var.set("Mapping active")
        else:
            # Some firmware versions return non-zero codes even when mapping starts.
            # Keep the UI alive and show SLAM data if it arrives.
            self.mapping_active = True
            self.status_var.set("Mapping requested (service returned error)")

    def _end_mapping(self) -> None:
        if self.slam_client is None:
            return
        path = self.save_path_var.get().strip()
        if not path:
            self._log("ERROR: save path is empty")
            return
        resp = self.slam_client.end_mapping(path)
        self._log(f"End mapping + save (api 1802, {path}): {self._decode_resp(resp)}")
        if self._resp_ok(resp):
            self.mapping_active = False
            self.status_var.set("Mapping stopped (save requested)")
            self._log("NOTE: PCD path is on robot filesystem.")
            self._auto_export_latest_map()
        else:
            self.status_var.set("Save failed")

    def _reset_slam(self) -> None:
        if self.slam_client is None:
            return
        resp = self.slam_client.close_slam()
        self._log(f"Close SLAM (api 1901): {self._decode_resp(resp)}")
        if not self._resp_ok(resp):
            self.status_var.set("Reset failed")
            return
        self._clear_selection()
        self._start_mapping()

    def _export_map(self) -> None:
        if self.latest_grid is None:
            self._log("ERROR: no map available to export")
            return
        out_path = self.export_var.get().strip()
        if not out_path:
            self._log("ERROR: export path empty")
            return
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            self.latest_grid.save(out_path)
            self._log(f"Exported map to {out_path}")
        except Exception as exc:
            self._log(f"ERROR: export failed: {exc}")

    def _load_map_path(self, p: str) -> None:
        try:
            grid = OccupancyGrid.load(p)
            self.latest_grid = grid
            self._log(f"Loaded map {p} ({grid.width_cells}x{grid.height_cells} @ {grid.resolution:.3f}m)")
            self.status_var.set("Loaded map from file")
        except Exception as exc:
            self._log(f"ERROR: failed to load map: {exc}")

    def _load_map(self) -> None:
        p = self.load_var.get().strip()
        if not p:
            p = filedialog.askopenfilename(
                title="Select map (.npz)",
                filetypes=[("NumPy map", "*.npz"), ("All files", "*")],
            )
            if not p:
                return
            self.load_var.set(p)
        self._load_map_path(p)

    def _auto_export_latest_map(self) -> None:
        if self.latest_grid is None:
            if self.slam_sub is not None:
                grid, _meta = self.slam_sub.to_occupancy(
                    height_threshold=self.args.slam_height_threshold,
                    max_height=self.args.slam_max_height,
                    origin_centered=self.args.slam_origin_centered,
                )
                if grid is not None:
                    self.latest_grid = grid
        if self.latest_grid is None:
            self._log("WARN: no SLAM map available to auto-export")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(self.args.export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"slam_all_{ts}.npz"
        try:
            self.latest_grid.save(str(out_path))
            self.export_var.set(str(out_path))
            self.load_var.set(str(out_path))
            self._log(f"Auto-exported map to {out_path}")
            self._load_map_path(str(out_path))
        except Exception as exc:
            self._log(f"ERROR: auto-export failed: {exc}")

    def _clear_selection(self) -> None:
        self.start_xy = None
        self.goal_xy = None
        self.sel_var.set("Start/Goal: not set")

    def _on_canvas_click(self, event: tk.Event) -> None:
        if self.latest_grid is None:
            return
        wx, wy = self._pixel_to_world(event.x, event.y, self.latest_grid)
        if self.start_xy is None:
            self.start_xy = (wx, wy)
        elif self.goal_xy is None:
            self.goal_xy = (wx, wy)
        else:
            self.start_xy = (wx, wy)
            self.goal_xy = None
        self._update_selection_label()

    def _update_selection_label(self) -> None:
        s = f"start={self.start_xy}" if self.start_xy else "start=unset"
        g = f"goal={self.goal_xy}" if self.goal_xy else "goal=unset"
        self.sel_var.set(f"Start/Goal: {s}, {g}")

    def _add_slam_goal(self) -> None:
        if self.goal_xy is None:
            self._log("ERROR: set GOAL first (left click map)")
            return
        try:
            goal_yaw = float(self.goal_yaw_var.get().strip())
        except Exception:
            goal_yaw = 0.0
        half = goal_yaw / 2.0
        qx, qy, qz, qw = 0.0, 0.0, math.sin(half), math.cos(half)
        gx, gy = self.goal_xy
        self.slam_task_list.append(
            {
                "x": float(gx),
                "y": float(gy),
                "z": 0.0,
                "q_x": qx,
                "q_y": qy,
                "q_z": qz,
                "q_w": qw,
            }
        )
        self.task_var.set(f"SLAM tasks: {len(self.slam_task_list)}")
        self._log(f"Added SLAM goal ({gx:+.2f}, {gy:+.2f}, yaw={goal_yaw:+.2f})")

    def _clear_slam_tasks(self) -> None:
        self.slam_task_stop.set()
        self.slam_task_list.clear()
        self.task_var.set("SLAM tasks: 0")
        self._log("Cleared SLAM task list")

    def _run_slam_tasks(self) -> None:
        if self.slam_client is None:
            self._log("ERROR: slam client unavailable")
            return
        if not self.slam_task_list:
            self._log("ERROR: no SLAM tasks queued (use 'Add SLAM Goal')")
            return
        if self.slam_task_thread is not None and self.slam_task_thread.is_alive():
            self._log("WARN: SLAM task runner already active")
            return
        self.slam_task_stop.clear()
        self.slam_task_thread = threading.Thread(target=self._slam_task_worker, daemon=True)
        self.slam_task_thread.start()

    def _slam_task_worker(self) -> None:
        for idx, pose in enumerate(list(self.slam_task_list)):
            if self.slam_task_stop.is_set():
                self._log("SLAM task runner stopped")
                return
            self.slam_task_arrived = False
            self.slam_task_last_ts = time.time()
            resp = self.slam_client.pose_nav(
                pose["x"],
                pose["y"],
                pose["z"],
                pose["q_x"],
                pose["q_y"],
                pose["q_z"],
                pose["q_w"],
                mode=1,
            )
            self._log(f"SLAM task {idx+1}/{len(self.slam_task_list)} pose_nav: {self._decode_resp(resp)}")
            if not self._resp_ok(resp):
                self._log("ERROR: SLAM pose_nav rejected, aborting task list")
                return
            t0 = time.time()
            while time.time() - t0 < self.args.slam_nav_timeout:
                if self.slam_task_stop.is_set():
                    self._log("SLAM task runner stopped")
                    return
                if self.slam_task_arrived:
                    break
                time.sleep(0.1)
            if not self.slam_task_arrived:
                self._log("WARN: SLAM task timeout")
                return
        self._log("SLAM task list complete")

    def _ensure_local_nav_ready(self) -> bool:
        if self.walker is not None and self.detector is not None:
            return True
        try:
            loco = hanger_boot_sequence(iface=self.args.iface)
            detector = ObstacleDetector(topic=self.args.sport_topic)
            detector.start()
            time.sleep(0.6)
            if detector.is_stale():
                self._log("ERROR: no SportModeState data for local navigation")
                return False
            self.detector = detector
            self.walker = Locomotion(loco, detector, max_vx=self.args.max_speed)
            self._log("Local navigation initialized (LocoClient + obstacle detector)")
            return True
        except Exception as exc:
            self._log(f"ERROR: local navigation init failed: {exc}")
            return False

    def _navigate(self) -> None:
        if self.nav_thread is not None and self.nav_thread.is_alive():
            self._log("WARN: navigation already running")
            return
        if self.goal_xy is None:
            self._log("ERROR: set GOAL first (left click map)")
            return
        if self.latest_grid is None:
            self._log("ERROR: no map available")
            return

        self.nav_stop.clear()
        self.nav_thread = threading.Thread(target=self._nav_worker, daemon=True)
        self.nav_thread.start()

    def _stop_navigation(self) -> None:
        self.nav_stop.set()
        if self.walker is not None:
            try:
                self.walker.stop()
            except Exception:
                pass
        self.status_var.set("Navigation stop requested")

    def _nav_worker(self) -> None:
        try:
            self.status_var.set("Navigating...")
            goal_x, goal_y = self.goal_xy if self.goal_xy is not None else (None, None)
            if goal_x is None or goal_y is None:
                self.status_var.set("Navigation aborted")
                return

            goal_yaw = 0.0
            try:
                goal_yaw = float(self.goal_yaw_var.get().strip())
            except Exception:
                goal_yaw = 0.0

            if not self.use_local_nav.get():
                self._nav_worker_slam(goal_x, goal_y, goal_yaw)
                return

            if not self._ensure_local_nav_ready():
                self.status_var.set("Local nav init failed")
                return

            assert self.walker is not None
            assert self.detector is not None
            grid = self.latest_grid if self.latest_grid is not None else create_empty_map()

            sx, sy, syaw = self.detector.get_pose()
            if self.start_xy is not None and math.hypot(self.start_xy[0] - sx, self.start_xy[1] - sy) < 0.6:
                sx, sy = self.start_xy

            inflated = grid.inflate(radius_cells=self.args.inflation)
            start_cell = grid.world_to_grid(sx, sy)
            goal_cell = grid.world_to_grid(goal_x, goal_y)
            path = astar(inflated, start_cell, goal_cell)
            if path is None:
                self._log(f"ERROR: no path from {start_cell} to {goal_cell}")
                self.status_var.set("No path")
                return
            smoothed = smooth_path(path, inflated)
            waypoints = grid_path_to_world_waypoints(smoothed, grid, spacing_m=self.args.spacing)
            self._log(
                f"Local nav plan: {len(path)} raw -> {len(smoothed)} smooth -> {len(waypoints)} waypoints"
            )

            for i, (wx, wy) in enumerate(waypoints):
                if self.nav_stop.is_set():
                    self.status_var.set("Navigation stopped")
                    return
                is_last = i == len(waypoints) - 1

                def _check_obs() -> bool:
                    if self.nav_stop.is_set():
                        return True
                    if not self.use_obstacle_avoid.get():
                        return False
                    if self.detector is None:
                        return False
                    return self.detector.front_blocked()

                reached = self.walker.walk_to(
                    wx,
                    wy,
                    final_yaw=goal_yaw if is_last else None,
                    timeout=30.0,
                    check_obstacle=_check_obs,
                )
                if not reached:
                    self._log(f"Waypoint {i+1}/{len(waypoints)} aborted (obstacle/timeout)")
                    self.status_var.set("Navigation aborted")
                    return

            self.status_var.set("Navigation complete")
            self._log("Navigation complete")

        except Exception as exc:
            self.status_var.set("Navigation error")
            self._log(f"ERROR in navigation: {exc}")

    def _nav_worker_slam(self, goal_x: float, goal_y: float, goal_yaw: float) -> None:
        if self.slam_client is None:
            self._log("ERROR: slam client unavailable")
            self.status_var.set("Navigation failed")
            return

        half = goal_yaw / 2.0
        qx, qy, qz, qw = 0.0, 0.0, math.sin(half), math.cos(half)
        resp = self.slam_client.pose_nav(goal_x, goal_y, 0.0, qx, qy, qz, qw, mode=1)
        self._log(f"Pose nav (api 1102): {self._decode_resp(resp)}")
        if not self._resp_ok(resp):
            self.status_var.set("SLAM pose nav rejected")
            return

        t0 = time.time()
        while time.time() - t0 < self.args.slam_nav_timeout:
            if self.nav_stop.is_set():
                self.status_var.set("Navigation stopped")
                return
            if self.odom_sub is not None and not self.odom_sub.is_stale(1.0):
                x, y, _ = self.odom_sub.get_pose()
                if math.hypot(x - goal_x, y - goal_y) < self.args.goal_tolerance:
                    self.status_var.set("Navigation complete")
                    self._log("SLAM pose navigation reached goal")
                    return
            time.sleep(0.2)

        self.status_var.set("SLAM nav timeout")
        self._log("WARN: SLAM pose navigation timeout")

    def _decode_points(self, sub: SlamPointCloudSubscriber | None, stride: int, zmin: float, zmax: float) -> list[tuple[float, float]]:
        if sub is None:
            return []
        msg, _ts = sub.get_latest()
        if msg is None:
            return []
        try:
            fields = {f.name: f for f in msg.fields}
            if "x" not in fields or "y" not in fields:
                return []
            point_step = int(msg.point_step)
            if point_step <= 0:
                return []
            data = bytes(msg.data)
            if not data:
                return []
            xoff = int(fields["x"].offset)
            yoff = int(fields["y"].offset)
            zoff = int(fields["z"].offset) if "z" in fields else xoff + 8
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
            pts: list[tuple[float, float]] = []
            for x, y, z in zip(xs, ys, zs):
                if z < zmin or z > zmax:
                    continue
                pts.append((float(x), float(y)))
            return pts
        except Exception:
            return []

    def _decode_points_xyz(self, sub: SlamPointCloudSubscriber | None, stride: int, zmin: float, zmax: float) -> Optional[np.ndarray]:
        if sub is None:
            return None
        msg, _ts = sub.get_latest()
        if msg is None:
            return None
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
            return pts
        except Exception:
            return None

    def _periodic_update(self) -> None:
        try:
            if self.slam_sub is not None:
                grid, _meta = self.slam_sub.to_occupancy(
                    height_threshold=self.args.slam_height_threshold,
                    max_height=self.args.slam_max_height,
                    origin_centered=self.args.slam_origin_centered,
                )
                if grid is not None:
                    self.latest_grid = grid
                    obs = int(np.count_nonzero(grid.grid))
                    total = grid.grid.size
                    self.map_var.set(
                        f"Map: {grid.width_cells}x{grid.height_cells}, res={grid.resolution:.3f}m, obs={100.0*obs/max(1,total):.1f}%"
                    )

            if self.odom_sub is not None and not self.odom_sub.is_stale(1.2):
                pose = self.odom_sub.get_pose()
                self.latest_pose = pose
                self.pose_var.set(f"Pose: x={pose[0]:+.2f}, y={pose[1]:+.2f}, yaw={math.degrees(pose[2]):+.1f}deg")
            else:
                self.pose_var.set("Pose: stale/no data")
            slam_pts = self._decode_points(
                self.slam_pts_sub,
                stride=self.args.slam_points_stride,
                zmin=self.args.points_z_min,
                zmax=self.args.points_z_max,
            )
            lidar_pts = self._decode_points(
                self.lidar_pts_sub,
                stride=self.args.lidar_stride,
                zmin=self.args.points_z_min,
                zmax=self.args.points_z_max,
            )
            self.overlay_var.set(f"Overlay: slam={len(slam_pts)} lidar={len(lidar_pts)}")

            if self.slam_info_sub is not None:
                info = self.slam_info_sub.get_info()
                if info:
                    try:
                        payload = json.loads(info)
                        if payload.get("type") == "pos_info":
                            cur = payload.get("data", {}).get("currentPose", {})
                            self.slam_pose = {
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

                key = self.slam_info_sub.get_key()
                if key:
                    try:
                        payload = json.loads(key)
                        if payload.get("type") == "task_result":
                            arrived = bool(payload.get("data", {}).get("is_arrived", False))
                            if arrived and (time.time() - self.slam_task_last_ts) < self.args.slam_nav_timeout:
                                self.slam_task_arrived = True
                    except Exception:
                        pass

            if self.o3d_viewer is not None:
                slam_xyz = self._decode_points_xyz(
                    self.slam_pts_sub,
                    stride=self.args.slam_points_stride,
                    zmin=self.args.points_z_min,
                    zmax=self.args.points_z_max,
                )
                lidar_xyz = None
                if slam_xyz is None:
                    lidar_xyz = self._decode_points_xyz(
                        self.lidar_pts_sub,
                        stride=self.args.lidar_stride,
                        zmin=self.args.points_z_min,
                        zmax=self.args.points_z_max,
                    )
                cloud = slam_xyz if slam_xyz is not None else lidar_xyz
                pose_mat = None
                if self.latest_pose is not None:
                    x, y, yaw = self.latest_pose
                    c = math.cos(yaw)
                    s = math.sin(yaw)
                    pose_mat = np.array(
                        [
                            [c, -s, 0.0, x],
                            [s, c, 0.0, y],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        dtype=float,
                    )
                if cloud is not None:
                    self.o3d_viewer.push(cloud, pose_mat)
                if not self.o3d_viewer.tick():
                    self.o3d_viewer.close()
                    self.o3d_viewer = None

            stale_bits = []
            if self.slam_sub is not None and self.slam_sub.is_stale(1.5):
                stale_bits.append("map stale")
            if self.odom_sub is not None and self.odom_sub.is_stale(1.5):
                stale_bits.append("odom stale")
            self.health_var.set("DDS: OK" if not stale_bits else "DDS: " + ", ".join(stale_bits))

            def _topic_status(sub: SlamPointCloudSubscriber | None, label: str) -> str:
                if sub is None:
                    return f"{label}=off"
                msg, ts = sub.get_latest()
                if msg is None or ts == 0.0:
                    return f"{label}=empty"
                age = time.time() - ts
                if age > 1.5:
                    return f"{label}=stale"
                return f"{label}=ok"

            def _topic_status_map(sub: SlamMapSubscriber | None, label: str) -> str:
                if sub is None:
                    return f"{label}=off"
                _msg, ts = sub.get_latest()
                if ts == 0.0:
                    return f"{label}=empty"
                age = time.time() - ts
                if age > 1.5:
                    return f"{label}=stale"
                return f"{label}=ok"

            def _topic_status_odom(sub: SlamOdomSubscriber | None, label: str) -> str:
                if sub is None:
                    return f"{label}=off"
                if sub.is_stale(1.5):
                    return f"{label}=stale"
                return f"{label}=ok"

            topic_bits = [
                _topic_status_map(self.slam_sub, "map"),
                _topic_status_odom(self.odom_sub, "odom"),
                _topic_status(self.slam_pts_sub, "slam_pts"),
                _topic_status(self.lidar_pts_sub, "lidar_pts"),
            ]
            self.topic_var.set("Topics: " + ", ".join(topic_bits))

            if self.latest_grid is not None:
                self._draw(self.latest_grid, self.latest_pose, slam_pts, lidar_pts)

        except Exception as exc:
            self._log(f"ERROR in periodic update: {exc}")

        self.root.after(self.args.update_ms, self._periodic_update)

    def _world_to_pixel(self, grid: OccupancyGrid, wx: float, wy: float) -> tuple[int, int]:
        row, col = grid.world_to_grid(wx, wy)
        px = int(col * self.canvas_w / max(1, grid.width_cells))
        py = int((grid.height_cells - 1 - row) * self.canvas_h / max(1, grid.height_cells))
        return px, py

    def _pixel_to_world(self, px: int, py: int, grid: OccupancyGrid) -> tuple[float, float]:
        col = int(px * grid.width_cells / max(1, self.canvas_w))
        row = int((grid.height_cells - 1) - (py * grid.height_cells / max(1, self.canvas_h)))
        row = max(0, min(grid.height_cells - 1, row))
        col = max(0, min(grid.width_cells - 1, col))
        return grid.grid_to_world(row, col)

    def _draw(self, grid: OccupancyGrid, pose: tuple[float, float, float] | None, slam_pts: list[tuple[float, float]], lidar_pts: list[tuple[float, float]]) -> None:
        self.map_canvas.delete("all")

        g = grid.grid
        h, w = g.shape
        if h <= 0 or w <= 0:
            return

        cell_w = self.canvas_w / float(w)
        cell_h = self.canvas_h / float(h)

        stride = max(1, int(math.ceil(max(w, h) / 250.0)))
        obs = np.argwhere(g > 0)
        if stride > 1:
            obs = obs[::stride]

        for row, col in obs:
            x0 = col * cell_w
            y0 = (h - 1 - row) * cell_h
            x1 = x0 + max(1.0, cell_w * stride)
            y1 = y0 + max(1.0, cell_h * stride)
            self.map_canvas.create_rectangle(x0, y0, x1, y1, fill="#3f79ff", outline="")

        for px, py in slam_pts:
            sx, sy = self._world_to_pixel(grid, px, py)
            self.map_canvas.create_oval(sx - 1, sy - 1, sx + 1, sy + 1, fill="#ff7f2a", outline="")

        if pose is not None:
            x, y, yaw = pose
            rx, ry = self._world_to_pixel(grid, x, y)
            rr = 5
            self.map_canvas.create_oval(rx - rr, ry - rr, rx + rr, ry + rr, fill="#00ff99", outline="")
            hx = rx + 18 * math.cos(yaw)
            hy = ry - 18 * math.sin(yaw)
            self.map_canvas.create_line(rx, ry, hx, hy, fill="#00ff99", width=2)

            if not slam_pts and lidar_pts:
                cy = math.cos(yaw)
                sy = math.sin(yaw)
                for lx, ly in lidar_pts:
                    wx = x + (lx * cy - ly * sy)
                    wy = y + (lx * sy + ly * cy)
                    px2, py2 = self._world_to_pixel(grid, wx, wy)
                    self.map_canvas.create_oval(px2 - 1, py2 - 1, px2 + 1, py2 + 1, fill="#f2c230", outline="")

        if self.start_xy is not None:
            sx, sy = self._world_to_pixel(grid, self.start_xy[0], self.start_xy[1])
            self.map_canvas.create_oval(sx - 6, sy - 6, sx + 6, sy + 6, outline="#00e5ff", width=2)
            self.map_canvas.create_text(sx + 14, sy - 10, text="START", fill="#00e5ff", anchor="w")
        if self.goal_xy is not None:
            gx, gy = self._world_to_pixel(grid, self.goal_xy[0], self.goal_xy[1])
            self.map_canvas.create_rectangle(gx - 6, gy - 6, gx + 6, gy + 6, outline="#ff5a5a", width=2)
            self.map_canvas.create_text(gx + 14, gy - 10, text="GOAL", fill="#ff5a5a", anchor="w")

    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}\n"
        if self.log_text is None:
            print(line, end="")
            return
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        try:
            self.nav_stop.set()
            if self.walker is not None:
                self.walker.stop()
        except Exception:
            pass
        if self.o3d_viewer is not None:
            try:
                self.o3d_viewer.close()
            except Exception:
                pass
        self.root.destroy()

    def run(self) -> None:
        self._log(
            f"Started with iface={self.args.iface}, domain_id={self.args.domain_id}, "
            f"save_path={self.args.save_path}"
        )
        self._log("Left click map: set START then GOAL. Right click: clear start/goal.")
        self._log("NOTE: PCD save path is on robot storage.")
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="All-in-one SLAM map + navigation utility for G1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iface", default="eth0", help="Network interface for DDS")
    p.add_argument("--domain-id", type=int, default=0, help="DDS domain id")
    p.add_argument("--sport-topic", default="rt/odommodestate", help="SportModeState topic")

    p.add_argument("--slam-map-topic", default="rt/utlidar/map_state", help="SLAM map topic")
    p.add_argument("--slam-odom-topic", default="rt/unitree/slam_mapping/odom", help="SLAM odom topic")
    p.add_argument("--slam-points-topic", default="rt/unitree/slam_mapping/points", help="SLAM points topic")
    p.add_argument("--lidar-topic", default="rt/utlidar/cloud_livox_mid360", help="Lidar points topic")
    p.add_argument("--slam-info-topic", default="rt/slam_info", help="SLAM info topic")
    p.add_argument("--slam-key-topic", default="rt/slam_key_info", help="SLAM key info topic")

    p.add_argument("--save-path", default="/home/unitree/test1.pcd", help="PCD save path on robot")
    p.add_argument("--export-npz", default="/tmp/slam_all_map.npz", help="Export path for local NPZ map")
    p.add_argument("--export-dir", default="./maps", help="Directory for auto-exported NPZ maps")

    p.add_argument("--slam-height-threshold", type=float, default=0.15, help="HeightMap obstacle threshold")
    p.add_argument("--slam-max-height", type=float, default=None, help="Optional max height clamp")
    p.add_argument("--slam-origin-centered", action="store_true", help="Center map around (0,0)")

    p.add_argument("--slam-points-stride", type=int, default=6, help="SLAM points stride")
    p.add_argument("--lidar-stride", type=int, default=6, help="Lidar points stride")
    p.add_argument("--points-z-min", type=float, default=-0.5, help="Min z for points")
    p.add_argument("--points-z-max", type=float, default=1.5, help="Max z for points")
    p.add_argument("--no-open3d", action="store_true", help="Disable Open3D point cloud viewer")

    p.add_argument("--inflation", type=int, default=3, help="Local planning inflation radius (cells)")
    p.add_argument("--spacing", type=float, default=0.5, help="Local planning waypoint spacing (m)")
    p.add_argument("--max-speed", type=float, default=0.25, help="Local navigation max vx")
    p.add_argument("--goal-tolerance", type=float, default=0.35, help="SLAM pose-nav completion tolerance (m)")
    p.add_argument("--slam-nav-timeout", type=float, default=60.0, help="SLAM pose-nav timeout (s)")

    p.add_argument("--update-ms", type=int, default=300, help="UI update period (ms)")
    p.add_argument("--no-lidar-switch", action="store_true", help="Do not publish rt/utlidar/switch ON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = SlamAllApp(args)
    app.run()


if __name__ == "__main__":
    main()
