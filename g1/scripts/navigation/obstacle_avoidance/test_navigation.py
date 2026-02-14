#!/usr/bin/env python3
"""
test_navigation.py
==================

Minimal navigation test for G1 using the SLAM occupancy grid + A*.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc

from create_map import OccupancyGrid
from path_planner import astar, smooth_path, grid_path_to_world_waypoints
from obstacle_detection import ObstacleDetector
from locomotion import Locomotion
from slam_map import SlamMapSubscriber, SlamInfoSubscriber
from slam_service import SlamOperateClient, SlamResponse
from safety.hanger_boot_sequence import hanger_boot_sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simple SLAM map navigation test (A* + locomotion)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--nav-json", default="navigation.json", help="Path to navigation JSON (start/goal)")
    p.add_argument("--iface", default="eth0", help="Network interface for DDS")
    p.add_argument("--load-map", default="", help="Path to saved map (.pcd/.ply/.npz)")
    p.add_argument("--map-resolution", type=float, default=0.1, help="Resolution for PCD/PLY map (m)")
    p.add_argument("--map-padding", type=float, default=0.5, help="Padding around PCD/PLY bounds (m)")
    p.add_argument("--map-origin-centered", action="store_true", help="Center PCD/PLY map around (0,0)")
    p.add_argument("--domain-id", type=int, default=0, help="DDS domain id")
    p.add_argument("--sport-topic", default="rt/odommodestate", help="SportModeState topic")
    p.add_argument("--slam-map-topic", default="rt/utlidar/map_state", help="SLAM map topic")
    p.add_argument("--slam-info-topic", default="rt/slam_info", help="SLAM info topic")
    p.add_argument("--slam-key-topic", default="rt/slam_key_info", help="SLAM key info topic")
    p.add_argument("--slam-height-threshold", type=float, default=0.15, help="HeightMap obstacle threshold")
    p.add_argument("--slam-max-height", type=float, default=None, help="Optional max height clamp")
    p.add_argument("--slam-origin-centered", action="store_true", help="Center map around (0,0)")
    p.add_argument("--slam-timeout", type=float, default=6.0, help="Seconds to wait for SLAM map")
    p.add_argument("--goal-x", type=float, default=None, help="Goal X (metres)")
    p.add_argument("--goal-y", type=float, default=None, help="Goal Y (metres)")
    p.add_argument("--goal-yaw", type=float, default=0.0, help="Final yaw (radians)")
    p.add_argument("--inflation", type=int, default=3, help="Inflation radius (cells)")
    p.add_argument("--spacing", type=float, default=0.5, help="Waypoint spacing (m)")
    p.add_argument("--max-speed", type=float, default=0.25, help="Max forward speed (m/s)")
    p.add_argument("--with-avoidance", action="store_true", help="Enable obstacle avoidance checks")
    p.add_argument("--use-slam-nav", action="store_true", help="Use slam_operate pose navigation")
    return p.parse_args()


def _pose_from_dict(data: dict) -> Optional[tuple[float, float, float]]:
    try:
        x = float(data.get("x", 0.0))
        y = float(data.get("y", 0.0))
    except Exception:
        return None
    if "yaw_deg" in data:
        try:
            yaw = math.radians(float(data["yaw_deg"]))
            return (x, y, yaw)
        except Exception:
            return None
    if "yaw" in data:
        try:
            yaw = float(data["yaw"])
            return (x, y, yaw)
        except Exception:
            return None
    if {"q_x", "q_y", "q_z", "q_w"}.issubset(data.keys()):
        try:
            qx = float(data["q_x"])
            qy = float(data["q_y"])
            qz = float(data["q_z"])
            qw = float(data["q_w"])
        except Exception:
            return None
        # Yaw from quaternion (Z axis)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return (x, y, yaw)
    return (x, y, 0.0)


def _load_nav_json(path: str) -> tuple[Optional[tuple[float, float, float]], Optional[tuple[float, float, float]]]:
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return (None, None)
    start = None
    goal = None
    for key in ("start_pose", "start"):
        if key in payload and isinstance(payload[key], dict):
            start = _pose_from_dict(payload[key])
            break
    for key in ("goal_pose", "goal", "end_pose", "end"):
        if key in payload and isinstance(payload[key], dict):
            goal = _pose_from_dict(payload[key])
            break
    return (start, goal)


def _load_occ_map(
    map_path: str,
    resolution: float,
    padding_m: float,
    height_threshold: float,
    max_height: Optional[float],
    origin_centered: bool,
) -> OccupancyGrid:
    suffix = Path(map_path).suffix.lower()
    if suffix == ".npz":
        return OccupancyGrid.load(map_path)

    if suffix not in {".pcd", ".ply"}:
        raise SystemExit(f"Unsupported map format: {map_path}")

    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        raise SystemExit("open3d is required to load PCD/PLY maps.") from exc

    cloud = o3d.io.read_point_cloud(map_path)
    if len(cloud.points) == 0:
        raise SystemExit(f"Loaded point cloud is empty: {map_path}")

    pts = np.asarray(cloud.points)
    if pts.shape[1] != 3:
        raise SystemExit(f"Unexpected point cloud shape: {pts.shape}")

    if max_height is not None:
        pts = pts[pts[:, 2] <= max_height]
        if pts.size == 0:
            raise SystemExit("All points filtered by max_height.")

    min_xy = pts[:, :2].min(axis=0) - padding_m
    max_xy = pts[:, :2].max(axis=0) + padding_m
    width_m = float(max_xy[0] - min_xy[0])
    height_m = float(max_xy[1] - min_xy[1])
    if width_m <= 0 or height_m <= 0 or resolution <= 0:
        raise SystemExit("Invalid map bounds/resolution.")

    if origin_centered:
        origin_x = -width_m / 2.0
        origin_y = -height_m / 2.0
    else:
        origin_x = float(min_xy[0])
        origin_y = float(min_xy[1])

    occ_grid = OccupancyGrid(width_m, height_m, resolution, origin_x, origin_y)
    grid = np.zeros((occ_grid.height_cells, occ_grid.width_cells), dtype=np.int8)

    # Mark occupied cells for points above the threshold.
    if height_threshold is not None:
        pts = pts[pts[:, 2] >= height_threshold]
    if pts.size > 0:
        gx = ((pts[:, 0] - origin_x) / resolution).astype(int)
        gy = ((pts[:, 1] - origin_y) / resolution).astype(int)
        valid = (
            (gx >= 0) & (gx < occ_grid.width_cells) &
            (gy >= 0) & (gy < occ_grid.height_cells)
        )
        gx = gx[valid]
        gy = gy[valid]
        grid[gy, gx] = 1

    occ_grid.grid = grid
    return occ_grid


def _wait_for_slam_map(
    sub: SlamMapSubscriber,
    timeout: float,
    height_threshold: float,
    max_height: Optional[float],
    origin_centered: bool,
) -> Optional[OccupancyGrid]:
    t0 = time.time()
    while time.time() - t0 < timeout:
        grid, _meta = sub.to_occupancy(
            height_threshold=height_threshold,
            max_height=max_height,
            origin_centered=origin_centered,
        )
        if grid is not None:
            return grid
        time.sleep(0.05)
    return None


def main() -> None:
    args = parse_args()

    ChannelFactoryInitialize(args.domain_id, args.iface)

    slam_sub: Optional[SlamMapSubscriber] = None
    if not args.load_map:
        slam_sub = SlamMapSubscriber(args.slam_map_topic)
        slam_sub.start()
    slam_info_sub = SlamInfoSubscriber(args.slam_info_topic, args.slam_key_topic)
    slam_info_sub.start()

    detector = ObstacleDetector(topic=args.sport_topic)
    detector.start()
    time.sleep(0.6)
    if detector.is_stale():
        raise SystemExit("No SportModeState data; check DDS/iface.")

    if args.load_map:
        grid = _load_occ_map(
            args.load_map,
            resolution=args.map_resolution,
            padding_m=args.map_padding,
            height_threshold=args.slam_height_threshold,
            max_height=args.slam_max_height,
            origin_centered=args.map_origin_centered,
        )
    else:
        if slam_sub is None:
            raise SystemExit("SLAM subscriber not initialized.")
        grid = _wait_for_slam_map(
            slam_sub,
            timeout=args.slam_timeout,
            height_threshold=args.slam_height_threshold,
            max_height=args.slam_max_height,
            origin_centered=args.slam_origin_centered,
        )
        if grid is None and not args.use_slam_nav:
            raise SystemExit("Timed out waiting for SLAM map.")

    nav_start, nav_goal = _load_nav_json(args.nav_json)

    goal_x = args.goal_x
    goal_y = args.goal_y
    goal_yaw = args.goal_yaw

    if nav_goal is not None and (goal_x is None or goal_y is None):
        goal_x, goal_y, goal_yaw = nav_goal
    if goal_x is None or goal_y is None:
        info = slam_info_sub.get_info()
        if info:
            try:
                payload = json.loads(info)
                if payload.get("type") == "ctrl_info":
                    target = payload.get("data", {}).get("targetPose", {})
                    goal_x = float(target.get("x", 0.0))
                    goal_y = float(target.get("y", 0.0))
            except Exception:
                pass
    if goal_x is None or goal_y is None:
        raise SystemExit("Goal not provided and no targetPose found in slam_info.")

    loco = hanger_boot_sequence(iface=args.iface)
    walker = Locomotion(loco, detector, max_vx=args.max_speed)

    if nav_start is not None:
        sx, sy, _syaw = nav_start
    else:
        sx, sy, _syaw = detector.get_pose()
    goal_x, goal_y = float(goal_x), float(goal_y)

    if args.use_slam_nav:
        slam_client = SlamOperateClient()
        slam_client.Init()
        slam_client.SetTimeout(10.0)
        half = goal_yaw / 2.0
        qx, qy, qz, qw = 0.0, 0.0, math.sin(half), math.cos(half)
        resp = slam_client.pose_nav(goal_x, goal_y, 0.0, qx, qy, qz, qw, mode=1)
        if resp.code != 0:
            raise SystemExit(f"pose_nav failed: code={resp.code} raw={resp.raw}")
        print("pose_nav sent.")
        return

    def _check_obs() -> bool:
        if not args.with_avoidance:
            return False
        return detector.front_blocked()

    inflated = grid.inflate(radius_cells=args.inflation)
    start_cell = grid.world_to_grid(sx, sy)
    goal_cell = grid.world_to_grid(goal_x, goal_y)
    if start_cell == goal_cell:
        reached = walker.walk_to(
            goal_x,
            goal_y,
            final_yaw=goal_yaw,
            timeout=30.0,
            check_obstacle=_check_obs,
        )
        if not reached:
            raise SystemExit("Start and goal are the same cell; rotation/hold failed.")
        print("Navigation complete (same-cell goal).")
        return
    path = astar(inflated, start_cell, goal_cell)
    if path is None:
        raise SystemExit(f"No path from {start_cell} to {goal_cell}")

    smoothed = smooth_path(path, inflated)
    waypoints = grid_path_to_world_waypoints(smoothed, grid, spacing_m=args.spacing)
    if not waypoints:
        raise SystemExit("No waypoints generated.")

    for i, (wx, wy) in enumerate(waypoints):
        is_last = i == len(waypoints) - 1
        reached = walker.walk_to(
            wx,
            wy,
            final_yaw=goal_yaw if is_last else None,
            timeout=30.0,
            check_obstacle=_check_obs,
        )
        if not reached:
            raise SystemExit(f"Waypoint {i+1}/{len(waypoints)} aborted.")

    print("Navigation complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
