#!/usr/bin/env python3
"""
plan_path_view.py
=================

Load a saved occupancy map (.npz) that includes start/goal poses, run A*,
and visualize the planned path.
"""
from __future__ import annotations

import argparse
from typing import Optional

import cv2

from create_map import OccupancyGrid
from map_viewer import MapViewer
from path_planner import astar, grid_path_to_world_waypoints, smooth_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plan a path on a saved map and visualize it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--map", required=True, help="Path to saved map (.npz)")
    p.add_argument("--start-x", type=float, default=None, help="Override start x (m)")
    p.add_argument("--start-y", type=float, default=None, help="Override start y (m)")
    p.add_argument("--goal-x", type=float, default=None, help="Override goal x (m)")
    p.add_argument("--goal-y", type=float, default=None, help="Override goal y (m)")
    p.add_argument("--inflation", type=int, default=3, help="Inflation radius (cells)")
    p.add_argument("--allow-diagonal", action="store_true", help="Allow diagonal moves (8-connected)")
    p.add_argument("--smooth", action="store_true", help="Smooth the grid path")
    p.add_argument("--max-skip", type=int, default=5, help="Max skip for smoothing")
    p.add_argument("--spacing", type=float, default=0.5, help="Waypoint spacing (m)")
    p.add_argument("--scale", type=int, default=4, help="Viewer scale (px/cell)")
    return p.parse_args()


def _get_pose(occ: OccupancyGrid, key: str) -> Optional[tuple[float, float, float]]:
    if key == "start":
        return occ.start_pose
    if key == "goal":
        return occ.goal_pose
    return None


def main() -> None:
    args = _parse_args()

    occ = OccupancyGrid.load(args.map)
    start = _get_pose(occ, "start")
    goal = _get_pose(occ, "goal")

    if args.start_x is not None and args.start_y is not None:
        start = (args.start_x, args.start_y, 0.0)
    if args.goal_x is not None and args.goal_y is not None:
        goal = (args.goal_x, args.goal_y, 0.0)

    if start is None or goal is None:
        raise SystemExit("ERROR: start_pose or goal_pose missing. Use --start-x/--start-y and --goal-x/--goal-y.")

    start_rc = occ.world_to_grid(start[0], start[1])
    goal_rc = occ.world_to_grid(goal[0], goal[1])

    if args.inflation > 0:
        plan_grid = occ.inflate(args.inflation)
    else:
        plan_grid = occ.grid.copy()

    path = astar(plan_grid, start_rc, goal_rc, allow_diagonal=args.allow_diagonal)
    if not path:
        raise SystemExit("ERROR: A* failed to find a path.")

    if args.smooth:
        path = smooth_path(path, plan_grid, max_skip=args.max_skip)

    world_path = [occ.grid_to_world(r, c) for r, c in path]
    waypoints = grid_path_to_world_waypoints(path, occ, spacing_m=args.spacing)

    viewer = MapViewer(
        occ,
        window_name="Planned Path",
        scale=args.scale,
        inflation_radius=max(0, args.inflation),
    )
    viewer.set_start(start[0], start[1], start[2])
    viewer.set_goal(goal[0], goal[1])
    viewer.set_path(world_path)
    viewer.set_waypoints(waypoints)

    print(f"Start: ({start[0]:+.2f}, {start[1]:+.2f})")
    print(f"Goal:  ({goal[0]:+.2f}, {goal[1]:+.2f})")
    print(f"Path cells: {len(path)} | waypoints: {len(waypoints)}")
    print("Press 'q' or ESC to quit.")

    try:
        while True:
            img = viewer.render_image(start[0], start[1], start[2])
            cv2.imshow(viewer.window_name, img)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
