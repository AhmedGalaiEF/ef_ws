"""
navigate.py
===========

Main orchestrator for obstacle-avoidance navigation on the Unitree G1.

Connects to the robot over the network via the Unitree SDK2 Python library,
builds (or loads) an occupancy grid map, plans an A* path from the robot's
current position to a goal, divides the path into ~0.5 m waypoints, and
walks the robot along them.  At each step the robot checks for obstacles --
if one is detected it is added to a "dynamic" map and the path is replanned.

Prerequisites
-------------
* Robot powered on, standing (FSM-200), and reachable on the network.
* ``ChannelFactoryInitialize`` is called once here; do NOT call it elsewhere
  in the same process.

Usage
-----
::

    python navigate.py --iface eth0 --goal_x 3.0 --goal_y 2.0
    python navigate.py --iface eth0 --goal_x 3.0 --goal_y 2.0 --goal_yaw 1.57
    python navigate.py --iface eth0 --goal_x 5.0 --goal_y 0.0 --load_map saved.npz
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed.  Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc

from create_map import OccupancyGrid, create_empty_map
from path_planner import astar, smooth_path, grid_path_to_world_waypoints
from obstacle_detection import ObstacleDetector
from locomotion import Locomotion
from map_viewer import MapViewer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G1 obstacle-avoidance navigation (A* + dynamic replanning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iface", default="eth0",
                        help="Network interface connected to the robot")
    parser.add_argument("--goal_x", type=float, required=True,
                        help="Goal X position in metres")
    parser.add_argument("--goal_y", type=float, required=True,
                        help="Goal Y position in metres")
    parser.add_argument("--goal_yaw", type=float, default=None,
                        help="Optional final heading in radians")
    parser.add_argument("--map_width", type=float, default=10.0,
                        help="Map width in metres")
    parser.add_argument("--map_height", type=float, default=10.0,
                        help="Map height in metres")
    parser.add_argument("--resolution", type=float, default=0.1,
                        help="Map resolution in metres per cell")
    parser.add_argument("--load_map", type=str, default=None,
                        help="Path to a saved .npz map file")
    parser.add_argument("--max_speed", type=float, default=0.3,
                        help="Maximum forward speed (m/s)")
    parser.add_argument("--inflation", type=int, default=3,
                        help="Obstacle inflation radius in cells")
    parser.add_argument("--spacing", type=float, default=0.5,
                        help="Waypoint spacing in metres (~1 step)")
    parser.add_argument("--max_replans", type=int, default=20,
                        help="Maximum replan attempts before giving up")
    parser.add_argument("--viz", action="store_true",
                        help="Show a live map window (OpenCV) while navigating")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # --- 1. SDK initialisation ---------------------------------------------
    print(f"Initialising SDK on interface '{args.iface}' ...")
    ChannelFactoryInitialize(0, args.iface)

    # --- 2. LocoClient (robot must already be standing / FSM-200) ----------
    loco = LocoClient()
    loco.SetTimeout(10.0)
    loco.Init()

    # --- 3. Obstacle detector (subscribes to rt/sportmodestate) ------------
    detector = ObstacleDetector(warn_distance=0.8, stop_distance=0.4)
    detector.start()

    # Wait for first pose reading
    print("Waiting for SportModeState_ ...")
    time.sleep(1.0)
    if detector.is_stale():
        sys.exit("ERROR: no SportModeState_ data received.  "
                 "Is the robot connected and standing?")

    # --- 4. Map ------------------------------------------------------------
    if args.load_map:
        occ_grid = OccupancyGrid.load(args.load_map)
        print(f"Loaded map from {args.load_map}  "
              f"({occ_grid.width_cells}x{occ_grid.height_cells} cells)")
    else:
        occ_grid = create_empty_map(
            width_m=args.map_width,
            height_m=args.map_height,
            resolution=args.resolution,
            origin_x=-args.map_width / 2,
            origin_y=-args.map_height / 2,
        )
        print(f"Created empty {args.map_width}x{args.map_height} m map, "
              f"resolution={args.resolution} m "
              f"({occ_grid.width_cells}x{occ_grid.height_cells} cells)")

    # --- 5. Locomotion wrapper ---------------------------------------------
    walker = Locomotion(loco, detector, max_vx=args.max_speed)

    # --- 5b. Optional live map viewer -------------------------------------
    viewer: MapViewer | None = None
    if args.viz:
        viewer = MapViewer(occ_grid, scale=4, inflation_radius=args.inflation)
        print("Live map viewer enabled (press 'q' in the window to quit).")

    # --- 6. Starting pose --------------------------------------------------
    sx, sy, syaw = detector.get_pose()
    print(f"Start : ({sx:+.2f}, {sy:+.2f}), yaw={math.degrees(syaw):+.1f} deg")
    print(f"Goal  : ({args.goal_x:+.2f}, {args.goal_y:+.2f})"
          + (f", yaw={math.degrees(args.goal_yaw):.1f} deg" if args.goal_yaw else ""))

    # --- 7. Navigation loop ------------------------------------------------
    goal_x, goal_y = args.goal_x, args.goal_y
    replan_count = 0
    goal_reached = False

    try:
        while replan_count < args.max_replans and not goal_reached:
            # 7a. Current pose
            cx, cy, cyaw = detector.get_pose()

            # 7b. Update map with latest obstacle readings
            ranges = detector.get_ranges()
            occ_grid.mark_obstacle_from_range(cx, cy, cyaw, ranges)

            # 7c. Inflate for planning
            inflated = occ_grid.inflate(radius_cells=args.inflation)

            # 7d. Grid coordinates
            start_cell = occ_grid.world_to_grid(cx, cy)
            goal_cell = occ_grid.world_to_grid(goal_x, goal_y)

            # 7e. A*
            print(f"\nPlanning (attempt {replan_count + 1}/{args.max_replans}) ...")
            print(f"  From cell {start_cell} to {goal_cell}")
            raw_path = astar(inflated, start_cell, goal_cell)

            if raw_path is None:
                print("ERROR: no path found.  Goal may be unreachable.")
                break

            # 7f. Smooth + world waypoints
            smoothed = smooth_path(raw_path, inflated)
            waypoints = grid_path_to_world_waypoints(
                smoothed, occ_grid, spacing_m=args.spacing
            )
            print(f"  {len(raw_path)} cells -> {len(smoothed)} smoothed "
                  f"-> {len(waypoints)} waypoints")

            # 7f-viz. Update viewer overlays with the new plan
            if viewer is not None:
                viewer.set_goal(goal_x, goal_y)
                # Full smoothed path in world coords for display
                full_world_path = [occ_grid.grid_to_world(r, c) for r, c in smoothed]
                viewer.set_path(full_world_path)
                viewer.set_waypoints(waypoints)

            # 7g. Walk each waypoint
            aborted = False
            for i, (wx, wy) in enumerate(waypoints):
                dist_to_goal = math.hypot(wx - goal_x, wy - goal_y)
                is_last = i == len(waypoints) - 1
                final_yaw = args.goal_yaw if is_last else None

                print(f"  [{i + 1}/{len(waypoints)}] -> ({wx:+.2f}, {wy:+.2f})"
                      f"  dist_to_goal={dist_to_goal:.2f} m")

                def _check_and_viz() -> bool:
                    """Obstacle check + live viewer update (called every tick)."""
                    if viewer is not None:
                        vx, vy, vyaw = detector.get_pose()
                        vranges = detector.get_ranges()
                        occ_grid.mark_obstacle_from_range(vx, vy, vyaw, vranges)
                        viewer.update(vx, vy, vyaw, vranges)
                    return detector.front_blocked()

                reached = walker.walk_to(
                    wx, wy,
                    final_yaw=final_yaw,
                    timeout=30.0,
                    check_obstacle=_check_and_viz,
                )

                if not reached:
                    print("  ** Obstacle or timeout -- replanning **")
                    aborted = True
                    replan_count += 1

                    # Record newly sensed obstacles
                    obs_positions = detector.get_obstacle_world_positions()
                    for ox, oy in obs_positions:
                        occ_grid.set_obstacle_world(ox, oy)
                    print(f"  Added {len(obs_positions)} obstacle(s) to map")
                    break

            if not aborted:
                goal_reached = True

        # --- 8. Result -----------------------------------------------------
        print()
        if goal_reached:
            fx, fy, fyaw = detector.get_pose()
            dist = math.hypot(fx - goal_x, fy - goal_y)
            print(f"Goal reached!  Final pos: ({fx:+.2f}, {fy:+.2f}), "
                  f"error={dist:.2f} m")
        else:
            print(f"Navigation failed after {replan_count} replans.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # --- 9. Cleanup ----------------------------------------------------
        walker.stop()
        if viewer is not None:
            viewer.close()
        save_path = "/tmp/final_obstacle_map.npz"
        occ_grid.save(save_path)
        print(f"Robot stopped.  Map saved to {save_path}")


if __name__ == "__main__":
    main()
