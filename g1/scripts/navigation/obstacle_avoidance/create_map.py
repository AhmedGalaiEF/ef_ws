"""
create_map.py
=============

2D occupancy grid map for obstacle avoidance navigation on the Unitree G1.

Grid cells are ``int8``: 0 = free, 1 = obstacle.  The map stores a
world-coordinate origin so that conversions between grid indices and metres
are consistent across all modules.

No SDK dependency for the core grid class -- pure numpy (+ scipy for
inflation). Optional helpers can build a grid from the built-in SLAM
HeightMap DDS topic.
"""
from __future__ import annotations

import math
import time

import numpy as np


class OccupancyGrid:
    """2D occupancy grid.  0 = free, 1 = obstacle."""

    def __init__(
        self,
        width_m: float,
        height_m: float,
        resolution: float = 0.1,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ):
        """
        Args:
            width_m:    Physical width in metres (x-axis extent).
            height_m:   Physical height in metres (y-axis extent).
            resolution: Metres per cell (default 0.1 m).
            origin_x:   World x-coordinate of the grid's (row=0, col=0) cell.
            origin_y:   World y-coordinate of the grid's (row=0, col=0) cell.
        """
        self.resolution = resolution
        self.origin = (origin_x, origin_y)
        self.width_cells = int(width_m / resolution)
        self.height_cells = int(height_m / resolution)
        self.grid = np.zeros((self.height_cells, self.width_cells), dtype=np.int8)
        # Optional annotations (world x, y, yaw in radians)
        self.start_pose: tuple[float, float, float] | None = None
        self.goal_pose: tuple[float, float, float] | None = None

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world (x, y) metres to grid (row, col).  Clamped to bounds."""
        col = int((wx - self.origin[0]) / self.resolution)
        row = int((wy - self.origin[1]) / self.resolution)
        row = max(0, min(self.height_cells - 1, row))
        col = max(0, min(self.width_cells - 1, col))
        return (row, col)

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert grid (row, col) to world (x, y) at the cell centre."""
        wx = col * self.resolution + self.origin[0] + self.resolution / 2
        wy = row * self.resolution + self.origin[1] + self.resolution / 2
        return (wx, wy)

    # ------------------------------------------------------------------
    # Cell queries
    # ------------------------------------------------------------------

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height_cells and 0 <= col < self.width_cells

    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.grid[row, col] == 0

    # ------------------------------------------------------------------
    # Cell mutations
    # ------------------------------------------------------------------

    def set_obstacle(self, row: int, col: int) -> None:
        if self.in_bounds(row, col):
            self.grid[row, col] = 1

    def set_obstacle_world(self, wx: float, wy: float) -> None:
        row, col = self.world_to_grid(wx, wy)
        self.set_obstacle(row, col)

    def set_free(self, row: int, col: int) -> None:
        if self.in_bounds(row, col):
            self.grid[row, col] = 0

    def add_rectangle(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> None:
        """Mark all cells within a world-coordinate rectangle as obstacle."""
        r_min, c_min = self.world_to_grid(x_min, y_min)
        r_max, c_max = self.world_to_grid(x_max, y_max)
        r_lo, r_hi = min(r_min, r_max), max(r_min, r_max)
        c_lo, c_hi = min(c_min, c_max), max(c_min, c_max)
        self.grid[r_lo : r_hi + 1, c_lo : c_hi + 1] = 1

    def mark_obstacle_from_range(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        range_obstacle: list[float],
    ) -> None:
        """Use ``range_obstacle[4]`` from ``SportModeState_`` to mark cells.

        Index mapping (verify empirically on real robot):
            0 = front, 1 = right, 2 = rear, 3 = left

        Skips values ``<= 0.01`` (invalid) or ``>= 5.0`` (out of range).
        """
        offsets = [0.0, -math.pi / 2, math.pi, math.pi / 2]
        for i, offset in enumerate(offsets):
            if i >= len(range_obstacle):
                break
            dist = range_obstacle[i]
            if dist <= 0.01 or dist >= 5.0:
                continue
            angle = robot_yaw + offset
            obs_x = robot_x + dist * math.cos(angle)
            obs_y = robot_y + dist * math.sin(angle)
            self.set_obstacle_world(obs_x, obs_y)

    # ------------------------------------------------------------------
    # Inflation (for path planning safety margin)
    # ------------------------------------------------------------------

    def inflate(self, radius_cells: int = 2) -> np.ndarray:
        """Return a *copy* of the grid with obstacles dilated.  Does NOT
        modify ``self.grid``."""
        from scipy.ndimage import binary_dilation

        kernel_size = 2 * radius_cells + 1
        struct = np.ones((kernel_size, kernel_size), dtype=bool)
        dilated = binary_dilation(self.grid > 0, structure=struct)
        return dilated.astype(np.int8)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        return self.grid.copy()

    def save(self, filepath: str) -> None:
        payload = {
            "grid": self.grid,
            "resolution": np.float64(self.resolution),
            "origin_x": np.float64(self.origin[0]),
            "origin_y": np.float64(self.origin[1]),
        }
        if self.start_pose is not None:
            payload["start_pose"] = np.array(self.start_pose, dtype=np.float64)
        if self.goal_pose is not None:
            payload["goal_pose"] = np.array(self.goal_pose, dtype=np.float64)
        np.savez(filepath, **payload)

    @classmethod
    def load(cls, filepath: str) -> "OccupancyGrid":
        try:
            data = np.load(filepath)
        except ValueError as exc:
            # Some legacy maps were saved with object arrays; allow pickle for those.
            if "pickled" in str(exc):
                data = np.load(filepath, allow_pickle=True)
            else:
                raise
        resolution = float(data["resolution"])
        origin_x = float(data["origin_x"])
        origin_y = float(data["origin_y"])
        grid_data = data["grid"]
        h, w = grid_data.shape
        obj = cls(w * resolution, h * resolution, resolution, origin_x, origin_y)
        obj.grid = grid_data.astype(np.int8)
        try:
            if "start_pose" in data:
                sp = np.array(data["start_pose"]).astype(np.float64).flatten()
                if sp.size >= 3:
                    obj.start_pose = (float(sp[0]), float(sp[1]), float(sp[2]))
            if "goal_pose" in data:
                gp = np.array(data["goal_pose"]).astype(np.float64).flatten()
                if gp.size >= 3:
                    obj.goal_pose = (float(gp[0]), float(gp[1]), float(gp[2]))
        except Exception:
            pass
        return obj


def load_from_point_cloud(
    map_path: str,
    resolution: float = 0.1,
    padding_m: float = 0.5,
    height_threshold: float | None = 0.15,
    max_height: float | None = None,
    origin_centered: bool = False,
) -> OccupancyGrid:
    """Load a PCD/PLY point cloud into an OccupancyGrid."""
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        raise RuntimeError("open3d is required to load PCD/PLY maps.") from exc

    cloud = o3d.io.read_point_cloud(map_path)
    if len(cloud.points) == 0:
        raise RuntimeError(f"Loaded point cloud is empty: {map_path}")

    pts = np.asarray(cloud.points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError(f"Unexpected point cloud shape: {pts.shape}")

    if max_height is not None:
        pts = pts[pts[:, 2] <= max_height]
        if pts.size == 0:
            raise RuntimeError("All points filtered by max_height.")

    min_xy = pts[:, :2].min(axis=0) - padding_m
    max_xy = pts[:, :2].max(axis=0) + padding_m
    width_m = float(max_xy[0] - min_xy[0])
    height_m = float(max_xy[1] - min_xy[1])
    if width_m <= 0 or height_m <= 0 or resolution <= 0:
        raise RuntimeError("Invalid map bounds/resolution.")

    if origin_centered:
        origin_x = -width_m / 2.0
        origin_y = -height_m / 2.0
    else:
        origin_x = float(min_xy[0])
        origin_y = float(min_xy[1])

    occ_grid = OccupancyGrid(width_m, height_m, resolution, origin_x, origin_y)
    grid = np.zeros((occ_grid.height_cells, occ_grid.width_cells), dtype=np.int8)

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


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_empty_map(
    width_m: float = 10.0,
    height_m: float = 10.0,
    resolution: float = 0.1,
    origin_x: float = -5.0,
    origin_y: float = -5.0,
) -> OccupancyGrid:
    """Create a blank map with the robot roughly centred."""
    return OccupancyGrid(width_m, height_m, resolution, origin_x, origin_y)


def create_from_slam(
    timeout: float = 5.0,
    height_threshold: float = 0.15,
    max_height: float | None = None,
    origin_centered: bool = True,
) -> OccupancyGrid:
    """Build an OccupancyGrid from the built-in SLAM HeightMap DDS topic.

    This is a best-effort helper that waits up to *timeout* seconds for a
    HeightMap_ message on ``rt/utlidar/map_state``.
    """
    try:
        from slam_map import SlamMapSubscriber
    except Exception as exc:
        raise RuntimeError("slam_map helper unavailable") from exc

    sub = SlamMapSubscriber()
    sub.start()

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

    raise RuntimeError("No SLAM map received within timeout")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create occupancy maps.")
    parser.add_argument("--source", choices=["empty", "slam_map", "slam_points", "lidar"], default="empty")
    parser.add_argument("--iface", default="enp1s0", help="network interface for DDS")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id")
    parser.add_argument("--width-m", type=float, default=10.0, help="map width (m)")
    parser.add_argument("--height-m", type=float, default=10.0, help="map height (m)")
    parser.add_argument("--resolution", type=float, default=0.1, help="map resolution (m)")
    parser.add_argument("--origin-x", type=float, default=-5.0, help="map origin x")
    parser.add_argument("--origin-y", type=float, default=-5.0, help="map origin y")
    parser.add_argument("--duration", type=float, default=5.0, help="seconds to accumulate points")
    parser.add_argument("--z-min", type=float, default=-0.5, help="min z for points")
    parser.add_argument("--z-max", type=float, default=1.5, help="max z for points")
    parser.add_argument("--stride", type=int, default=6, help="subsample points")
    parser.add_argument("--topic", type=str, default="", help="override DDS topic")
    parser.add_argument("--save", type=str, default="/tmp/occ_map.npz", help="output .npz path")
    args = parser.parse_args()

    if args.source == "empty":
        grid = create_empty_map(args.width_m, args.height_m, args.resolution, args.origin_x, args.origin_y)
        grid.add_rectangle(1.0, -1.0, 1.3, 1.0)
        print(
            f"Grid: {grid.width_cells}x{grid.height_cells} cells, "
            f"resolution={grid.resolution}m"
        )
        print(f"Obstacles after adding wall: {np.sum(grid.grid > 0)} cells")
    elif args.source == "slam_map":
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        except Exception as exc:
            raise SystemExit("unitree_sdk2py required for DDS SLAM map") from exc
        ChannelFactoryInitialize(args.domain_id, args.iface)
        grid = create_from_slam(
            timeout=args.duration,
            height_threshold=0.15,
            max_height=None,
            origin_centered=True,
        )
        print(f"SLAM map received: {grid.width_cells}x{grid.height_cells} cells")
    else:
        # DDS pointcloud accumulation (slam_points or lidar)
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
            from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
        except Exception as exc:
            raise SystemExit("unitree_sdk2py required for DDS pointcloud sources") from exc

        if args.source == "slam_points":
            topic = args.topic or "rt/unitree/slam_mapping/points"
        else:
            topic = args.topic or "rt/utlidar/cloud_livox_mid360"

        ChannelFactoryInitialize(args.domain_id, args.iface)
        grid = create_empty_map(args.width_m, args.height_m, args.resolution, args.origin_x, args.origin_y)

        latest = {"msg": None}

        def _cb(msg):
            latest["msg"] = msg

        sub = ChannelSubscriber(topic, PointCloud2_)
        sub.Init(_cb, 10)

        t0 = time.time()
        print(f"Accumulating points from {topic} for {args.duration:.1f}s ...")
        while time.time() - t0 < args.duration:
            msg = latest["msg"]
            if msg is None:
                time.sleep(0.05)
                continue
            try:
                fields = {f.name: f for f in msg.fields}
                if "x" not in fields or "y" not in fields:
                    time.sleep(0.05)
                    continue
                point_step = int(msg.point_step)
                if point_step <= 0:
                    time.sleep(0.05)
                    continue
                data = bytes(msg.data)
                if not data:
                    time.sleep(0.05)
                    continue
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
                xs = arr["x"][:: args.stride]
                ys = arr["y"][:: args.stride]
                zs = arr["z"][:: args.stride]
                for x, y, z in zip(xs, ys, zs):
                    if z < args.z_min or z > args.z_max:
                        continue
                    # Points are assumed to be in the robot frame (lidar) or world frame (slam_points).
                    grid.set_obstacle_world(float(x), float(y))
            except Exception:
                pass
            time.sleep(0.05)

        print(f"Obstacles after accumulation: {np.sum(grid.grid > 0)} cells")

    grid.save(args.save)
    print(f"Saved map to {args.save}")
