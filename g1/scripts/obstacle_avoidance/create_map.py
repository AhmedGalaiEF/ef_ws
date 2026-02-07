"""
create_map.py
=============

2D occupancy grid map for obstacle avoidance navigation on the Unitree G1.

Grid cells are ``int8``: 0 = free, 1 = obstacle.  The map stores a
world-coordinate origin so that conversions between grid indices and metres
are consistent across all modules.

No SDK dependency -- pure numpy (+ scipy for inflation).
"""
from __future__ import annotations

import math

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
        np.savez(
            filepath,
            grid=self.grid,
            resolution=np.float64(self.resolution),
            origin_x=np.float64(self.origin[0]),
            origin_y=np.float64(self.origin[1]),
        )

    @classmethod
    def load(cls, filepath: str) -> OccupancyGrid:
        data = np.load(filepath)
        resolution = float(data["resolution"])
        origin_x = float(data["origin_x"])
        origin_y = float(data["origin_y"])
        grid_data = data["grid"]
        h, w = grid_data.shape
        obj = cls(w * resolution, h * resolution, resolution, origin_x, origin_y)
        obj.grid = grid_data.astype(np.int8)
        return obj


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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid = create_empty_map(10.0, 10.0, 0.1, -5.0, -5.0)
    print(f"Grid: {grid.width_cells}x{grid.height_cells} cells, "
          f"resolution={grid.resolution}m")

    grid.add_rectangle(1.0, -1.0, 1.3, 1.0)
    print(f"Obstacles after adding wall: {np.sum(grid.grid > 0)} cells")

    r, c = grid.world_to_grid(0.0, 0.0)
    wx, wy = grid.grid_to_world(r, c)
    print(f"World (0,0) -> grid ({r},{c}) -> world ({wx:.2f},{wy:.2f})")

    grid.mark_obstacle_from_range(0.0, 0.0, 0.0, [2.0, 1.5, 3.0, 1.0])
    print(f"Obstacles after range marking: {np.sum(grid.grid > 0)} cells")

    inflated = grid.inflate(3)
    print(f"Inflated obstacles: {np.sum(inflated > 0)} cells")

    grid.save("/tmp/test_map.npz")
    loaded = OccupancyGrid.load("/tmp/test_map.npz")
    assert np.array_equal(grid.grid, loaded.grid)
    print("Save/load round-trip: OK")
    print("All tests passed.")
