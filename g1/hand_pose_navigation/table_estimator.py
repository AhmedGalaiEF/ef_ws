"""
Obstacle perception — table plane estimation
==============================================
Fits an approximate horizontal table plane from a single RGB-D frame so the
trajectory checker (obstacle_checker.py) can keep the arm from driving into
it. This is a lightweight RANSAC-style plane fit over a subsampled point
cloud — good enough for a height + rough footprint, not exact table edges.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TablePlane:
    z_base_m: float                    # table surface height in base_link frame
    x_range_m: Tuple[float, float]      # rough horizontal footprint, base frame
    y_range_m: Tuple[float, float]
    inlier_count: int
    confidence: float                   # inlier fraction of sampled points


def estimate_table_plane(
    depth_m: np.ndarray,
    K: np.ndarray,
    T_base_camera: np.ndarray,
    *,
    stride: int = 8,
    min_depth_m: float = 0.2,
    max_depth_m: float = 1.5,
    z_normal_tol: float = 0.25,
    min_inliers: int = 200,
    ransac_iters: int = 60,
    dist_tol_m: float = 0.015,
) -> Optional[TablePlane]:
    """
    Estimate a horizontal plane in base_link frame from a depth image.

    Returns None if there aren't enough valid depth samples, or no
    sufficiently horizontal/confident plane is found (e.g. camera pointed
    at open floor or a scene with nothing in range).
    """
    if depth_m is None or depth_m.ndim < 2:
        return None
    h, w = depth_m.shape[:2]
    ys = np.arange(0, h, max(1, stride))
    xs = np.arange(0, w, max(1, stride))
    grid_x, grid_y = np.meshgrid(xs, ys)
    d = depth_m[grid_y, grid_x].astype(np.float64)
    valid = np.isfinite(d) & (d > min_depth_m) & (d < max_depth_m)
    if int(np.count_nonzero(valid)) < min_inliers:
        return None

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = grid_x[valid].astype(np.float64)
    v = grid_y[valid].astype(np.float64)
    dz = d[valid]
    x_cam = (u - cx) * dz / fx
    y_cam = (v - cy) * dz / fy
    pts_cam_h = np.stack([x_cam, y_cam, dz, np.ones_like(dz)], axis=1)  # N x 4

    pts_base = (T_base_camera @ pts_cam_h.T).T[:, :3]  # N x 3
    n_pts = pts_base.shape[0]
    if n_pts < min_inliers:
        return None

    rng = np.random.default_rng(0)
    best_count = 0
    best_z: Optional[float] = None
    best_mask: Optional[np.ndarray] = None
    for _ in range(ransac_iters):
        idx = rng.choice(n_pts, size=3, replace=False)
        p0, p1, p2 = pts_base[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        if abs(normal[2]) < (1.0 - z_normal_tol):
            continue  # not close enough to horizontal
        plane_d = -np.dot(normal, p0)
        dist = np.abs(pts_base @ normal + plane_d)
        inliers = dist < dist_tol_m
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_z = float(np.median(pts_base[inliers, 2]))
            best_mask = inliers

    if best_mask is None or best_count < min_inliers:
        return None

    inlier_pts = pts_base[best_mask]
    return TablePlane(
        z_base_m=float(best_z),
        x_range_m=(float(inlier_pts[:, 0].min()), float(inlier_pts[:, 0].max())),
        y_range_m=(float(inlier_pts[:, 1].min()), float(inlier_pts[:, 1].max())),
        inlier_count=best_count,
        confidence=float(best_count) / float(n_pts),
    )
