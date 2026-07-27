#!/usr/bin/env python3
"""
VLA.py — Vision-Language-Action grasp pipeline for the Unitree G1
====================================================================

RGB-D + lidar -> fused point cloud in base_link -> VLM grounding + 3D
segmentation -> keyframe policy (VLA stand-in) -> safety filter -> IK
(arm_sdk.ArmSdk) -> joint trajectory smoothing -> closed-loop replan.

Stage map (see the architecture note this file was written against):
  1. Calibration            -> RobotCalibration / CameraIntrinsics
  2. Fused 3D observation   -> backproject_depth_to_base, lidar_points_to_base
  3. Detect + localize      -> ground_object_in_frame, segment_object_points,
                               build_grasp_targets (T_base_bottle/grasp/pregrasp)
  4. VLA predicts an action -> VLAPolicy / GeometricKeyframePolicy
  5. Staged manipulation    -> BottleGraspSequencer (Stage enum)
  6. Safety/reachability    -> SafetyFilter
  7. IK                     -> arm_sdk.ArmSdk.ik_move_EE (DLS IK + joint limits)
  8. Trajectory smoothing   -> ArmSdk's internal ramp + small clamped EE deltas
  9. Grasp success check    -> BottleGraspSequencer._verify_grasp

This module reuses the robot SDK in ../../ (modules/): sdk_client.Robot for
sensors/hands/locomotion/VLM detection, and arm_sdk.ArmSdk for Cartesian IK
(DLS solver over the 7-DOF arm chain, joint-limit clamped, ramped publish).
It does not reimplement FK/IK — see modules/hand_pose_navigation/arm_fk.py
and arm_ik.py, and modules/WBC/ik.md, for that layer's internals.

The "VLA" in GeometricKeyframePolicy is a deterministic geometric baseline,
not a learned network. It satisfies the same Observation -> ActionDelta
contract a trained policy would, so it can be swapped in without touching
perception, safety, IK, or sequencing.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

# ── Path bootstrap: modules/ (../../ from this file) holds sdk_client.py,
# arm_sdk.py, dds_env.py — the actual G1 SDK wrappers.
_OLLAMA_AI_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_OLLAMA_AI_DIR)
_MODULES_DIR = os.path.dirname(_SCRIPTS_DIR)
for _p in (_MODULES_DIR, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sdk_client import Robot          # rt/* high-level SDK wrapper
from arm_sdk import ArmSdk            # rt/arm_sdk Cartesian IK controller

logger = logging.getLogger("VLA")


# ============================================================================
# Rotation / transform helpers
# ============================================================================

def _Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _homogeneous(R: np.ndarray, t: Tuple[float, float, float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _apply_pose_increment(T: np.ndarray, inc: np.ndarray) -> np.ndarray:
    """Apply [dx,dy,dz,droll,dpitch,dyaw] to a 4x4 pose.

    Mirrors arm_sdk._apply_pose_increment's convention so the safety filter's
    pre-check matches what ik_move_EE will actually execute.
    """
    T_new = T.copy()
    T_new[:3, 3] = T_new[:3, 3] + inc[:3]
    T_new[:3, :3] = _Rz(inc[5]) @ _Ry(inc[4]) @ _Rx(inc[3]) @ T_new[:3, :3]
    return T_new


def _transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    homog = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    return (T @ homog.T).T[:, :3]


# ============================================================================
# Stage 1 — Calibration: tie the RGB-D camera and lidar into base_link
# ============================================================================

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def default_d435_640x480(cls) -> "CameraIntrinsics":
        """Placeholder D435-class values. scripts/real_sense.py prints the
        real fx/fy/cx/cy for the attached unit at stream start — copy those
        in before trusting any metric grasp computed from this file."""
        return cls(fx=615.0, fy=615.0, cx=320.0, cy=240.0, width=640, height=480)


@dataclass
class RobotCalibration:
    """Static extrinsics tying every sensor into base_link.

    CALIBRATE HERE: T_base_camera and T_base_lidar are mount-geometry
    placeholders (chest-mounted RGBD looking forward/level; lidar treated as
    coincident with base_link, which is approximately true for the G1's
    head-mounted Mid-360 whose rt/utlidar/cloud_deskewed output is already
    published close to base_link). Measure/CAD-derive the real offsets and
    replace them — every downstream 3D position depends on these two.
    T_base_camera maps the RealSense *optical* frame (x-right, y-down,
    z-forward) into base_link, matching backproject_depth_to_base's use of it.
    """

    camera_intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics.default_d435_640x480)
    T_base_camera: np.ndarray = field(
        default_factory=lambda: _homogeneous(_Ry(0.0) @ _Rx(-np.pi / 2) @ _Rz(np.pi / 2), (0.06, 0.0, 0.45))
    )
    T_base_lidar: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    max_depth_m: float = 3.0


# ============================================================================
# Stage 2 — Fused 3D observation: RGB-D + lidar -> colored cloud in base_link
# ============================================================================

def backproject_depth_to_base(
    depth_m: np.ndarray,
    rgb_rgb: np.ndarray,
    calib: RobotCalibration,
    *,
    bbox_px: Optional[Tuple[int, int, int, int]] = None,
    stride: int = 2,
) -> Dict[str, np.ndarray]:
    """RGB-D -> colored point cloud in base_link frame.

    Returns points_base (N,3), colors (N,3 uint8), and pixel_uv (N,2) so a
    2D detection bbox can later be used to select the matching 3D points.
    """
    empty = {
        "points_base": np.zeros((0, 3), np.float32),
        "colors": np.zeros((0, 3), np.uint8),
        "pixel_uv": np.zeros((0, 2), np.int32),
    }
    if depth_m.size == 0:
        return empty
    if rgb_rgb.shape[:2] != depth_m.shape[:2]:
        import cv2

        rgb_rgb = cv2.resize(rgb_rgb, (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_AREA)

    K = calib.camera_intrinsics
    h, w = depth_m.shape[:2]
    if bbox_px is not None:
        x1, y1, x2, y2 = bbox_px
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
    else:
        x1, y1, x2, y2 = 0, 0, w, h
    if x2 <= x1 or y2 <= y1:
        return empty

    cols = np.arange(x1, x2, max(1, stride))
    rows = np.arange(y1, y2, max(1, stride))
    uu, vv = np.meshgrid(cols, rows)
    uu, vv = uu.ravel(), vv.ravel()

    z = depth_m[vv, uu]
    valid = np.isfinite(z) & (z > 0.05) & (z < calib.max_depth_m)
    uu, vv, z = uu[valid], vv[valid], z[valid]
    if uu.size == 0:
        return empty

    x = (uu.astype(np.float64) - K.cx) / K.fx * z
    y = (vv.astype(np.float64) - K.cy) / K.fy * z
    pts_cam = np.stack([x, y, z.astype(np.float64)], axis=1)
    pts_base = _transform_points(calib.T_base_camera, pts_cam).astype(np.float32)

    return {
        "points_base": pts_base,
        "colors": rgb_rgb[vv, uu].astype(np.uint8),
        "pixel_uv": np.stack([uu, vv], axis=1).astype(np.int32),
    }


def lidar_points_to_base(points: List[Dict[str, float]], calib: RobotCalibration) -> np.ndarray:
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.array([[p["x"], p["y"], p["z"]] for p in points], dtype=np.float64)
    if np.allclose(calib.T_base_lidar, np.eye(4)):
        return arr.astype(np.float32)
    return _transform_points(calib.T_base_lidar, arr).astype(np.float32)


def fit_plane_ransac(
    points: np.ndarray,
    *,
    n_iters: int = 200,
    dist_thresh: float = 0.015,
    min_inliers: int = 50,
    seed: int = 0,
) -> Optional[Tuple[np.ndarray, float]]:
    """RANSAC plane fit for the table / floor. Returns (unit normal, d) with
    normal.z >= 0, for the plane n . p + d = 0, or None if no plane is found.
    """
    n = points.shape[0]
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    best_inliers = 0
    best_plane: Optional[Tuple[np.ndarray, float]] = None
    for _ in range(n_iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(np.dot(normal, p0))
        inliers = int(np.sum(np.abs(points @ normal + d) < dist_thresh))
        if inliers > best_inliers:
            best_inliers, best_plane = inliers, (normal, d)
    if best_plane is None or best_inliers < min_inliers:
        return None

    # Refine with all inliers via SVD (more accurate than the 3-point sample).
    normal, d = best_plane
    inlier_pts = points[np.abs(points @ normal + d) < dist_thresh]
    centroid = inlier_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_pts - centroid)
    refined_normal = vt[-1]
    if refined_normal[2] < 0:
        refined_normal = -refined_normal
    refined_d = -float(np.dot(refined_normal, centroid))
    return refined_normal, refined_d


# ============================================================================
# Stage 3 — Detect + localize the object, build grasp/pregrasp targets
# ============================================================================

@dataclass
class Detection2D:
    object_name: str
    confidence: float
    bbox_norm: Tuple[float, float, float, float]  # x1,y1,x2,y2 in 0..1
    reason: str


def ground_object_in_frame(
    robot: Robot,
    object_name: str,
    rgb_jpeg: bytes,
    *,
    model: Optional[str] = None,
    timeout: float = 20.0,
) -> Optional[Detection2D]:
    """VLM grounding of `object_name` in the *same* RGB frame the depth map
    came from. robot.detect() prefers the Unitree VideoClient image, which
    can be a different camera than the RGBD/RealSense stream — using it here
    would silently break the pixel<->depth correspondence needed to convert
    the bbox into a metric 3D position, so this talks to Ollama directly on
    the RGBD frame instead of calling robot.detect().
    """
    import urllib.error
    import urllib.request

    selected_model = model or robot.vision_model
    prompt = (
        "You are a visual object detector for a robot end-effector. "
        f"Find the {object_name!r} in the image and answer only valid JSON: "
        '{"present":true|false,"confidence":0.0,"bbox":[x1,y1,x2,y2],"reason":"short"}. '
        "bbox is normalized 0..1 image coordinates tightly around the object, "
        "or null if the object is not visible."
    )
    body = {
        "model": selected_model,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [base64.b64encode(rgb_jpeg).decode("ascii")],
        }],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 150},
    }
    request = urllib.request.Request(
        f"{robot.ollama_url}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        logger.warning("Vision grounding request failed: %s", exc)
        return None

    text = str(payload.get("message", {}).get("content", "")).strip()
    cleaned = text
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start:end + 1])
    except Exception:
        return None
    if not isinstance(parsed, dict) or not parsed.get("present"):
        return None
    bbox = parsed.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [max(0.0, min(1.0, float(v))) for v in bbox]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except Exception:
        confidence = 0.0
    return Detection2D(
        object_name=object_name,
        confidence=confidence,
        bbox_norm=(x1, y1, x2, y2),
        reason=str(parsed.get("reason", "")),
    )


@dataclass
class ObjectObservation:
    points_base: np.ndarray   # (N,3) trimmed object point cloud, base_link
    centroid: np.ndarray       # (3,)
    axis: np.ndarray           # (3,) principal axis (bottle "up"), unit, z >= 0
    extent_along_axis: float
    radius_perp: float
    confidence: float


def segment_object_points(
    detection: Detection2D,
    cloud: Dict[str, np.ndarray],
    image_shape: Tuple[int, int],
    *,
    min_points: int = 30,
) -> Optional[ObjectObservation]:
    """Keep fused-cloud points whose source pixel falls inside the 2D bbox,
    then trim outliers so background bleeding into a loose bbox doesn't
    corrupt the centroid/axis estimate. This is the "see geometry, not just
    images" step — the object pose below is computed from these 3D points.
    """
    h, w = image_shape
    x1, y1, x2, y2 = detection.bbox_norm
    px1, py1, px2, py2 = x1 * w, y1 * h, x2 * w, y2 * h

    uv = cloud["pixel_uv"]
    if uv.shape[0] == 0:
        return None
    mask = (uv[:, 0] >= px1) & (uv[:, 0] <= px2) & (uv[:, 1] >= py1) & (uv[:, 1] <= py2)
    pts = cloud["points_base"][mask].astype(np.float64)
    if pts.shape[0] < min_points:
        return None

    median = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - median, axis=1)
    mad = float(np.median(np.abs(dist - np.median(dist)))) + 1e-6
    pts = pts[dist < (np.median(dist) + 4.0 * mad)]
    if pts.shape[0] < min_points:
        return None

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = (centered.T @ centered) / max(1, pts.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    axis = eigvecs[:, -1]
    if axis[2] < 0:
        axis = -axis
    along = centered @ axis
    extent = float(along.max() - along.min())
    perp = centered - np.outer(along, axis)
    radius = float(np.percentile(np.linalg.norm(perp, axis=1), 90))

    return ObjectObservation(
        points_base=pts,
        centroid=centroid,
        axis=axis,
        extent_along_axis=max(extent, 0.02),
        radius_perp=max(radius, 0.01),
        confidence=detection.confidence,
    )


@dataclass
class GraspGeometry:
    """CALIBRATE HERE against the gripper's actual palm-axis convention.

    build_object_frame assumes the wrist approach direction is the object
    frame's local +X — verify this against the physical hand_palm_link axes
    in g1_29dof_with_hand_rev_1_0_pkg.urdf before trusting the orientation.
    """

    pregrasp_offset_m: float = 0.12   # 10-15 cm back along the approach axis
    grasp_height_frac: float = 0.55   # fraction up the bottle body to center the hand
    lift_height_m: float = 0.15


def build_object_frame(obs: ObjectObservation, ee_current_pos: np.ndarray) -> np.ndarray:
    """T_base_bottle: origin at the bottle centroid, z = bottle vertical
    axis, x = horizontal direction from the current EE toward the bottle
    (becomes the wrist approach axis)."""
    z_axis = obs.axis / (np.linalg.norm(obs.axis) + 1e-9)
    to_object = obs.centroid - ee_current_pos
    to_object_horiz = to_object - np.dot(to_object, z_axis) * z_axis
    norm = np.linalg.norm(to_object_horiz)
    if norm < 1e-6:
        to_object_horiz, norm = np.array([1.0, 0.0, 0.0]), 1.0
    x_axis = to_object_horiz / norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
    x_axis = np.cross(y_axis, z_axis)  # re-orthogonalize for a clean right-handed basis

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    T[:3, 3] = obs.centroid
    return T


def build_grasp_targets(
    obs: ObjectObservation,
    ee_current_pos: np.ndarray,
    geometry: GraspGeometry,
) -> Dict[str, np.ndarray]:
    """T_base_bottle / T_base_grasp / T_base_pregrasp."""
    T_object = build_object_frame(obs, ee_current_pos)
    R = T_object[:3, :3]
    x_axis, z_axis = R[:, 0], R[:, 2]

    half_extent = obs.extent_along_axis / 2.0
    grasp_z_offset = -half_extent + geometry.grasp_height_frac * obs.extent_along_axis

    T_grasp = T_object.copy()
    T_grasp[:3, 3] = obs.centroid + grasp_z_offset * z_axis

    T_pregrasp = T_grasp.copy()
    T_pregrasp[:3, 3] = T_grasp[:3, 3] - geometry.pregrasp_offset_m * x_axis

    return {"T_base_bottle": T_object, "T_base_grasp": T_grasp, "T_base_pregrasp": T_pregrasp}


# ============================================================================
# Stage 6 — Safety / reachability filter
# ============================================================================

@dataclass
class SafetyLimits:
    """CALIBRATE HERE against the physical G1 arm reach / workspace.

    shoulder_in_base values are the documented G1 shoulder origins in
    base_link at zero waist angle (see hand_pose_navigation/arm_fk.py)."""

    shoulder_in_base: Dict[str, np.ndarray] = field(default_factory=lambda: {
        "right": np.array([0.0, -0.100, 0.292]),
        "left": np.array([0.0, 0.100, 0.292]),
    })
    min_reach_m: float = 0.12
    max_reach_m: float = 0.62
    min_z_base_m: float = 0.05        # floor/lap clearance
    max_step_pos_m: float = 0.03      # "use small EE deltas, not large jumps"
    max_step_rot_rad: float = 0.15
    table_clearance_m: float = 0.02


class SafetyFilter:
    """Coarse pre-IK gate. Joint limits and singularity handling are already
    enforced inside arm_sdk.ArmSdk.ik_move_EE (DLS solve clamped to URDF
    limits, per-joint step clamp, and it reports failure instead of moving on
    an unreachable/singular solve) — this filter catches obviously bad
    targets (out of reach, below the table, oversized steps) before spending
    an IK solve on them, and rejects table collisions IK alone can't see.
    """

    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits
        self.table_plane: Optional[Tuple[np.ndarray, float]] = None  # (normal, d)

    def set_table_plane(self, plane: Optional[Tuple[np.ndarray, float]]) -> None:
        self.table_plane = plane

    def clamp_step(self, delta: np.ndarray) -> np.ndarray:
        out = delta.copy()
        pos_norm = float(np.linalg.norm(out[:3]))
        if pos_norm > self.limits.max_step_pos_m:
            out[:3] *= self.limits.max_step_pos_m / pos_norm
        rot_norm = float(np.linalg.norm(out[3:]))
        if rot_norm > self.limits.max_step_rot_rad:
            out[3:] *= self.limits.max_step_rot_rad / rot_norm
        return out

    def check_target(self, T_base_target: np.ndarray, arm: str) -> List[str]:
        problems: List[str] = []
        shoulder = self.limits.shoulder_in_base.get(arm, self.limits.shoulder_in_base["right"])
        reach = float(np.linalg.norm(T_base_target[:3, 3] - shoulder))
        if reach > self.limits.max_reach_m:
            problems.append(f"target {reach:.3f} m from shoulder exceeds max_reach_m={self.limits.max_reach_m}")
        if reach < self.limits.min_reach_m:
            problems.append(f"target {reach:.3f} m from shoulder is below min_reach_m={self.limits.min_reach_m}")
        z = float(T_base_target[2, 3])
        if z < self.limits.min_z_base_m:
            problems.append(f"target z={z:.3f} m is below min_z_base_m={self.limits.min_z_base_m}")
        if self.table_plane is not None:
            normal, d = self.table_plane
            height_above_table = float(np.dot(normal, T_base_target[:3, 3]) + d)
            if height_above_table < self.limits.table_clearance_m:
                problems.append(
                    f"target is {height_above_table:.3f} m above the fitted table plane "
                    f"(< table_clearance_m={self.limits.table_clearance_m})"
                )
        return problems


# ============================================================================
# Stage 4 — VLA policy: Observation -> next EE SE(3) delta + gripper command
# ============================================================================

@dataclass
class Proprioception:
    T_base_ee: np.ndarray
    gripper_open: bool


@dataclass
class Observation:
    """input = image tokens + point tokens + language + proprioception."""
    object_obs: Optional[ObjectObservation]
    proprio: Proprioception
    T_base_keyframe_target: np.ndarray
    instruction: str


@dataclass
class ActionDelta:
    """output = next EE SE(3) delta + gripper command."""
    delta_pose: np.ndarray             # [dx,dy,dz,droll,dpitch,dyaw], base_link frame
    gripper: Optional[str]             # "open" | "close" | None
    converged: bool


class VLAPolicy(Protocol):
    def predict(self, obs: Observation) -> ActionDelta: ...


class GeometricKeyframePolicy:
    """Deterministic short-horizon keyframe-tracking baseline.

    Stands in for a learned VLA network behind the Observation ->
    ActionDelta contract above: swap this class out for a real policy
    (image + point-token transformer, diffusion policy, etc.) without
    touching perception, safety, IK, or sequencing.
    """

    def __init__(
        self,
        *,
        pos_gain: float = 0.6,
        rot_gain: float = 0.6,
        pos_tol_m: float = 0.01,
        rot_tol_rad: float = 0.05,
    ) -> None:
        self.pos_gain = pos_gain
        self.rot_gain = rot_gain
        self.pos_tol_m = pos_tol_m
        self.rot_tol_rad = rot_tol_rad

    def predict(self, obs: Observation) -> ActionDelta:
        T_cur = obs.proprio.T_base_ee
        T_tgt = obs.T_base_keyframe_target
        pos_err = T_tgt[:3, 3] - T_cur[:3, 3]
        R_err = T_tgt[:3, :3] @ T_cur[:3, :3].T
        rot_err = 0.5 * np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ])
        pos_norm = float(np.linalg.norm(pos_err))
        rot_norm = float(np.linalg.norm(rot_err))
        converged = pos_norm < self.pos_tol_m and rot_norm < self.rot_tol_rad
        delta = np.concatenate([pos_err * self.pos_gain, rot_err * self.rot_gain])
        return ActionDelta(delta_pose=delta, gripper=None, converged=converged)


# ============================================================================
# Stage 2/3 orchestration — perception pipeline
# ============================================================================

class PerceptionPipeline:
    def __init__(self, robot: Robot, calib: RobotCalibration, *, vision_model: Optional[str] = None) -> None:
        self.robot = robot
        self.calib = calib
        self.vision_model = vision_model

    def observe(
        self,
        object_name: str,
        *,
        stride: int = 2,
    ) -> Tuple[Optional[ObjectObservation], Optional[Tuple[np.ndarray, float]]]:
        frame = self.robot.get_rgbd(timeout=2.0)
        rgb_rgb = frame["rgb_rgb"]
        depth_m = frame["depth_m"]
        h, w = depth_m.shape[:2]

        detection = ground_object_in_frame(
            self.robot, object_name, frame["rgb_jpeg"], model=self.vision_model,
        )
        cloud = backproject_depth_to_base(depth_m, rgb_rgb, self.calib, stride=stride)

        table_plane: Optional[Tuple[np.ndarray, float]] = None
        try:
            lidar_pts = lidar_points_to_base(self.robot.get_lidar_points(max_points=4000), self.calib)
            if lidar_pts.shape[0] >= 50:
                table_plane = fit_plane_ransac(lidar_pts)
        except Exception as exc:
            logger.debug("Lidar plane fit skipped: %s", exc)
        if table_plane is None and cloud["points_base"].shape[0] >= 200:
            table_plane = fit_plane_ransac(cloud["points_base"][::3].astype(np.float64))

        if detection is None:
            return None, table_plane

        obs = segment_object_points(detection, cloud, (h, w))
        return obs, table_plane


# ============================================================================
# Stage 5/9 — staged manipulation with closed-loop correction + verification
# ============================================================================

class Stage(str, Enum):
    OBSERVE = "observe"
    APPROACH_PREGRASP = "approach_pregrasp"
    ALIGN_WRIST = "align_wrist"
    DESCEND = "descend"
    CLOSE_GRIPPER = "close_gripper"
    LIFT = "lift"
    VERIFY = "verify"
    DONE = "done"
    FAILED = "failed"


@dataclass
class SequencerConfig:
    arm: str = "right"
    control_hz: float = 15.0           # IK/motion loop rate (10-30 Hz)
    perception_period_s: float = 0.8   # VLM grounding is far slower than the
                                        # control loop; re-observe on this
                                        # cadence and hold the last known
                                        # object pose between updates.
    stage_timeout_s: float = 12.0
    max_retries: int = 2


class BottleGraspSequencer:
    def __init__(
        self,
        robot: Robot,
        arm_sdk: ArmSdk,
        perception: PerceptionPipeline,
        safety: SafetyFilter,
        policy: VLAPolicy,
        geometry: GraspGeometry,
        config: SequencerConfig,
    ) -> None:
        self.robot = robot
        self.arm_sdk = arm_sdk
        self.perception = perception
        self.safety = safety
        self.policy = policy
        self.geometry = geometry
        self.config = config

    def _current_ee_pose(self) -> np.ndarray:
        return self.arm_sdk.get_ee_pose(self.config.arm)

    def _step_toward(self, T_target: np.ndarray, object_obs: Optional[ObjectObservation], instruction: str) -> bool:
        """One closed-loop tick: policy -> safety clamp/check -> IK."""
        T_cur = self._current_ee_pose()
        proprio = Proprioception(T_base_ee=T_cur, gripper_open=True)
        obs = Observation(
            object_obs=object_obs, proprio=proprio,
            T_base_keyframe_target=T_target, instruction=instruction,
        )
        action = self.policy.predict(obs)
        delta = self.safety.clamp_step(action.delta_pose)

        T_next = _apply_pose_increment(T_cur, delta)
        problems = self.safety.check_target(T_next, self.config.arm)
        if problems:
            logger.warning("%s: safety filter rejected step: %s", instruction, "; ".join(problems))
            return action.converged

        info = self.arm_sdk.ik_move_EE(
            delta.tolist(), arm=self.config.arm, mirror=False,
            ramp_speed_rad_s=0.4, ramp_rate_hz=50.0,
        )
        if not info.get("success"):
            logger.debug("%s: IK step did not converge: %s", instruction, info)
        return action.converged

    def _run_stage_to_target(self, object_name: str, target_key: str, instruction: str) -> bool:
        deadline = time.monotonic() + self.config.stage_timeout_s
        next_perceive = 0.0
        obs_cache: Optional[ObjectObservation] = None
        dt = 1.0 / self.config.control_hz
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_perceive:
                obs_cache, table_plane = self.perception.observe(object_name)
                self.safety.set_table_plane(table_plane)
                next_perceive = now + self.config.perception_period_s
                if obs_cache is None:
                    logger.warning("%s: %s not visible this update", instruction, object_name)
            if obs_cache is None:
                time.sleep(dt)
                continue

            T_target = build_grasp_targets(
                obs_cache, self._current_ee_pose()[:3, 3], self.geometry,
            )[target_key]
            if self._step_toward(T_target, obs_cache, instruction):
                return True
            time.sleep(dt)
        return False

    def run(self, object_name: str = "bottle") -> Dict[str, Any]:
        result: Dict[str, Any] = {"object": object_name, "stage": Stage.OBSERVE.value, "ok": False}
        attempt = 0
        while attempt <= self.config.max_retries:
            attempt += 1
            obs, table_plane = self.perception.observe(object_name)
            self.safety.set_table_plane(table_plane)
            if obs is None:
                result.update(stage=Stage.OBSERVE.value, error=f"{object_name} not detected")
                continue

            ee_pos = self._current_ee_pose()[:3, 3]
            targets = build_grasp_targets(obs, ee_pos, self.geometry)
            problems = self.safety.check_target(targets["T_base_grasp"], self.config.arm)
            if problems:
                result.update(stage=Stage.OBSERVE.value, error=f"grasp target unreachable: {'; '.join(problems)}")
                continue

            result["stage"] = Stage.APPROACH_PREGRASP.value
            if not self._run_stage_to_target(object_name, "T_base_pregrasp", "approach_pregrasp"):
                result["error"] = "approach_pregrasp stage timed out"
                continue

            # Re-check wrist alignment against the pregrasp pose before the
            # blind descend below — usually converges immediately since the
            # previous stage already tracked position and orientation together.
            result["stage"] = Stage.ALIGN_WRIST.value
            if not self._run_stage_to_target(object_name, "T_base_pregrasp", "align_wrist"):
                result["error"] = "align_wrist stage timed out"
                continue

            result["stage"] = Stage.DESCEND.value
            if not self._run_stage_to_target(object_name, "T_base_grasp", "descend"):
                result["error"] = "descend stage timed out"
                continue

            result["stage"] = Stage.CLOSE_GRIPPER.value
            self.robot.hand_close(self.config.arm, hold_s=0.8)
            time.sleep(0.5)

            result["stage"] = Stage.LIFT.value
            self._lift()

            result["stage"] = Stage.VERIFY.value
            if self._verify_grasp(object_name):
                result.update(ok=True, stage=Stage.DONE.value, attempts=attempt)
                return result

            logger.info("Grasp verification failed on attempt %d; re-localizing and retrying.", attempt)
            self.robot.hand_open(self.config.arm, hold_s=0.6)

        result["stage"] = Stage.FAILED.value
        result["attempts"] = attempt
        return result

    def _lift(self) -> None:
        T_lift = self._current_ee_pose()
        T_lift[:3, 3] = T_lift[:3, 3] + np.array([0.0, 0.0, self.geometry.lift_height_m])
        dt = 1.0 / self.config.control_hz
        for _ in range(int(self.config.control_hz * 2)):
            T_cur = self._current_ee_pose()
            pos_err = T_lift[:3, 3] - T_cur[:3, 3]
            if float(np.linalg.norm(pos_err)) < 0.015:
                break
            delta = self.safety.clamp_step(np.concatenate([pos_err, np.zeros(3)]))
            self.arm_sdk.ik_move_EE(delta.tolist(), arm=self.config.arm, mirror=False)
            time.sleep(dt)

    def _verify_grasp(self, object_name: str) -> bool:
        """Lift-and-check: tactile pressure and/or object no longer on table."""
        holding_by_touch = False
        try:
            pressures = self.robot.get_tactile_pressures(self.config.arm)
            if pressures:
                flat = [v for row in pressures for v in row if isinstance(v, (int, float))]
                holding_by_touch = bool(flat) and max(flat) > 0.05  # CALIBRATE: sensor-specific threshold
        except Exception as exc:
            logger.debug("Tactile readback unavailable: %s", exc)

        still_on_table = False
        try:
            still_on_table = float(self.robot.detect(object_name)) > 0.5
        except Exception as exc:
            logger.debug("Re-detection unavailable: %s", exc)

        return holding_by_touch or not still_on_table


# ============================================================================
# Top-level pipeline + CLI
# ============================================================================

class VLAGraspPipeline:
    def __init__(
        self,
        *,
        iface: str = "eth0",
        domain_id: int = 0,
        arm: str = "right",
        dry_run: bool = False,
        rgbd_host: Optional[str] = None,
        rgbd_port: Optional[int] = None,
        vision_model: Optional[str] = None,
        calib: Optional[RobotCalibration] = None,
        geometry: Optional[GraspGeometry] = None,
        safety_limits: Optional[SafetyLimits] = None,
        sequencer_config: Optional[SequencerConfig] = None,
    ) -> None:
        self.dry_run = dry_run
        self.arm = arm
        self.calib = calib or RobotCalibration()
        self.geometry = geometry or GraspGeometry()
        self.safety = SafetyFilter(safety_limits or SafetyLimits())
        self.policy: VLAPolicy = GeometricKeyframePolicy()
        self.sequencer_config = sequencer_config or SequencerConfig(arm=arm)

        self.robot: Optional[Robot] = None
        self.arm_sdk: Optional[ArmSdk] = None
        self.perception: Optional[PerceptionPipeline] = None
        self.sequencer: Optional[BottleGraspSequencer] = None
        if dry_run:
            return

        robot_kwargs: Dict[str, Any] = {"iface": iface, "domain_id": domain_id, "auto_start_sensors": True}
        if rgbd_host:
            robot_kwargs["rgbd_host"] = rgbd_host
        if rgbd_port:
            robot_kwargs["rgbd_port"] = rgbd_port
        if vision_model:
            robot_kwargs["vision_model"] = vision_model

        self.robot = Robot(**robot_kwargs)
        self.arm_sdk = ArmSdk(iface=iface, domain_id=domain_id)
        self.arm_sdk.resync()
        self.perception = PerceptionPipeline(self.robot, self.calib, vision_model=vision_model)
        self.sequencer = BottleGraspSequencer(
            self.robot, self.arm_sdk, self.perception, self.safety,
            self.policy, self.geometry, self.sequencer_config,
        )

    def grasp(self, object_name: str = "bottle") -> Dict[str, Any]:
        if self.dry_run:
            return {"ok": True, "dry_run": True, "object": object_name, "note": "no robot attached"}
        assert self.robot is not None and self.sequencer is not None
        logger.info("Handing rt/arm_sdk authority for the %s arm", self.arm)
        self.robot.unrelease_arms()
        try:
            return self.sequencer.run(object_name)
        finally:
            self.robot.hand_open(self.arm, hold_s=0.4)
            self.robot.release_arms()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G1 vision-language-action grasp pipeline.")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--object", default="bottle")
    parser.add_argument("--rgbd-host", default=None)
    parser.add_argument("--rgbd-port", type=int, default=None)
    parser.add_argument("--vision-model", default=None)
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build the pipeline without touching DDS/hardware.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    _configure_logging(args.log_level)

    pipeline = VLAGraspPipeline(
        iface=args.iface,
        domain_id=args.domain_id,
        arm=args.arm,
        dry_run=args.dry_run,
        rgbd_host=args.rgbd_host,
        rgbd_port=args.rgbd_port,
        vision_model=args.vision_model,
        sequencer_config=SequencerConfig(
            arm=args.arm, control_hz=args.control_hz, max_retries=args.max_retries,
        ),
    )
    try:
        result = pipeline.grasp(args.object)
    except KeyboardInterrupt:
        logger.warning("Interrupted; releasing arm authority.")
        if pipeline.robot is not None:
            pipeline.robot.release_arms()
        return 130

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
