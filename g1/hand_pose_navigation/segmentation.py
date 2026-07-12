"""
Lightweight, model-free segmentation and pose fallback for untagged objects.
================================================================================
Companion to target_detector.py's ArUco/color/center methods. For objects
that were found by an open-vocabulary 2D box detector (vision_detector.py)
rather than a printed tag, we don't have a learned mask or a pose model —
instead we cut a mask out of the depth image around the box's own depth
(the "object" is whatever in the box sits near its own median depth, the
background/table beyond it is not) and estimate an oriented pose from the
resulting 3-D point cloud via PCA. No extra model, no GPU.

This intentionally trades accuracy for simplicity: it is a reasonable
top-down tabletop-pick approximation, not a 6-DoF pose network. Prefer an
ArUco tag (target_detector.py) wherever millimeter accuracy matters.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .target_detector import CameraIntrinsics, DetectionResult, _make_T


# ---------------------------------------------------------------------------
# Depth-cut mask
# ---------------------------------------------------------------------------

def mask_from_box_depth(
    depth_m: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    depth_tol_m: float = 0.05,
) -> np.ndarray:
    """Boolean H×W mask: pixels inside the box whose depth is close to the
    box's own median depth (i.e. the foreground object, not the table/wall
    behind it).

    Returns an all-False mask if the box has no valid depth.
    """
    h, w = depth_m.shape[:2]
    x0, y0, x1, y1 = box_xyxy
    x0 = max(0, min(w - 1, int(x0)))
    x1 = max(0, min(w, int(x1)))
    y0 = max(0, min(h - 1, int(y0)))
    y1 = max(0, min(h, int(y1)))
    mask = np.zeros((h, w), dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return mask

    roi = depth_m[y0:y1, x0:x1]
    valid = roi[roi > 0.05]
    if valid.size == 0:
        return mask

    median_d = float(np.median(valid))
    roi_mask = (roi > 0.05) & (np.abs(roi - median_d) <= depth_tol_m)
    mask[y0:y1, x0:x1] = roi_mask
    return mask


# ---------------------------------------------------------------------------
# PCA pose from a mask
# ---------------------------------------------------------------------------

def pose_from_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    K: CameraIntrinsics,
    min_points: int = 30,
) -> Optional[DetectionResult]:
    """Back-project the masked pixels to camera-frame 3-D points and fit an
    oriented pose via PCA.

    Position = point-cloud centroid. Orientation: largest-variance axis is
    treated as the object's long/lateral axis, smallest-variance axis as its
    local "up" (surface normal for a roughly flat top). This is a coarse
    approximation intended for a top-down or side pinch grasp, not a precise
    6-DoF estimate.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < min_points:
        return None

    depths = depth_m[ys, xs]
    valid = depths > 0.05
    xs, ys, depths = xs[valid], ys[valid], depths[valid]
    if xs.size < min_points:
        return None

    X = (xs.astype(np.float64) - K.cx) * depths / K.fx
    Y = (ys.astype(np.float64) - K.cy) * depths / K.fy
    Z = depths.astype(np.float64)
    pts = np.stack([X, Y, Z], axis=1)

    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # PCA via SVD, most to least variance.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    e0, e1, e2 = vt[0], vt[1], vt[2]

    # Orient e2 ("up"/normal) to point back toward the camera, not away.
    if np.dot(e2, centroid) > 0:
        e2 = -e2
    # Re-derive e1 to keep a strict right-handed, orthonormal frame.
    e1 = np.cross(e2, e0)
    e1 /= max(1e-9, np.linalg.norm(e1))
    e0 = np.cross(e1, e2)
    e0 /= max(1e-9, np.linalg.norm(e0))

    R = np.column_stack([e0, e1, e2])
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1

    d_center = float(np.median(depths))
    u_center = float(xs.mean())
    v_center = float(ys.mean())

    return DetectionResult(
        T_camera_object=_make_T(centroid, R),
        confidence=0.6,
        method="mask_pca",
        pixel_uv=(u_center, v_center),
        depth_m=d_center,
    )


# ---------------------------------------------------------------------------
# Drawing helpers for the recognition UI panels
# ---------------------------------------------------------------------------

_PALETTE: List[Tuple[int, int, int]] = [
    (66, 135, 245), (245, 130, 49), (60, 180, 75), (230, 25, 75),
    (145, 30, 180), (255, 225, 25), (70, 240, 240), (240, 50, 230),
]


def _color_for(index: int) -> Tuple[int, int, int]:
    return _PALETTE[index % len(_PALETTE)]


def draw_detection_boxes(
    rgb_bgr: np.ndarray,
    detections: List[Dict],
) -> np.ndarray:
    """Draw labeled bounding rectangles.

    Each entry in `detections` is a dict with keys "box" (x0,y0,x1,y1),
    "label" (str), and optionally "score" (float) and "selected" (bool).
    """
    out = rgb_bgr.copy()
    for i, det in enumerate(detections):
        x0, y0, x1, y1 = (int(v) for v in det["box"])
        color = _color_for(i)
        thickness = 3 if det.get("selected") else 2
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
        label = str(det.get("label", "?"))
        score = det.get("score")
        text = f"{label} {score:.2f}" if score is not None else label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty0 = max(0, y0 - th - 6)
        cv2.rectangle(out, (x0, ty0), (x0 + tw + 6, ty0 + th + 6), color, -1)
        cv2.putText(
            out, text, (x0 + 3, ty0 + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def draw_segmentation_overlay(
    rgb_bgr: np.ndarray,
    masks: List[Tuple[str, np.ndarray]],
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend each (label, boolean mask) as a semi-transparent color region."""
    out = rgb_bgr.copy().astype(np.float32)
    for i, (_label, mask) in enumerate(masks):
        color = np.array(_color_for(i), dtype=np.float32)
        out[mask] = (1 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_aruco_overlay(
    rgb_bgr: np.ndarray,
    tags: Dict[int, "DetectionResult"],
    K: CameraIntrinsics,
    axis_len_m: float = 0.03,
) -> np.ndarray:
    """Draw a small 3-axis gizmo + id label at each detected tag's pixel."""
    out = rgb_bgr.copy()
    cam_mat = np.array(
        [[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1]], dtype=np.float64
    )
    dist = np.zeros((4,), dtype=np.float64)
    for marker_id, result in tags.items():
        R = result.rotation
        t = result.position
        rvec, _ = cv2.Rodrigues(R)
        try:
            cv2.drawFrameAxes(out, cam_mat, dist, rvec, t.reshape(3, 1), axis_len_m)
        except Exception:
            pass
        if result.pixel_uv is not None:
            u, v = int(result.pixel_uv[0]), int(result.pixel_uv[1])
            cv2.putText(
                out, f"id={marker_id}", (u + 8, v - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA,
            )
    return out
