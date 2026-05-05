"""
Step 2 — Detect target pose in RGB-D
======================================
Estimates T_camera_object from the combined RGB + depth stream provided by
the G1 Robot SDK (robot.get_rgbd()) or directly from a pyrealsense2 pipeline.

Detection strategies (selectable via ``method`` parameter):
    "aruco"  — ArUco marker (reliable, requires printed marker)
    "color"  — HSV colour-blob centroid (tunable, no special target needed)
    "center" — Fixed image-centre (useful for teleoperation / debugging)

The output is a 4×4 homogeneous transform T expressed in the
``camera_color_optical_frame`` coordinate system (Z forward, X right, Y down).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    fx: float = 615.0
    fy: float = 615.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480

    def deproject(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """Back-project pixel (u,v) + depth to 3-D point in camera frame."""
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        return np.array([x, y, depth_m], dtype=np.float64)


@dataclass
class DetectionResult:
    T_camera_object: np.ndarray          # 4×4 homogeneous transform
    confidence: float = 1.0
    method: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    pixel_uv: Optional[Tuple[float, float]] = None
    depth_m: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return self.T_camera_object[:3, 3]

    @property
    def rotation(self) -> np.ndarray:
        return self.T_camera_object[:3, :3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_T(position: np.ndarray, R: Optional[np.ndarray] = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = position
    return T


def _median_depth_roi(depth_m: np.ndarray, u: int, v: int, half: int = 5) -> float:
    """Median depth in a small ROI around (u,v) — robust to holes."""
    h, w = depth_m.shape[:2]
    u0, u1 = max(0, u - half), min(w, u + half + 1)
    v0, v1 = max(0, v - half), min(h, v + half + 1)
    roi = depth_m[v0:v1, u0:u1]
    valid = roi[roi > 0.05]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class TargetDetector:
    """
    Step 2: Estimate T_camera_object from RGB + depth.

    Args:
        method:      "aruco" | "color" | "center"
        intrinsics:  CameraIntrinsics (leave None for 640×480 RealSense defaults)
        aruco_dict:  cv2.aruco dict constant (default DICT_4X4_50)
        aruco_id:    which marker ID to track (default 0)
        hsv_lower:   lower HSV bound for color-blob detection
        hsv_upper:   upper HSV bound for color-blob detection
        min_area_px: ignore blobs smaller than this
    """

    def __init__(
        self,
        method: str = "aruco",
        intrinsics: Optional[CameraIntrinsics] = None,
        aruco_dict: int = cv2.aruco.DICT_4X4_50,
        aruco_id: int = 0,
        marker_size_m: float = 0.05,
        hsv_lower: Tuple[int, int, int] = (100, 150, 50),
        hsv_upper: Tuple[int, int, int] = (130, 255, 255),
        min_area_px: int = 500,
    ) -> None:
        self.method = method
        self.K = intrinsics or CameraIntrinsics()
        self.aruco_id = aruco_id
        self.marker_size_m = marker_size_m
        self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)
        self.min_area_px = min_area_px

        # ArUco detector (OpenCV 4.7+)
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

        # Camera matrix and dist coeffs for solvePnP
        self._cam_mat = np.array([
            [self.K.fx, 0, self.K.cx],
            [0, self.K.fy, self.K.cy],
            [0, 0, 1],
        ], dtype=np.float64)
        self._dist = np.zeros((4,), dtype=np.float64)

    # ------------------------------------------------------------------
    def detect(
        self,
        rgb_bgr: np.ndarray,
        depth_m: np.ndarray,
    ) -> Optional[DetectionResult]:
        """
        Run detection on a single RGB+depth frame pair.

        Args:
            rgb_bgr:  H×W×3 BGR image (uint8)
            depth_m:  H×W float32/64 depth in metres (0 = invalid)

        Returns:
            DetectionResult or None if no target found.
        """
        if self.method == "aruco":
            return self._detect_aruco(rgb_bgr, depth_m)
        elif self.method == "color":
            return self._detect_color_blob(rgb_bgr, depth_m)
        elif self.method == "center":
            return self._detect_center(rgb_bgr, depth_m)
        else:
            raise ValueError(f"Unknown detection method: {self.method!r}")

    # ------------------------------------------------------------------
    # ArUco-marker detection (solvePnP gives full 6-DOF pose)
    # ------------------------------------------------------------------

    def _detect_aruco(
        self, rgb_bgr: np.ndarray, depth_m: np.ndarray
    ) -> Optional[DetectionResult]:
        gray = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)

        if ids is None:
            return None

        for idx, marker_id in enumerate(ids.flatten()):
            if marker_id != self.aruco_id:
                continue

            s = self.marker_size_m / 2.0
            obj_pts = np.array([
                [-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]
            ], dtype=np.float64)

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, corners[idx].reshape(4, 2),
                self._cam_mat, self._dist,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok:
                continue

            R, _ = cv2.Rodrigues(rvec)
            T = _make_T(tvec.flatten(), R)

            cx_px = float(corners[idx][0, :, 0].mean())
            cy_px = float(corners[idx][0, :, 1].mean())
            d = _median_depth_roi(depth_m, int(cx_px), int(cy_px))

            return DetectionResult(
                T_camera_object=T,
                confidence=0.95,
                method="aruco",
                pixel_uv=(cx_px, cy_px),
                depth_m=d,
            )
        return None

    # ------------------------------------------------------------------
    # HSV colour-blob detection (centroid only — orientation is identity)
    # ------------------------------------------------------------------

    def _detect_color_blob(
        self, rgb_bgr: np.ndarray, depth_m: np.ndarray
    ) -> Optional[DetectionResult]:
        hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        largest = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_area_px:
            return None

        M = cv2.moments(largest)
        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]

        d = _median_depth_roi(depth_m, int(u), int(v))
        if d <= 0.0:
            return None

        pos = self.K.deproject(u, v, d)
        return DetectionResult(
            T_camera_object=_make_T(pos),
            confidence=0.7,
            method="color",
            pixel_uv=(u, v),
            depth_m=d,
        )

    # ------------------------------------------------------------------
    # Image-centre fallback — useful for teleoperation debugging
    # ------------------------------------------------------------------

    def _detect_center(
        self, rgb_bgr: np.ndarray, depth_m: np.ndarray
    ) -> Optional[DetectionResult]:
        u, v = self.K.cx, self.K.cy
        d = _median_depth_roi(depth_m, int(u), int(v), half=10)
        if d <= 0.0:
            return None
        pos = self.K.deproject(u, v, d)
        return DetectionResult(
            T_camera_object=_make_T(pos),
            confidence=0.5,
            method="center",
            pixel_uv=(u, v),
            depth_m=d,
        )

    # ------------------------------------------------------------------
    # Convenience: update intrinsics from a pyrealsense2 profile
    # ------------------------------------------------------------------

    @staticmethod
    def intrinsics_from_rs_profile(profile) -> CameraIntrinsics:  # type: ignore[valid-type]
        """Extract CameraIntrinsics from a pyrealsense2 video stream profile."""
        try:
            intr = profile.as_video_stream_profile().get_intrinsics()
            return CameraIntrinsics(
                fx=intr.fx, fy=intr.fy,
                cx=intr.ppx, cy=intr.ppy,
                width=intr.width, height=intr.height,
            )
        except Exception:
            return CameraIntrinsics()
