#!/usr/bin/env python3
"""
Recognition layer — sensors-to-UI web app for the G1 + Dex3 grasp pipeline.

Shows the RGB-D feed, a segmentation overlay, a labeled-detections overlay
(open-vocabulary vision model, NL-prompted), and an ArUco tag overlay
(object tags + the optional Dex3 hand tag) side by side. Pick a detected
object from the list, hit Grab, and direct_nav.DirectHandPoseNav drives the
arm to it (no ROS 2) and closes the hand. Release Arms and Damp are also
exposed as one-click safety controls (sdk_client.Robot.release_arms() /
.damp()).

The ROS 2/TF backend (hand_pose_nav_node.HandPoseNavNode via grab_ros2.py)
was removed from this app: it was never exercised past the initial scaffold
and its camera extrinsic was disconnected from the one calibrated/drawn here
(see camera_tf_publisher.py's own hardcoded default). Use
run_hand_pose_nav.py --use-ros-tf directly if you need that pipeline.

Run:
    python3 recognition_app.py [--iface eth0] [--domain-id 0] [--mock]
Then open http://<host>:8060 in a browser on the same network.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ── path setup ────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_G1_DIR = os.path.abspath(os.path.join(_DIR, ".."))
for _p in (_G1_DIR, os.path.join(_G1_DIR, "modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hand_pose_navigation.target_detector import (
    TargetDetector, CameraIntrinsics, DetectionResult,
)
from hand_pose_navigation import aruco_assets
from hand_pose_navigation.segmentation import (
    mask_from_box_depth, pose_from_mask,
    draw_detection_boxes, draw_segmentation_overlay, draw_aruco_overlay,
)
from hand_pose_navigation.vision_detector import VisionDetector
from hand_pose_navigation.direct_nav import DirectHandPoseNav, _make_transform
from hand_pose_navigation.grasp_planner import GraspPlanner
from hand_pose_navigation.reachability_checker import ReachabilityChecker
from hand_pose_navigation.arm_fk import ArmFK
from hand_pose_navigation.arm_ik import ArmIK
from hand_pose_navigation.arm_executor import ArmExecutor
from hand_pose_navigation.arm_fk import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, JOINT_LIMITS

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, ALL
import dash_bootstrap_components as dbc

try:
    from sdk_client import Robot
    _ROBOT_AVAILABLE = True
except ImportError:
    Robot = None
    _ROBOT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    id: str
    label: str
    source: str  # "aruco" | "vision"
    score: float
    T_camera_object: np.ndarray
    box: Optional[Tuple[int, int, int, int]] = None
    marker_id: Optional[int] = None
    role: str = "object"  # "object" | "hand" (aruco only)


# Unitree G1 depth-camera drawing:
# optical ray is 42 deg from vertical -> 48 deg below horizontal.
_UNITREE_CAMERA_X_M = 47.64571478 / 1000.0
_UNITREE_CAMERA_Z_M = 462.68178553 / 1000.0
_UNITREE_CAMERA_DOWN_PITCH_RAD = math.radians(90.0 - 42.0)
DEFAULT_CAMERA_EXTRINSIC = {
    "x": _UNITREE_CAMERA_X_M,
    "y": 0.0,
    "z": _UNITREE_CAMERA_Z_M,
    "roll": -math.pi / 2.0 - _UNITREE_CAMERA_DOWN_PITCH_RAD,
    "pitch": 0.0,
    "yaw": -math.pi / 2.0,
}
DEFAULT_MAX_JOINT_STEP_RAD = 0.05
ABSOLUTE_MAX_JOINT_STEP_RAD = 0.10
DEFAULT_MAX_JOINT_SPEED_RAD_S = 0.15
ABSOLUTE_MAX_JOINT_SPEED_RAD_S = 0.60
BODY_JOINT_INDEX_BY_LABEL = {
    "left_arm.shoulder_pitch": 15,
    "left_arm.shoulder_roll": 16,
    "left_arm.shoulder_yaw": 17,
    "left_arm.elbow": 18,
    "left_arm.wrist_roll": 19,
    "left_arm.wrist_pitch": 20,
    "left_arm.wrist_yaw": 21,
    "right_arm.shoulder_pitch": 22,
    "right_arm.shoulder_roll": 23,
    "right_arm.shoulder_yaw": 24,
    "right_arm.elbow": 25,
    "right_arm.wrist_roll": 26,
    "right_arm.wrist_pitch": 27,
    "right_arm.wrist_yaw": 28,
}


def _copy_detection(det: Detection) -> Detection:
    return Detection(
        id=det.id,
        label=det.label,
        source=det.source,
        score=float(det.score),
        T_camera_object=det.T_camera_object.copy(),
        box=None if det.box is None else tuple(det.box),
        marker_id=det.marker_id,
        role=det.role,
    )


def _joint_entry_index(label: str, data: Dict[str, Any]) -> int:
    if "index" in data:
        try:
            return int(data.get("index", -1))
        except Exception:
            return -1
    return int(BODY_JOINT_INDEX_BY_LABEL.get(str(label), -1))


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.rgb_bgr: Optional[np.ndarray] = None
        self.depth_m: Optional[np.ndarray] = None
        self.detection_rgb_bgr: Optional[np.ndarray] = None
        self.detection_depth_m: Optional[np.ndarray] = None
        self.detections: Dict[str, Detection] = {}
        self.boxes_for_draw: List[Dict] = []
        self.masks_for_draw: List[Tuple[str, str, np.ndarray]] = []
        self.tags_for_draw: Dict[int, DetectionResult] = {}
        self.hand_tag_T_camera: Optional[np.ndarray] = None
        self.hand_tag_ts: float = 0.0
        self.max_visible_detections = 5
        self.max_joint_step_rad = DEFAULT_MAX_JOINT_STEP_RAD
        self.max_joint_speed_rad_s = DEFAULT_MAX_JOINT_SPEED_RAD_S
        self.selected_id: Optional[str] = None
        self.camera_extrinsic = dict(DEFAULT_CAMERA_EXTRINSIC)
        self.camera_tf_status = "camera TF: Unitree drawing fallback"
        self.use_ros_camera_tf = False
        self.use_aruco = True
        self.arm_override = "auto"
        self.auto_step_base = False
        self.vision_classes: List[str] = []
        self.status_msg = "starting…"
        self.grab_log: List[str] = []
        self.grab_running = False
        self.arm_motion_running = False
        self.arm_motion_label = ""
        self.last_frame_ts: float = 0.0
        self.last_detection_ts: float = 0.0
        self.frame_seq: int = 0
        self.hand_fk_base: Dict[str, np.ndarray] = {}
        self.hand_fk_T_base: Dict[str, np.ndarray] = {}
        self.hand_fk_ts: float = 0.0
        self.hand_fk_status: str = "FK hand: starting"


STATE = SharedState()
ARM_CONTROL_LOCK = threading.Lock()
ARM_CANCEL_EVENT = threading.Event()
ACTIVE_NAV_LOCK = threading.Lock()
ACTIVE_DIRECT_NAV = None
BASE_K = CameraIntrinsics()  # default 640x480 RealSense-ish intrinsics
K = CameraIntrinsics()
KNOWN_INTRINSICS = {
    (424, 240): CameraIntrinsics(
        fx=302.1,
        fy=302.5,
        cx=214.3,
        cy=121.4,
        width=424,
        height=240,
    ),
    (640, 480): CameraIntrinsics(
        fx=604.2,
        fy=605.1,
        cx=324.7,
        cy=242.9,
        width=640,
        height=480,
    ),
}

# The object list must reflect what the robot can see *right now*, not the
# last thing it happened to see. If the perception loop hasn't produced a
# fresh frame+detection pass within this window, the UI treats the list as
# stale rather than silently keeping possibly-gone objects selectable.
STALE_AFTER_S = 2.0
DEFAULT_GRASP_STANDOFF_M = 0.08
DEFAULT_MAX_BASE_STEP_M = 0.30
DEFAULT_REACH_MARGIN_M = 0.04
DEFAULT_DIRECT_MAX_REACH_M = 0.52
DEFAULT_SLOW_GRAB_RATE_HZ = 3.0
DEFAULT_SLOW_GRAB_TIMEOUT_S = 180.0
DEFAULT_EXTEND_STAGE_DURATION_S = 1.2


class _MockRobot:
    """Frame source + no-op safety calls, used with --mock."""

    def get_rgbd(self, timeout: float = 2.0) -> Dict[str, Any]:
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(rgb, "MOCK CAMERA", (160, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
        depth = np.full((480, 640), 0.8, dtype=np.float32)
        return {"rgb_bgr": rgb, "depth_m": depth}

    def release_arms(self, **kw) -> Dict:
        time.sleep(0.5)
        return {"mock": True}

    def unrelease_arms(self, **kw) -> Dict:
        time.sleep(0.2)
        return {"mock": True}

    def damp(self) -> None:
        pass


class _ZmqRgbdRobot:
    """Persistent RGB-D ZMQ frame source used by the recognition UI."""

    def __init__(self, host: str, port: int, topic: str = "") -> None:
        import zmq  # type: ignore

        self.host = str(host)
        self.port = int(port)
        self.topic = str(topic)
        self._zmq = zmq
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, self.topic.encode("utf-8"))
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.setsockopt(zmq.RCVHWM, 2)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def get_rgbd(self, timeout: float = 2.0) -> Dict[str, Any]:
        import struct

        deadline = time.time() + max(0.2, float(timeout))
        last_error = ""
        while time.time() < deadline:
            try:
                parts = self._socket.recv_multipart()
                # Detection can be slower than the RGB-D publisher. Drain any
                # queued multipart frames so the UI works from the newest one.
                while True:
                    try:
                        parts = self._socket.recv_multipart(flags=self._zmq.NOBLOCK)
                    except self._zmq.Again:
                        break
            except self._zmq.Again:
                continue
            except Exception as exc:
                last_error = str(exc)
                continue
            if len(parts) >= 4:
                parts = parts[-3:]
            if len(parts) < 2:
                last_error = f"expected RGBD multipart frame, got {len(parts)} part(s)"
                continue

            rgb_jpeg = bytes(parts[0])
            depth_png = bytes(parts[1])
            depth_scale = 0.001
            if len(parts) >= 3 and len(parts[2]) >= 4:
                try:
                    depth_scale = float(struct.unpack("f", parts[2][:4])[0])
                except Exception:
                    depth_scale = 0.001

            rgb = cv2.imdecode(np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            depth_raw = cv2.imdecode(np.frombuffer(depth_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if rgb is None:
                last_error = "failed to decode RGB JPEG"
                continue
            if depth_raw is None:
                last_error = "failed to decode depth PNG"
                continue
            if depth_raw.ndim == 3:
                depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
            depth_m = depth_raw.astype("float32") * float(depth_scale)
            return {
                "source": f"zmq://{self.host}:{self.port}",
                "rgb_jpeg": rgb_jpeg,
                "depth_png": depth_png,
                "rgb_bgr": rgb,
                "rgb_rgb": cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
                "depth_raw": depth_raw,
                "depth_m": depth_m,
                "depth_scale_m_per_unit": float(depth_scale),
                "valid_depth_fraction": float((depth_raw > 0).mean()) if depth_raw.size else 0.0,
            }

        detail = f" Last error: {last_error}" if last_error else ""
        raise RuntimeError(
            f"No RGBD frames received from tcp://{self.host}:{self.port} "
            f"within {timeout:.1f}s.{detail}"
        )

    def release_arms(self, **kw) -> Dict:
        return {"rgbd_only": True, "message": "release_arms unavailable in RGBD-only UI mode"}

    def unrelease_arms(self, **kw) -> Dict:
        return {"rgbd_only": True, "message": "unrelease_arms unavailable in RGBD-only UI mode"}

    def damp(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Background perception loop
# ---------------------------------------------------------------------------

def _update_intrinsics_for_frame(rgb: np.ndarray, detector: TargetDetector) -> None:
    global K
    h, w = rgb.shape[:2]
    if w == K.width and h == K.height:
        return
    if (w, h) in KNOWN_INTRINSICS:
        K = KNOWN_INTRINSICS[(w, h)]
    else:
        sx = float(w) / float(BASE_K.width)
        sy = float(h) / float(BASE_K.height)
        K = CameraIntrinsics(
            fx=BASE_K.fx * sx,
            fy=BASE_K.fy * sy,
            cx=BASE_K.cx * sx,
            cy=BASE_K.cy * sy,
            width=w,
            height=h,
        )
    detector.set_intrinsics(K)


def _camera_loop(robot, detector: TargetDetector, rate_hz: float) -> None:
    period = 1.0 / max(1.0, rate_hz)
    while True:
        t0 = time.time()
        try:
            frame = robot.get_rgbd(timeout=2.0)
            rgb = frame["rgb_bgr"]
            depth = frame["depth_m"]
            _update_intrinsics_for_frame(rgb, detector)
        except Exception as exc:
            with STATE.lock:
                STATE.status_msg = f"camera error: {exc}"
            time.sleep(period)
            continue

        with STATE.lock:
            STATE.rgb_bgr = rgb
            STATE.depth_m = depth
            STATE.last_frame_ts = time.time()
            STATE.frame_seq += 1

        dt = time.time() - t0
        time.sleep(max(0.0, period - dt))


def _perception_loop(detector: TargetDetector, vision: VisionDetector, rate_hz: float) -> None:
    period = 1.0 / max(0.5, rate_hz)
    last_processed_seq = -1
    while True:
        t0 = time.time()
        with STATE.lock:
            seq = STATE.frame_seq
            rgb = None if STATE.rgb_bgr is None else STATE.rgb_bgr.copy()
            depth = None if STATE.depth_m is None else STATE.depth_m.copy()
        if rgb is None or depth is None or seq == last_processed_seq:
            time.sleep(min(0.05, period))
            continue
        last_processed_seq = seq

        _update_intrinsics_for_frame(rgb, detector)

        with STATE.lock:
            use_aruco = STATE.use_aruco
        tags = detector.detect_all_aruco(rgb, depth) if use_aruco else {}
        vis_dets = vision.detect(rgb) if vision.available else []

        detections: Dict[str, Detection] = {}
        for marker_id, result in tags.items():
            role = "hand" if marker_id == aruco_assets.HAND_MARKER_ID else "object"
            det_id = f"aruco:{marker_id}"
            detections[det_id] = Detection(
                id=det_id,
                label=f"{role} tag #{marker_id}",
                source="aruco",
                score=result.confidence,
                T_camera_object=result.T_camera_object,
                marker_id=marker_id,
                role=role,
            )

        boxes_for_draw: List[Dict] = []
        masks_for_draw: List[Tuple[str, str, np.ndarray]] = []
        with STATE.lock:
            selected_id = STATE.selected_id
        for i, vd in enumerate(vis_dets):
            mask = mask_from_box_depth(depth, vd.box_xyxy)
            pose = pose_from_mask(mask, depth, K)
            det_id = f"vision:{i}"
            boxes_for_draw.append({
                "id": det_id, "box": vd.box_xyxy, "label": vd.label, "score": vd.score,
                "selected": det_id == selected_id,
            })
            if pose is None:
                continue
            detections[det_id] = Detection(
                id=det_id, label=vd.label, source="vision", score=vd.score,
                T_camera_object=pose.T_camera_object, box=vd.box_xyxy,
            )
            masks_for_draw.append((det_id, vd.label, mask))

        with STATE.lock:
            STATE.detection_rgb_bgr = rgb
            STATE.detection_depth_m = depth
            STATE.detections = detections
            STATE.boxes_for_draw = boxes_for_draw
            STATE.masks_for_draw = masks_for_draw
            STATE.tags_for_draw = tags
            hand_tag = tags.get(aruco_assets.HAND_MARKER_ID)
            if hand_tag is not None:
                STATE.hand_tag_T_camera = hand_tag.T_camera_object.copy()
                STATE.hand_tag_ts = time.time()
            elif STATE.hand_tag_ts and time.time() - STATE.hand_tag_ts > STALE_AFTER_S:
                STATE.hand_tag_T_camera = None
                STATE.hand_tag_ts = 0.0
            STATE.last_detection_ts = time.time()
            n_vision = "on" if vision.available else f"off ({vision.error})"
            STATE.status_msg = (
                f"{len(detections)} detections  |  vision model: {n_vision}  |  frame #{seq}"
            )

        dt = time.time() - t0
        time.sleep(max(0.0, period - dt))


def _hand_fk_loop(iface: str, domain_id: int) -> None:
    try:
        from dds_env import ensure_channel_factory_initialized
        from sdk_sensors import LatestSubscriber, resolve_lowstate_type, lowstate_snapshot_from_msg

        ensure_channel_factory_initialized(int(domain_id), iface)
        lowstate_type = resolve_lowstate_type()
        if lowstate_type is None:
            raise RuntimeError("Could not resolve Unitree LowState_ message type.")
        sub = LatestSubscriber("rt/lowstate", lowstate_type)
        sub.start(queue_len=10)
        fk = {"left": ArmFK("left"), "right": ArmFK("right")}
    except Exception as exc:
        with STATE.lock:
            STATE.hand_fk_status = f"FK hand unavailable: {exc}"
        return

    while True:
        msg, ts = sub.get_latest()
        if msg is None:
            with STATE.lock:
                STATE.hand_fk_status = "FK hand: waiting for rt/lowstate"
            time.sleep(0.2)
            continue
        try:
            snap = lowstate_snapshot_from_msg(msg)
            q_full = np.zeros(30, dtype=np.float64)
            n = min(len(snap.joint_positions), q_full.size)
            q_full[:n] = np.asarray(snap.joint_positions[:n], dtype=np.float64)
            hand_T = {
                side: fk[side].compute(q_full).copy()
                for side in ("left", "right")
            }
            hands = {
                side: T_base_hand[:3, 3].copy()
                for side, T_base_hand in hand_T.items()
            }
            with STATE.lock:
                STATE.hand_fk_base = hands
                STATE.hand_fk_T_base = hand_T
                STATE.hand_fk_ts = float(ts or time.time())
                STATE.hand_fk_status = "FK hand: rt/lowstate"
        except Exception as exc:
            with STATE.lock:
                STATE.hand_fk_status = f"FK hand error: {exc}"
        time.sleep(0.1)


def _ros_camera_tf_loop(
    base_frame: str,
    camera_frame: str,
    timeout_s: float,
    period_s: float = 1.0,
) -> None:
    probe = os.path.join(_DIR, "ros2_camera_tf_probe.py")
    while True:
        with STATE.lock:
            enabled = STATE.use_ros_camera_tf
        if not enabled:
            time.sleep(0.2)
            continue

        cmd = [
            sys.executable,
            probe,
            "--base-frame",
            base_frame,
            "--camera-frame",
            camera_frame,
            "--timeout-s",
            str(timeout_s),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=max(1.0, timeout_s + 1.0),
            )
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip().splitlines()
                msg = detail[-1] if detail else f"exit {proc.returncode}"
                with STATE.lock:
                    STATE.camera_tf_status = f"ROS TF unavailable: {msg[:160]}"
                time.sleep(period_s)
                continue
            data = json.loads(proc.stdout)
            cam = {
                "x": float(data["x"]),
                "y": float(data["y"]),
                "z": float(data["z"]),
                "roll": float(data["roll"]),
                "pitch": float(data["pitch"]),
                "yaw": float(data["yaw"]),
            }
            age = float(data.get("age_s", 0.0))
            with STATE.lock:
                STATE.camera_extrinsic = cam
                STATE.camera_tf_status = (
                    f"camera TF: ROS 2 {base_frame}<-{camera_frame}, age={age:.2f}s"
                )
        except Exception as exc:
            with STATE.lock:
                STATE.camera_tf_status = f"ROS TF error: {exc}"
        time.sleep(period_s)


def _init_unitree_dds_once(iface: str, domain_id: int) -> str:
    try:
        from dds_env import ensure_channel_factory_initialized
        ensure_channel_factory_initialized(int(domain_id), iface)
        return f"Unitree DDS initialized: iface={iface} domain_id={domain_id}"
    except Exception as exc:
        return f"Unitree DDS init failed: {exc}"


def _make_control_robot(iface: str, domain_id: int):
    if not _ROBOT_AVAILABLE:
        return None, "sdk_client.Robot unavailable"
    try:
        robot = Robot(
            iface=iface,
            domain_id=domain_id,
            auto_start_sensors=True,
        )
        return robot, "control robot ready"
    except Exception as exc:
        return None, f"control robot unavailable: {exc!r}"


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------

def _encode_jpeg_src(bgr: np.ndarray, quality: int = 85) -> str:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    payload = base64.b64encode(enc.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _placeholder_src(text: str) -> str:
    img = np.full((240, 320, 3), 30, dtype=np.uint8)
    cv2.putText(img, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    return _encode_jpeg_src(img)


def _colorize_depth(depth_m: np.ndarray, max_m: float = 3.0) -> np.ndarray:
    clipped = np.clip(depth_m, 0.0, max_m)
    scaled = (clipped / max_m * 255.0).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def _draw_pose_axis(
    image: np.ndarray,
    T_camera_pose: np.ndarray,
    label: str,
    color: Tuple[int, int, int],
    axis_len_m: float = 0.05,
) -> bool:
    cam_mat = np.array(
        [[K.fx, 0.0, K.cx], [0.0, K.fy, K.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4,), dtype=np.float64)
    R = T_camera_pose[:3, :3]
    t = T_camera_pose[:3, 3].reshape(3, 1)
    if float(t[2, 0]) <= 0.05:
        return False
    try:
        rvec, _ = cv2.Rodrigues(R)
        cv2.drawFrameAxes(image, cam_mat, dist, rvec, t, float(axis_len_m))
        uv, _ = cv2.projectPoints(
            np.zeros((1, 3), dtype=np.float64),
            rvec,
            t,
            cam_mat,
            dist,
        )
        u, v = (int(x) for x in uv.reshape(-1)[:2])
        cv2.putText(
            image,
            label,
            (u + 8, v + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
        return True
    except Exception:
        return False


def _draw_detection_gizmos(
    image: np.ndarray,
    selected: Optional[Detection],
    cam: Dict[str, float],
    arm_override: str,
    hand_fk_T_base: Dict[str, np.ndarray],
    hand_tag_T_camera: Optional[np.ndarray],
) -> np.ndarray:
    out = image.copy()
    T_base_camera = _make_transform(
        xyz=(cam["x"], cam["y"], cam["z"]),
        rpy=(cam["roll"], cam["pitch"], cam["yaw"]),
    )
    T_camera_base = np.linalg.inv(T_base_camera)
    arm = arm_override
    if selected is not None:
        arm, _T_base_camera, _T_base_object = _resolve_arm_and_base_pose(
            selected,
            cam,
            arm_override,
        )
        _draw_pose_axis(out, selected.T_camera_object, "object", (0, 220, 255), axis_len_m=0.055)
    if arm == "auto":
        arm = "right"
    T_base_hand = hand_fk_T_base.get(arm)
    drew_hand = False
    if T_base_hand is not None:
        drew_hand = _draw_pose_axis(
            out,
            T_camera_base @ T_base_hand,
            f"{arm} hand",
            (255, 255, 80),
            axis_len_m=0.06,
        )
    if not drew_hand and hand_tag_T_camera is not None:
        _draw_pose_axis(
            out,
            hand_tag_T_camera,
            f"{arm} hand tag",
            (255, 255, 80),
            axis_len_m=0.06,
        )
    return out


def _resolve_arm_and_base_pose(
    det: Detection,
    cam: Dict[str, float],
    arm_override: str,
) -> Tuple[str, np.ndarray, np.ndarray]:
    T_base_camera = _make_transform(
        xyz=(cam["x"], cam["y"], cam["z"]),
        rpy=(cam["roll"], cam["pitch"], cam["yaw"]),
    )
    T_base_object = T_base_camera @ det.T_camera_object
    arm = arm_override
    if arm == "auto":
        arm = "left" if T_base_object[1, 3] > 0 else "right"
    return arm, T_base_camera, T_base_object


def _reach_preview(
    det: Detection,
    cam: Dict[str, float],
    arm_override: str,
) -> Tuple[str, float, float, float, float]:
    arm, _T_base_camera, T_base_object = _resolve_arm_and_base_pose(det, cam, arm_override)
    checker = ReachabilityChecker(arm=arm, max_reach_m=DEFAULT_DIRECT_MAX_REACH_M)
    T_base_desired = GraspPlanner(
        arm=arm,
        standoff_m=DEFAULT_GRASP_STANDOFF_M,
    ).compute(T_base_object)
    shoulder_y = 0.10 if arm == "left" else -0.10
    shoulder = np.array([0.0, shoulder_y, 0.292], dtype=np.float64)
    reach_dist = float(np.linalg.norm(T_base_desired[:3, 3] - shoulder))
    excess_m = max(0.0, reach_dist - checker.max_reach_m)
    suggested_step_m = min(
        max(0.0, excess_m + DEFAULT_REACH_MARGIN_M),
        DEFAULT_MAX_BASE_STEP_M,
    )
    return arm, reach_dist, checker.max_reach_m, excess_m, suggested_step_m


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "G1 Recognition Layer"


def _instructions_card() -> dbc.Card:
    hand_marker = aruco_assets.default_hand_marker()
    obj_markers = aruco_assets.default_object_markers()[:4]  # keep the page light
    marker_thumbs = [
        html.Div([
            html.Img(src=m.data_uri(), style={"width": "90px", "border": "1px solid #555"}),
            html.Div(f"id {m.marker_id} · {m.size_mm:.0f}mm · {m.role}",
                      style={"fontSize": "11px", "textAlign": "center"}),
        ], style={"display": "inline-block", "margin": "6px"})
        for m in [hand_marker] + obj_markers
    ]
    return dbc.Card(dbc.CardBody([
        html.H5("ArUco setup"),
        html.Div(marker_thumbs),
        dcc.Markdown(aruco_assets.PLACEMENT_INSTRUCTIONS, style={"fontSize": "13px"}),
        html.Small(
            "Full sheet (all object tags) is generated on the fly — "
            "increase OBJECT_MARKER_IDS in aruco_assets.py if you need more.",
            className="text-muted",
        ),
    ]), className="mb-3")


def _options_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H5("Perception options"),
        dbc.Checkbox(
            id="use-aruco",
            label="Use ArUco tags",
            value=STATE.use_aruco,
            className="mb-2",
        ),
        dbc.Checkbox(
            id="use-ros-camera-tf",
            label="Use ROS 2 camera TF",
            value=STATE.use_ros_camera_tf,
            className="mb-2",
        ),
        dbc.Label("Maximum visible detections", className="mt-2"),
        dbc.Input(
            id="max-visible-detections",
            type="number",
            min=1,
            step=1,
            value=STATE.max_visible_detections,
            className="mb-2",
        ),
        dbc.Label("Max arm joint increment per command (rad)", className="mt-2"),
        dbc.Input(
            id="max-joint-step-rad",
            type="number",
            min=0.005,
            max=ABSOLUTE_MAX_JOINT_STEP_RAD,
            step=0.005,
            value=STATE.max_joint_step_rad,
            className="mb-2",
        ),
        dbc.Label("Max arm joint speed (rad/s)", className="mt-2"),
        dbc.Input(
            id="max-joint-speed-rad-s",
            type="number",
            min=0.02,
            max=ABSOLUTE_MAX_JOINT_SPEED_RAD_S,
            step=0.01,
            value=STATE.max_joint_speed_rad_s,
            className="mb-2",
        ),
        html.Small(
            "When ROS 2 TF is enabled, the app reads base_link<-camera_color_optical_frame "
            "from a separate probe process. Otherwise it uses the built-in fixed transform.",
            className="text-muted",
        ),
        html.Div(id="camera-tf-status", className="mt-2", style={"fontSize": "12px"}),
    ]), className="mb-3")


def _image_panel(panel_id: str, title: str) -> dbc.Col:
    return dbc.Col(dbc.Card([
        dbc.CardHeader(title, style={"fontSize": "13px"}),
        html.Img(id=panel_id, style={"width": "100%"}),
    ]), width=6, className="mb-3")


def _view_panel() -> dbc.Card:
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(html.Div("Live view", style={"fontSize": "13px"}), width=3),
                dbc.Col(dcc.Dropdown(
                    id="view-select",
                    options=[
                        {"label": "RGB raw", "value": "rgb"},
                        {"label": "Depth", "value": "depth"},
                        {"label": "Vision detections", "value": "detections"},
                        {"label": "Segmentation", "value": "segmentation"},
                        {"label": "ArUco tags", "value": "aruco"},
                    ],
                    value="detections",
                    clearable=False,
                    style={"color": "#111"},
                ), width=9),
            ], align="center"),
        ]),
        dbc.CardBody(html.Img(id="panel-view", style={"width": "100%"})),
    ], className="mb-3")


def _limit_visible_detections(
    detections: Dict[str, Detection],
    boxes: List[Dict],
    masks: List[Tuple[str, str, np.ndarray]],
    tags: Dict[int, DetectionResult],
    max_count: int,
) -> Tuple[Dict[str, Detection], List[Dict], List[Tuple[str, str, np.ndarray]], Dict[int, DetectionResult]]:
    try:
        limit = max(1, int(max_count))
    except Exception:
        limit = 5
    if len(detections) <= limit:
        return detections, boxes, masks, tags

    ranked = sorted(
        detections.items(),
        key=lambda item: (-float(item[1].score), item[0]),
    )
    allowed_ids = {det_id for det_id, _det in ranked[:limit]}
    limited_detections = {
        det_id: det
        for det_id, det in detections.items()
        if det_id in allowed_ids
    }
    limited_boxes = [
        box for box in boxes
        if str(box.get("id", "")) in allowed_ids
    ]
    limited_masks = [
        mask_item for mask_item in masks
        if mask_item[0] in allowed_ids
    ]
    allowed_marker_ids = {
        det.marker_id
        for det in limited_detections.values()
        if det.source == "aruco" and det.marker_id is not None
    }
    limited_tags = {
        marker_id: result
        for marker_id, result in tags.items()
        if marker_id in allowed_marker_ids
    }
    return limited_detections, limited_boxes, limited_masks, limited_tags


app.layout = dbc.Container([
    html.H3("G1 Recognition Layer", className="mt-3"),
    html.Div(id="global-status", className="mb-2", style={"fontSize": "13px", "color": "#9c9"}),

    dbc.Row([
        dbc.Col(_instructions_card(), width=6),
        dbc.Col(_options_card(), width=6),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Label("Describe what to look for"),
            dbc.InputGroup([
                dbc.Input(id="nl-prompt", placeholder="e.g. red mug, soda can"),
                dbc.Button("Set prompt", id="set-prompt-btn", color="primary"),
            ]),
            html.Div(id="prompt-status", className="mt-1", style={"fontSize": "12px"}),
        ], width=8),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col(_view_panel(), width=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("Detections — select one"),
            html.Div(id="detection-list"),
        ], width=6),
        dbc.Col([
            html.H5("Grab"),
            dbc.RadioItems(
                id="arm-select",
                options=[
                    {"label": "Auto (pick by target side)", "value": "auto"},
                    {"label": "Left", "value": "left"},
                    {"label": "Right", "value": "right"},
                ],
                value="auto", inline=True, className="mb-2",
            ),
            dbc.Checkbox(
                id="auto-step-base",
                label="Step base closer if target is outside arm reach",
                value=False,
                className="mb-2",
            ),
            dbc.Button("Grab selected object", id="grab-btn", color="success",
                        className="mb-2", disabled=True),
            html.Div(id="grab-selected-label", className="mb-2", style={"fontSize": "13px"}),
            html.Hr(),
            dbc.ButtonGroup([
                dbc.Button("Prepare", id="prepare-btn", color="secondary"),
                dbc.Button("Walk", id="walk-btn", color="secondary"),
                dbc.Button("Stop moving", id="stop-moving-btn", color="dark"),
                dbc.Button("Extend arm", id="extend-arm-btn", color="info"),
                dbc.Button("Stop grabbing", id="stop-grabbing-btn", color="danger"),
                dbc.Button("Release arms", id="release-arms-btn", color="warning"),
                dbc.Button("Re-engage arms", id="unrelease-arms-btn", color="primary"),
                dbc.Button("Damp", id="damp-btn", color="danger"),
            ]),
            html.Div(id="safety-status", className="mt-2", style={"fontSize": "12px"}),
            html.Hr(),
            html.H6("Grab log"),
            html.Pre(id="grab-log", style={
                "maxHeight": "220px", "overflow": "auto", "fontSize": "11px",
                "background": "#111", "padding": "8px",
            }),
        ], width=6),
    ]),

    dcc.Store(id="selected-store"),
    dcc.Interval(id="refresh-interval", interval=500, n_intervals=0),
], fluid=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("panel-view", "src"),
    Output("detection-list", "children"),
    Output("global-status", "children"),
    Output("camera-tf-status", "children"),
    Output("grab-log", "children"),
    Output("grab-btn", "disabled"),
    Output("grab-selected-label", "children"),
    Input("refresh-interval", "n_intervals"),
    Input("view-select", "value"),
)
def _refresh(_n, view_name):
    with STATE.lock:
        rgb = None if STATE.rgb_bgr is None else STATE.rgb_bgr.copy()
        depth = None if STATE.depth_m is None else STATE.depth_m.copy()
        detection_rgb = None if STATE.detection_rgb_bgr is None else STATE.detection_rgb_bgr.copy()
        detection_depth = None if STATE.detection_depth_m is None else STATE.detection_depth_m.copy()
        detections = dict(STATE.detections)
        boxes = list(STATE.boxes_for_draw)
        masks = list(STATE.masks_for_draw)
        tags = dict(STATE.tags_for_draw)
        selected_id = STATE.selected_id
        status_msg = STATE.status_msg
        grab_log = list(STATE.grab_log[-100:])
        grab_running = STATE.grab_running
        arm_motion_running = STATE.arm_motion_running
        arm_motion_label = STATE.arm_motion_label
        last_frame_ts = STATE.last_frame_ts
        last_detection_ts = STATE.last_detection_ts
        cam = dict(STATE.camera_extrinsic)
        arm_override = STATE.arm_override
        hand_fk_T_base = {k: v.copy() for k, v in STATE.hand_fk_T_base.items()}
        hand_tag_T_camera = (
            None if STATE.hand_tag_T_camera is None else STATE.hand_tag_T_camera.copy()
        )
        hand_tag_ts = STATE.hand_tag_ts
        camera_tf_status = STATE.camera_tf_status
        max_visible_detections = STATE.max_visible_detections

    if rgb is None:
        ph = _placeholder_src("waiting for camera…")
        return ph, "No detections yet.", status_msg, camera_tf_status, "\n".join(grab_log), True, ""

    # A frozen/erroring camera feed must not keep offering stale objects as
    # if the robot could still see them — the list only reflects detections
    # from a genuinely recent perception pass.
    now = time.time()
    frame_is_stale = (now - last_frame_ts) > STALE_AFTER_S
    detections_are_stale = (
        last_detection_ts <= 0.0 or (now - last_detection_ts) > STALE_AFTER_S
    )
    is_stale = frame_is_stale or detections_are_stale
    if is_stale:
        detections = {}
        boxes = []
        masks = []
        tags = {}
        if frame_is_stale:
            status_msg = f"STALE — no fresh frame in >{STALE_AFTER_S:.0f}s. {status_msg}"
        else:
            status_msg = f"DETECTING — latest detection is >{STALE_AFTER_S:.0f}s old. {status_msg}"
    else:
        total_detections = len(detections)
        detections, boxes, masks, tags = _limit_visible_detections(
            detections, boxes, masks, tags, max_visible_detections,
        )
        if len(detections) < total_detections:
            status_msg = (
                f"{status_msg} | showing {len(detections)}/{total_detections} "
                f"detections"
            )
    if hand_tag_ts > 0.0 and (time.time() - hand_tag_ts) <= STALE_AFTER_S:
        camera_tf_status = f"{camera_tf_status} | hand tag visible"

    view_name = view_name or "detections"
    if view_name == "rgb":
        view_src = _encode_jpeg_src(rgb)
    elif view_name == "depth":
        view_src = _encode_jpeg_src(_colorize_depth(depth)) if depth is not None else _placeholder_src("no depth")
    elif view_name == "segmentation":
        base = detection_rgb if detection_rgb is not None else rgb
        view_src = _encode_jpeg_src(draw_segmentation_overlay(base, masks)) if masks else _placeholder_src("no masks")
    elif view_name == "aruco":
        base = detection_rgb if detection_rgb is not None else rgb
        view_src = _encode_jpeg_src(draw_aruco_overlay(base, tags, K))
    else:
        base = detection_rgb if detection_rgb is not None else rgb
        selected = detections.get(selected_id or "")
        boxed = draw_detection_boxes(base, boxes)
        view_src = _encode_jpeg_src(_draw_detection_gizmos(
            boxed,
            selected,
            cam,
            arm_override,
            hand_fk_T_base,
            hand_tag_T_camera,
        ))

    items = []
    for det_id, det in sorted(detections.items()):
        active = det_id == selected_id
        badge_color = "info" if det.source == "aruco" else "primary"
        items.append(dbc.ListGroupItem(
            [
                dbc.Badge(det.source, color=badge_color, className="me-2"),
                f"{det.label}  (score={det.score:.2f})",
            ],
            id={"type": "det-item", "index": det_id},
            action=True, active=active, n_clicks=0,
        ))
    if items:
        det_list = dbc.ListGroup(items)
    elif is_stale:
        det_list = html.Div(
            "Camera feed is stale — the robot isn't reporting fresh frames, "
            "so no objects can be confirmed as currently visible.",
            className="text-danger",
        )
    else:
        det_list = html.Div(
            "No detections. Set an NL prompt and/or stick an ArUco tag on an object.",
            className="text-muted",
        )

    grab_disabled = (selected_id not in detections) or grab_running or arm_motion_running
    selected_label = ""
    if selected_id in detections:
        det = detections[selected_id]
        with STATE.lock:
            cam = dict(STATE.camera_extrinsic)
            arm_override = STATE.arm_override
            auto_step_base = STATE.auto_step_base
        arm, reach_dist, max_reach, excess_m, suggested_step_m = _reach_preview(
            det, cam, arm_override,
        )
        selected_label = (
            f"Selected: {det.label} ({det.source}) | arm={arm} | "
            f"wrist reach={reach_dist:.3f}/{max_reach:.3f} m"
        )
        if excess_m > 0.0:
            if auto_step_base:
                selected_label += f" | base step planned ~{suggested_step_m:.3f} m"
            else:
                selected_label += (
                    f" | outside arm reach by {excess_m:.3f} m; "
                    "enable base step or move closer"
                )
        if arm_motion_running:
            selected_label += f" | arm busy: {arm_motion_label}"
    elif selected_id is not None:
        selected_label = "Selected object is no longer visible."

    return (
        view_src, det_list, status_msg, camera_tf_status,
        "\n".join(grab_log), grab_disabled, selected_label,
    )


@app.callback(
    Output("prompt-status", "children"),
    Input("set-prompt-btn", "n_clicks"),
    State("nl-prompt", "value"),
    prevent_initial_call=True,
)
def _set_prompt(_n, text):
    text = text or ""
    classes = _VISION.set_prompt(text)
    with STATE.lock:
        STATE.vision_classes = classes
    if not _VISION.available:
        return f"Vision model unavailable: {_VISION.error}"
    return f"Looking for: {', '.join(classes)}"


@app.callback(
    Output("selected-store", "data"),
    Input({"type": "det-item", "index": ALL}, "n_clicks"),
    State({"type": "det-item", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def _select_detection(_clicks, ids):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        det_id = json.loads(triggered_id)["index"]
    except Exception:
        return no_update
    with STATE.lock:
        STATE.selected_id = det_id
    return det_id


@app.callback(
    Output("arm-select", "value"),
    Input("arm-select", "value"),
)
def _sync_arm(value):
    with STATE.lock:
        STATE.arm_override = value
    return value


@app.callback(
    Output("auto-step-base", "value"),
    Input("auto-step-base", "value"),
)
def _sync_auto_step(value):
    enabled = bool(value)
    with STATE.lock:
        STATE.auto_step_base = enabled
    return enabled


@app.callback(
    Output("use-aruco", "value"),
    Input("use-aruco", "value"),
)
def _sync_use_aruco(value):
    enabled = bool(value)
    with STATE.lock:
        STATE.use_aruco = enabled
        if not enabled:
            STATE.detections = {
                det_id: det
                for det_id, det in STATE.detections.items()
                if det.source != "aruco"
            }
            STATE.tags_for_draw = {}
            if STATE.selected_id and STATE.selected_id.startswith("aruco:"):
                STATE.selected_id = None
    return enabled


@app.callback(
    Output("max-visible-detections", "value"),
    Input("max-visible-detections", "value"),
)
def _sync_max_visible_detections(value):
    try:
        limit = max(1, int(value))
    except Exception:
        limit = 5
    with STATE.lock:
        STATE.max_visible_detections = limit
    return limit


@app.callback(
    Output("max-joint-step-rad", "value"),
    Input("max-joint-step-rad", "value"),
)
def _sync_max_joint_step_rad(value):
    try:
        step = float(value)
    except Exception:
        step = DEFAULT_MAX_JOINT_STEP_RAD
    step = max(0.005, min(ABSOLUTE_MAX_JOINT_STEP_RAD, step))
    with STATE.lock:
        STATE.max_joint_step_rad = step
    return step


@app.callback(
    Output("max-joint-speed-rad-s", "value"),
    Input("max-joint-speed-rad-s", "value"),
)
def _sync_max_joint_speed_rad_s(value):
    try:
        speed = float(value)
    except Exception:
        speed = DEFAULT_MAX_JOINT_SPEED_RAD_S
    speed = max(0.02, min(ABSOLUTE_MAX_JOINT_SPEED_RAD_S, speed))
    with STATE.lock:
        STATE.max_joint_speed_rad_s = speed
    return speed


@app.callback(
    Output("use-ros-camera-tf", "value"),
    Input("use-ros-camera-tf", "value"),
)
def _sync_use_ros_camera_tf(value):
    enabled = bool(value)
    with STATE.lock:
        STATE.use_ros_camera_tf = enabled
        if enabled:
            STATE.camera_tf_status = "camera TF: waiting for ROS 2 TF"
        else:
            STATE.camera_tf_status = "camera TF: Unitree drawing fallback"
    return enabled


@app.callback(
    Output("safety-status", "children"),
    Input("grab-btn", "n_clicks"),
    Input("prepare-btn", "n_clicks"),
    Input("walk-btn", "n_clicks"),
    Input("stop-moving-btn", "n_clicks"),
    Input("extend-arm-btn", "n_clicks"),
    Input("stop-grabbing-btn", "n_clicks"),
    Input("release-arms-btn", "n_clicks"),
    Input("unrelease-arms-btn", "n_clicks"),
    Input("damp-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _buttons(
    grab_clicks,
    prepare_clicks,
    walk_clicks,
    stop_moving_clicks,
    extend_clicks,
    stop_grabbing_clicks,
    release_clicks,
    unrelease_clicks,
    damp_clicks,
):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "grab-btn":
        _log("[recognition_app] Grab button clicked.")
        threading.Thread(target=_run_grab, daemon=True).start()
        return "Grab started — see log below."
    if trigger == "prepare-btn":
        threading.Thread(target=_run_prepare, daemon=True).start()
        return "Prepare sent..."
    if trigger == "walk-btn":
        threading.Thread(target=_run_walk, daemon=True).start()
        return "Walk sent..."
    if trigger == "stop-moving-btn":
        threading.Thread(target=_run_stop_moving, daemon=True).start()
        return "Stop moving sent..."
    if trigger == "extend-arm-btn":
        threading.Thread(target=_run_extend_arm, daemon=True).start()
        return "Extending arm..."
    if trigger == "stop-grabbing-btn":
        threading.Thread(target=_run_stop_grabbing, daemon=True).start()
        return "Stopping grab..."
    if trigger == "release-arms-btn":
        threading.Thread(target=_run_release_arms, daemon=True).start()
        return "Releasing arms…"
    if trigger == "unrelease-arms-btn":
        threading.Thread(target=_run_unrelease_arms, daemon=True).start()
        return "Re-engaging arms..."
    if trigger == "damp-btn":
        threading.Thread(target=_run_damp, daemon=True).start()
        return "Damping…"
    return no_update


# ---------------------------------------------------------------------------
# Background actions
# ---------------------------------------------------------------------------

def _run_release_arms() -> None:
    ARM_CANCEL_EVENT.set()
    acquired = ARM_CONTROL_LOCK.acquire(timeout=3.0)
    if not acquired:
        with STATE.lock:
            STATE.status_msg = "release_arms() waiting: arm controller is busy"
        ARM_CONTROL_LOCK.acquire()
    try:
        with STATE.lock:
            STATE.arm_motion_running = True
            STATE.arm_motion_label = "release arms"
        if _CONTROL_ROBOT is None:
            raise RuntimeError(_CONTROL_STATUS)
        result = _CONTROL_ROBOT.release_arms()
        with STATE.lock:
            STATE.status_msg = f"release_arms() done: {result}"
    except Exception as exc:
        with STATE.lock:
            STATE.status_msg = f"release_arms() failed: {exc}"
    finally:
        with STATE.lock:
            STATE.arm_motion_running = False
            STATE.arm_motion_label = ""
        ARM_CANCEL_EVENT.clear()
        ARM_CONTROL_LOCK.release()


def _run_unrelease_arms() -> None:
    if not ARM_CONTROL_LOCK.acquire(blocking=False):
        with STATE.lock:
            STATE.status_msg = "unrelease_arms() skipped: arm controller is busy"
        return
    try:
        ARM_CANCEL_EVENT.clear()
        with STATE.lock:
            STATE.arm_motion_running = True
            STATE.arm_motion_label = "re-engage arms"
        if _CONTROL_ROBOT is None:
            raise RuntimeError(_CONTROL_STATUS)
        if not hasattr(_CONTROL_ROBOT, "unrelease_arms"):
            raise AttributeError("control robot has no unrelease_arms()")
        result = _CONTROL_ROBOT.unrelease_arms()
        with STATE.lock:
            STATE.status_msg = f"unrelease_arms() done: {result}"
    except Exception as exc:
        with STATE.lock:
            STATE.status_msg = f"unrelease_arms() failed: {exc}"
    finally:
        with STATE.lock:
            STATE.arm_motion_running = False
            STATE.arm_motion_label = ""
        ARM_CONTROL_LOCK.release()


def _run_damp() -> None:
    ARM_CANCEL_EVENT.set()
    acquired = ARM_CONTROL_LOCK.acquire(timeout=3.0)
    if not acquired:
        with STATE.lock:
            STATE.status_msg = "damp() waiting: arm controller is busy"
        ARM_CONTROL_LOCK.acquire()
    try:
        with STATE.lock:
            STATE.arm_motion_running = True
            STATE.arm_motion_label = "damp"
        if _CONTROL_ROBOT is None:
            raise RuntimeError(_CONTROL_STATUS)
        _CONTROL_ROBOT.damp()
        with STATE.lock:
            STATE.status_msg = "damp() sent."
    except Exception as exc:
        with STATE.lock:
            STATE.status_msg = f"damp() failed: {exc}"
    finally:
        with STATE.lock:
            STATE.arm_motion_running = False
            STATE.arm_motion_label = ""
        ARM_CANCEL_EVENT.clear()
        ARM_CONTROL_LOCK.release()


def _run_prepare() -> None:
    try:
        if _CONTROL_ROBOT is None:
            raise RuntimeError(_CONTROL_STATUS)
        client = getattr(_CONTROL_ROBOT, "_client", None)
        if client is None or not hasattr(client, "SetFsmId"):
            raise AttributeError("control robot has no SetFsmId client")
        client.SetFsmId(4)
        with STATE.lock:
            STATE.status_msg = "prepare FSM 4 sent."
    except Exception as exc:
        with STATE.lock:
            STATE.status_msg = f"prepare failed: {exc}"


def _run_walk() -> None:
    try:
        if _CONTROL_ROBOT is None:
            raise RuntimeError(_CONTROL_STATUS)
        if hasattr(_CONTROL_ROBOT, "walk_mode"):
            _CONTROL_ROBOT.walk_mode()
        else:
            client = getattr(_CONTROL_ROBOT, "_client", None)
            if client is None or not hasattr(client, "SetFsmId"):
                raise AttributeError("control robot has no walk_mode or SetFsmId")
            client.SetFsmId(501)
        with STATE.lock:
            STATE.status_msg = "walk FSM 501 sent."
    except Exception as exc:
        with STATE.lock:
            STATE.status_msg = f"walk failed: {exc}"


def _run_stop_moving() -> None:
    try:
        if _CONTROL_ROBOT is None:
            raise RuntimeError(_CONTROL_STATUS)
        if hasattr(_CONTROL_ROBOT, "loco_move"):
            _CONTROL_ROBOT.loco_move(0.0, 0.0, 0.0)
        else:
            client = getattr(_CONTROL_ROBOT, "_client", None)
            if client is None or not hasattr(client, "Move"):
                raise AttributeError("control robot has no loco_move or Move")
            client.Move(0.0, 0.0, 0.0, continous_move=False)
        with STATE.lock:
            STATE.status_msg = "loco velocity zero sent."
    except Exception as exc:
        with STATE.lock:
            STATE.status_msg = f"stop moving failed: {exc}"


def _run_stop_grabbing() -> None:
    ARM_CANCEL_EVENT.set()
    _log("[recognition_app] Stop grabbing requested.")
    with ACTIVE_NAV_LOCK:
        nav = ACTIVE_DIRECT_NAV
    if nav is not None:
        try:
            nav.shutdown()
            _log("[recognition_app] Active direct nav stopped.")
        except Exception as exc:
            _log(f"[recognition_app] Active direct nav stop failed: {exc}")

    acquired = ARM_CONTROL_LOCK.acquire(timeout=3.0)
    if not acquired:
        with STATE.lock:
            STATE.status_msg = "stop grabbing waiting: arm controller is busy"
        ARM_CONTROL_LOCK.acquire()
    try:
        with STATE.lock:
            STATE.grab_running = False
            STATE.arm_motion_running = True
            STATE.arm_motion_label = "stop grabbing"
        if _CONTROL_ROBOT is None:
            raise RuntimeError(_CONTROL_STATUS)
        if not hasattr(_CONTROL_ROBOT, "release_arms"):
            raise AttributeError("control robot has no release_arms()")
        result = _CONTROL_ROBOT.release_arms(duration_s=0.5)
        _log(f"[recognition_app] stop grabbing release_arms() done: {result}")
        with STATE.lock:
            STATE.status_msg = "grab stopped; arms released."
    except Exception as exc:
        _log(f"[recognition_app] stop grabbing failed: {exc}")
        with STATE.lock:
            STATE.status_msg = f"stop grabbing failed: {exc}"
    finally:
        with STATE.lock:
            STATE.arm_motion_running = False
            STATE.arm_motion_label = ""
        ARM_CANCEL_EVENT.clear()
        ARM_CONTROL_LOCK.release()


def _read_current_arm_q(robot, arm: str) -> np.ndarray:
    js = robot.get_joint_states()
    joints = js.get("joints", {}) if js else {}
    joint_indices = LEFT_ARM_JOINTS if arm == "left" else RIGHT_ARM_JOINTS
    seen_indices = set()
    q_full = np.zeros(30, dtype=np.float64)
    for name, data in joints.items():
        idx = _joint_entry_index(name, data)
        if 0 <= idx < q_full.size and data.get("position") is not None:
            q_full[idx] = float(data.get("position"))
            seen_indices.add(idx)
    missing = [idx for idx in joint_indices if idx not in seen_indices]
    if missing:
        raise RuntimeError(f"cannot read current {arm} arm joints from lowstate; missing indices {missing}")
    return q_full[joint_indices]


def _extend_arm_forward_joint_waypoints(q_start: np.ndarray, arm: str) -> List[Tuple[str, np.ndarray]]:
    side = str(arm).strip().lower()
    if side not in ("left", "right"):
        raise ValueError("arm must be 'left' or 'right'")
    limits = JOINT_LIMITS[side]

    def clamp_i(i: int, value: float) -> float:
        lo, hi = limits[i]
        return max(lo, min(hi, float(value)))

    roll_delta = 0.50 if side == "left" else -0.50
    pitch_delta = -0.35
    restore_fraction = 0.45

    start = np.asarray(q_start, dtype=np.float64).copy()
    roll_pose = start.copy()
    roll_pose[1] = clamp_i(1, start[1] + roll_delta)

    pitch_pose = roll_pose.copy()
    pitch_pose[0] = clamp_i(0, roll_pose[0] + pitch_delta)

    restored_roll_pose = pitch_pose.copy()
    restored_roll_pose[1] = clamp_i(1, roll_pose[1] - (roll_delta * restore_fraction))

    target_pose = restored_roll_pose.copy()
    target_pose[3] = clamp_i(3, start[3] - 0.90)
    target_pose[4] = clamp_i(4, start[4] + 0.40)
    target_pose[5] = clamp_i(5, start[5] - 0.40)

    final_pose = target_pose.copy()
    final_pose[0] = clamp_i(0, target_pose[0] - 0.08)
    final_pose[1] = clamp_i(1, target_pose[1] - np.copysign(0.10, roll_delta))
    final_pose[3] = clamp_i(3, target_pose[3] - 0.10)
    final_pose[5] = clamp_i(5, target_pose[5] - 0.08)

    return [
        ("shoulder_roll_clearance", roll_pose),
        ("shoulder_pitch_forward", pitch_pose),
        ("partial_shoulder_roll_restore", restored_roll_pose),
        ("elbow_and_wrist_pitch", target_pose),
        ("final_forward_up_and_in", final_pose),
    ]


def _run_extend_arm_ee_ik(
    robot,
    arm: str,
    max_joint_step_rad: float,
    max_joint_speed_rad_s: float,
) -> Dict[str, Any]:
    if hasattr(robot, "unrelease_arms"):
        robot.unrelease_arms(duration_s=0.4)
    q_cur = _read_current_arm_q(robot, arm)
    fk = ArmFK(arm=arm, backend="urdf")
    ik = ArmIK(
        arm=arm,
        solver="dls",
        tol_pos_m=0.012,
        tol_rot_rad=0.20,
    )
    executor = ArmExecutor(
        robot,
        arm=arm,
        kp=30.0,
        kd=1.5,
        max_reach_m=DEFAULT_DIRECT_MAX_REACH_M,
        max_joint_step_rad=max_joint_step_rad,
        max_joint_speed_rad_s=max_joint_speed_rad_s,
    )
    results = []
    for stage_name, q_stage_nominal in _extend_arm_forward_joint_waypoints(q_cur, arm):
        if ARM_CANCEL_EVENT.is_set():
            return {"success": False, "reason": "cancelled", "stage_results": results}
        T_desired = fk.compute_arm(q_stage_nominal)
        q_solution, info = ik.solve(T_desired, q_init=q_cur)
        if q_solution is None:
            return {
                "success": False,
                "reason": f"ik failed at {stage_name}",
                "ik_info": info,
                "stage_results": results,
            }
        result = executor.execute(
            q_solution,
            duration_s=DEFAULT_EXTEND_STAGE_DURATION_S,
            q_arm_start=q_cur,
            T_base_desired=T_desired,
            stop_event=ARM_CANCEL_EVENT,
        )
        results.append({"stage": stage_name, "ik": info, "execute": result})
        _log(
            "[recognition_app] extend stage "
            f"{stage_name}: success={result.get('success')} "
            f"duration={result.get('duration_s', 0.0):.2f}s "
            f"requested={result.get('requested_duration_s', 0.0):.2f}s "
            f"final_err={result.get('final_error_rad', 0.0):.4f}rad "
            f"capped_samples={result.get('capped_samples', 0)}"
        )
        if not result.get("success"):
            return {"success": False, "reason": result.get("reason"), "stage_results": results}
        q_cur = np.asarray(result.get("final_q", q_solution), dtype=np.float64)
    return {
        "success": True,
        "arm": arm,
        "controller": "ee_ik",
        "stages": [item["stage"] for item in results],
    }


def _run_extend_arm() -> None:
    if not ARM_CONTROL_LOCK.acquire(blocking=False):
        with STATE.lock:
            STATE.status_msg = "extend arm skipped: arm controller is busy"
        return
    try:
        ARM_CANCEL_EVENT.clear()
        with STATE.lock:
            STATE.arm_motion_running = True
            STATE.arm_motion_label = "extend arm"
        if _CONTROL_ROBOT is None:
            raise RuntimeError(_CONTROL_STATUS)
        with STATE.lock:
            arm_override = STATE.arm_override
            selected_id = STATE.selected_id
            det = STATE.detections.get(selected_id) if selected_id else None
            cam = dict(STATE.camera_extrinsic)
            max_joint_step_rad = STATE.max_joint_step_rad
            max_joint_speed_rad_s = STATE.max_joint_speed_rad_s
        arm = arm_override
        if arm == "auto":
            if det is not None:
                arm, _T_base_camera, _T_base_object = _resolve_arm_and_base_pose(det, cam, arm_override)
            else:
                arm = "right"
        result = _run_extend_arm_ee_ik(
            _CONTROL_ROBOT,
            arm,
            max_joint_step_rad,
            max_joint_speed_rad_s,
        )
        with STATE.lock:
            STATE.status_msg = f"extend arm EE IK ({arm}) done: {result}"
    except Exception as exc:
        with STATE.lock:
            STATE.status_msg = f"extend arm failed: {exc}"
    finally:
        with STATE.lock:
            STATE.arm_motion_running = False
            STATE.arm_motion_label = ""
        ARM_CONTROL_LOCK.release()


def _log(line: str) -> None:
    with STATE.lock:
        STATE.grab_log.append(line)
        STATE.grab_log = STATE.grab_log[-300:]


def _run_direct_grab_inline(
    det: Detection,
    arm: str,
    cam: Dict[str, float],
    auto_step_base: bool,
    T_base_object: np.ndarray,
    max_joint_step_rad: float,
    max_joint_speed_rad_s: float,
) -> None:
    global ACTIVE_DIRECT_NAV
    if _CONTROL_ROBOT is None:
        raise RuntimeError(_CONTROL_STATUS)

    if not ARM_CONTROL_LOCK.acquire(blocking=False):
        _log("[recognition_app] Waiting for current arm motion to finish before grab control.")
        ARM_CONTROL_LOCK.acquire()
    try:
        if ARM_CANCEL_EVENT.is_set():
            _log("[recognition_app] Direct grab cancelled before control handoff.")
            return
        ARM_CANCEL_EVENT.clear()
        with STATE.lock:
            STATE.arm_motion_running = True
            STATE.arm_motion_label = "direct grab"

        _arm, reach_dist, max_reach, excess_m, suggested_step_m = _reach_preview(
            det, cam, arm,
        )
        T_base_camera = _make_transform(
            xyz=(cam["x"], cam["y"], cam["z"]),
            rpy=(cam["roll"], cam["pitch"], cam["yaw"]),
        )
        T_camera_object = det.T_camera_object.copy()
        base_step_m = 0.0
        if excess_m > 0.0:
            _log(
                "[recognition_app] target outside arm workspace: "
                f"reach_dist={reach_dist:.3f} m max={max_reach:.3f} m "
                f"excess={excess_m:.3f} m"
            )
            if not auto_step_base:
                _log("[recognition_app] Auto-step disabled; move closer or enable base step.")
                return
            base_step_m = suggested_step_m
            if base_step_m <= 0.0:
                _log("[recognition_app] Auto-step requested, but allowed step distance is zero.")
                return
            speed_m_s = 1.5
            duration_s = base_step_m / speed_m_s
            _log(
                f"[recognition_app] stepping base forward {base_step_m:.3f} m "
                f"at {speed_m_s:.3f} m/s for {duration_s:.2f} s"
            )
            _CONTROL_ROBOT.move_for(duration=duration_s, vx=speed_m_s, vy=0.0, vyaw=0.0)
            T_base_object = T_base_object.copy()
            T_base_object[0, 3] -= base_step_m
            T_camera_object = np.linalg.inv(T_base_camera) @ T_base_object

        fixed_result = DetectionResult(
            T_camera_object=T_camera_object,
            confidence=float(det.score),
            method="fixed",
        )
        config = {
            "arm": arm,
            "detection_method": "fixed",
            "standoff_m": DEFAULT_GRASP_STANDOFF_M,
            "rate_hz": DEFAULT_SLOW_GRAB_RATE_HZ,
            "timeout_s": DEFAULT_SLOW_GRAB_TIMEOUT_S,
            "ik_solver": "dls",
            "iface": _ARGS.iface,
            "domain_id": _ARGS.domain_id,
            "mock": _ARGS.mock,
            "camera_x": float(cam.get("x", 0.0)),
            "camera_y": float(cam.get("y", 0.0)),
            "camera_z": float(cam.get("z", 0.0)),
            "camera_roll": float(cam.get("roll", 0.0)),
            "camera_pitch": float(cam.get("pitch", 0.0)),
            "camera_yaw": float(cam.get("yaw", 0.0)),
            "ik_tol_pos_m": 0.035 if det.source == "vision" else 0.003,
            "ik_tol_rot_rad": 3.14 if det.source == "vision" else 0.01,
            "convergence_pos_m": 0.035 if det.source == "vision" else 0.015,
            "convergence_rot_rad": 3.14 if det.source == "vision" else 0.05,
            "max_joint_step_rad": max_joint_step_rad,
            "max_joint_speed_rad_s": max_joint_speed_rad_s,
            "max_reach_m": DEFAULT_DIRECT_MAX_REACH_M,
        }
        _log(
            "[recognition_app] object_base_xyz="
            f"({T_base_object[0, 3]:+.3f}, {T_base_object[1, 3]:+.3f}, {T_base_object[2, 3]:+.3f}) m "
            f"base_step={base_step_m:.3f} m"
        )
        _log(
            "[recognition_app] using fixed selected-object XYZ for this grab; "
            "camera detections will not retarget the arm mid-grab."
        )
        _log(
            "[recognition_app] slow grab config: "
            f"rate={config['rate_hz']:.1f}Hz "
            f"max_joint_step={config['max_joint_step_rad']:.3f}rad "
            f"max_joint_speed={config['max_joint_speed_rad_s']:.3f}rad/s "
            f"timeout={config['timeout_s']:.1f}s"
        )
        if hasattr(_CONTROL_ROBOT, "unrelease_arms"):
            _log("[recognition_app] Re-engaging arm SDK control before direct grab.")
            try:
                result = _CONTROL_ROBOT.unrelease_arms(duration_s=0.5)
                _log(f"[recognition_app] unrelease_arms() done: {result}")
            except Exception as exc:
                _log(f"[recognition_app] unrelease_arms() before grab failed: {exc}")
        nav = DirectHandPoseNav(config, fixed_result=fixed_result, robot=_CONTROL_ROBOT)
        with ACTIVE_NAV_LOCK:
            ACTIVE_DIRECT_NAV = nav
        ok = False
        last_status_log_t = 0.0
        try:
            deadline = time.time() + float(config["timeout_s"]) + 2.0
            while time.time() < deadline:
                if ARM_CANCEL_EVENT.is_set():
                    _log("[recognition_app] Direct grab cancelled by safety command.")
                    break
                status = nav.status_snapshot()
                now = time.time()
                if now - last_status_log_t >= 1.0:
                    last_status_log_t = now
                    loop_log = status.get("log") or []
                    last_loop_msg = loop_log[-1] if loop_log else ""
                    _log(
                        "[recognition_app] direct nav status: "
                        f"running={status.get('running')} "
                        f"iter={status.get('iteration')} "
                        f"converged={status.get('converged')} "
                        f"pos_err={status.get('last_error_pos_m')} "
                        f"ik_failures={status.get('ik_failures')} "
                        f"safety_rejections={status.get('safety_rejections')} "
                        f"detection_failures={status.get('detection_failures')} "
                        f"last={last_loop_msg}"
                    )
                if status.get("converged"):
                    ok = True
                    break
                if not status.get("running", True):
                    _log(f"[recognition_app] direct nav stopped: {status}")
                    break
                time.sleep(0.2)
        finally:
            nav.shutdown()
            with ACTIVE_NAV_LOCK:
                if ACTIVE_DIRECT_NAV is nav:
                    ACTIVE_DIRECT_NAV = None

        if not ok:
            _log("[recognition_app] Direct grab did not converge; not closing hand.")
            return
        _log("[recognition_app] Direct grab converged.")
        try:
            if hasattr(_CONTROL_ROBOT, "hand_close"):
                _CONTROL_ROBOT.hand_close(hand=arm, hold_s=0.6)
                _log(f"[recognition_app] Closed {arm} hand.")
        except Exception as exc:
            _log(f"[recognition_app] Hand close failed: {exc}")
    finally:
        with STATE.lock:
            STATE.arm_motion_running = False
            STATE.arm_motion_label = ""
        ARM_CONTROL_LOCK.release()


def _run_grab_impl() -> None:
    with STATE.lock:
        selected_id = STATE.selected_id
        live_det = STATE.detections.get(selected_id) if selected_id else None
        det = _copy_detection(live_det) if live_det is not None else None
        detection_age_s = time.time() - STATE.last_detection_ts if STATE.last_detection_ts else float("inf")
        cam = dict(STATE.camera_extrinsic)
        arm_override = STATE.arm_override
        auto_step_base = STATE.auto_step_base
        max_joint_step_rad = STATE.max_joint_step_rad
        max_joint_speed_rad_s = STATE.max_joint_speed_rad_s
        STATE.grab_running = True

    if det is None or detection_age_s > STALE_AFTER_S:
        if det is None:
            _log("[recognition_app] No object selected — aborting grab.")
        else:
            _log(
                "[recognition_app] Selected detection is stale "
                f"({detection_age_s:.1f}s old) — aborting grab."
            )
        with STATE.lock:
            STATE.grab_running = False
        return

    arm, _T_base_camera, T_base_object = _resolve_arm_and_base_pose(
        det, cam, arm_override,
    )
    _arm, reach_dist, max_reach, excess_m, suggested_step_m = _reach_preview(
        det, cam, arm_override,
    )
    if excess_m > 0.0 and not auto_step_base:
        _log(
            "[recognition_app] Selected target is outside arm-only reach: "
            f"reach_dist={reach_dist:.3f} m max={max_reach:.3f} m "
            f"excess={excess_m:.3f} m. Enable 'Step base closer' "
            f"or move the object/robot about {suggested_step_m:.3f} m closer."
        )

    p_cam = det.T_camera_object[:3, 3]
    p_base = T_base_object[:3, 3]
    _log(
        "[recognition_app] Grab target coordinate snapshot: "
        f"id={det.id!r} label={det.label!r} source={det.source} "
        f"confidence={det.score:.2f}"
    )
    _log(
        f"[recognition_app] arm={arm} label={det.label!r} "
        f"auto_step_base={auto_step_base}"
    )
    _log(
        "[recognition_app] object camera xyz="
        f"({p_cam[0]:+.3f}, {p_cam[1]:+.3f}, {p_cam[2]:+.3f}) m  "
        "base xyz="
        f"({p_base[0]:+.3f}, {p_base[1]:+.3f}, {p_base[2]:+.3f}) m"
    )

    try:
        _run_direct_grab_inline(
            det,
            arm,
            cam,
            auto_step_base,
            T_base_object,
            max_joint_step_rad,
            max_joint_speed_rad_s,
        )
    finally:
        with STATE.lock:
            STATE.grab_running = False


def _run_grab() -> None:
    try:
        _run_grab_impl()
    except Exception as exc:
        _log(f"[recognition_app] Grab worker failed before launch: {exc!r}")
        with STATE.lock:
            STATE.grab_running = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1 recognition layer web app.")
    p.add_argument("--iface", default="eth0")
    p.add_argument("--domain-id", type=int, default=0)
    p.add_argument("--rgbd-host", default="192.168.2.41")
    p.add_argument("--rgbd-port", type=int, default=5555)
    p.add_argument("--rgbd-topic", default="")
    p.add_argument(
        "--sdk-rgbd",
        action="store_true",
        help="Use sdk_client.Robot.get_rgbd() instead of the persistent ZMQ receiver.",
    )
    p.add_argument("--rate-hz", type=float, default=3.0, help="perception loop rate")
    p.add_argument("--camera-rate-hz", type=float, default=15.0, help="raw RGB-D capture loop rate")
    p.add_argument("--vision-model", default="yolov8s-world.pt")
    p.add_argument("--ros-base-frame", default="base_link")
    p.add_argument("--ros-camera-frame", default="camera_color_optical_frame")
    p.add_argument("--ros-tf-timeout-s", type=float, default=0.5)
    p.add_argument("--mock", action="store_true", help="use a synthetic camera feed")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8060)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


_ARGS = _parse_args()

_DDS_STATUS = "Unitree DDS skipped in mock mode"
if not _ARGS.mock:
    _DDS_STATUS = _init_unitree_dds_once(_ARGS.iface, _ARGS.domain_id)
with STATE.lock:
    STATE.hand_fk_status = _DDS_STATUS

if _ARGS.mock:
    _ROBOT = _MockRobot()
elif _ARGS.sdk_rgbd and _ROBOT_AVAILABLE:
    _ROBOT = Robot(
        iface=_ARGS.iface,
        domain_id=_ARGS.domain_id,
        auto_start_sensors=True,
        rgbd_host=_ARGS.rgbd_host,
        rgbd_port=_ARGS.rgbd_port,
    )
else:
    _ROBOT = _ZmqRgbdRobot(
        host=_ARGS.rgbd_host,
        port=_ARGS.rgbd_port,
        topic=_ARGS.rgbd_topic,
    )

_CONTROL_ROBOT = _ROBOT if (Robot is not None and isinstance(_ROBOT, Robot)) else None
_CONTROL_STATUS = "control robot unavailable"
if _ARGS.mock:
    _CONTROL_ROBOT = _ROBOT
    _CONTROL_STATUS = "mock control robot"
elif _DDS_STATUS.startswith("Unitree DDS initialized"):
    if _CONTROL_ROBOT is None:
        _CONTROL_ROBOT, _CONTROL_STATUS = _make_control_robot(_ARGS.iface, _ARGS.domain_id)
    else:
        _CONTROL_STATUS = "control robot ready"
else:
    _CONTROL_STATUS = _DDS_STATUS
with STATE.lock:
    if _CONTROL_ROBOT is None:
        STATE.status_msg = _CONTROL_STATUS

_DETECTOR = TargetDetector(
    method="aruco",  # unused directly; detect_all_aruco() is what's called
    marker_sizes={
        aruco_assets.HAND_MARKER_ID: aruco_assets.HAND_MARKER_SIZE_M,
        **{mid: aruco_assets.OBJECT_MARKER_SIZE_M for mid in aruco_assets.OBJECT_MARKER_IDS},
    },
    intrinsics=K,
)
_VISION = VisionDetector(model_name=_ARGS.vision_model)

threading.Thread(
    target=_camera_loop, args=(_ROBOT, _DETECTOR, _ARGS.camera_rate_hz),
    daemon=True,
).start()

threading.Thread(
    target=_perception_loop, args=(_DETECTOR, _VISION, _ARGS.rate_hz),
    daemon=True,
).start()

if not _ARGS.mock and _DDS_STATUS.startswith("Unitree DDS initialized"):
    threading.Thread(
        target=_hand_fk_loop, args=(_ARGS.iface, _ARGS.domain_id),
        daemon=True,
    ).start()

threading.Thread(
    target=_ros_camera_tf_loop,
    args=(_ARGS.ros_base_frame, _ARGS.ros_camera_frame, _ARGS.ros_tf_timeout_s),
    daemon=True,
).start()


if __name__ == "__main__":
    app.run(host=_ARGS.host, port=_ARGS.port, debug=_ARGS.debug)
