#!/usr/bin/env python3
"""
Recognition layer — sensors-to-UI web app for the G1 + Dex3 grasp pipeline.

Shows the RGB-D feed, a segmentation overlay, a labeled-detections overlay
(open-vocabulary vision model, NL-prompted), and an ArUco tag overlay
(object tags + the optional Dex3 hand tag) side by side. Pick a detected
object from the list, hit Grab, and one of the two existing hand_pose_navigation
backends (direct_nav.DirectHandPoseNav or hand_pose_nav_node.HandPoseNavNode,
launched as a separate subprocess via grab_direct.py / grab_ros2.py) drives
the arm to it and closes the hand. Release Arms and Damp are also exposed as
one-click safety controls (sdk_client.Robot.release_arms() / .damp()).

Run:
    python3 recognition_app.py [--iface eth0] [--domain-id 0] [--mock]
Then open http://<host>:8060 in a browser on the same network.
"""
from __future__ import annotations

import argparse
import base64
import json
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
from hand_pose_navigation.direct_nav import _make_transform

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


class SharedState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.rgb_bgr: Optional[np.ndarray] = None
        self.depth_m: Optional[np.ndarray] = None
        self.detections: Dict[str, Detection] = {}
        self.boxes_for_draw: List[Dict] = []
        self.masks_for_draw: List[Tuple[str, np.ndarray]] = []
        self.tags_for_draw: Dict[int, DetectionResult] = {}
        self.selected_id: Optional[str] = None
        self.camera_extrinsic = {
            "x": 0.0, "y": 0.0, "z": 0.30,
            "roll": -1.5708, "pitch": 0.0, "yaw": -1.5708,
        }
        self.arm_override = "auto"
        self.backend = "direct"
        self.vision_classes: List[str] = []
        self.status_msg = "starting…"
        self.grab_log: List[str] = []
        self.grab_running = False
        self.last_frame_ts: float = 0.0


STATE = SharedState()
BASE_K = CameraIntrinsics()  # default 640x480 RealSense-ish intrinsics
K = CameraIntrinsics()

# The object list must reflect what the robot can see *right now*, not the
# last thing it happened to see. If the perception loop hasn't produced a
# fresh frame+detection pass within this window, the UI treats the list as
# stale rather than silently keeping possibly-gone objects selectable.
STALE_AFTER_S = 2.0


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

    def damp(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Background perception loop
# ---------------------------------------------------------------------------

def _perception_loop(robot, detector: TargetDetector, vision: VisionDetector, rate_hz: float) -> None:
    global K
    period = 1.0 / max(0.5, rate_hz)
    while True:
        t0 = time.time()
        try:
            frame = robot.get_rgbd(timeout=0.5)
            rgb = frame["rgb_bgr"]
            depth = frame["depth_m"]
        except Exception as exc:
            with STATE.lock:
                STATE.status_msg = f"camera error: {exc}"
            time.sleep(period)
            continue

        h, w = rgb.shape[:2]
        if w != K.width or h != K.height:
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

        tags = detector.detect_all_aruco(rgb, depth)
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
        masks_for_draw: List[Tuple[str, np.ndarray]] = []
        with STATE.lock:
            selected_id = STATE.selected_id
        for i, vd in enumerate(vis_dets):
            mask = mask_from_box_depth(depth, vd.box_xyxy)
            pose = pose_from_mask(mask, depth, K)
            det_id = f"vision:{i}"
            boxes_for_draw.append({
                "box": vd.box_xyxy, "label": vd.label, "score": vd.score,
                "selected": det_id == selected_id,
            })
            if pose is None:
                continue
            detections[det_id] = Detection(
                id=det_id, label=vd.label, source="vision", score=vd.score,
                T_camera_object=pose.T_camera_object, box=vd.box_xyxy,
            )
            masks_for_draw.append((vd.label, mask))

        with STATE.lock:
            STATE.rgb_bgr = rgb
            STATE.depth_m = depth
            STATE.detections = detections
            STATE.boxes_for_draw = boxes_for_draw
            STATE.masks_for_draw = masks_for_draw
            STATE.tags_for_draw = tags
            STATE.last_frame_ts = time.time()
            n_vision = "on" if vision.available else f"off ({vision.error})"
            STATE.status_msg = (
                f"{len(detections)} detections  |  vision model: {n_vision}"
            )

        dt = time.time() - t0
        time.sleep(max(0.0, period - dt))


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


def _camera_calib_card() -> dbc.Card:
    fields = ["x", "y", "z", "roll", "pitch", "yaw"]
    inputs = [
        dbc.Col([
            dbc.Label(f, size="sm"),
            dbc.Input(id=f"calib-{f}", type="number", value=STATE.camera_extrinsic[f],
                       step=0.005, size="sm"),
        ], width=2) for f in fields
    ]
    return dbc.Card(dbc.CardBody([
        html.H5("Camera → base_link extrinsic"),
        html.Small(
            "Position (m) and roll/pitch/yaw (rad) of the camera optical "
            "frame in the robot base frame. Defaults are a rough guess — "
            "refine using the hand tag overlay vs. FK as a sanity check.",
            className="text-muted",
        ),
        dbc.Row(inputs, className="mt-2"),
        dbc.Button("Save calibration", id="save-calib-btn", size="sm", color="secondary",
                    className="mt-2"),
        html.Div(id="calib-status", className="mt-1", style={"fontSize": "12px"}),
    ]), className="mb-3")


def _image_panel(panel_id: str, title: str) -> dbc.Col:
    return dbc.Col(dbc.Card([
        dbc.CardHeader(title, style={"fontSize": "13px"}),
        html.Img(id=panel_id, style={"width": "100%"}),
    ]), width=6, className="mb-3")


app.layout = dbc.Container([
    html.H3("G1 Recognition Layer", className="mt-3"),
    html.Div(id="global-status", className="mb-2", style={"fontSize": "13px", "color": "#9c9"}),

    dbc.Row([
        dbc.Col(_instructions_card(), width=6),
        dbc.Col(_camera_calib_card(), width=6),
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
        _image_panel("panel-rgb", "RGB (raw)"),
        _image_panel("panel-depth", "Depth"),
    ]),
    dbc.Row([
        _image_panel("panel-detections", "Detected objects (vision model)"),
        _image_panel("panel-segmentation", "Segmentation overlay"),
    ]),
    dbc.Row([
        _image_panel("panel-aruco", "ArUco tags (objects + Dex3 hand)"),
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("Detections — select one"),
            html.Div(id="detection-list"),
        ], width=6),
        dbc.Col([
            html.H5("Grab"),
            dbc.RadioItems(
                id="backend-select",
                options=[
                    {"label": "Direct (no ROS 2)", "value": "direct"},
                    {"label": "ROS 2 / TF (needs ROS 2 set up)", "value": "ros2"},
                ],
                value="direct", inline=True, className="mb-2",
            ),
            dbc.RadioItems(
                id="arm-select",
                options=[
                    {"label": "Auto (pick by target side)", "value": "auto"},
                    {"label": "Left", "value": "left"},
                    {"label": "Right", "value": "right"},
                ],
                value="auto", inline=True, className="mb-2",
            ),
            dbc.Button("Grab selected object", id="grab-btn", color="success",
                        className="mb-2", disabled=True),
            html.Div(id="grab-selected-label", className="mb-2", style={"fontSize": "13px"}),
            html.Hr(),
            dbc.ButtonGroup([
                dbc.Button("Release arms", id="release-arms-btn", color="warning"),
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
    Output("panel-rgb", "src"),
    Output("panel-depth", "src"),
    Output("panel-detections", "src"),
    Output("panel-segmentation", "src"),
    Output("panel-aruco", "src"),
    Output("detection-list", "children"),
    Output("global-status", "children"),
    Output("grab-log", "children"),
    Output("grab-btn", "disabled"),
    Output("grab-selected-label", "children"),
    Input("refresh-interval", "n_intervals"),
)
def _refresh(_n):
    with STATE.lock:
        rgb = None if STATE.rgb_bgr is None else STATE.rgb_bgr.copy()
        depth = None if STATE.depth_m is None else STATE.depth_m.copy()
        detections = dict(STATE.detections)
        boxes = list(STATE.boxes_for_draw)
        masks = list(STATE.masks_for_draw)
        tags = dict(STATE.tags_for_draw)
        selected_id = STATE.selected_id
        status_msg = STATE.status_msg
        grab_log = list(STATE.grab_log[-100:])
        grab_running = STATE.grab_running
        last_frame_ts = STATE.last_frame_ts

    if rgb is None:
        ph = _placeholder_src("waiting for camera…")
        return ph, ph, ph, ph, ph, "No detections yet.", status_msg, "\n".join(grab_log), True, ""

    # A frozen/erroring camera feed must not keep offering stale objects as
    # if the robot could still see them — the list only reflects detections
    # from a genuinely recent perception pass.
    is_stale = (time.time() - last_frame_ts) > STALE_AFTER_S
    if is_stale:
        detections = {}
        boxes = []
        masks = []
        tags = {}
        status_msg = f"STALE — no fresh frame in >{STALE_AFTER_S:.0f}s. {status_msg}"

    rgb_src = _encode_jpeg_src(rgb)
    depth_src = _encode_jpeg_src(_colorize_depth(depth)) if depth is not None else _placeholder_src("no depth")
    det_src = _encode_jpeg_src(draw_detection_boxes(rgb, boxes))
    seg_src = _encode_jpeg_src(draw_segmentation_overlay(rgb, masks)) if masks else _placeholder_src("no masks")
    aruco_src = _encode_jpeg_src(draw_aruco_overlay(rgb, tags, K))

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

    grab_disabled = (selected_id not in detections) or grab_running
    selected_label = ""
    if selected_id in detections:
        selected_label = f"Selected: {detections[selected_id].label} ({detections[selected_id].source})"
    elif selected_id is not None:
        selected_label = "Selected object is no longer visible."

    return (
        rgb_src, depth_src, det_src, seg_src, aruco_src,
        det_list, status_msg, "\n".join(grab_log), grab_disabled, selected_label,
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
    Output("calib-status", "children"),
    Input("save-calib-btn", "n_clicks"),
    State("calib-x", "value"), State("calib-y", "value"), State("calib-z", "value"),
    State("calib-roll", "value"), State("calib-pitch", "value"), State("calib-yaw", "value"),
    prevent_initial_call=True,
)
def _save_calib(_n, x, y, z, roll, pitch, yaw):
    with STATE.lock:
        STATE.camera_extrinsic = {
            "x": float(x or 0.0), "y": float(y or 0.0), "z": float(z or 0.0),
            "roll": float(roll or 0.0), "pitch": float(pitch or 0.0), "yaw": float(yaw or 0.0),
        }
    return "Saved."


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
    Output("backend-select", "value"),
    Input("backend-select", "value"),
)
def _sync_backend(value):
    with STATE.lock:
        STATE.backend = value
    return value


@app.callback(
    Output("arm-select", "value"),
    Input("arm-select", "value"),
)
def _sync_arm(value):
    with STATE.lock:
        STATE.arm_override = value
    return value


@app.callback(
    Output("safety-status", "children"),
    Input("grab-btn", "n_clicks"),
    Input("release-arms-btn", "n_clicks"),
    Input("damp-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _buttons(grab_clicks, release_clicks, damp_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "grab-btn":
        threading.Thread(target=_run_grab, daemon=True).start()
        return "Grab started — see log below."
    if trigger == "release-arms-btn":
        threading.Thread(target=_run_release_arms, daemon=True).start()
        return "Releasing arms…"
    if trigger == "damp-btn":
        threading.Thread(target=_run_damp, daemon=True).start()
        return "Damping…"
    return no_update


# ---------------------------------------------------------------------------
# Background actions
# ---------------------------------------------------------------------------

def _run_release_arms() -> None:
    try:
        result = _ROBOT.release_arms()
        with STATE.lock:
            STATE.status_msg = f"release_arms() done: {result}"
    except Exception as exc:
        with STATE.lock:
            STATE.status_msg = f"release_arms() failed: {exc}"


def _run_damp() -> None:
    try:
        _ROBOT.damp()
        with STATE.lock:
            STATE.status_msg = "damp() sent."
    except Exception as exc:
        with STATE.lock:
            STATE.status_msg = f"damp() failed: {exc}"


def _log(line: str) -> None:
    with STATE.lock:
        STATE.grab_log.append(line)
        STATE.grab_log = STATE.grab_log[-300:]


def _run_grab() -> None:
    with STATE.lock:
        selected_id = STATE.selected_id
        det = STATE.detections.get(selected_id) if selected_id else None
        cam = dict(STATE.camera_extrinsic)
        arm_override = STATE.arm_override
        backend = STATE.backend
        STATE.grab_running = True

    if det is None:
        _log("[recognition_app] No object selected — aborting grab.")
        with STATE.lock:
            STATE.grab_running = False
        return

    arm = arm_override
    if arm == "auto":
        T_base_camera = _make_transform(
            xyz=(cam["x"], cam["y"], cam["z"]),
            rpy=(cam["roll"], cam["pitch"], cam["yaw"]),
        )
        T_base_object = T_base_camera @ det.T_camera_object
        arm = "left" if T_base_object[1, 3] > 0 else "right"
    else:
        T_base_camera = _make_transform(
            xyz=(cam["x"], cam["y"], cam["z"]),
            rpy=(cam["roll"], cam["pitch"], cam["yaw"]),
        )
        T_base_object = T_base_camera @ det.T_camera_object

    target = {
        "arm": arm,
        "T_camera_object": det.T_camera_object.tolist(),
        "camera_extrinsic": cam,
        "standoff_m": 0.08,
        "label": det.label,
        "source": det.source,
        "confidence": det.score,
    }

    fd, path = tempfile.mkstemp(prefix="grab_target_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(target, f)

    script = "grab_direct.py" if backend == "direct" else "grab_ros2.py"
    script_path = os.path.join(_DIR, script)
    cmd = [sys.executable, script_path, "--target-json", path,
           "--iface", _ARGS.iface, "--domain-id", str(_ARGS.domain_id)]
    if _ARGS.mock and backend == "direct":
        cmd.append("--mock")

    p_cam = det.T_camera_object[:3, 3]
    p_base = T_base_object[:3, 3]
    _log(f"[recognition_app] arm={arm} label={det.label!r} backend={backend}")
    _log(
        "[recognition_app] object camera xyz="
        f"({p_cam[0]:+.3f}, {p_cam[1]:+.3f}, {p_cam[2]:+.3f}) m  "
        "base xyz="
        f"({p_base[0]:+.3f}, {p_base[1]:+.3f}, {p_base[2]:+.3f}) m"
    )
    _log(f"[recognition_app] $ {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            _log(line.rstrip())
        proc.wait()
        _log(f"[recognition_app] Grab process exited with code {proc.returncode}")
    except Exception as exc:
        _log(f"[recognition_app] Failed to launch grab script: {exc}")
    finally:
        os.remove(path) if os.path.exists(path) else None
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
    p.add_argument("--rate-hz", type=float, default=3.0, help="perception loop rate")
    p.add_argument("--vision-model", default="yolov8s-world.pt")
    p.add_argument("--mock", action="store_true", help="use a synthetic camera feed")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8060)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


_ARGS = _parse_args()

if _ARGS.mock or not _ROBOT_AVAILABLE:
    _ROBOT = _MockRobot()
else:
    _ROBOT = Robot(
        iface=_ARGS.iface,
        domain_id=_ARGS.domain_id,
        auto_start_sensors=True,
        rgbd_host=_ARGS.rgbd_host,
        rgbd_port=_ARGS.rgbd_port,
    )

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
    target=_perception_loop, args=(_ROBOT, _DETECTOR, _VISION, _ARGS.rate_hz),
    daemon=True,
).start()


if __name__ == "__main__":
    app.run(host=_ARGS.host, port=_ARGS.port, debug=_ARGS.debug)
