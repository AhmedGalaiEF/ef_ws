"""Prompt-driven RGB-D grasp wrapper using OpenAI localization + existing IK nav."""
from __future__ import annotations

import base64
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from .vision import VisionError, ZmqRgbdSource


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


class VisionGrabError(RuntimeError):
    pass


def _bootstrap_hand_pose_navigation() -> None:
    here = Path(__file__).resolve()
    ef_ws_root = next((parent for parent in here.parents if (parent / "g1" / "hand_pose_navigation").exists()), None)
    if ef_ws_root is None:
        ef_ws_root = next((parent for parent in here.parents if (parent / "hand_pose_navigation").exists()), None)
        g1_dir = ef_ws_root if ef_ws_root is not None else here.parents[4] / "g1"
    else:
        g1_dir = ef_ws_root / "g1"
    for path in (g1_dir, g1_dir / "modules", g1_dir / "modules" / "scripts"):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


class OpenAIVisionGrabber:
    def __init__(self, *, robot: Any, iface: str = "eth0", domain_id: int = 0) -> None:
        self.robot = robot
        self.iface = iface
        self.domain_id = int(domain_id)
        self._source: Optional[ZmqRgbdSource] = None
        self._source_key: Optional[tuple[str, int, str]] = None

    def grab(self, *, settings: Any, prompt: str, arm: str = "auto") -> str:
        prompt = _clean_grab_prompt(prompt)
        if not settings.rgbd_enabled:
            raise VisionGrabError(
                "RGB-D vision input is disabled. Enable vision.rgbd_enabled before using prompt-based grab."
            )
        frame = self._get_source(settings).get_rgbd(timeout=settings.rgbd_timeout_s)
        loc = self._localize_with_openai(
            model=str(settings.openai_model),
            rgb_jpeg=frame["rgb_jpeg"],
            prompt=prompt,
        )
        if not loc.get("found"):
            reason = loc.get("reason") or "target was not localized"
            raise VisionGrabError(f"could not find {prompt!r}: {reason}")

        return self._run_ik_grab(
            frame=frame,
            label=str(loc.get("label") or prompt),
            confidence=float(loc.get("confidence") or 0.0),
            box_xyxy=_coerce_box(loc.get("box_xyxy")),
            arm_override=arm,
        )

    def _get_source(self, settings: Any) -> ZmqRgbdSource:
        key = (str(settings.rgbd_host), int(settings.rgbd_port), str(settings.rgbd_topic))
        if self._source is None or self._source_key != key:
            self._source = ZmqRgbdSource(*key)
            self._source_key = key
        return self._source

    @staticmethod
    def _localize_with_openai(*, model: str, rgb_jpeg: bytes, prompt: str) -> dict[str, Any]:
        _bootstrap_llm_transport()
        try:
            from llm_client import chat as chat_module
        except Exception as exc:
            raise VisionGrabError(f"llm_client.chat is unavailable: {exc}") from exc
        try:
            import llm_client.secrets  # noqa: F401
        except Exception:
            pass
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise VisionGrabError("OPENAI_API_KEY is not set; OpenAI vision localization cannot authenticate.")
        chat_module.dnabot_auth = SimpleNamespace(
            get_auth_header=lambda: {"Authorization": f"Bearer {api_key}"}
        )
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "vision_grab_target",
                "schema": {
                    "type": "object",
                    "properties": {
                        "found": {"type": "boolean"},
                        "label": {"type": "string"},
                        "confidence": {"type": "number"},
                        "box_xyxy": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["found", "label", "confidence", "box_xyxy", "reason"],
                    "additionalProperties": False,
                },
                "strict": False,
            },
        }
        image_b64 = base64.b64encode(rgb_jpeg).decode("ascii")
        content, _ = chat_module.send_chat_with_tool_usage(
            model_key=model,
            base="https://api.openai.com/v1",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Locate the requested grasp target in the image. Return only JSON. "
                        "box_xyxy must be pixel coordinates [x1,y1,x2,y2] around the visible object."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Target to grasp: {prompt}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                },
            ],
            extra_body={"response_format": schema, "max_tokens": 200},
        )
        try:
            return json.loads(content)
        except Exception as exc:
            raise VisionGrabError(f"OpenAI vision localization returned invalid JSON: {content!r}") from exc

    def _run_ik_grab(
        self,
        *,
        frame: dict[str, Any],
        label: str,
        confidence: float,
        box_xyxy: tuple[int, int, int, int],
        arm_override: str,
    ) -> str:
        _bootstrap_hand_pose_navigation()
        try:
            import numpy as np
            from hand_pose_navigation.direct_nav import DirectHandPoseNav, _make_transform
            from hand_pose_navigation.grasp_planner import GraspPlanner
            from hand_pose_navigation.reachability_checker import ReachabilityChecker
            from hand_pose_navigation.target_detector import CameraIntrinsics, DetectionResult
        except Exception as exc:
            raise VisionGrabError(f"hand_pose_navigation IK modules are unavailable: {exc}") from exc

        depth_m = frame["depth_m"]
        u, v, z = _box_depth_center(depth_m, box_xyxy)
        if z <= 0.05:
            raise VisionGrabError(f"no valid depth for localized {label!r} at box={box_xyxy}")

        h, w = depth_m.shape[:2]
        intr = CameraIntrinsics(width=int(w), height=int(h), cx=float(w) / 2.0, cy=float(h) / 2.0)
        p_camera = intr.deproject(float(u), float(v), float(z))
        T_base_camera = _make_transform(
            xyz=(DEFAULT_CAMERA_EXTRINSIC["x"], DEFAULT_CAMERA_EXTRINSIC["y"], DEFAULT_CAMERA_EXTRINSIC["z"]),
            rpy=(DEFAULT_CAMERA_EXTRINSIC["roll"], DEFAULT_CAMERA_EXTRINSIC["pitch"], DEFAULT_CAMERA_EXTRINSIC["yaw"]),
        )
        p_h = np.ones(4, dtype=np.float64)
        p_h[:3] = p_camera
        p_base = (T_base_camera @ p_h)[:3]
        if not _plausible_object_base_point(p_base):
            raise VisionGrabError(
                "localized target has implausible base-frame position "
                f"camera_xyz=({p_camera[0]:+.3f},{p_camera[1]:+.3f},{p_camera[2]:+.3f})m "
                f"base_xyz=({p_base[0]:+.3f},{p_base[1]:+.3f},{p_base[2]:+.3f})m "
                f"box={box_xyxy}. Move the object into the reachable area or check camera extrinsics/depth alignment."
            )
        T_base_object = _stable_vision_object_pose(p_base)
        side = str(arm_override).strip().lower()
        if side not in {"left", "right"}:
            side = "left" if p_base[1] > 0.12 else "right"
        standoff_m, T_base_desired, reach_reason = _choose_reachable_standoff(
            T_base_object,
            side,
            GraspPlanner=GraspPlanner,
            ReachabilityChecker=ReachabilityChecker,
        )
        if standoff_m is None:
            raise VisionGrabError(
                f"localized {label!r}, but no reachable wrist target was found: {reach_reason}. "
                f"object_base_xyz=({p_base[0]:+.3f},{p_base[1]:+.3f},{p_base[2]:+.3f})m "
                f"camera_xyz=({p_camera[0]:+.3f},{p_camera[1]:+.3f},{p_camera[2]:+.3f})m "
                f"box={box_xyxy}"
            )
        T_camera_object = np.linalg.inv(T_base_camera) @ T_base_object

        fixed_result = DetectionResult(
            T_camera_object=T_camera_object,
            confidence=confidence,
            method="fixed",
            pixel_uv=(float(u), float(v)),
            depth_m=float(z),
        )
        config = {
            "arm": side,
            "detection_method": "fixed",
            "standoff_m": float(standoff_m),
            "rate_hz": 8.0,
            "timeout_s": 18.0,
            "ik_solver": "dls",
            "iface": self.iface,
            "domain_id": self.domain_id,
            "mock": False,
            "camera_x": DEFAULT_CAMERA_EXTRINSIC["x"],
            "camera_y": DEFAULT_CAMERA_EXTRINSIC["y"],
            "camera_z": DEFAULT_CAMERA_EXTRINSIC["z"],
            "camera_roll": DEFAULT_CAMERA_EXTRINSIC["roll"],
            "camera_pitch": DEFAULT_CAMERA_EXTRINSIC["pitch"],
            "camera_yaw": DEFAULT_CAMERA_EXTRINSIC["yaw"],
            "ik_tol_pos_m": 0.06,
            "ik_tol_rot_rad": 3.14,
            "convergence_pos_m": 0.06,
            "convergence_rot_rad": 3.14,
            "max_joint_step_rad": 0.06,
            "max_joint_speed_rad_s": 0.15,
            "max_reach_m": 0.42,
        }
        nav = DirectHandPoseNav(config, fixed_result=fixed_result, robot=self.robot)
        ok = False
        last_status: dict[str, Any] = {}
        try:
            deadline = time.time() + float(config["timeout_s"]) + 2.0
            while time.time() < deadline:
                last_status = nav.status_snapshot()
                if last_status.get("converged"):
                    ok = True
                    break
                if not last_status.get("running", True):
                    break
                time.sleep(0.2)
        finally:
            nav.shutdown()

        if not ok:
            raise VisionGrabError(
                f"IK did not converge for {label!r}; "
                f"object_base_xyz=({p_base[0]:+.3f},{p_base[1]:+.3f},{p_base[2]:+.3f})m "
                f"desired_wrist_xyz=({T_base_desired[0, 3]:+.3f},{T_base_desired[1, 3]:+.3f},{T_base_desired[2, 3]:+.3f})m "
                f"standoff={float(standoff_m):.3f}m last_status={_compact_status(last_status)}"
            )
        close_msg = _close_hand_best_effort(self.robot, side)
        return (
            f"localized {label!r} at pixel=({u},{v}) depth={z:.2f}m; "
            f"moved {side} end effector toward target with IK. {close_msg}"
        )


def _bootstrap_llm_transport() -> None:
    here = Path(__file__).resolve()
    scripts_dir = next((parent for parent in here.parents if (parent / "llm_client").exists()), None)
    if scripts_dir is None:
        scripts_dir = here.parents[1]
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _clean_grab_prompt(text: str) -> str:
    cleaned = str(text).strip()
    prefixes = (
        "please grab ",
        "please grba ",
        "the grab ",
        "the grba ",
        "grab ",
        "grba ",
        "grasp ",
        "pick up ",
        "nimm ",
        "greif ",
        "greife ",
    )
    while True:
        lowered = cleaned.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        else:
            break
    lowered = cleaned.lower()
    for marker in (" grab ", " grba ", " grasp ", " pick up ", " greif ", " greife ", " nimm "):
        idx = lowered.rfind(marker)
        if idx >= 0:
            cleaned = cleaned[idx + len(marker):].strip()
            break
    return cleaned or "object"


def _coerce_box(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise VisionGrabError(f"invalid localization box: {value!r}")
    x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    if x2 <= x1 or y2 <= y1:
        raise VisionGrabError(f"invalid localization box: {value!r}")
    return x1, y1, x2, y2


def _box_depth_center(depth_m: Any, box: tuple[int, int, int, int]) -> tuple[int, int, float]:
    import numpy as np

    h, w = depth_m.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        raise VisionGrabError(f"localization box is outside image: {box!r}")
    cx0 = int(x1 + 0.30 * (x2 - x1))
    cx1 = int(x1 + 0.70 * (x2 - x1))
    cy0 = int(y1 + 0.30 * (y2 - y1))
    cy1 = int(y1 + 0.70 * (y2 - y1))
    roi = depth_m[max(y1, cy0):min(y2, cy1), max(x1, cx0):min(x2, cx1)]
    valid = roi[roi > 0.05]
    if valid.size == 0:
        roi = depth_m[y1:y2, x1:x2]
        valid = roi[roi > 0.05]
    if valid.size == 0:
        return ((x1 + x2) // 2, (y1 + y2) // 2, 0.0)
    # Use a near-surface depth, not the median. Boxes around cups often include
    # table/floor behind the object; the median can put the target far past the
    # actual grasp surface.
    return ((x1 + x2) // 2, (y1 + y2) // 2, float(np.percentile(valid, 25)))


def _stable_vision_object_pose(p_base: Any) -> Any:
    import numpy as np

    T = np.eye(4, dtype=np.float64)
    pos = np.asarray(p_base, dtype=np.float64).copy()
    pos[2] += 0.04
    z_axis = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    y_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= max(1e-9, np.linalg.norm(x_axis))
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(1e-9, np.linalg.norm(y_axis))
    T[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T[:3, 3] = pos
    return T


def _plausible_object_base_point(p_base: Any) -> bool:
    x, y, z = [float(v) for v in p_base[:3]]
    return 0.05 <= x <= 0.75 and -0.55 <= y <= 0.55 and -0.05 <= z <= 0.80


def _choose_reachable_standoff(
    T_base_object: Any,
    arm: str,
    *,
    GraspPlanner: Any,
    ReachabilityChecker: Any,
) -> tuple[Optional[float], Any, str]:
    checker = ReachabilityChecker(arm=arm, max_reach_m=0.42)
    best: tuple[float, Any, list[str]] | None = None
    # For near objects, a smaller standoff keeps the wrist out of the torso.
    # For far objects, a larger standoff pulls the wrist back toward the robot.
    for standoff_m in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
        T_desired = GraspPlanner(arm=arm, standoff_m=standoff_m).compute(T_base_object)
        result = checker.check_target_reachable(T_desired)
        if result.safe:
            return standoff_m, T_desired, "reachable"
        if best is None or len(result.reasons) < len(best[2]):
            best = (standoff_m, T_desired, list(result.reasons))
    if best is None:
        return None, None, "no standoff candidates evaluated"
    return None, best[1], "; ".join(best[2])


def _close_hand_best_effort(robot: Any, arm: str) -> str:
    if hasattr(robot, "hand_close"):
        try:
            robot.hand_close(hand=arm, hold_s=0.6, ramp_s=0.25)
            return f"Closed {arm} hand."
        except TypeError:
            try:
                robot.hand_close(arm)
                return f"Closed {arm} hand."
            except Exception as exc:
                return f"Hand close failed: {exc}"
        except Exception as exc:
            return f"Hand close failed: {exc}"
    return "Hand close skipped: robot has no hand_close method."


def _compact_status(status: dict[str, Any]) -> dict[str, Any]:
    keys = ("running", "converged", "last_error_pos_m", "last_error_rot_rad", "ik_failures", "safety_rejections")
    return {key: status.get(key) for key in keys if key in status}
