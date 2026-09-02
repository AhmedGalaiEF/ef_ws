"""Heavy reusable support for the G1 Academy notebooks.

The notebooks intentionally use Unitree SDK clients and DDS channels directly.
This module contains only items that are impractical to retype: documented
mode/joint/hand target maps, Piper WAV conversion plus SDK audio playback, and
integration points for academy-supplied perception and IK pipelines.
"""
from __future__ import annotations

import audioop
import importlib
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

FSM_IDS = {
    "zero_torque": 0,
    "damp": 1,
    "sit": 3,
    "prepare": 4,
    "walk": 501,
    "run": 802,
}
JOINT_GROUPS = {
    "left_leg": list(range(0, 6)),
    "right_leg": list(range(6, 12)),
    "waist": list(range(12, 15)),
    "left_arm": list(range(15, 22)),
    "right_arm": list(range(22, 29)),
}
HAND_JOINT_NAMES = ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"]
HAND_OPEN = {
    "left": [-0.0993, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75],
    "right": [-0.0351, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0],
}
HAND_CLOSED = {
    "left": [-0.0993, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0],
    "right": [-0.0351, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75],
}
PIPER_VOICES = {
    "en": "en_US-lessac-medium",
    "de": "de_DE-thorsten-medium",
    "fr": "fr_FR-siwis-medium",
    "es": "es_ES-davefx-medium",
}


def play_piper_text(audio_client: Any, text: str, language: str = "en", volume: int = 100) -> int:
    """Synthesize text with Piper and play robot-compatible PCM through AudioClient."""
    language = language.lower().replace("-", "_")
    voice = PIPER_VOICES.get(language, PIPER_VOICES.get(language[:2]))
    if voice is None:
        raise ValueError(f"Unsupported configured Piper language: {language}")
    piper = os.environ.get("G1_PIPER_BIN", "piper")
    model = Path(os.environ.get("G1_PIPER_VOICE_DIR", str(Path.home() / ".local/share/piper/voices"))) / voice / f"{voice}.onnx"
    if not model.exists():
        raise FileNotFoundError(f"Piper model not found: {model}")
    with tempfile.TemporaryDirectory(prefix="g1_piper_") as tmp:
        wav_path = Path(tmp) / "speech.wav"
        subprocess.run([piper, "--model", str(model), "--output-file", str(wav_path)], input=text, text=True, check=True)
        with wave.open(str(wav_path), "rb") as src:
            channels, width, rate = src.getnchannels(), src.getsampwidth(), src.getframerate()
            pcm = src.readframes(src.getnframes())
        if channels == 2:
            pcm = audioop.tomono(pcm, width, 0.5, 0.5)
        if width != 2:
            pcm = audioop.lin2lin(pcm, width, 2)
        if rate != 16000:
            pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 16000, None)
        audio_client.SetVolume(int(volume))
        result = audio_client.PlayStream("academy_piper", "academy-piper", pcm)
        return int(result[0] if isinstance(result, tuple) else result)


def load_provided_pipeline(module_name: str):
    """Load an academy-provided perception/pose/IK module by its documented name."""
    return importlib.import_module(module_name)


# Delivery-pipeline boilerplate.  It deliberately imports optional heavy
# packages only at call time, so basic DDS notebooks do not require them.
from dataclasses import dataclass
import base64
import json as _json


@dataclass(frozen=True)
class Detection2D:
    label: str
    confidence: float
    center_px: tuple[float, float]
    raw: dict


@dataclass(frozen=True)
class PoseEstimate:
    translation_m: tuple[float, float, float]
    rotation_rvec: tuple[float, float, float] | None = None
    frame: str = "camera"


class DeliveryPipeline:
    """Reusable OpenAI/ArUco/transform/IK orchestration for notebook 4.

    Supply camera intrinsics, camera_to_base and wrist_to_palm 4x4 transforms
    from the calibrated academy configuration.  Supply ik_increment as a
    bounded direct arm function accepting (dx, dy, dz, side=...).  The class
    handles perception result parsing, pose transforms, and small-step
    invocation; it does not hide robot command ownership.
    """

    def __init__(self, camera_matrix, distortion, camera_to_base, wrist_to_palm, openai_client=None, model=None):
        import numpy as np
        self.np = np
        self.camera_matrix = np.asarray(camera_matrix, dtype=float).reshape(3, 3)
        self.distortion = np.asarray(distortion, dtype=float)
        self.camera_to_base = np.asarray(camera_to_base, dtype=float).reshape(4, 4)
        self.wrist_to_palm = np.asarray(wrist_to_palm, dtype=float).reshape(4, 4)
        self.client = openai_client
        self.model = model or os.environ.get("OPENAI_VISION_MODEL", "gpt-4.1-mini")

    def _client(self):
        if self.client is None:
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "OpenAI vision is unavailable. Install the optional 'openai' package "
                    "or pass an initialized openai_client to DeliveryPipeline."
                ) from exc
            self.client = OpenAI()
        return self.client

    def detect_object(self, rgb_jpeg: bytes, prompt: str) -> Detection2D:
        """Ask OpenAI for one object center; response is strict JSON."""
        image_url = "data:image/jpeg;base64," + base64.b64encode(rgb_jpeg).decode("ascii")
        request = {
            "label": "short object label",
            "confidence": "number 0..1",
            "center_px": "[x, y] pixel center in the supplied image",
        }
        try:
            client = self._client()
        except RuntimeError as exc:
            return Detection2D(
                label="unavailable",
                confidence=0.0,
                center_px=(float("nan"), float("nan")),
                raw={"available": False, "reason": str(exc)},
            )

        response = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": f"{prompt}. Return JSON matching: {request}"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
        )
        raw = _json.loads(response.choices[0].message.content)
        center = raw["center_px"]
        return Detection2D(str(raw["label"]), float(raw["confidence"]), (float(center[0]), float(center[1])), raw)

    def aruco_pose(self, bgr_image, marker_length_m: float, dictionary_name: str = "DICT_4X4_50") -> PoseEstimate:
        """Estimate the first detected ArUco marker pose in the camera frame."""
        import cv2
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV was installed without the contrib ArUco module.")
        aruco = cv2.aruco
        dictionary_id = getattr(aruco, dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
        dictionary = aruco.getPredefinedDictionary(dictionary_id)
        if hasattr(aruco, "ArucoDetector"):
            corners, ids, _ = aruco.ArucoDetector(dictionary).detectMarkers(bgr_image)
        elif hasattr(aruco, "detectMarkers"):
            corners, ids, _ = aruco.detectMarkers(bgr_image, dictionary)
        else:
            raise RuntimeError("This OpenCV ArUco build exposes no marker detector.")
        if ids is None or not len(corners):
            raise RuntimeError("No ArUco marker detected.")
        if hasattr(aruco, "estimatePoseSingleMarkers"):
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, float(marker_length_m), self.camera_matrix, self.distortion
            )
            rvec, tvec = rvecs[0].reshape(3), tvecs[0].reshape(3)
        else:
            half = float(marker_length_m) / 2.0
            object_points = self.np.array(
                [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
                dtype=self.np.float32,
            )
            image_points = self.np.asarray(corners[0], dtype=self.np.float32).reshape(4, 2)
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.distortion,
                flags=getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
            )
            if not success:
                raise RuntimeError("Unable to estimate the ArUco marker pose.")
            rvec, tvec = rvec.reshape(3), tvec.reshape(3)
        return PoseEstimate(tuple(float(x) for x in tvec), tuple(float(x) for x in rvec), "camera")

    def transform_pose(self, pose: PoseEstimate, transform, frame: str) -> PoseEstimate:
        matrix = self.np.asarray(transform, dtype=float).reshape(4, 4)
        point = matrix @ self.np.array([*pose.translation_m, 1.0])
        return PoseEstimate(tuple(float(x) for x in point[:3]), pose.rotation_rvec, frame)

    def marker_to_palm_target(self, marker_pose: PoseEstimate) -> PoseEstimate:
        """Camera marker pose -> base wrist target -> base palm target."""
        base_marker = self.transform_pose(marker_pose, self.camera_to_base, "base")
        return self.transform_pose(base_marker, self.wrist_to_palm, "base_palm")

    def execute_incremental_ik(self, ik_increment, current_palm_xyz, target: PoseEstimate, side: str = "right", max_step_m: float = 0.025):
        """Call a bounded direct IK increment function until target translation is reached."""
        current = self.np.asarray(current_palm_xyz, dtype=float)
        goal = self.np.asarray(target.translation_m, dtype=float)
        results = []
        while True:
            delta = goal - current
            distance = float(self.np.linalg.norm(delta))
            if distance <= 1e-4:
                return results
            step = delta * min(1.0, float(max_step_m) / distance)
            result = ik_increment(float(step[0]), float(step[1]), float(step[2]), side=side)
            results.append(result)
            current = current + step


class RecognitionIkIncrement:
    """Relative Cartesian IK executor adapted from recognition_app_v3.

    robot must provide the recognition stack's get_joint_states() and
    move_upper_body_joint(...) interface (for example its sdk_client.Robot).
    The executor reads the current 7-DoF arm state, computes FK, offsets the
    target translation, solves DLS IK, then executes a safety-gated, speed-
    limited interpolated joint trajectory through ArmExecutor.
    """

    def __init__(self, robot, arm="right", max_reach_m=0.42, max_joint_step_rad=0.08, max_joint_speed_rad_s=0.15, kp=30.0, kd=1.5):
        from hand_pose_navigation.arm_fk import ArmFK
        from hand_pose_navigation.arm_ik import ArmIK
        from hand_pose_navigation.arm_executor import ArmExecutor
        self.robot, self.arm = robot, arm
        self.fk = ArmFK(arm=arm, backend="urdf")
        self.ik = ArmIK(arm=arm, solver="dls", tol_pos_m=0.005, tol_rot_rad=0.05)
        self.executor = ArmExecutor(
            robot, arm=arm, kp=kp, kd=kd, max_reach_m=max_reach_m,
            max_joint_step_rad=max_joint_step_rad,
            max_joint_speed_rad_s=max_joint_speed_rad_s,
        )

    def current_palm_xyz(self):
        q_current = self.executor._read_current_arm_q()
        if q_current is None:
            raise RuntimeError("Current arm joint state is unavailable.")
        transform = self.fk.compute_arm(q_current)
        return tuple(float(value) for value in transform[:3, 3])

    def __call__(self, dx, dy, dz, side=None, duration_s=1.0, stop_event=None):
        if side is not None and str(side).lower() != self.arm:
            raise ValueError(f"This incrementer controls {self.arm}, not {side}.")
        q_current = self.executor._read_current_arm_q()
        if q_current is None:
            return {"success": False, "reason": "joint_state_unavailable"}
        target_transform = self.fk.compute_arm(q_current).copy()
        target_transform[0, 3] += float(dx)
        target_transform[1, 3] += float(dy)
        target_transform[2, 3] += float(dz)
        q_solution, ik_info = self.ik.solve(target_transform, q_init=q_current)
        if q_solution is None:
            return {"success": False, "reason": "ik_failed", "ik_info": ik_info}
        result = self.executor.execute(
            q_solution, duration_s=duration_s, q_arm_start=q_current,
            T_base_desired=target_transform, stop_event=stop_event,
        )
        result["ik_info"] = ik_info
        result["target_translation_m"] = tuple(float(value) for value in target_transform[:3, 3])
        return result


def make_recognition_ik_increment(robot, arm="right", **kwargs):
    """Create the concrete ik_increment callback expected by DeliveryPipeline."""
    return RecognitionIkIncrement(robot, arm=arm, **kwargs)
