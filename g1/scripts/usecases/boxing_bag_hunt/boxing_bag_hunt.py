#!/usr/bin/env python3
"""
Walk, search for a pink boxing bag, then replay boxing_4 arm motion when found.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from unitree_sdk2py.go2.video.video_client import VideoClient

import open_clip

try:
    from rich.console import Console
except Exception:  # pragma: no cover
    Console = None

SCRIPTS_ROOT = Path(__file__).resolve().parents[2]
DEV_DIR = SCRIPTS_ROOT / "dev"
if str(DEV_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_DIR))

from ef_client import Robot  # type: ignore  # noqa: E402


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


def _generate_boxes(width: int, height: int, step_frac: float) -> List[Box]:
    boxes: List[Box] = []
    min_dim = min(width, height)
    scales = [0.6, 0.8, 1.0]
    for scale in scales:
        size = int(min_dim * scale)
        if size <= 0:
            continue
        step = max(12, int(size * step_frac))
        for y in range(0, height - size + 1, step):
            for x in range(0, width - size + 1, step):
                boxes.append(Box(x, y, x + size, y + size))
    if not boxes:
        boxes.append(Box(0, 0, width, height))
    return boxes


def _best_clip_box(
    image_bgr: np.ndarray,
    model,
    preprocess,
    text_features: torch.Tensor,
    device: torch.device,
    boxes: List[Box],
) -> Tuple[float, Box]:
    best_score = -1.0
    best_box = boxes[0]
    for box in boxes:
        crop = image_bgr[box.y1:box.y2, box.x1:box.x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image_input = preprocess(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
        with torch.inference_mode():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            score = float(logits[0, 0].item())
        if score > best_score:
            best_score = score
            best_box = box
    return best_score, best_box


class BoxingBagHunter:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.console = Console() if Console is not None else None
        self.robot = Robot(iface=args.iface, safety_boot=True, auto_start_sensors=True)
        self.device = torch.device(args.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self._text_cache: Dict[str, torch.Tensor] = {}

        self.scale = max(0.1, min(1.0, args.downscale))
        self.det_w = int(args.width * self.scale)
        self.det_h = int(args.height * self.scale)
        self.boxes = _generate_boxes(self.det_w, self.det_h, args.step_frac)

        self.video_client = VideoClient()
        self.video_client.SetTimeout(self.args.camera_timeout)
        self.video_client.Init()

        self.audio_client = self._init_audio_client()
        self._last_light_ts = 0.0

        if self.args.show_feed:
            try:
                cv2.namedWindow(self.args.window_name, cv2.WINDOW_NORMAL)
            except cv2.error as exc:
                self._log(f"[warn] disabling --show-feed (OpenCV display unavailable: {exc})", style="bold yellow")
                self.args.show_feed = False

    def _log(self, message: str, style: str = "white") -> None:
        if self.console is not None:
            self.console.print(message, style=style, markup=False)
        else:
            print(message)

    def _init_audio_client(self):
        try:
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

            client = AudioClient()
            client.SetTimeout(3.0)
            client.Init()
            return client
        except Exception as exc:
            self._log(f"[warn] audio/headlight client unavailable: {exc}", style="bold yellow")
            return None

    def _state_text(self, state: str) -> str:
        return {
            "walking": "walking forward",
            "detecting": "scanning with CLIP",
            "turning": "turning to continue search",
            "boxing": "boxing bag found, replaying motion",
            "not_found": "search finished, target not found",
        }.get(state, state)

    def _set_state_light(self, state: str) -> None:
        states = {
            "walking": ((0, 80, 255), "bold blue"),
            "detecting": ((255, 210, 0), "bold yellow"),
            "turning": ((255, 140, 0), "bold magenta"),
            "boxing": ((255, 0, 0), "bold red"),
            "not_found": ((180, 180, 180), "bold white"),
        }
        rgb, style = states.get(state, ((255, 255, 255), "white"))
        self._log(f"[{state}] {self._state_text(state)}", style=style)

        if self.audio_client is None:
            return

        now = time.time()
        # LedControl requires >=200ms between calls.
        if now - self._last_light_ts < 0.22:
            time.sleep(0.22 - (now - self._last_light_ts))

        code = self.audio_client.LedControl(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        self._last_light_ts = time.time()
        if code != 0:
            self._log(f"[warn] LedControl failed with code={code}", style="bold yellow")

    def _capture_frame(self) -> np.ndarray:
        code, data = self.video_client.GetImageSample()
        if code != 0:
            raise RuntimeError(f"VideoClient.GetImageSample failed with code {code}")
        frame = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode camera frame")
        return frame

    def _text_features_for(self, target: str) -> torch.Tensor:
        key = target.strip().lower()
        if key in self._text_cache:
            return self._text_cache[key]
        positive = f"a photo of {target}"
        negative = f"a photo without {target}"
        text_tokens = self.tokenizer([positive, negative]).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._text_cache[key] = text_features
        return text_features

    def _render_detection(
        self,
        frame: np.ndarray,
        step_idx: int,
        total_steps: int,
        det_box: Box | None = None,
        score: float | None = None,
        detected: bool | None = None,
        scan_frame_idx: int | None = None,
        scan_total_frames: int | None = None,
    ) -> None:
        if not self.args.show_feed:
            return

        color = (0, 180, 255)
        if det_box is not None and score is not None and detected is not None:
            h, w = frame.shape[:2]
            sx = w / float(self.det_w)
            sy = h / float(self.det_h)
            x1 = int(det_box.x1 * sx)
            y1 = int(det_box.y1 * sy)
            x2 = int(det_box.x2 * sx)
            y2 = int(det_box.y2 * sy)
            color = (0, 220, 0) if detected else (0, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"pink boxing bag: {score:.3f}"
        else:
            label = "pink boxing bag: waiting for next CLIP pass"

        status = f"step {step_idx}/{total_steps}"
        if scan_frame_idx is not None and scan_total_frames is not None:
            status = f"{status} | frame {scan_frame_idx}/{scan_total_frames}"
        if detected is not None:
            status = f"{status} | {'DETECTED' if detected else 'SEARCHING'}"

        cv2.putText(frame, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, status, (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.imshow(self.args.window_name, frame)
        cv2.waitKey(1)

    def detect(self, target: str, step_idx: int, total_steps: int) -> Tuple[bool, float]:
        frame = None
        for i in range(1, self.args.detect_every_frames + 1):
            frame = self._capture_frame()
            self._render_detection(
                frame=frame,
                step_idx=step_idx,
                total_steps=total_steps,
                scan_frame_idx=i,
                scan_total_frames=self.args.detect_every_frames,
            )

        assert frame is not None
        small = cv2.resize(frame, (self.det_w, self.det_h), interpolation=cv2.INTER_AREA)
        text_features = self._text_features_for(target)
        score, box = _best_clip_box(
            image_bgr=small,
            model=self.model,
            preprocess=self.preprocess,
            text_features=text_features,
            device=self.device,
            boxes=self.boxes,
        )
        detected = score >= self.args.threshold
        self._render_detection(
            frame=frame,
            det_box=box,
            score=score,
            detected=detected,
            step_idx=step_idx,
            total_steps=total_steps,
            scan_frame_idx=self.args.detect_every_frames,
            scan_total_frames=self.args.detect_every_frames,
        )
        return detected, score

    def _resolve_motion_file(self) -> Path:
        candidates = [
            SCRIPTS_ROOT / "arm_motion" / "pbd" / "motion_databse" / "boxing_4.csv",
            SCRIPTS_ROOT / "arm_motion" / "pbd" / "motion_database" / "boxing_4.csv",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not find boxing_4.csv in: {', '.join(str(p) for p in candidates)}")

    def _replay_boxing_motion(self) -> None:
        motion_file = self._resolve_motion_file()
        pbd_script = SCRIPTS_ROOT / "arm_motion" / "pbd" / "pbd_reproduce.py"
        cmd = [
            sys.executable,
            str(pbd_script),
            "--iface",
            self.args.iface,
            "--file",
            str(motion_file),
            "--arm",
            "both",
            "--mode",
            "joint",
        ]
        self._log(f"[boxing] replaying: {motion_file}", style="bold red")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"pbd_reproduce failed with exit code {result.returncode}")

    def _cleanup(self) -> None:
        if self.args.show_feed:
            cv2.destroyAllWindows()

    def run(self) -> int:
        try:
            self.robot.say("looking for boxing bag")
            self._set_state_light("walking")
            self._log("[motion] walking forward 6.0 meters", style="bold cyan")
            self.robot.run_for(6.0)

            steps = max(1, math.ceil(self.args.full_rotation_deg / self.args.turn_step_deg))
            for i in range(steps):
                self._set_state_light("detecting")
                detected, score = self.detect("pink boxing bag", i + 1, steps)
                self._log(
                    f"[detect] attempt {i + 1}/{steps} detected={detected} clip_prob={score:.3f}",
                    style="bold green" if detected else "yellow",
                )
                if detected:
                    self.robot.say("boxing bag is found")
                    self._set_state_light("boxing")
                    self._replay_boxing_motion()
                    return 0

                self.robot.say("I am looking for the box")
                self._set_state_light("turning")
                self._log(
                    f"[motion] turning {self.args.turn_step_deg:.1f} degrees to continue search",
                    style="bold magenta",
                )
                self.robot.turn_for(self.args.turn_step_deg)

            self._set_state_light("not_found")
            self.robot.say("no boxing bag found")
            return 1
        finally:
            self._cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run, search for pink boxing bag, and execute boxing_4 motion when detected.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iface", default="eth0", help="Network interface for robot DDS")
    parser.add_argument("--threshold", type=float, default=0.72, help="Detection threshold (0-1)")
    parser.add_argument("--camera-timeout", type=float, default=3.0, help="VideoClient timeout in seconds")
    parser.add_argument("--turn-step-deg", type=float, default=20.0, help="Turn step between detection attempts")
    parser.add_argument("--full-rotation-deg", type=float, default=360.0, help="Total sweep before giving up")
    parser.add_argument("--width", type=int, default=640, help="Camera width assumption")
    parser.add_argument("--height", type=int, default=480, help="Camera height assumption")
    parser.add_argument("--downscale", type=float, default=0.5, help="Detection downscale factor (0.1-1)")
    parser.add_argument("--step-frac", type=float, default=0.25, help="Sliding-window step fraction")
    parser.add_argument(
        "--detect-every-frames",
        type=int,
        default=150,
        help="Run CLIP detection once per this many frames (~5s at 30 FPS)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device")
    parser.add_argument(
        "--no-show-feed",
        dest="show_feed",
        action="store_false",
        help="Disable RGB feed window with CLIP box/probability overlay",
    )
    parser.add_argument("--window-name", default="boxing_bag_hunt", help="OpenCV window title for feed display")
    parser.set_defaults(show_feed=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.turn_step_deg <= 0:
        raise SystemExit("--turn-step-deg must be > 0")
    if args.full_rotation_deg <= 0:
        raise SystemExit("--full-rotation-deg must be > 0")
    if args.detect_every_frames <= 0:
        raise SystemExit("--detect-every-frames must be > 0")
    if not (0.0 <= args.threshold <= 1.0):
        raise SystemExit("--threshold must be in [0, 1]")

    hunter = BoxingBagHunter(args)
    return hunter.run()


if __name__ == "__main__":
    raise SystemExit(main())
