#!/usr/bin/env python3
"""
Walk 3m forward, detect an object, then run a PBD arm motion sequence.

Sequence:
1) Safe boot to standing/walking-ready (FSM-200).
2) Walk a configurable forward distance (default 3.0m) using live pose.
3) Capture one camera frame and run CLIP zero-shot detection.
4) If detection is positive, replay a predefined motion CSV via pbd_reproduce.py.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

from unitree_sdk2py.go2.video.video_client import VideoClient

_SCRIPTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

for _subdir in (
    os.path.join("navigation", "obstacle_avoidance"),
    "obj_detection",
    os.path.join("arm_motion", "pbd"),
    os.path.join("basic", "safety"),
):
    _path = os.path.join(_SCRIPTS_DIR, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from hanger_boot_sequence import hanger_boot_sequence
from obstacle_detection import ObstacleDetector
from locomotion import Locomotion
from soda_can_detect import load_clip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G1 usecase: walk 3m, detect object, touch object.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iface", default="eth0", help="Network interface connected to robot")
    parser.add_argument("--distance", type=float, default=3.0, help="Forward distance in meters")
    parser.add_argument("--max-speed", type=float, default=0.18, help="Max walking speed (m/s)")
    parser.add_argument("--walk-timeout", type=float, default=40.0, help="Timeout for walking (s)")
    parser.add_argument("--threshold", type=float, default=0.6, help="CLIP confidence threshold (0-1)")
    parser.add_argument(
        "--label",
        default="a pink object",
        help="Target object text label for CLIP (e.g. 'a pink object')",
    )
    parser.add_argument("--camera-timeout", type=float, default=3.0, help="Camera RPC timeout (s)")
    parser.add_argument(
        "--motion-file",
        default=os.path.join("..", "arm_motion", "pbd", "motion_database_boxing_1.csv"),
        help="Motion CSV/NPZ file for pbd_reproduce.py",
    )
    parser.add_argument("--no-touch", action="store_true", help="Detect only; do not replay arm motion")
    return parser.parse_args()


def capture_frame_no_init(timeout: float = 3.0) -> np.ndarray:
    client = VideoClient()
    client.SetTimeout(timeout)
    client.Init()

    code, data = client.GetImageSample()
    if code != 0:
        raise RuntimeError(f"VideoClient.GetImageSample failed with error code {code}")

    jpeg_bytes = np.frombuffer(bytes(data), dtype=np.uint8)
    frame = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("Failed to decode camera frame")
    return frame


def classify_target(
    frame: np.ndarray,
    model: Any,
    processor: Any,
    device: str,
    target_label: str,
    threshold: float,
) -> dict[str, Any]:
    labels = [target_label, "a scene without a pink object"]
    rgb = np.ascontiguousarray(frame[..., ::-1])

    with torch.no_grad():
        inputs = processor(text=labels, images=rgb, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    target_score = float(probs[0])
    best_idx = int(np.argmax(probs))
    return {
        "detected": target_score >= threshold,
        "confidence": target_score,
        "label": labels[best_idx],
        "scores": {labels[i]: float(probs[i]) for i in range(len(labels))},
    }


def run_pbd_motion_sequence(iface: str, motion_file: str) -> None:
    script_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "arm_motion", "pbd", "pbd_reproduce.py"))
    requested_file = os.path.normpath(os.path.join(os.path.dirname(__file__), motion_file))
    fallback_file = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "arm_motion", "pbd", "motion_databse", "boxing_1.csv")
    )

    if os.path.exists(requested_file):
        motion_path = requested_file
    elif os.path.exists(fallback_file):
        motion_path = fallback_file
        print(f"Motion file not found at '{requested_file}', using fallback '{motion_path}'.")
    else:
        raise SystemExit(
            f"Motion file not found. Checked '{requested_file}' and '{fallback_file}'."
        )

    cmd = [
        sys.executable,
        script_path,
        "--iface",
        iface,
        "--file",
        motion_path,
        "--arm",
        "both",
        "--mode",
        "joint",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if args.distance <= 0:
        raise SystemExit("--distance must be > 0")
    if not (0.0 <= args.threshold <= 1.0):
        raise SystemExit("--threshold must be in [0, 1]")

    print(f"Initialising robot on interface '{args.iface}' ...")
    loco = hanger_boot_sequence(iface=args.iface)

    detector = ObstacleDetector(warn_distance=0.8, stop_distance=0.4)
    detector.start()
    time.sleep(1.0)
    if detector.is_stale():
        raise SystemExit("No SportModeState data received; check network/robot state.")

    walker = Locomotion(loco, detector, max_vx=args.max_speed)

    print("Loading CLIP model ...")
    model, processor, device = load_clip()

    x0, y0, yaw0 = detector.get_pose()
    target_x = x0 + args.distance * math.cos(yaw0)
    target_y = y0 + args.distance * math.sin(yaw0)

    print(f"Walking {args.distance:.2f}m forward: ({x0:+.2f}, {y0:+.2f}) -> ({target_x:+.2f}, {target_y:+.2f})")
    walked = walker.walk_to(target_x, target_y, timeout=args.walk_timeout)
    if not walked:
        walker.stop()
        raise SystemExit("Walking failed (obstacle/timeout/stale data).")

    print("Capturing image for object detection ...")
    frame = capture_frame_no_init(timeout=args.camera_timeout)
    result = classify_target(
        frame=frame,
        model=model,
        processor=processor,
        device=device,
        target_label=args.label,
        threshold=args.threshold,
    )
    print(f"Detected={result['detected']} confidence={result['confidence']:.1%} best_label='{result['label']}'")

    if result["detected"] and not args.no_touch:
        print("Target detected. Replaying motion via pbd_reproduce.py ...")
        run_pbd_motion_sequence(iface=args.iface, motion_file=args.motion_file)
        print("Motion replay complete.")
    elif args.no_touch:
        print("--no-touch enabled; skipping arm motion replay.")
    else:
        print("Target not detected; skipping arm motion replay.")

    walker.stop()
    print("Done.")


if __name__ == "__main__":
    main()
