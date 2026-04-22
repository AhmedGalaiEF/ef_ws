#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sdk_client import Robot


try:
    import torch
except ImportError as exc:
    raise SystemExit("torch is not installed. Install it with: pip install torch") from exc

try:
    from transformers import CLIPModel, CLIPProcessor  # type: ignore
except ImportError as exc:
    raise SystemExit("transformers is not installed. Install it with: pip install transformers") from exc


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 4 perception example: capture a robot camera frame and run CLIP zero-shot perception."
    )
    parser.add_argument("--iface", default="eth0", help="Network interface for the robot SDK.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--no-safety-boot",
        action="store_true",
        help="Skip the robot safety boot sequence during initialization.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="VideoClient timeout in seconds.",
    )
    parser.add_argument(
        "--target",
        default="a soda can",
        help="Positive text prompt for CLIP zero-shot perception.",
    )
    parser.add_argument(
        "--negative",
        default="no target object, an empty scene",
        help="Negative/background text prompt for CLIP zero-shot perception.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for a positive detection.",
    )
    parser.add_argument(
        "--model-name",
        default="openai/clip-vit-base-patch32",
        help="HuggingFace CLIP model name.",
    )
    parser.add_argument(
        "--localize",
        action="store_true",
        help="Also run a sliding-window CLIP search to estimate a target region.",
    )
    parser.add_argument(
        "--downscale",
        type=float,
        default=1.0,
        help="Optional image downscale factor before CLIP inference (0-1].",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional path for the annotated output image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated result in an OpenCV window.",
    )
    return parser.parse_args()


def capture_frame(robot: Robot, timeout: float) -> np.ndarray:
    try:
        robot._get_video_client().SetTimeout(float(timeout))
    except Exception:
        pass
    frame = robot.get_camera_frame_bgr()
    if frame is None or frame.size == 0:
        raise RuntimeError("Robot camera returned an empty frame.")
    return frame


def load_clip(model_name: str, device: str | None = None) -> tuple[CLIPModel, CLIPProcessor, str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model '{model_name}' on '{device}' ...")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor, device


def classify_frame(
    frame_bgr: np.ndarray,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    positive_prompt: str,
    negative_prompt: str,
    threshold: float,
) -> dict[str, Any]:
    prompts = [positive_prompt, negative_prompt]
    rgb = np.ascontiguousarray(frame_bgr[..., ::-1])

    with torch.no_grad():
        inputs = processor(text=prompts, images=rgb, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    scores = {label: float(score) for label, score in zip(prompts, probs)}
    positive_score = scores[positive_prompt]
    best_idx = int(np.argmax(probs))
    return {
        "detected": positive_score >= threshold,
        "confidence": positive_score,
        "label": prompts[best_idx],
        "scores": scores,
    }


def resize_for_inference(frame_bgr: np.ndarray, downscale: float) -> tuple[np.ndarray, float]:
    scale = max(0.05, min(1.0, float(downscale)))
    if scale >= 0.999:
        return frame_bgr, 1.0
    resized = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def generate_boxes(width: int, height: int, step_frac: float = 0.25) -> list[Box]:
    boxes: list[Box] = []
    min_dim = min(width, height)
    for scale in (0.6, 0.8, 1.0):
        size = int(min_dim * scale)
        if size <= 0:
            continue
        step = max(12, int(size * step_frac))
        for y in range(0, max(1, height - size + 1), step):
            for x in range(0, max(1, width - size + 1), step):
                boxes.append(Box(x, y, x + size, y + size))
    return boxes or [Box(0, 0, width, height)]


def find_best_box(
    frame_bgr: np.ndarray,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
    positive_prompt: str,
    negative_prompt: str,
) -> tuple[float, Box]:
    height, width = frame_bgr.shape[:2]
    prompts = [positive_prompt, negative_prompt]
    boxes = generate_boxes(width, height)

    best_score = -1.0
    best_box = boxes[0]

    with torch.no_grad():
        for box in boxes:
            crop = frame_bgr[box.y1:box.y2, box.x1:box.x2]
            if crop.size == 0:
                continue
            crop_rgb = np.ascontiguousarray(crop[..., ::-1])
            inputs = processor(text=prompts, images=crop_rgb, return_tensors="pt", padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            positive_score = float(probs[0])
            if positive_score > best_score:
                best_score = positive_score
                best_box = box

    return best_score, best_box


def upscale_box(box: Box, inverse_scale: float) -> Box:
    return Box(
        x1=int(round(box.x1 * inverse_scale)),
        y1=int(round(box.y1 * inverse_scale)),
        x2=int(round(box.x2 * inverse_scale)),
        y2=int(round(box.y2 * inverse_scale)),
    )


def annotate_frame(
    frame_bgr: np.ndarray,
    result: dict[str, Any],
    positive_prompt: str,
    localized_box: Box | None = None,
    localized_score: float | None = None,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    detected = bool(result["detected"])
    color = (0, 255, 0) if detected else (0, 0, 255)
    status = "DETECTED" if detected else "NOT DETECTED"

    cv2.putText(
        annotated,
        f"Target: {status}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )
    cv2.putText(
        annotated,
        f"Prompt: {positive_prompt}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated,
        f"Confidence: {result['confidence']:.1%}",
        (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    if localized_box is not None and localized_score is not None:
        cv2.rectangle(
            annotated,
            (localized_box.x1, localized_box.y1),
            (localized_box.x2, localized_box.y2),
            (255, 200, 0),
            2,
        )
        cv2.putText(
            annotated,
            f"Best region: {localized_score:.1%}",
            (localized_box.x1, max(20, localized_box.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
        )

    return annotated


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.threshold <= 1.0):
        print("Error: --threshold must be in the range [0, 1].")
        return 1

    try:
        robot = Robot(
            iface=args.iface,
            domain_id=args.domain_id,
            safety_boot=not args.no_safety_boot,
            auto_start_sensors=False,
        )
    except Exception as exc:
        print(f"Failed to connect to robot: {exc}")
        return 1

    try:
        frame_bgr = capture_frame(robot, timeout=args.timeout)
    except Exception as exc:
        print(f"Camera capture failed: {exc}")
        return 1

    print(f"Captured frame: {frame_bgr.shape[1]}x{frame_bgr.shape[0]} pixels")
    inference_frame, scale = resize_for_inference(frame_bgr, args.downscale)

    try:
        model, processor, device = load_clip(args.model_name)
    except Exception as exc:
        print(f"Failed to load CLIP model: {exc}")
        return 1

    try:
        result = classify_frame(
            inference_frame,
            model,
            processor,
            device,
            positive_prompt=args.target,
            negative_prompt=args.negative,
            threshold=args.threshold,
        )
    except Exception as exc:
        print(f"CLIP classification failed: {exc}")
        return 1

    localized_box = None
    localized_score = None
    if args.localize:
        try:
            localized_score, box_small = find_best_box(
                inference_frame,
                model,
                processor,
                device,
                positive_prompt=args.target,
                negative_prompt=args.negative,
            )
            localized_box = upscale_box(box_small, inverse_scale=(1.0 / scale))
        except Exception as exc:
            print(f"Localization step failed: {exc}")

    print("\nPerception result:")
    print(f"  Detected  : {result['detected']}")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Best label: {result['label']}")
    for label, score in result["scores"].items():
        print(f"  Score for '{label}': {score:.1%}")
    if localized_score is not None:
        print(f"  Best region score: {localized_score:.1%}")

    annotated = annotate_frame(
        frame_bgr,
        result,
        positive_prompt=args.target,
        localized_box=localized_box,
        localized_score=localized_score,
    )

    if args.save:
        cv2.imwrite(args.save, annotated)
        print(f"\nAnnotated image saved to: {args.save}")

    if args.show:
        cv2.imshow("Day 4 Perception", annotated)
        print("Press any key to close the window ...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if not args.save and not args.show:
        print("\nTip: use --save result.jpg or --show to inspect the annotated result.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
