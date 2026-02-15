"""receive_realsense_gst_clip_can.py  –  GStreamer client + CLIP detection

This is a copy of receive_realsense_gst.py with CLIP-based energy drink can
detection overlaid on the live RGB stream.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

import open_clip

# GStreamer
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp


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
    for s in scales:
        size = int(min_dim * s)
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
        pil = Image.fromarray(crop_rgb)
        image_input = preprocess(pil).unsqueeze(0).to(device)
        with torch.inference_mode():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            positive_prob = float(logits[0, 0].item())
        if positive_prob > best_score:
            best_score = positive_prob
            best_box = box

    return best_score, best_box


def build_rgb_sink(port: int) -> tuple[GstApp.AppSink, Gst.Pipeline]:
    pipeline = Gst.parse_launch(
        f"udpsrc port={port} caps=application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=true sync=false drop=true"
    )
    sink = pipeline.get_by_name("sink")
    return sink, pipeline


def build_depth_sink(port: int) -> tuple[GstApp.AppSink, Gst.Pipeline]:
    pipeline = Gst.parse_launch(
        f"udpsrc port={port} caps=application/x-rtp,media=video,encoding-name=H264,payload=97 ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=true sync=false drop=true"
    )
    sink = pipeline.get_by_name("sink")
    return sink, pipeline


def _build_navigation_command(args: argparse.Namespace) -> list[str]:
    scripts_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    nav_script = os.path.join(
        scripts_root,
        "navigation",
        "obstacle_avoidance",
        "real_time_path_steps_dynamic.py",
    )
    cmd = [
        sys.executable,
        nav_script,
        "--map",
        args.nav_map,
        "--iface",
        args.nav_iface,
        "--domain-id",
        str(args.nav_domain_id),
        "--sport-topic",
        args.nav_sport_topic,
        "--slam-map-topic",
        args.nav_slam_map_topic,
    ]
    if args.nav_start_x is not None and args.nav_start_y is not None:
        cmd.extend(["--start-x", str(args.nav_start_x), "--start-y", str(args.nav_start_y)])
    if args.nav_goal_x is not None and args.nav_goal_y is not None:
        cmd.extend(["--goal-x", str(args.nav_goal_x), "--goal-y", str(args.nav_goal_y)])
    if args.nav_extra_args:
        cmd.extend(shlex.split(args.nav_extra_args))
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GStreamer RealSense receiver with CLIP energy drink can detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rgb-port", type=int, default=5600, help="UDP port for RGB stream")
    parser.add_argument("--depth-port", type=int, default=5602, help="UDP port for depth stream")
    parser.add_argument("--width", type=int, default=640, help="RGB width")
    parser.add_argument("--height", type=int, default=480, help="RGB height")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS")
    parser.add_argument("--threshold", type=float, default=0.85, help="Detection threshold (0-1)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--every", type=int, default=15, help="Run CLIP every N frames")
    parser.add_argument("--downscale", type=float, default=0.5, help="Downscale RGB for detection (0.1-1.0)")
    parser.add_argument("--step-frac", type=float, default=0.25, help="Sliding window step as fraction of size")
    parser.add_argument(
        "--positive",
        default="a photo of an energy drink can",
        help="Positive text prompt",
    )
    parser.add_argument(
        "--negative",
        default="a photo without an energy drink can",
        help="Negative text prompt",
    )
    parser.add_argument(
        "--trigger-nav-on-detect",
        action="store_true",
        help="Launch real_time_path_steps_dynamic.py when detection is above threshold",
    )
    parser.add_argument("--nav-map", default=None, help="Map path for navigation script")
    parser.add_argument("--nav-start-x", type=float, default=None, help="Optional nav start x")
    parser.add_argument("--nav-start-y", type=float, default=None, help="Optional nav start y")
    parser.add_argument("--nav-goal-x", type=float, default=None, help="Optional nav goal x")
    parser.add_argument("--nav-goal-y", type=float, default=None, help="Optional nav goal y")
    parser.add_argument("--nav-iface", default="eth0", help="DDS interface passed to navigation")
    parser.add_argument("--nav-domain-id", type=int, default=0, help="DDS domain id for navigation")
    parser.add_argument("--nav-sport-topic", default="rt/odommodestate", help="Sport topic for navigation")
    parser.add_argument("--nav-slam-map-topic", default="rt/utlidar/map_state", help="SLAM map topic for navigation")
    parser.add_argument("--nav-cooldown", type=float, default=30.0, help="Seconds between nav launches")
    parser.add_argument("--nav-repeat", action="store_true", help="Allow multiple navigation launches")
    parser.add_argument(
        "--nav-extra-args",
        default="",
        help="Extra args appended to real_time_path_steps_dynamic.py",
    )
    args = parser.parse_args()

    if args.trigger_nav_on_detect and not args.nav_map:
        raise SystemExit("--nav-map is required when --trigger-nav-on-detect is set")

    nav_cmd: list[str] | None = None
    if args.trigger_nav_on_detect:
        nav_cmd = _build_navigation_command(args)
        print("[nav] trigger enabled")
        print("[nav] command:", " ".join(shlex.quote(c) for c in nav_cmd))

    Gst.init(None)

    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text_tokens = tokenizer([args.positive, args.negative]).to(device)
    with torch.inference_mode():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    rgb_sink, rgb_pipeline = build_rgb_sink(args.rgb_port)
    depth_sink, depth_pipeline = build_depth_sink(args.depth_port)

    for p in (rgb_pipeline, depth_pipeline):
        p.set_state(Gst.State.PLAYING)

    scale = max(0.1, min(1.0, args.downscale))
    det_w = int(args.width * scale)
    det_h = int(args.height * scale)
    boxes = _generate_boxes(det_w, det_h, args.step_frac)

    last = time.perf_counter()
    last_score = -1.0
    last_box = None
    frame_idx = 0
    nav_proc: subprocess.Popen | None = None
    nav_last_launch = 0.0
    nav_started_once = False

    try:
        while True:
            sample_rgb = rgb_sink.emit("try-pull-sample", Gst.SECOND // args.fps)
            sample_d = depth_sink.emit("try-pull-sample", Gst.SECOND // args.fps)

            if not sample_rgb or not sample_d:
                time.sleep(0.005)
                continue

            buf_rgb = sample_rgb.get_buffer()
            buf_d = sample_d.get_buffer()

            rgb = np.frombuffer(buf_rgb.extract_dup(0, buf_rgb.get_size()), dtype=np.uint8)
            rgb = rgb.reshape((args.height, args.width, 3)).copy()

            depth_bgr = np.frombuffer(buf_d.extract_dup(0, buf_d.get_size()), dtype=np.uint8)
            depth_bgr = depth_bgr.reshape((args.height, args.width, 3))

            if frame_idx % max(1, args.every) == 0:
                if scale < 1.0:
                    small = cv2.resize(
                        rgb,
                        (det_w, det_h),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    small = rgb
                score, box = _best_clip_box(small, model, preprocess, text_features, device, boxes)
                if scale < 1.0:
                    box = Box(
                        int(box.x1 / scale),
                        int(box.y1 / scale),
                        int(box.x2 / scale),
                        int(box.y2 / scale),
                    )
                last_score = score
                last_box = box

            if last_box is not None:
                label = f"energy drink can prob: {last_score:.2f}"
                if last_score >= args.threshold:
                    cv2.rectangle(
                        rgb,
                        (last_box.x1, last_box.y1),
                        (last_box.x2, last_box.y2),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        rgb,
                        label,
                        (last_box.x1, max(20, last_box.y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        rgb,
                        f"no can (best {label})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            if nav_proc is not None and nav_proc.poll() is not None:
                print(f"[nav] exited with code {nav_proc.returncode}")
                nav_proc = None

            should_start_nav = (
                nav_cmd is not None
                and last_box is not None
                and last_score >= args.threshold
                and (args.nav_repeat or not nav_started_once)
                and nav_proc is None
                and (time.time() - nav_last_launch) >= max(0.0, args.nav_cooldown)
            )
            if should_start_nav:
                nav_proc = subprocess.Popen(nav_cmd)
                nav_last_launch = time.time()
                nav_started_once = True
                print("[nav] launched dynamic path navigation")

            combo = cv2.hconcat([rgb, depth_bgr])

            now = time.perf_counter()
            fps = 1.0 / (now - last)
            last = now
            cv2.putText(combo, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            if nav_cmd is not None:
                nav_status = "running" if nav_proc is not None else ("ready" if args.nav_repeat or not nav_started_once else "done")
                cv2.putText(combo, f"NAV: {nav_status}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("RGB + Depth (CLIP)", combo)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

            frame_idx += 1
    finally:
        if nav_proc is not None and nav_proc.poll() is None:
            nav_proc.terminate()
        for p in (rgb_pipeline, depth_pipeline):
            p.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Error:", exc)
        sys.exit(1)
