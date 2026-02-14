"""real_time_detection.py - GStreamer RGB stream + periodic CLIP captioning.

Receives an RGB stream over UDP (H264) and, every N seconds, runs a CLIP
classification against a small prompt set to produce a short text description.
"""

from __future__ import annotations

import argparse
import sys
import time
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


def build_rgb_sink(port: int) -> tuple[GstApp.AppSink, Gst.Pipeline]:
    pipeline = Gst.parse_launch(
        f"udpsrc port={port} caps=application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
        "rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=true sync=false drop=true"
    )
    sink = pipeline.get_by_name("sink")
    return sink, pipeline


def _build_prompts(prompt_str: str) -> List[str]:
    prompts = [p.strip() for p in prompt_str.split("|") if p.strip()]
    if not prompts:
        prompts = ["a photo of a robot", "a photo of a person", "a photo of a room"]
    return prompts


def _classify_frame(
    image_bgr: np.ndarray,
    model,
    preprocess,
    text_features: torch.Tensor,
    device: torch.device,
) -> Tuple[str, float]:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)
    image_input = preprocess(pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        image_features = model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
    best_idx = int(torch.argmax(probs).item())
    best_prob = float(probs[best_idx].item())
    return best_idx, best_prob


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GStreamer RGB receiver with periodic CLIP text description",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rgb-port", type=int, default=5600, help="UDP port for RGB stream")
    parser.add_argument("--width", type=int, default=640, help="RGB width")
    parser.add_argument("--height", type=int, default=480, help="RGB height")
    parser.add_argument("--fps", type=int, default=30, help="Stream FPS")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interval", type=float, default=20.0, help="Seconds between descriptions")
    parser.add_argument("--downscale", type=float, default=0.5, help="Downscale RGB for detection (0.1-1.0)")
    parser.add_argument(
        "--prompts",
        default=(
            "a photo of an energy drink can|"
            "a photo of a soda can|"
            "a photo of a robot|"
            "a photo of a person|"
            "a photo of a floor|"
            "a photo of a chair|"
            "a photo of a desk|"
            "a photo of a room"
        ),
        help="Pipe-separated prompt list",
    )
    args = parser.parse_args()

    Gst.init(None)

    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    prompts = _build_prompts(args.prompts)
    text_tokens = tokenizer(prompts).to(device)
    with torch.inference_mode():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    rgb_sink, rgb_pipeline = build_rgb_sink(args.rgb_port)
    rgb_pipeline.set_state(Gst.State.PLAYING)

    last = time.perf_counter()
    last_caption_time = 0.0
    last_caption = "waiting..."
    last_prob = 0.0

    scale = max(0.1, min(1.0, args.downscale))
    det_w = int(args.width * scale)
    det_h = int(args.height * scale)

    try:
        while True:
            sample_rgb = rgb_sink.emit("try-pull-sample", Gst.SECOND // args.fps)
            if not sample_rgb:
                time.sleep(0.005)
                continue

            buf_rgb = sample_rgb.get_buffer()
            rgb = np.frombuffer(buf_rgb.extract_dup(0, buf_rgb.get_size()), dtype=np.uint8)
            rgb = rgb.reshape((args.height, args.width, 3)).copy()

            now = time.perf_counter()
            if now - last_caption_time >= max(1.0, args.interval):
                if scale < 1.0:
                    small = cv2.resize(
                        rgb,
                        (det_w, det_h),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    small = rgb

                best_idx, best_prob = _classify_frame(
                    small, model, preprocess, text_features, device
                )
                last_caption = prompts[best_idx]
                last_prob = best_prob
                last_caption_time = now
                print(f"[caption] {last_caption} ({last_prob:.2f})")

            fps = 1.0 / max(1e-6, now - last)
            last = now

            cv2.putText(
                rgb,
                f"desc: {last_caption} ({last_prob:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                rgb,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("RGB (CLIP caption)", rgb)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        rgb_pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Error:", exc)
        sys.exit(1)
