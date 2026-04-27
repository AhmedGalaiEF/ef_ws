#!/usr/bin/env python3
from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import open_clip
from PIL import Image
import torch
import zmq


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class DetectionResult:
    score: float
    detected: bool
    box: Box | None
    infer_started_at: float
    infer_finished_at: float


def _decode_color(payload: bytes) -> np.ndarray | None:
    arr = np.frombuffer(payload, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _generate_boxes(width: int, height: int, step_frac: float, scales: Sequence[float]) -> list[Box]:
    boxes: list[Box] = []
    min_dim = min(width, height)
    for scale in scales:
        size = int(min_dim * scale)
        if size <= 0:
            continue
        step = max(12, int(size * step_frac))
        for y in range(0, max(1, height - size + 1), step):
            for x in range(0, max(1, width - size + 1), step):
                boxes.append(Box(x, y, min(width, x + size), min(height, y + size)))
        if size < width or size < height:
            boxes.append(Box(max(0, width - size), max(0, height - size), width, height))
    if not boxes:
        boxes.append(Box(0, 0, width, height))
    return boxes


def _parse_scales(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("at least one window scale is required")
    return values


def _fit_display(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _best_clip_box(
    image_bgr: np.ndarray,
    model,
    preprocess,
    text_features: torch.Tensor,
    device: torch.device,
    step_frac: float,
    window_scales: Sequence[float],
    batch_size: int,
) -> tuple[float, Box]:
    boxes = _generate_boxes(image_bgr.shape[1], image_bgr.shape[0], step_frac, window_scales)
    best_score = -1.0
    best_box = boxes[0]

    for idx in range(0, len(boxes), batch_size):
        batch_boxes = boxes[idx : idx + batch_size]
        crops = []
        valid_boxes = []
        for box in batch_boxes:
            crop = image_bgr[box.y1 : box.y2, box.x1 : box.x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(preprocess(Image.fromarray(crop_rgb)))
            valid_boxes.append(box)

        if not crops:
            continue

        image_input = torch.stack(crops).to(device)
        with torch.inference_mode():
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[:, 0]

        batch_best = int(torch.argmax(probs).item())
        batch_score = float(probs[batch_best].item())
        if batch_score > best_score:
            best_score = batch_score
            best_box = valid_boxes[batch_best]

    return best_score, best_box


class InferenceWorker:
    def __init__(
        self,
        model,
        preprocess,
        text_features: torch.Tensor,
        device: torch.device,
        threshold: float,
        downscale: float,
        step_frac: float,
        window_scales: Sequence[float],
        batch_size: int,
    ) -> None:
        self._model = model
        self._preprocess = preprocess
        self._text_features = text_features
        self._device = device
        self._threshold = threshold
        self._downscale = downscale
        self._step_frac = step_frac
        self._window_scales = window_scales
        self._batch_size = batch_size
        self._jobs: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._latest_result: DetectionResult | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, infer_started_at: float) -> bool:
        if self._jobs.full():
            return False
        self._jobs.put_nowait((frame.copy(), infer_started_at))
        return True

    def latest_result(self) -> DetectionResult | None:
        with self._result_lock:
            return self._latest_result

    def close(self) -> None:
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            frame, infer_started_at = job
            scale = max(0.1, min(1.0, self._downscale))
            small = frame
            if scale < 1.0:
                small = cv2.resize(
                    frame,
                    (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            score, box = _best_clip_box(
                small,
                self._model,
                self._preprocess,
                self._text_features,
                self._device,
                self._step_frac,
                self._window_scales,
                self._batch_size,
            )

            if scale < 1.0:
                box = Box(
                    x1=int(box.x1 / scale),
                    y1=int(box.y1 / scale),
                    x2=int(box.x2 / scale),
                    y2=int(box.y2 / scale),
                )

            result = DetectionResult(
                score=score,
                detected=score >= self._threshold,
                box=box,
                infer_started_at=infer_started_at,
                infer_finished_at=time.perf_counter(),
            )
            with self._result_lock:
                self._latest_result = result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CLIP object detector for the RGB stream published by rgbd_client/image_server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", required=True, help="Target text prompt, for example 'a photo of a soda can'")
    parser.add_argument("--negative-prompt", default="", help="Optional negative comparison prompt")
    parser.add_argument("--host", "--robot-ip", dest="host", default="10.34.0.83", help="Publisher host/IP")
    parser.add_argument("--port", type=int, default=5555, help="Publisher TCP port")
    parser.add_argument("--topic", default="", help="ZMQ subscription prefix; empty subscribes to all")
    parser.add_argument("--timeout-ms", type=int, default=200, help="Receive timeout in milliseconds")
    parser.add_argument("--display-fps", type=float, default=30.0, help="Target display FPS")
    parser.add_argument("--infer-fps", type=float, default=2.0, help="How often to rerun CLIP inference")
    parser.add_argument("--threshold", type=float, default=0.60, help="Detection threshold in [0, 1]")
    parser.add_argument("--downscale", type=float, default=0.5, help="Downscale factor for CLIP inference")
    parser.add_argument("--step-frac", type=float, default=0.25, help="Sliding-window step as fraction of window size")
    parser.add_argument("--window-scales", default="0.5,0.7,0.9", help="Comma-separated sliding-window scales")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of crops to score per CLIP batch")
    parser.add_argument("--model", default="ViT-B-32", help="open_clip model name")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained tag")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-width", type=int, default=1280, help="Max display width")
    parser.add_argument("--window-height", type=int, default=720, help="Max display height")
    args = parser.parse_args()

    if args.display_fps <= 0 or args.infer_fps <= 0:
        print("display-fps and infer-fps must be > 0")
        return 2

    try:
        window_scales = _parse_scales(args.window_scales)
    except ValueError as exc:
        print(f"Invalid --window-scales: {exc}")
        return 2

    negative_prompt = args.negative_prompt.strip() or f"a photo without {args.prompt.strip()}"
    prompts = [args.prompt.strip(), negative_prompt]

    device = torch.device(args.device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)
    text_tokens = tokenizer(prompts).to(device)
    with torch.inference_mode():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    endpoint = f"tcp://{args.host}:{args.port}"
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, args.topic.encode("utf-8"))
    socket.setsockopt(zmq.RCVTIMEO, int(args.timeout_ms))
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect(endpoint)

    cv2.namedWindow("CLIP RGB Detection", cv2.WINDOW_NORMAL)

    worker = InferenceWorker(
        model=model,
        preprocess=preprocess,
        text_features=text_features,
        device=device,
        threshold=args.threshold,
        downscale=args.downscale,
        step_frac=args.step_frac,
        window_scales=window_scales,
        batch_size=max(1, args.batch_size),
    )

    latest_frame: np.ndarray | None = None
    last_display_at = 0.0
    last_infer_submit_at = 0.0
    display_last_tick = time.perf_counter()
    display_fps = 0.0
    frame_counter = 0

    print(f"Subscribed to {endpoint}. Press q or Esc to quit.")
    try:
        while True:
            try:
                parts = socket.recv_multipart()
                if len(parts) >= 1:
                    color = _decode_color(parts[0])
                    if color is not None:
                        latest_frame = color
            except zmq.Again:
                pass

            now = time.perf_counter()
            if latest_frame is not None and (now - last_infer_submit_at) >= (1.0 / args.infer_fps):
                if worker.submit(latest_frame, now):
                    last_infer_submit_at = now

            if (now - last_display_at) < (1.0 / args.display_fps):
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue

            last_display_at = now
            if latest_frame is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f"Waiting for RGB stream on {endpoint}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)
                cv2.putText(blank, args.prompt, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
                cv2.imshow("CLIP RGB Detection", blank)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue

            display = latest_frame.copy()
            result = worker.latest_result()

            frame_counter += 1
            elapsed = now - display_last_tick
            if elapsed >= 0.5:
                display_fps = frame_counter / elapsed
                frame_counter = 0
                display_last_tick = now

            status_text = "detecting..."
            status_color = (0, 255, 255)
            if result is not None:
                status_text = "DETECTED" if result.detected else "NOT DETECTED"
                status_color = (0, 255, 0) if result.detected else (0, 0, 255)
                if result.detected and result.box is not None:
                    cv2.rectangle(
                        display,
                        (result.box.x1, result.box.y1),
                        (result.box.x2, result.box.y2),
                        status_color,
                        2,
                    )

            infer_hz = 0.0
            infer_latency_ms = 0.0
            if result is not None:
                infer_latency_ms = (result.infer_finished_at - result.infer_started_at) * 1000.0
                infer_hz = 1.0 / max(result.infer_finished_at - result.infer_started_at, 1e-6)

            cv2.putText(display, f"prompt: {args.prompt}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(display, f"status: {status_text}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)
            cv2.putText(
                display,
                f"confidence: {0.0 if result is None else result.score:.2f}  threshold: {args.threshold:.2f}",
                (12, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                f"display: {display_fps:.1f} FPS  infer target: {args.infer_fps:.1f} FPS  infer compute: {infer_hz:.1f} FPS",
                (12, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                display,
                f"infer latency: {infer_latency_ms:.0f} ms",
                (12, 148),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            display = _fit_display(display, args.window_width, args.window_height)
            cv2.imshow("CLIP RGB Detection", display)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()
        socket.close(0)
        context.term()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
