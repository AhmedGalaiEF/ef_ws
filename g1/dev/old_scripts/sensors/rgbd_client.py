#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import struct
import time

def _fix_qt_fontdir() -> None:
    current = os.environ.get("QT_QPA_FONTDIR")
    if current and os.path.isdir(current):
        return

    for font_dir in (
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/dejavu",
        "/usr/share/fonts",
    ):
        if os.path.isdir(font_dir):
            os.environ["QT_QPA_FONTDIR"] = font_dir
            return


_fix_qt_fontdir()

import cv2
import numpy as np
import zmq

_fix_qt_fontdir()


def _decode_color(payload: bytes) -> np.ndarray | None:
    arr = np.frombuffer(payload, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _decode_depth(payload: bytes) -> np.ndarray | None:
    if payload == b"0":
        return None
    arr = np.frombuffer(payload, dtype=np.uint8)
    if arr.size == 0:
        return None
    depth = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    if depth.ndim == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    return depth


def _decode_scale(payload: bytes) -> float | None:
    if payload == b"0" or len(payload) != 4:
        return None
    return float(struct.unpack("f", payload)[0])


def _colorize_depth(depth: np.ndarray, max_depth_m: float, scale: float | None) -> np.ndarray:
    if depth.dtype == np.uint16 and scale is not None and scale > 0:
        depth_m = depth.astype(np.float32) * scale
        clipped = np.clip(depth_m / max_depth_m, 0.0, 1.0)
        disp = (clipped * 255.0).astype(np.uint8)
        disp[depth == 0] = 0
    else:
        valid = depth > 0
        if not np.any(valid):
            disp = np.zeros(depth.shape[:2], dtype=np.uint8)
        else:
            vals = depth[valid].astype(np.float32)
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            if hi <= lo:
                disp = np.zeros(depth.shape[:2], dtype=np.uint8)
            else:
                disp = np.clip(255.0 * (depth.astype(np.float32) - lo) / (hi - lo), 0, 255).astype(np.uint8)
                disp[~valid] = 0
    return cv2.applyColorMap(disp, cv2.COLORMAP_JET)


def _overlay_info(
    color: np.ndarray,
    depth_vis: np.ndarray,
    depth_raw: np.ndarray | None,
    depth_scale: float | None,
    fps: float,
    show_meter_at: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    color = color.copy()
    depth_vis = depth_vis.copy()
    cv2.putText(color, f"RGB  {fps:4.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(depth_vis, f"Depth  {fps:4.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if depth_scale is not None:
        cv2.putText(
            depth_vis,
            f"scale={depth_scale:.6f} m/unit",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    if depth_raw is None or show_meter_at is None:
        return color, depth_vis

    x, y = show_meter_at
    if 0 <= y < depth_raw.shape[0] and 0 <= x < depth_raw.shape[1]:
        raw = int(depth_raw[y, x])
        cv2.drawMarker(color, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.drawMarker(depth_vis, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        if depth_scale is not None and raw > 0:
            label = f"{raw * depth_scale:.3f} m @ ({x},{y})"
        else:
            label = f"raw={raw} @ ({x},{y})"
        cv2.putText(color, label, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(depth_vis, label, (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return color, depth_vis


def main() -> None:
    parser = argparse.ArgumentParser(description="ZeroMQ RGBD viewer for image_server.py RealSense streams.")
    parser.add_argument("--host", default="10.34.0.11", help="Publisher host/IP")
    parser.add_argument("--port", type=int, default=5555, help="Publisher TCP port")
    parser.add_argument("--topic", default="", help="ZMQ subscription prefix; empty subscribes to all")
    parser.add_argument("--timeout-ms", type=int, default=3000, help="Receive timeout in milliseconds")
    parser.add_argument("--max-depth-m", type=float, default=4.0, help="Upper range for depth visualization")
    parser.add_argument("--window-scale", type=float, default=1.0, help="Resize display windows by this factor")
    args = parser.parse_args()

    endpoint = f"tcp://{args.host}:{args.port}"
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, args.topic.encode("utf-8"))
    socket.setsockopt(zmq.RCVTIMEO, int(args.timeout_ms))
    socket.connect(endpoint)

    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            try:
                parts = socket.recv_multipart()
            except zmq.Again:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f"Waiting for stream on {endpoint}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 220), 2)
                cv2.putText(blank, "Make sure image_server.py is publishing.", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
                cv2.imshow("RGB", blank)
                cv2.imshow("Depth", blank)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
                continue

            if len(parts) < 3:
                continue

            color = _decode_color(parts[0])
            depth = _decode_depth(parts[1])
            depth_scale = _decode_scale(parts[2])
            if color is None:
                continue

            if depth is None:
                depth_vis = np.zeros_like(color)
                cv2.putText(depth_vis, "No depth payload", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)
                probe = None
            else:
                depth_vis = _colorize_depth(depth, args.max_depth_m, depth_scale)
                probe = (depth.shape[1] // 2, depth.shape[0] // 2)

            frame_count += 1
            elapsed = max(time.time() - t_start, 1e-6)
            fps = frame_count / elapsed
            color, depth_vis = _overlay_info(color, depth_vis, depth, depth_scale, fps, probe)

            if args.window_scale != 1.0:
                color = cv2.resize(color, None, fx=args.window_scale, fy=args.window_scale, interpolation=cv2.INTER_AREA)
                depth_vis = cv2.resize(depth_vis, None, fx=args.window_scale, fy=args.window_scale, interpolation=cv2.INTER_NEAREST)

            cv2.imshow("RGB", color)
            cv2.imshow("Depth", depth_vis)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(0)
        context.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
