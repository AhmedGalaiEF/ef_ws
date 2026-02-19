#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
except ImportError as exc:
    raise SystemExit("Could not import ChannelFactoryInitialize from unitree_sdk2py.") from exc


def _load_video_client():
    paths = [
        "unitree_sdk2py.g1.video.video_client",
        "unitree_sdk2py.go2.video.video_client",
    ]
    last_exc = None
    for path in paths:
        try:
            module = __import__(path, fromlist=["VideoClient"])
            return module.VideoClient
        except Exception as exc:
            last_exc = exc
    raise SystemExit(f"Could not import VideoClient from any known path: {paths}. Last error: {last_exc}")


RPC_ERR_CLIENT_SEND = 3102
RPC_ERR_CLIENT_API_TIMEOUT = 3104


def _rpc_hint(code: int) -> str:
    if code == RPC_ERR_CLIENT_SEND:
        return (
            "RPC 3102: request send failed. Check interface/IP routing to robot, "
            "DDS domain, and that robot videohub service is running."
        )
    if code == RPC_ERR_CLIENT_API_TIMEOUT:
        return "RPC 3104: request timed out. Link exists but service did not respond in time."
    return f"RPC error code {code}"


def _decode_bgr(data) -> np.ndarray:
    jpg = np.frombuffer(bytes(data), dtype=np.uint8)
    if jpg.size == 0:
        raise RuntimeError("Received empty image payload from GetImageSample().")
    frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)  # BGR
    if frame is None:
        raise RuntimeError("Failed to decode JPEG payload into BGR frame.")
    return frame


def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test G1 VideoClient BGR frame access.")
    parser.add_argument("--iface", default="enp1s0", help="Network interface connected to the robot.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID (0 for real robot).")
    parser.add_argument("--timeout", type=float, default=2.0, help="RPC timeout in seconds.")
    parser.add_argument("--count", type=int, default=30, help="Number of frames to fetch.")
    parser.add_argument("--sleep", type=float, default=0.03, help="Delay between frame requests in seconds.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live BGR and RGB video in OpenCV windows.",
    )
    parser.add_argument("--save", default="", help="Optional output image path for the last frame.")
    args = parser.parse_args()

    VideoClient = _load_video_client()
    ChannelFactoryInitialize(args.domain_id, args.iface)

    client = VideoClient()
    client.SetTimeout(float(args.timeout))
    client.Init()

    ok = 0
    t0 = time.time()
    last = None

    try:
        for i in range(args.count):
            code, data = client.GetImageSample()
            if code != 0:
                print(f"[{i + 1}/{args.count}] GetImageSample failed with code={code}")
                if i < 3 or (i + 1) % 10 == 0:
                    print("  " + _rpc_hint(code))
                time.sleep(args.sleep)
                continue

            try:
                frame = _decode_bgr(data)
            except Exception as exc:
                print(f"[{i + 1}/{args.count}] decode failed: {exc}")
                time.sleep(args.sleep)
                continue

            ok += 1
            last = frame
            rgb_frame = _bgr_to_rgb(frame)
            h, w, c = frame.shape
            print(f"[{i + 1}/{args.count}] BGR frame: shape=({h},{w},{c}) dtype={frame.dtype} bytes={len(data)}")

            if args.show:
                cv2.imshow("G1 VideoClient BGR Test", frame)
                cv2.imshow("G1 VideoClient RGB Test", rgb_frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    print("ESC pressed, exiting.")
                    break
            time.sleep(args.sleep)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    elapsed = max(time.time() - t0, 1e-6)
    print(f"\nFrames decoded: {ok}/{args.count} | approx decode FPS: {ok / elapsed:.2f}")

    if args.save and last is not None:
        if cv2.imwrite(args.save, last):
            print(f"Saved last BGR frame to: {args.save}")
        else:
            print(f"Failed to save frame to: {args.save}")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
