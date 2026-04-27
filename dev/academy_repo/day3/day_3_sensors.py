#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
import threading
import time
from typing import Any


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from sdk_client import Robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 3 sensor demo: VideoClient RGB feed and DDS RGBD stream viewer."
    )
    parser.add_argument("--mode", choices=["video", "rgbd"], default="video")
    parser.add_argument("--iface", default="eth0", help="Network interface for DDS/SDK traffic.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--no-safety-boot",
        action="store_true",
        help="Skip the robot safety boot sequence during Robot initialization.",
    )
    parser.add_argument(
        "--rpc-timeout",
        type=float,
        default=2.0,
        help="VideoClient RPC timeout in seconds for video mode.",
    )
    parser.add_argument(
        "--show-rgb-copy",
        action="store_true",
        help="Also show an RGB-converted window in video mode.",
    )
    parser.add_argument(
        "--rgb-topic",
        default="rt/frontvideostream",
        help="DDS RGB topic for rgbd mode.",
    )
    parser.add_argument(
        "--depth-topic",
        default="",
        help="DDS depth topic for rgbd mode. Leave empty for RGB-only DDS viewing.",
    )
    parser.add_argument(
        "--rgb-type",
        default="unitree_go::msg::dds_::Go2FrontVideoData_",
        help="Preferred DDS RGB message type for rgbd mode.",
    )
    parser.add_argument(
        "--depth-type",
        default="sensor_msgs::msg::dds_::Image_",
        help="Preferred DDS depth message type for rgbd mode.",
    )
    parser.add_argument(
        "--video-field",
        default="video720p",
        choices=["video720p", "video360p", "video180p"],
        help="Compressed image field to use when the DDS RGB message is a front-video message.",
    )
    return parser.parse_args()


def _resolve_type(path: str) -> Any:
    if "::" in path:
        parts = [p for p in path.split("::") if p]
        if len(parts) < 2:
            raise ValueError(path)
        module = importlib.import_module(".".join(parts[:-1]))
        return getattr(module, parts[-1])
    if ":" in path:
        module_name, class_name = path.split(":", 1)
    else:
        module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _resolve_first(candidates: list[str]) -> type | None:
    for candidate in candidates:
        try:
            return _resolve_type(candidate)
        except Exception:
            continue
    return None


def _rgb_type_candidates(user_type: str) -> list[str]:
    return [
        user_type,
        "unitree_go::msg::dds_::Go2FrontVideoData_",
        "unitree_sdk2py.idl.unitree_go.msg.dds_:Go2FrontVideoData_",
        "unitree_sdk2py.idl.unitree_go.msg.dds_.Go2FrontVideoData_",
        "sensor_msgs::msg::dds_::Image_",
        "unitree_sdk2py.idl.sensor_msgs.msg.dds_:Image_",
    ]


def _depth_type_candidates(user_type: str) -> list[str]:
    return [
        user_type,
        "sensor_msgs::msg::dds_::Image_",
        "unitree_sdk2py.idl.sensor_msgs.msg.dds_:Image_",
    ]


def _bytes_from_seq(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    return bytes(bytearray(data))


def _decode_ros_image(msg: Any):
    import cv2
    import numpy as np

    try:
        height = int(msg.height)
        width = int(msg.width)
        step = int(msg.step)
        encoding = str(getattr(msg, "encoding", "")).lower()
        buf = _bytes_from_seq(msg.data)
    except Exception:
        return None

    if height <= 0 or width <= 0 or not buf:
        return None

    if encoding in ("bgr8", "rgb8"):
        dtype, channels = np.uint8, 3
    elif encoding in ("bgra8", "rgba8"):
        dtype, channels = np.uint8, 4
    elif encoding in ("mono8", "8uc1"):
        dtype, channels = np.uint8, 1
    elif encoding in ("mono16", "16uc1", "z16"):
        dtype, channels = np.uint16, 1
    else:
        if len(buf) == height * width * 3:
            dtype, channels, step = np.uint8, 3, width * 3
        elif len(buf) == height * width * 2:
            dtype, channels, step = np.uint16, 1, width * 2
        else:
            return None

    elem_size = int(np.dtype(dtype).itemsize)
    min_step = width * channels * elem_size
    if step < min_step:
        step = min_step
    if len(buf) < height * step:
        return None

    if channels == 1:
        img = np.ndarray((height, width), dtype=dtype, buffer=buf, strides=(step, elem_size)).copy()
    else:
        img = np.ndarray(
            (height, width, channels),
            dtype=dtype,
            buffer=buf,
            strides=(step, channels * elem_size, elem_size),
        ).copy()

    if encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif encoding == "rgba8":
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif encoding == "bgra8":
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _decode_go2_front(msg: Any, preferred_field: str):
    import cv2
    import numpy as np

    for field in [preferred_field, "video720p", "video360p", "video180p"]:
        payload = getattr(msg, field, None)
        if payload is None:
            continue
        try:
            arr = np.frombuffer(_bytes_from_seq(payload), dtype=np.uint8)
        except Exception:
            continue
        if arr.size == 0:
            continue
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            return frame
    return None


def _decode_any_image(msg: Any, preferred_field: str):
    if hasattr(msg, "height") and hasattr(msg, "width") and hasattr(msg, "data"):
        return _decode_ros_image(msg)
    if hasattr(msg, "video720p") or hasattr(msg, "video360p") or hasattr(msg, "video180p"):
        return _decode_go2_front(msg, preferred_field)
    return None


def _depth_to_colormap(depth):
    import cv2
    import numpy as np

    if depth is None:
        return None
    if len(depth.shape) == 3 and depth.shape[2] == 3:
        return depth
    if depth.dtype == np.uint8:
        display = depth
    else:
        depth_f = depth.astype(np.float32, copy=False)
        valid = np.isfinite(depth_f)
        if not np.any(valid):
            return None
        vals = depth_f[valid]
        dmin = float(np.min(vals))
        dmax = float(np.max(vals))
        if dmax <= dmin:
            display = np.zeros_like(depth_f, dtype=np.uint8)
        else:
            display = (255.0 * (depth_f - dmin) / (dmax - dmin)).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(display, cv2.COLORMAP_JET)


class RGBDState:
    def __init__(self, video_field: str) -> None:
        self._lock = threading.Lock()
        self._video_field = video_field
        self._rgb = None
        self._depth = None
        self._rgb_count = 0
        self._depth_count = 0

    def rgb_cb(self, msg: Any) -> None:
        frame = _decode_any_image(msg, self._video_field)
        if frame is None:
            return
        with self._lock:
            self._rgb = frame
            self._rgb_count += 1

    def depth_cb(self, msg: Any) -> None:
        frame = _decode_any_image(msg, self._video_field)
        if frame is None:
            return
        with self._lock:
            self._depth = frame
            self._depth_count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "rgb": None if self._rgb is None else self._rgb.copy(),
                "depth": None if self._depth is None else self._depth.copy(),
                "rgb_count": self._rgb_count,
                "depth_count": self._depth_count,
            }


def run_video_mode(args: argparse.Namespace) -> int:
    import cv2

    robot = Robot(
        iface=args.iface,
        domain_id=args.domain_id,
        safety_boot=not args.no_safety_boot,
        auto_start_sensors=False,
    )

    try:
        robot._get_video_client().SetTimeout(float(args.rpc_timeout))
    except Exception:
        pass

    print("Starting VideoClient RGB viewer. Press ESC or q to exit.")
    while True:
        frame_bgr = robot.get_camera_frame_bgr()
        cv2.imshow("G1 VideoClient BGR", frame_bgr)

        if args.show_rgb_copy:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            cv2.imshow("G1 VideoClient RGB", frame_rgb)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
    return 0


def run_rgbd_mode(args: argparse.Namespace) -> int:
    import cv2
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

    ChannelFactoryInitialize(int(args.domain_id), args.iface)

    rgb_type = _resolve_first(_rgb_type_candidates(args.rgb_type))
    if rgb_type is None:
        raise RuntimeError("Could not resolve any RGB DDS message type for rgbd mode.")

    depth_type = None
    if args.depth_topic:
        depth_type = _resolve_first(_depth_type_candidates(args.depth_type))
        if depth_type is None:
            raise RuntimeError("Could not resolve any depth DDS message type for rgbd mode.")

    state = RGBDState(video_field=args.video_field)
    rgb_sub = ChannelSubscriber(args.rgb_topic, rgb_type)
    rgb_sub.Init(state.rgb_cb, 10)
    depth_sub = None
    if args.depth_topic and depth_type is not None:
        depth_sub = ChannelSubscriber(args.depth_topic, depth_type)
        depth_sub.Init(state.depth_cb, 10)

    print(f"DDS RGB subscribed: topic={args.rgb_topic}, type={rgb_type}")
    if depth_sub is not None:
        print(f"DDS Depth subscribed: topic={args.depth_topic}, type={depth_type}")
    else:
        print("Depth DDS disabled. Provide --depth-topic to view a streamed depth feed.")
    print("Starting DDS RGBD viewer. Press ESC or q to exit.")

    while True:
        snap = state.snapshot()
        rgb = snap["rgb"]
        depth = snap["depth"]

        if rgb is None:
            blank_rgb = __import__("numpy").zeros((360, 640, 3), dtype="uint8")
            cv2.putText(blank_rgb, "Waiting for RGB stream...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 220), 2)
            cv2.imshow("G1 DDS RGB", blank_rgb)
        else:
            cv2.imshow("G1 DDS RGB", rgb)

        if args.depth_topic:
            if depth is None:
                blank_depth = __import__("numpy").zeros((360, 640, 3), dtype="uint8")
                cv2.putText(blank_depth, "Waiting for Depth stream...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 180, 0), 2)
                cv2.imshow("G1 DDS Depth", blank_depth)
            else:
                depth_vis = _depth_to_colormap(depth)
                if depth_vis is not None:
                    cv2.imshow("G1 DDS Depth", depth_vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    del rgb_sub
    if depth_sub is not None:
        del depth_sub
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "video":
            return run_video_mode(args)
        return run_rgbd_mode(args)
    except KeyboardInterrupt:
        return 1
    except Exception as exc:
        print(f"Sensor demo failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
