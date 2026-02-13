#!/usr/bin/env python3
"""
rgbd_cam.py
===========

DDS RGBD viewer for Unitree robots.
Runs on a laptop with unitree_sdk2py + OpenCV + NumPy only.
No ROS2 and no local librealsense dependency are required.
"""
from __future__ import annotations

import argparse
import importlib
import time
from threading import Lock
from typing import Any, List, Optional

try:
    import cv2  # type: ignore
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing deps. Ensure numpy and opencv-python are installed.") from exc

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing unitree_sdk2py. Install it on this laptop first.") from exc


def _resolve_type(type_path: str) -> Any:
    """Resolve module.path:Class, module.path.Class, or cxx::namespace::Type."""
    if "::" in type_path:
        parts = [p for p in type_path.split("::") if p]
        if len(parts) < 2:
            raise ValueError(f"Invalid C++ style type path: {type_path}")
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    if ":" in type_path:
        module_path, class_name = type_path.split(":", 1)
    else:
        module_path, class_name = type_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _type_path_candidates(type_path: str) -> List[str]:
    if "::" not in type_path:
        return [type_path]

    parts = [p for p in type_path.split("::") if p]
    class_name = parts[-1]
    namespace = parts[:-1]
    ns_dot = ".".join(namespace)

    return [
        f"unitree_sdk2py.idl.{ns_dot}:{class_name}",
        f"unitree_sdk2py.idl.{ns_dot}.{class_name}",
        f"{ns_dot}:{class_name}",
        f"{ns_dot}.{class_name}",
    ]


def _image_type_fallback_candidates() -> List[str]:
    return [
        "unitree_sdk2py.idl.sensor_msgs.msg.dds_:Image_",
        "unitree_sdk2py.idl.sensor_msgs.msg.dds_.Image_",
        "unitree_sdk2py.idl.sensor_msgs.msg.dds_:Image",
        "unitree_sdk2py.idl.sensor_msgs.msg.dds_.Image",
        "sensor_msgs.msg.dds_:Image_",
        "sensor_msgs.msg.dds_.Image_",
        "sensor_msgs.msg.dds_:Image",
        "sensor_msgs.msg.dds_.Image",
    ]


def _video_type_fallback_candidates() -> List[str]:
    return [
        "unitree_sdk2py.idl.unitree_go.msg.dds_:Go2FrontVideoData_",
        "unitree_sdk2py.idl.unitree_go.msg.dds_.Go2FrontVideoData_",
    ]


def _resolve_msg_type(type_path: str, allow_image_fallback: bool = False, allow_video_fallback: bool = False) -> Optional[type]:
    attempted: List[str] = []

    for candidate in _type_path_candidates(type_path):
        attempted.append(candidate)
        try:
            return _resolve_type(candidate)
        except Exception:
            pass

    if allow_image_fallback:
        for candidate in _image_type_fallback_candidates():
            if candidate in attempted:
                continue
            attempted.append(candidate)
            try:
                return _resolve_type(candidate)
            except Exception:
                pass

    if allow_video_fallback:
        for candidate in _video_type_fallback_candidates():
            if candidate in attempted:
                continue
            attempted.append(candidate)
            try:
                return _resolve_type(candidate)
            except Exception:
                pass

    print("Failed to resolve message type. Attempted:")
    for c in attempted:
        print(f"  - {c}")
    return None


def _bytes_from_seq(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    return bytes(bytearray(data))


def _decode_ros_image(msg: Any) -> Optional[np.ndarray]:
    try:
        h = int(msg.height)
        w = int(msg.width)
        step = int(msg.step)
        enc = str(getattr(msg, "encoding", "")).lower()
        buf = _bytes_from_seq(msg.data)
    except Exception:
        return None

    if h <= 0 or w <= 0 or not buf:
        return None

    if enc in ("bgr8", "rgb8"):
        dtype = np.uint8
        ch = 3
    elif enc in ("bgra8", "rgba8"):
        dtype = np.uint8
        ch = 4
    elif enc in ("mono8", "8uc1"):
        dtype = np.uint8
        ch = 1
    elif enc in ("mono16", "16uc1", "z16"):
        dtype = np.uint16
        ch = 1
    elif enc in ("32fc1", "32sc1"):
        dtype = np.float32 if enc == "32fc1" else np.int32
        ch = 1
    else:
        if len(buf) == h * w * 3:
            dtype = np.uint8
            ch = 3
            step = w * 3
        elif len(buf) == h * w * 2:
            dtype = np.uint16
            ch = 1
            step = w * 2
        elif len(buf) == h * w:
            dtype = np.uint8
            ch = 1
            step = w
        else:
            return None

    elem = np.dtype(dtype).itemsize
    min_step = w * ch * elem
    if step < min_step:
        step = min_step
    needed = h * step
    if len(buf) < needed:
        return None

    if ch == 1:
        img = np.ndarray((h, w), dtype=dtype, buffer=buf, strides=(step, elem)).copy()
    else:
        img = np.ndarray((h, w, ch), dtype=dtype, buffer=buf, strides=(step, ch * elem, elem)).copy()

    if enc == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif enc == "rgba8":
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif enc == "bgra8":
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def _decode_front_video_msg(msg: Any, preferred_field: str) -> Optional[np.ndarray]:
    fields = [preferred_field, "video720p", "video360p", "video180p"]
    tried = set()
    for field in fields:
        if field in tried:
            continue
        tried.add(field)
        payload = getattr(msg, field, None)
        if payload is None:
            continue
        try:
            buf = _bytes_from_seq(payload)
        except Exception:
            continue
        if not buf:
            continue
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    return None


def _decode_any_image(msg: Any, video_field: str) -> Optional[np.ndarray]:
    if hasattr(msg, "height") and hasattr(msg, "width") and hasattr(msg, "data"):
        return _decode_ros_image(msg)
    if hasattr(msg, "video720p") or hasattr(msg, "video360p") or hasattr(msg, "video180p"):
        return _decode_front_video_msg(msg, video_field)
    return None


def _depth_to_colormap(depth: np.ndarray) -> Optional[np.ndarray]:
    if depth is None:
        return None
    if depth.dtype == np.uint8:
        disp = depth
    else:
        d = depth.astype(np.float32, copy=False)
        valid = np.isfinite(d)
        if not np.any(valid):
            return None
        v = d[valid]
        dmin = float(np.min(v))
        dmax = float(np.max(v))
        if dmax <= dmin:
            disp = np.zeros_like(d, dtype=np.uint8)
        else:
            disp = np.clip((255.0 * (d - dmin) / (dmax - dmin)), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(disp, cv2.COLORMAP_JET)


class RGBDViewer:
    def __init__(self, video_field: str) -> None:
        self._lock = Lock()
        self._rgb: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._video_field = video_field

        cv2.namedWindow("G1 RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow("G1 Depth", cv2.WINDOW_NORMAL)

    def rgb_cb(self, msg: Any) -> None:
        img = _decode_any_image(msg, self._video_field)
        if img is None:
            return
        with self._lock:
            self._rgb = img

    def depth_cb(self, msg: Any) -> None:
        img = _decode_any_image(msg, self._video_field)
        if img is None:
            return
        with self._lock:
            self._depth = img

    def render(self) -> None:
        with self._lock:
            rgb = self._rgb.copy() if self._rgb is not None else None
            depth = self._depth.copy() if self._depth is not None else None

        if rgb is not None:
            cv2.imshow("G1 RGB", rgb)
        else:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for RGB...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 220), 2)
            cv2.imshow("G1 RGB", blank)

        if depth is not None:
            depth_col = _depth_to_colormap(depth)
            if depth_col is not None:
                cv2.imshow("G1 Depth", depth_col)
        else:
            blank = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for Depth...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 180, 0), 2)
            cv2.imshow("G1 Depth", blank)

        cv2.waitKey(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 DDS RGBD camera viewer (no ROS required).")
    parser.add_argument("--iface", default="eth0", help="DDS network interface on this laptop")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id")
    parser.add_argument("--rgb-topic", default="rt/frontvideostream", help="DDS RGB topic")
    parser.add_argument("--depth-topic", default="", help="DDS depth topic (optional)")
    parser.add_argument(
        "--rgb-type",
        default="sensor_msgs::msg::dds_::Image_",
        help="DDS type for RGB topic (auto-falls back to Go2FrontVideoData_)",
    )
    parser.add_argument(
        "--depth-type",
        default="sensor_msgs::msg::dds_::Image_",
        help="DDS type for depth topic",
    )
    parser.add_argument(
        "--video-field",
        default="video720p",
        choices=["video720p", "video360p", "video180p"],
        help="Preferred frame field when topic uses Go2FrontVideoData_",
    )
    args = parser.parse_args()

    rgb_type = _resolve_msg_type(
        args.rgb_type,
        allow_image_fallback=True,
        allow_video_fallback=True,
    )
    if rgb_type is None:
        raise SystemExit(f"Could not resolve RGB type: {args.rgb_type}")

    depth_type = None
    if args.depth_topic:
        depth_type = _resolve_msg_type(
            args.depth_type,
            allow_image_fallback=True,
            allow_video_fallback=False,
        )
        if depth_type is None:
            raise SystemExit(f"Could not resolve depth type: {args.depth_type}")

    ChannelFactoryInitialize(args.domain_id, args.iface)
    viewer = RGBDViewer(video_field=args.video_field)

    rgb_sub = ChannelSubscriber(args.rgb_topic, rgb_type)
    rgb_sub.Init(viewer.rgb_cb, 10)

    print(f"RGB type resolved to: {rgb_type}")

    if args.depth_topic and depth_type is not None:
        depth_sub = ChannelSubscriber(args.depth_topic, depth_type)
        depth_sub.Init(viewer.depth_cb, 10)
        print(f"Listening RGB={args.rgb_topic}, Depth={args.depth_topic}, domain={args.domain_id}, iface={args.iface}")
    else:
        depth_sub = None
        print(f"Listening RGB={args.rgb_topic}, domain={args.domain_id}, iface={args.iface}")
        print("Depth topic not set. Use --depth-topic <topic> to enable depth display.")

    try:
        while True:
            viewer.render()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        del rgb_sub
        if depth_sub is not None:
            del depth_sub
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
