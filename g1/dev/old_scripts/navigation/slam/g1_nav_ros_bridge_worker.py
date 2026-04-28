#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import struct
import sys
import time
from typing import Any

import numpy as np

from dds_env import ensure_cyclonedds_environment


def _write_message(payload: dict[str, Any]) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.write(struct.pack("<I", len(data)))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _decode_pointcloud2(msg: Any) -> np.ndarray | None:
    width = int(getattr(msg, "width", 0) or 0)
    height = int(getattr(msg, "height", 0) or 0)
    point_step = int(getattr(msg, "point_step", 0) or 0)
    data = bytes(getattr(msg, "data", b""))
    fields = list(getattr(msg, "fields", []) or [])
    count = width * max(1, height)
    if count <= 0 or point_step <= 0 or len(data) < count * point_step:
        return None
    offsets: dict[str, int] = {}
    for field in fields:
        name = str(getattr(field, "name", ""))
        if name in {"x", "y", "z"}:
            offsets[name] = int(getattr(field, "offset", -1))
    if any(axis not in offsets for axis in ("x", "y", "z")):
        return None
    if any(offset < 0 or offset + 4 > point_step for offset in offsets.values()):
        return None
    cloud = np.empty((count, 3), dtype=np.float32)
    for idx in range(count):
        base = idx * point_step
        cloud[idx, 0] = np.frombuffer(data, dtype=np.float32, count=1, offset=base + offsets["x"])[0]
        cloud[idx, 1] = np.frombuffer(data, dtype=np.float32, count=1, offset=base + offsets["y"])[0]
        cloud[idx, 2] = np.frombuffer(data, dtype=np.float32, count=1, offset=base + offsets["z"])[0]
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if cloud.shape[0] > 2000:
        step = max(1, cloud.shape[0] // 2000)
        cloud = cloud[::step]
    return cloud


def _decode_image(msg: Any) -> np.ndarray | None:
    width = int(getattr(msg, "width", 0) or 0)
    height = int(getattr(msg, "height", 0) or 0)
    step = int(getattr(msg, "step", 0) or 0)
    encoding = str(getattr(msg, "encoding", "")).lower()
    data = bytes(getattr(msg, "data", b""))
    if width <= 0 or height <= 0 or step <= 0 or not data:
        return None
    if encoding in {"rgb8", "bgr8"}:
        channels = 3
        expected_row = width * channels
        if step < expected_row or len(data) < height * step:
            return None
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
        return np.ascontiguousarray(arr[:, :expected_row].reshape(height, width, channels))
    if encoding in {"mono8", "8uc1"}:
        if step < width or len(data) < height * step:
            return None
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
        return np.ascontiguousarray(arr[:, :width])
    if encoding in {"16uc1", "mono16"}:
        expected_row = width * 2
        if step < expected_row or len(data) < height * step:
            return None
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
        return np.ascontiguousarray(arr[:, :expected_row].view(np.uint16).reshape(height, width))
    if encoding == "32fc1":
        expected_row = width * 4
        if step < expected_row or len(data) < height * step:
            return None
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, step)
        return np.ascontiguousarray(arr[:, :expected_row].view(np.float32).reshape(height, width))
    return None


def _decode_rgb_image(msg: Any) -> np.ndarray | None:
    image = _decode_image(msg)
    if image is None:
        return None
    encoding = str(getattr(msg, "encoding", "")).lower()
    if encoding == "rgb8":
        return image
    if encoding == "bgr8":
        return image[..., ::-1].copy()
    if encoding in {"mono8", "8uc1"}:
        return np.repeat(image[..., None], 3, axis=2)
    return None


def _decode_depth_image(msg: Any) -> np.ndarray | None:
    image = _decode_image(msg)
    if image is None:
        return None
    encoding = str(getattr(msg, "encoding", "")).lower()
    if encoding in {"16uc1", "mono16"} and image.ndim == 2:
        return image.astype(np.uint16, copy=False)
    if encoding == "32fc1" and image.ndim == 2:
        return image.astype(np.float32, copy=False)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidar-topic", default="/livox/points")
    parser.add_argument("--rgb-topic", default="/rgbd/color/image_raw")
    parser.add_argument("--depth-topic", default="/rgbd/depth/image_raw")
    args = parser.parse_args()

    ensure_cyclonedds_environment()
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, PointCloud2

    class BridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("g1_nav_ros_sensor_bridge_worker")
            self._seen_cloud = False
            self._seen_rgb = False
            self._seen_depth = False
            self._last_graph_report = 0.0
            self.create_subscription(PointCloud2, args.lidar_topic, self._on_cloud, qos_profile_sensor_data)
            self.create_subscription(Image, args.rgb_topic, self._on_rgb, qos_profile_sensor_data)
            self.create_subscription(Image, args.depth_topic, self._on_depth, qos_profile_sensor_data)
            _write_message({"kind": "status", "ok": True, "message": "ready"})

        def _on_cloud(self, msg: PointCloud2) -> None:
            cloud = _decode_pointcloud2(msg)
            if cloud is not None:
                if not self._seen_cloud:
                    self._seen_cloud = True
                    _write_message({"kind": "status", "ok": True, "message": f"received first lidar message on {args.lidar_topic}"})
                _write_message({"kind": "points", "stamp": time.time(), "data": cloud})

        def _on_rgb(self, msg: Image) -> None:
            image = _decode_rgb_image(msg)
            if image is not None:
                if not self._seen_rgb:
                    self._seen_rgb = True
                    _write_message({"kind": "status", "ok": True, "message": f"received first rgb message on {args.rgb_topic}"})
                _write_message({"kind": "rgb", "stamp": time.time(), "data": image})

        def _on_depth(self, msg: Image) -> None:
            image = _decode_depth_image(msg)
            if image is not None:
                if not self._seen_depth:
                    self._seen_depth = True
                    _write_message({"kind": "status", "ok": True, "message": f"received first depth message on {args.depth_topic}"})
                _write_message({"kind": "depth", "stamp": time.time(), "data": image})

        def maybe_report_graph(self) -> None:
            now = time.time()
            if now - self._last_graph_report < 5.0:
                return
            self._last_graph_report = now
            topics = self.get_topic_names_and_types()
            wanted = {args.lidar_topic, args.rgb_topic, args.depth_topic}
            visible = []
            for name, types in topics:
                if name in wanted or "livox" in name or "rgbd" in name:
                    visible.append(f"{name}:{','.join(types)}")
            domain = os.environ.get("ROS_DOMAIN_ID", "unset")
            _write_message(
                {
                    "kind": "status",
                    "ok": True,
                    "message": f"graph domain={domain} visible={visible[:12]}",
                }
            )

    rclpy.init(args=None)
    node = BridgeNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.2)
            node.maybe_report_graph()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        _write_message({"kind": "status", "ok": False, "message": str(exc)})
        raise
