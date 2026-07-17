#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time


def _quat_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main() -> int:
    parser = argparse.ArgumentParser(description="Print one ROS 2 TF as camera extrinsic JSON.")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--timeout-s", type=float, default=0.5)
    args = parser.parse_args()

    try:
        import rclpy
        from rclpy.duration import Duration
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformListener
    except Exception as exc:
        print(f"ROS 2 TF imports failed: {exc}", file=sys.stderr)
        return 2

    rclpy.init(args=None)
    node = rclpy.create_node("camera_tf_probe")
    try:
        buf = Buffer()
        listener = TransformListener(buf, node)
        deadline = time.time() + max(0.1, float(args.timeout_s))
        transform = None
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            try:
                transform = buf.lookup_transform(
                    args.base_frame,
                    args.camera_frame,
                    Time(),
                    timeout=Duration(seconds=0.05),
                )
                break
            except Exception:
                continue
        if transform is None:
            print(
                f"no TF {args.base_frame}<-{args.camera_frame} within {args.timeout_s:.2f}s",
                file=sys.stderr,
            )
            return 1

        t = transform.transform.translation
        q = transform.transform.rotation
        roll, pitch, yaw = _quat_to_rpy(float(q.x), float(q.y), float(q.z), float(q.w))
        stamp = transform.header.stamp
        stamp_s = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        now_s = node.get_clock().now().nanoseconds * 1e-9
        print(json.dumps({
            "x": float(t.x),
            "y": float(t.y),
            "z": float(t.z),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "stamp_s": stamp_s,
            "age_s": max(0.0, now_s - stamp_s) if stamp_s > 0.0 else 0.0,
        }))
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
