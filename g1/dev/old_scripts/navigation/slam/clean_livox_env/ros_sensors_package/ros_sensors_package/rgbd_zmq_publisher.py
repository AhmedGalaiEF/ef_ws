from __future__ import annotations

import struct
from typing import Optional

import cv2
import numpy as np
import rclpy
import zmq
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def _cv_to_image_msg(array: np.ndarray, encoding: str, frame_id: str, stamp) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(array.shape[0])
    msg.width = int(array.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = False
    msg.step = int(array.strides[0])
    msg.data = array.tobytes()
    return msg


class RgbdZmqPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("rgbd_zmq_publisher")

        self._host = str(self.declare_parameter("host", "10.34.0.83").value)
        self._port = int(self.declare_parameter("port", 5555).value)
        self._subscription_prefix = str(self.declare_parameter("zmq_topic", "").value)
        self._rgb_topic = str(self.declare_parameter("rgb_topic", "/rgbd/color/image_raw").value)
        self._depth_topic = str(self.declare_parameter("depth_topic", "/rgbd/depth/image_raw").value)
        self._rgb_info_topic = str(self.declare_parameter("rgb_camera_info_topic", "/rgbd/color/camera_info").value)
        self._depth_info_topic = str(self.declare_parameter("depth_camera_info_topic", "/rgbd/depth/camera_info").value)
        self._rgb_frame_id = str(self.declare_parameter("rgb_frame_id", "rgb_camera_frame").value)
        self._depth_frame_id = str(self.declare_parameter("depth_frame_id", "depth_camera_frame").value)
        self._poll_period = float(self.declare_parameter("poll_period_s", 0.01).value)
        self._depth_scale_fallback = float(self.declare_parameter("depth_scale_fallback", 0.001).value)

        self._fx = float(self.declare_parameter("fx", 0.0).value)
        self._fy = float(self.declare_parameter("fy", 0.0).value)
        self._cx = float(self.declare_parameter("cx", 0.0).value)
        self._cy = float(self.declare_parameter("cy", 0.0).value)

        self._rgb_pub = self.create_publisher(Image, self._rgb_topic, 10)
        self._depth_pub = self.create_publisher(Image, self._depth_topic, 10)
        self._rgb_info_pub = self.create_publisher(CameraInfo, self._rgb_info_topic, 10)
        self._depth_info_pub = self.create_publisher(CameraInfo, self._depth_info_topic, 10)

        self._zmq_context = zmq.Context()
        self._socket = self._zmq_context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, self._subscription_prefix.encode("utf-8"))
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

        self._timer = self.create_timer(self._poll_period, self._poll_socket)
        self.get_logger().info(
            f"Publishing RGBD from tcp://{self._host}:{self._port} to {self._rgb_topic} and {self._depth_topic}"
        )

    def _camera_info(self, width: int, height: int, frame_id: str, stamp) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.width = int(width)
        msg.height = int(height)
        if all(value > 0.0 for value in (self._fx, self._fy)):
            msg.k = [
                self._fx, 0.0, self._cx,
                0.0, self._fy, self._cy,
                0.0, 0.0, 1.0,
            ]
            msg.p = [
                self._fx, 0.0, self._cx, 0.0,
                0.0, self._fy, self._cy, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ]
        return msg

    def _poll_socket(self) -> None:
        try:
            while True:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
                self._publish_frame(parts)
        except zmq.Again:
            return

    def _publish_frame(self, parts: list[bytes]) -> None:
        if len(parts) >= 4:
            parts = parts[-3:]
        if len(parts) < 2:
            return

        rgb_jpeg = bytes(parts[0])
        depth_png = bytes(parts[1])
        scale = self._depth_scale_fallback
        if len(parts) >= 3 and len(parts[2]) == 4:
            try:
                scale = float(struct.unpack("f", parts[2])[0])
            except Exception:
                scale = self._depth_scale_fallback

        rgb = cv2.imdecode(np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            return
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_raw: Optional[np.ndarray] = None
        if depth_png != b"0":
            depth_raw = cv2.imdecode(np.frombuffer(depth_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if depth_raw is not None and depth_raw.ndim == 3:
                depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)

        stamp = self.get_clock().now().to_msg()
        self._rgb_pub.publish(_cv_to_image_msg(rgb, "rgb8", self._rgb_frame_id, stamp))
        self._rgb_info_pub.publish(self._camera_info(rgb.shape[1], rgb.shape[0], self._rgb_frame_id, stamp))

        if depth_raw is not None:
            if depth_raw.dtype != np.uint16:
                depth_raw = depth_raw.astype(np.uint16)
            self._depth_pub.publish(_cv_to_image_msg(depth_raw, "16UC1", self._depth_frame_id, stamp))
            info = self._camera_info(depth_raw.shape[1], depth_raw.shape[0], self._depth_frame_id, stamp)
            info.header.frame_id = self._depth_frame_id
            info.d.append(float(scale))
            self._depth_info_pub.publish(info)

    def destroy_node(self) -> bool:
        try:
            self._socket.close(0)
            self._zmq_context.term()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = RgbdZmqPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
