from __future__ import annotations

from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def _array_to_image_msg(array: np.ndarray, encoding: str, frame_id: str, stamp) -> Image:
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


class RgbdUsbPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("rgbd_usb_publisher")

        self._width = int(self.declare_parameter("width", 1280).value)
        self._height = int(self.declare_parameter("height", 720).value)
        self._fps = int(self.declare_parameter("fps", 30).value)
        self._align_depth_to_color = bool(self.declare_parameter("align_depth_to_color", True).value)
        self._rgb_topic = str(self.declare_parameter("rgb_topic", "/rgbd/color/image_raw").value)
        self._depth_topic = str(self.declare_parameter("depth_topic", "/rgbd/depth/image_raw").value)
        self._rgb_info_topic = str(self.declare_parameter("rgb_camera_info_topic", "/rgbd/color/camera_info").value)
        self._depth_info_topic = str(self.declare_parameter("depth_camera_info_topic", "/rgbd/depth/camera_info").value)
        self._rgb_frame_id = str(self.declare_parameter("rgb_frame_id", "rgb_camera_frame").value)
        self._depth_frame_id = str(self.declare_parameter("depth_frame_id", "depth_camera_frame").value)
        self._fx = float(self.declare_parameter("fx", 623.53829072479584).value)
        self._fy = float(self.declare_parameter("fy", 623.53829072479584).value)
        self._cx = float(self.declare_parameter("cx", 639.5).value)
        self._cy = float(self.declare_parameter("cy", 359.5).value)

        try:
            import pyrealsense2 as rs
        except Exception as exc:
            raise RuntimeError(
                "rgbd_usb_publisher requires pyrealsense2 on the Jetson."
            ) from exc

        self._rs = rs
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
        self._config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)
        self._profile = self._pipeline.start(self._config)
        self._depth_scale = float(
            self._profile.get_device().first_depth_sensor().get_depth_scale()
        )
        self._align = rs.align(rs.stream.color) if self._align_depth_to_color else None

        self._rgb_pub = self.create_publisher(Image, self._rgb_topic, 10)
        self._depth_pub = self.create_publisher(Image, self._depth_topic, 10)
        self._rgb_info_pub = self.create_publisher(CameraInfo, self._rgb_info_topic, 10)
        self._depth_info_pub = self.create_publisher(CameraInfo, self._depth_info_topic, 10)

        self._timer = self.create_timer(max(1.0 / max(self._fps, 1), 0.001), self._publish_once)
        self.get_logger().info(
            f"Publishing local USB RGBD at {self._width}x{self._height}@{self._fps}Hz "
            f"with depth_scale={self._depth_scale:.6f} m/unit"
        )

    def _camera_info(self, width: int, height: int, frame_id: str, stamp, depth_scale: Optional[float] = None) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.width = int(width)
        msg.height = int(height)
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
        if depth_scale is not None:
            msg.d.append(float(depth_scale))
        return msg

    def _publish_once(self) -> None:
        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        color_rgb = color_bgr[:, :, ::-1].copy()

        stamp = self.get_clock().now().to_msg()
        self._rgb_pub.publish(_array_to_image_msg(color_rgb, "rgb8", self._rgb_frame_id, stamp))
        self._depth_pub.publish(_array_to_image_msg(depth_raw, "16UC1", self._depth_frame_id, stamp))
        self._rgb_info_pub.publish(self._camera_info(color_rgb.shape[1], color_rgb.shape[0], self._rgb_frame_id, stamp))
        self._depth_info_pub.publish(
            self._camera_info(depth_raw.shape[1], depth_raw.shape[0], self._depth_frame_id, stamp, self._depth_scale)
        )

    def destroy_node(self) -> bool:
        try:
            self._pipeline.stop()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = RgbdUsbPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
