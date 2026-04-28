from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField

from .livox_sdk2 import Livox2


def _builtin_now(node: Node) -> Time:
    return node.get_clock().now().to_msg()


def _xyz_to_cloud_msg(points_xyz: np.ndarray, frame_id: str, stamp: Time) -> PointCloud2:
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = int(points_xyz.shape[0])
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = bool(points_xyz.size > 0)
    msg.data = np.asarray(points_xyz, dtype=np.float32).tobytes()
    return msg


class _LivoxRosBridge(Livox2):
    def __init__(
        self,
        node: Node,
        config_path: Path,
        host_ip: str,
        topic_name: str,
        frame_id: str,
        frame_time: float,
        frame_packets: int,
    ) -> None:
        self._node = node
        self._frame_id = frame_id
        self._publisher = node.create_publisher(PointCloud2, topic_name, 10)
        self._latest_cloud: Optional[PointCloud2] = None
        super().__init__(config_path, host_ip, frame_time=frame_time, frame_packets=frame_packets)

    def handle_points(self, xyz: np.ndarray) -> None:
        self._latest_cloud = _xyz_to_cloud_msg(xyz, self._frame_id, _builtin_now(self._node))

    def publish_pending(self) -> None:
        cloud = self._latest_cloud
        if cloud is None:
            return
        self._latest_cloud = None
        self._publisher.publish(cloud)


class LivoxPointsPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("livox_points_publisher")

        default_config = Path(__file__).resolve().parents[2] / "mid360_config.json"
        config_path = Path(self.declare_parameter("config_path", str(default_config)).value).expanduser()
        host_ip = str(self.declare_parameter("host_ip", "192.168.123.164").value)
        topic_name = str(self.declare_parameter("topic", "/livox/points").value)
        frame_id = str(self.declare_parameter("frame_id", "livox_frame").value)
        frame_time = float(self.declare_parameter("frame_time", 0.20).value)
        frame_packets = int(self.declare_parameter("frame_packets", 120).value)
        self._bridge = _LivoxRosBridge(
            self,
            config_path=config_path,
            host_ip=host_ip,
            topic_name=topic_name,
            frame_id=frame_id,
            frame_time=frame_time,
            frame_packets=frame_packets,
        )
        self._timer = self.create_timer(0.01, self._bridge.publish_pending)
        self.get_logger().info(
            f"Publishing Livox points to {topic_name} using config={config_path} host_ip={host_ip}"
        )

    def destroy_node(self) -> bool:
        try:
            self._bridge.shutdown()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = LivoxPointsPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
