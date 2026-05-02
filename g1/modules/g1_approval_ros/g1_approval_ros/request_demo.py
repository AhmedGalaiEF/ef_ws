from __future__ import annotations

from datetime import datetime, timezone
import json
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RequestDemo(Node):
    def __init__(self, action: str, source_ip: str, requester: str) -> None:
        super().__init__("request_demo")
        self._publisher = self.create_publisher(String, "/g1/command_request", 20)
        self._action = action
        self._source_ip = source_ip
        self._requester = requester
        self._timer = self.create_timer(0.5, self._publish_once)
        self._published = False

    def _publish_once(self) -> None:
        if self._published:
            return
        payload = {
            "action": self._action,
            "parameters": {"distance_m": 0.4, "angle_deg": 20.0, "hand": "right"},
            "source_ip": self._source_ip,
            "requester": self._requester,
            "submitted_at": utc_now(),
            "code_summary": f"Mapped gateway action '{self._action}' to a fixed robot SDK call.",
        }
        self._publisher.publish(String(data=json.dumps(payload, ensure_ascii=True, sort_keys=True)))
        self.get_logger().info(f"Published demo request: {payload}")
        self._published = True


def main() -> None:
    action = sys.argv[1] if len(sys.argv) > 1 else "walk_distance"
    source_ip = sys.argv[2] if len(sys.argv) > 2 else "192.168.1.50"
    requester = sys.argv[3] if len(sys.argv) > 3 else "demo-client"

    rclpy.init()
    node = RequestDemo(action=action, source_ip=source_ip, requester=requester)
    try:
        rclpy.spin_once(node, timeout_sec=1.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()
