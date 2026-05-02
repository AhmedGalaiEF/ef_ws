from __future__ import annotations

from datetime import datetime, timezone
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .messages import ApprovalResponse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApprovalConsole(Node):
    def __init__(self) -> None:
        super().__init__("approval_console")
        self.declare_parameter("auto_approve_low_risk", False)
        self._response_pub = self.create_publisher(String, "/g1/approval_response", 20)
        self._request_sub = self.create_subscription(
            String, "/g1/approval_request", self._on_request, 20
        )

    def _on_request(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().error(f"Invalid approval request payload: {exc}")
            return

        pretty = json.dumps(payload, indent=2, sort_keys=True)
        print("\n=== Approval Request ===")
        print(pretty)
        print("Approve? Type 'yes' to approve, anything else to deny:")
        try:
            answer = input("> ").strip().lower()
        except EOFError:
            answer = ""
        approved = answer == "yes"
        print("Approver name:")
        try:
            approver = input("> ").strip() or "console_operator"
        except EOFError:
            approver = "console_operator"
        print("Optional note:")
        try:
            note = input("> ").strip()
        except EOFError:
            note = ""

        response = ApprovalResponse(
            request_id=str(payload.get("request_id", "")),
            approved=approved,
            approver=approver,
            decided_at=utc_now(),
            note=note,
        )
        self._response_pub.publish(String(data=response.to_json()))


def main() -> None:
    rclpy.init()
    node = ApprovalConsole()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
