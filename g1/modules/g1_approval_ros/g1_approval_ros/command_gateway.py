from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .audit import AuditLogger
from .executor import RobotCommandExecutor
from .messages import ApprovalRequest, ApprovalResponse, CommandRequest, CommandResult
from .policy import classify_action


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CommandGateway(Node):
    def __init__(self) -> None:
        super().__init__("command_gateway")
        self.declare_parameter("audit_log_path", "/tmp/g1_approval_audit.jsonl")
        self.declare_parameter("approval_timeout_sec", 120.0)

        audit_log_path = self.get_parameter("audit_log_path").get_parameter_value().string_value
        self._approval_timeout = (
            self.get_parameter("approval_timeout_sec").get_parameter_value().double_value
        )
        self._audit = AuditLogger(audit_log_path)
        self._executor = RobotCommandExecutor()
        self._pending: dict[str, dict[str, Any]] = {}
        self._recent_results: deque[dict[str, Any]] = deque(maxlen=20)

        self._request_sub = self.create_subscription(
            String, "/g1/command_request", self._on_command_request, 20
        )
        self._approval_request_pub = self.create_publisher(String, "/g1/approval_request", 20)
        self._approval_response_sub = self.create_subscription(
            String, "/g1/approval_response", self._on_approval_response, 20
        )
        self._result_pub = self.create_publisher(String, "/g1/command_result", 20)
        self._status_pub = self.create_publisher(String, "/g1/gateway_status", 10)
        self._status_srv = self.create_service(
            Trigger, "/g1/gateway_status_text", self._status_service)
        self._timer = self.create_timer(1.0, self._tick)

    def _on_command_request(self, msg: String) -> None:
        try:
            request = CommandRequest.from_json(msg.data)
        except Exception as exc:
            self.get_logger().error(f"Invalid command request payload: {exc}")
            return

        if not request.submitted_at:
            request.submitted_at = utc_now()

        policy = classify_action(request.action)
        self._audit.write(
            "command_received",
            {
                "request_id": request.request_id,
                "action": request.action,
                "parameters": request.parameters,
                "source_ip": request.source_ip,
                "requester": request.requester,
                "submitted_at": request.submitted_at,
                "code_summary": request.code_summary,
                "risk": policy.risk,
                "policy_reason": policy.reason,
            },
        )

        if not policy.allowed:
            self._publish_result(
                CommandResult(
                    request_id=request.request_id,
                    status="rejected",
                    message=policy.reason,
                    action=request.action,
                    risk=policy.risk,
                )
            )
            return

        if policy.requires_approval:
            approval = ApprovalRequest(
                request_id=request.request_id,
                action=request.action,
                source_ip=request.source_ip,
                requester=request.requester,
                risk=policy.risk,
                reason=policy.reason,
                parameters=request.parameters,
                code_summary=request.code_summary,
                submitted_at=request.submitted_at,
            )
            self._pending[request.request_id] = {
                "request": request,
                "risk": policy.risk,
                "deadline_monotonic": self.get_clock().now().nanoseconds / 1e9 + self._approval_timeout,
            }
            self._approval_request_pub.publish(String(data=approval.to_json()))
            self._audit.write(
                "approval_requested",
                {
                    "request_id": request.request_id,
                    "action": request.action,
                    "risk": policy.risk,
                    "source_ip": request.source_ip,
                    "requester": request.requester,
                },
            )
            return

        self._execute_request(request, risk=policy.risk)

    def _on_approval_response(self, msg: String) -> None:
        try:
            response = ApprovalResponse.from_json(msg.data)
        except Exception as exc:
            self.get_logger().error(f"Invalid approval response payload: {exc}")
            return

        pending = self._pending.pop(response.request_id, None)
        if pending is None:
            self.get_logger().warning(
                f"No pending request for approval response {response.request_id}")
            return

        request: CommandRequest = pending["request"]
        risk = str(pending["risk"])
        self._audit.write(
            "approval_decided",
            {
                "request_id": response.request_id,
                "approved": response.approved,
                "approver": response.approver,
                "decided_at": response.decided_at or utc_now(),
                "note": response.note,
            },
        )

        if not response.approved:
            self._publish_result(
                CommandResult(
                    request_id=request.request_id,
                    status="denied",
                    message=f"Denied by {response.approver}: {response.note}".strip(),
                    action=request.action,
                    risk=risk,
                )
            )
            return

        self._execute_request(request, risk=risk)

    def _execute_request(self, request: CommandRequest, risk: str) -> None:
        ok, message = self._executor.execute(request.action, request.parameters)
        status = "executed" if ok else "failed"
        self._audit.write(
            "command_executed",
            {
                "request_id": request.request_id,
                "action": request.action,
                "parameters": request.parameters,
                "status": status,
                "message": message,
                "risk": risk,
            },
        )
        self._publish_result(
            CommandResult(
                request_id=request.request_id,
                status=status,
                message=message,
                action=request.action,
                risk=risk,
            )
        )

    def _publish_result(self, result: CommandResult) -> None:
        payload = result.to_json()
        self._result_pub.publish(String(data=payload))
        self._recent_results.appendleft(
            {
                "request_id": result.request_id,
                "status": result.status,
                "action": result.action,
                "risk": result.risk,
                "message": result.message,
            }
        )

    def _tick(self) -> None:
        now_monotonic = self.get_clock().now().nanoseconds / 1e9
        expired = [
            request_id
            for request_id, pending in self._pending.items()
            if now_monotonic > float(pending["deadline_monotonic"])
        ]
        for request_id in expired:
            pending = self._pending.pop(request_id)
            request: CommandRequest = pending["request"]
            self._audit.write(
                "approval_timeout",
                {"request_id": request_id, "action": request.action, "risk": pending["risk"]},
            )
            self._publish_result(
                CommandResult(
                    request_id=request_id,
                    status="expired",
                    message="Approval timed out.",
                    action=request.action,
                    risk=str(pending["risk"]),
                )
            )

        status = {
            "pending_count": len(self._pending),
            "recent_results": list(self._recent_results),
        }
        self._status_pub.publish(String(data=str(status)))

    def _status_service(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = True
        response.message = f"pending={len(self._pending)} recent={len(self._recent_results)}"
        return response


def main() -> None:
    rclpy.init()
    node = CommandGateway()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
