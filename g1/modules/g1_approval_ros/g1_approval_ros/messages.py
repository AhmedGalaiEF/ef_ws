from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class CommandRequest:
    request_id: str = field(default_factory=lambda: uuid4().hex)
    action: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    source_ip: str = "unknown"
    requester: str = "unknown"
    submitted_at: str = ""
    code_summary: str = ""

    @classmethod
    def from_json(cls, payload: str) -> "CommandRequest":
        raw = json.loads(payload)
        return cls(
            request_id=str(raw.get("request_id") or uuid4().hex),
            action=str(raw.get("action", "")),
            parameters=dict(raw.get("parameters", {}) or {}),
            source_ip=str(raw.get("source_ip", "unknown")),
            requester=str(raw.get("requester", "unknown")),
            submitted_at=str(raw.get("submitted_at", "")),
            code_summary=str(raw.get("code_summary", "")),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, sort_keys=True)


@dataclass
class ApprovalRequest:
    request_id: str
    action: str
    source_ip: str
    requester: str
    risk: str
    reason: str
    parameters: dict[str, Any]
    code_summary: str
    submitted_at: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, sort_keys=True)


@dataclass
class ApprovalResponse:
    request_id: str
    approved: bool
    approver: str
    decided_at: str
    note: str = ""

    @classmethod
    def from_json(cls, payload: str) -> "ApprovalResponse":
        raw = json.loads(payload)
        return cls(
            request_id=str(raw.get("request_id", "")),
            approved=bool(raw.get("approved", False)),
            approver=str(raw.get("approver", "unknown")),
            decided_at=str(raw.get("decided_at", "")),
            note=str(raw.get("note", "")),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, sort_keys=True)


@dataclass
class CommandResult:
    request_id: str
    status: str
    message: str
    action: str
    risk: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, sort_keys=True)
