"""Small, viz-app-only glue shared by scripts in this directory.

Everything that is an actual robot capability (an RPC, a subscription, a
motion primitive) belongs on `sdk_wrapper.G1`, not here -- this module only
holds presentation/adapter code that has no business being in the SDK
wrapper: turning a G1 RPC result into a normalized ok/code/raw dict, or a
service-list response into display rows. Keep it that way; if a helper here
starts doing DDS I/O of its own, it should move to sdk_wrapper.G1 instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def normalize_rpc(result: dict[str, Any]) -> dict[str, Any]:
    """Turns a `sdk_wrapper.G1` RPC result (`{"code": int, "raw": ...}`) into
    `{"code", "ok", "raw"}`, decoding `raw` from JSON where possible and
    treating a non-zero `errorCode` / `succeed: false` in the decoded body as
    failure too, not just a non-zero top-level RPC `code`.

    Equivalent to slam_web_app.py's old `response_dict()`, adapted for
    sdk_wrapper.G1's plain-dict return shape instead of sdk_slam.SlamResponse
    objects."""
    code = int(result.get("code", -1))
    raw = result.get("raw")
    try:
        raw = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        pass
    ok = code == 0
    if isinstance(raw, dict):
        ok = ok and int(raw.get("errorCode", 0)) == 0 and bool(raw.get("succeed", True))
    return {"code": code, "ok": bool(ok), "raw": raw}


@dataclass(frozen=True)
class ServiceRow:
    name: str
    description: str
    status: int | None = None
    protected: bool | None = None


#: status code -> (label, color-ish name a renderer can map to its own
#: palette). Not Rich-specific on purpose -- service_view.py maps these to
#: rich.text.Text, but a Dash-based viewer could use the same table.
SERVICE_STATUS_LABELS = {
    0: ("ON", "green"),
    1: ("OFF", "red"),
    5: ("PROTECTED", "yellow"),
}

#: robot_state / service RPC error codes worth a human-readable hint.
SERVICE_ERROR_HINTS = {
    0: "success",
    3001: "RPC unknown error",
    3102: "RPC client send error",
    3103: "RPC API not registered",
    3104: "RPC timeout",
    3105: "RPC API mismatch",
    3106: "RPC client data error",
    3201: "RPC server send error",
    3202: "RPC server internal error",
    3203: "RPC API not implemented",
    3204: "RPC server parameter error",
    5201: "service switch execution error",
    5202: "service is protected",
}


def service_error_text(code: int | None) -> str:
    if code is None:
        return ""
    return f"{code}: {SERVICE_ERROR_HINTS.get(int(code), 'unknown')}"


def service_rows(g1: Any, service: str | None = None) -> list[ServiceRow]:
    """Adapts `G1.get_service()`'s list-of-dicts into `ServiceRow`s."""
    rows = g1.get_service(service)
    rows = [rows] if isinstance(rows, dict) else (rows or [])
    return [
        ServiceRow(
            name=row["name"],
            description=row.get("description", ""),
            status=row.get("status"),
            protected=row.get("protected"),
        )
        for row in rows
    ]


def resolve_service_name(token: str, rows: list[ServiceRow]) -> str:
    """Resolves a service_view.py command token (a 1-based table index or a
    service name, case-insensitive) against the currently displayed rows."""
    value = token.strip()
    if not value:
        raise ValueError("missing service name or table number")
    if value.isdigit():
        index = int(value)
        if not 1 <= index <= len(rows):
            raise ValueError(f"service number {index} is outside the table")
        return rows[index - 1].name
    for row in rows:
        if row.name == value:
            return row.name
    for row in rows:
        if row.name.lower() == value.lower():
            return row.name
    raise ValueError(f"unknown service: {value}")


__all__ = [
    "normalize_rpc",
    "ServiceRow",
    "SERVICE_STATUS_LABELS",
    "SERVICE_ERROR_HINTS",
    "service_error_text",
    "service_rows",
    "resolve_service_name",
]
