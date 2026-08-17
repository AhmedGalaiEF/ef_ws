from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


OPENAIAPI_ROOT = Path(__file__).resolve().parents[1] / "g1" / "modules" / "scripts" / "openaiapi"
if str(OPENAIAPI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENAIAPI_ROOT))

from agent.llctl import LlctlAdapter  # noqa: E402
from agent.navigation import NavigationAdapter  # noqa: E402


def _settings(*, joint: bool = True, ik: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        llctl=SimpleNamespace(
            enabled=True,
            allow_joint_control=joint,
            allow_ik_control=ik,
            require_explicit_enable_each_session=False,
            session_timeout_s=60.0,
        )
    )


def test_llctl_rejects_non_finite_targets_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = LlctlAdapter()
    adapter.state.session_enabled = True
    adapter.state.last_activity_at = time.time()
    monkeypatch.setattr(adapter, "ensure_backend", lambda: True)

    ok, reason = adapter.validate_joint_command(_settings(), joint="waist", q=math.nan)
    assert not ok and "finite" in reason

    ok, reason = adapter.validate_ee_command(_settings(), side="left", x=0.0, y=0.0, z=math.inf)
    assert not ok and "finite" in reason


def test_llctl_command_path_cannot_bypass_joint_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = LlctlAdapter()
    adapter.state.session_enabled = True
    adapter.state.last_activity_at = time.time()
    monkeypatch.setattr(adapter, "ensure_backend", lambda: True)
    adapter._robot_link = object()

    result = adapter.command_joint(_settings(), joint="waist", q=math.nan)

    assert "finite" in result


def test_navigation_snapshot_marks_fresh_canonical_topics() -> None:
    class Backend:
        def status(self) -> str:
            return (
                '{"slam_running": true, "relocation_ready": true, "last_action": "go_to", '
                '"pose": {"x": 1, "y": 2, "yaw": 0.5}, "fresh_topics": ["rt/lowstate"]}'
            )

    snapshot = NavigationAdapter(slam_backend=Backend()).snapshot().as_dict()
    assert snapshot["slam"] == "running"
    assert snapshot["navigation"] == "active"
    assert snapshot["topics"]["/lowstate"]["alive"] is True
    assert snapshot["current_pose"] == "x=1.00 y=2.00 yaw=0.50"
