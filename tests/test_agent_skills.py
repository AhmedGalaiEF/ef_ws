from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")
if not pydantic.VERSION.startswith("2"):
    pytest.skip(f"agent package requires pydantic>=2 (found {pydantic.VERSION})", allow_module_level=True)

AGENT_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "g1" / "modules" / "scripts"
if str(AGENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_SCRIPTS_DIR))

from agent.capabilities import CapabilityResolver  # noqa: E402
from agent.models import RobotStateSnapshot  # noqa: E402
from agent.settings.models import AgentSettings  # noqa: E402
from agent.skills import (  # noqa: E402
    build_offline_registry,
    invoke_with_capability_check,
    resolve_and_maybe_invoke,
)


def _state(**overrides: object) -> RobotStateSnapshot:
    payload = dict(timestamp=time.time(), arm_control_state="commandable", active_faults=[], source="mock")
    payload.update(overrides)
    return RobotStateSnapshot(**payload)


def test_offline_registry_includes_core_and_new_skills() -> None:
    registry = build_offline_registry()
    names = set(registry.names())
    assert {"move", "hand_open", "hand_close", "gesture", "release_arms", "announce"} <= names
    assert {"request_sleep", "request_charge"} <= names


def test_request_charge_is_an_honest_stub() -> None:
    registry = build_offline_registry()
    result = registry.invoke("request_charge")
    assert result.ok is False
    assert result.detail.get("stub") is True


def test_unknown_skill_name_returns_error_result_not_raise() -> None:
    registry = build_offline_registry()
    result = registry.invoke("does_not_exist")
    assert result.ok is False
    assert "unknown skill" in result.message


def test_resolve_and_maybe_invoke_disabled_denies_without_confirmation() -> None:
    registry = build_offline_registry()
    resolver = CapabilityResolver()
    settings = AgentSettings()
    settings.set_skill_mode("release_arms", "disabled")

    outcome = resolve_and_maybe_invoke(
        registry, resolver, "release_arms", settings=settings, robot_state=_state()
    )
    assert outcome.status == "denied"


def test_resolve_and_maybe_invoke_confirm_mode_needs_confirmation_first() -> None:
    registry = build_offline_registry()
    resolver = CapabilityResolver()
    settings = AgentSettings()
    settings.set_skill_mode("release_arms", "confirm")

    first = resolve_and_maybe_invoke(
        registry, resolver, "release_arms", settings=settings, robot_state=_state()
    )
    assert first.status == "needs_confirmation"
    assert first.result is None

    second = resolve_and_maybe_invoke(
        registry, resolver, "release_arms", settings=settings, robot_state=_state(), confirmed=True
    )
    assert second.status == "executed"
    assert second.result is not None and second.result.ok


def test_approval_floor_upgrades_auto_to_confirm_when_capability_requires_approval() -> None:
    """An 'auto' skill mode must not bypass a safety-driven requires_approval result."""
    registry = build_offline_registry()
    settings = AgentSettings()
    settings.motion.allow_arm_sdk = False
    settings.motion.allow_low_cmd = True
    settings.set_skill_mode("reach_forward", "auto")  # operator asked for auto...
    resolver = CapabilityResolver(low_cmd_available=lambda: True)  # ...but only the risky /low_cmd path exists

    outcome = resolve_and_maybe_invoke(
        registry, resolver, "reach_forward", settings=settings, robot_state=_state()
    )
    assert outcome.policy.requires_approval
    assert outcome.status == "needs_confirmation"  # not silently auto-executed


def test_invoke_with_capability_check_denies_requires_approval_outright() -> None:
    """The low-stakes announcement path has no confirmation loop at all -- a
    requires_approval capability result is a hard deny there, not a prompt."""
    registry = build_offline_registry()
    settings = AgentSettings()
    settings.motion.allow_arm_sdk = False
    settings.motion.allow_low_cmd = True
    resolver = CapabilityResolver(low_cmd_available=lambda: True)

    decision, result = invoke_with_capability_check(
        registry, resolver, "reach_forward", settings=settings, robot_state=_state()
    )
    assert not decision.allowed
    assert result is None
