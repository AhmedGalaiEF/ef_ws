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


def _state(**overrides: object) -> RobotStateSnapshot:
    payload = dict(timestamp=time.time(), arm_control_state="commandable", active_faults=[], source="mock")
    payload.update(overrides)
    return RobotStateSnapshot(**payload)


def test_arm_motion_denied_when_setting_disabled() -> None:
    settings = AgentSettings()
    settings.motion.allow_arm_motion = False
    resolver = CapabilityResolver(arm_sdk_available=lambda: True)
    decision = resolver.resolve_arm_motion(settings=settings, robot_state=_state())
    assert not decision.allowed
    assert "allow_arm_motion" in decision.reason


def test_arm_motion_denied_when_arm_released() -> None:
    resolver = CapabilityResolver(arm_sdk_available=lambda: True)
    decision = resolver.resolve_arm_motion(settings=AgentSettings(), robot_state=_state(arm_control_state="released"))
    assert not decision.allowed
    assert "released" in decision.reason


def test_arm_motion_denied_and_needs_approval_when_faults_present() -> None:
    resolver = CapabilityResolver(arm_sdk_available=lambda: True)
    decision = resolver.resolve_arm_motion(
        settings=AgentSettings(), robot_state=_state(active_faults=["overheat"])
    )
    assert not decision.allowed
    assert decision.requires_approval
    assert decision.risk == "high"


def test_arm_motion_allowed_via_arm_sdk_when_available() -> None:
    resolver = CapabilityResolver(arm_sdk_available=lambda: True, low_cmd_available=lambda: False)
    decision = resolver.resolve_arm_motion(settings=AgentSettings(), robot_state=_state())
    assert decision.allowed
    assert not decision.requires_approval
    assert "/arm_sdk" in decision.reason


def test_arm_motion_denied_when_arm_sdk_permitted_but_unavailable() -> None:
    """Matches the spec's worked example: '/arm_sdk is permitted but is currently unavailable.'"""
    resolver = CapabilityResolver()  # default probes: both unavailable
    decision = resolver.resolve_arm_motion(settings=AgentSettings(), robot_state=_state())
    assert not decision.allowed
    assert "/arm_sdk is permitted" in decision.reason


def test_arm_motion_falls_back_to_low_cmd_and_requires_approval() -> None:
    settings = AgentSettings()
    settings.motion.allow_arm_sdk = False
    settings.motion.allow_low_cmd = True
    resolver = CapabilityResolver(low_cmd_available=lambda: True)
    decision = resolver.resolve_arm_motion(settings=settings, robot_state=_state())
    assert decision.allowed
    assert decision.requires_approval
    assert decision.risk == "high"


def test_arm_motion_denied_when_neither_backend_permitted() -> None:
    settings = AgentSettings()
    settings.motion.allow_arm_sdk = False
    settings.motion.allow_low_cmd = False
    resolver = CapabilityResolver(arm_sdk_available=lambda: True, low_cmd_available=lambda: True)
    decision = resolver.resolve_arm_motion(settings=settings, robot_state=_state())
    assert not decision.allowed
    assert "Neither /arm_sdk nor /low_cmd" in decision.reason


def test_high_level_arm_action_does_not_require_arm_sdk_probe() -> None:
    """gesture/release_arms use G1ArmActionClient, not /arm_sdk or /low_cmd."""
    resolver = CapabilityResolver()  # both probes default to unavailable
    decision = resolver.resolve_high_level_arm_action(settings=AgentSettings(), robot_state=_state())
    assert decision.allowed


def test_high_level_arm_action_denied_when_motion_disabled() -> None:
    settings = AgentSettings()
    settings.motion.allow_arm_motion = False
    resolver = CapabilityResolver()
    decision = resolver.resolve_high_level_arm_action(settings=settings, robot_state=_state())
    assert not decision.allowed


def test_resolve_skill_routes_low_level_vs_high_level_correctly() -> None:
    resolver = CapabilityResolver()  # both backends unavailable by default
    settings = AgentSettings()
    state = _state()

    low_level = resolver.resolve_skill("reach_forward", settings=settings, robot_state=state)
    assert not low_level.allowed  # no backend available -> denied

    high_level = resolver.resolve_skill("gesture", settings=settings, robot_state=state)
    assert high_level.allowed  # no backend needed -> allowed


def test_resolve_skill_has_no_gate_for_unrelated_skills() -> None:
    resolver = CapabilityResolver()
    decision = resolver.resolve_skill("announce", settings=AgentSettings(), robot_state=_state())
    assert decision.allowed
    assert not decision.requires_approval
