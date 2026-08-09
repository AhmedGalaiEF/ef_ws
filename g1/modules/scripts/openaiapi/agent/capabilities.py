"""Deterministic capability resolver (spec sections 17, 18).

    MODEL           understands what is being requested
       |
    RUNTIME (here)  determines what is physically and permissibly true
       |
    MODEL/ANNOUNCE   communicates or acts on the grounded result

Models ``g1_approval_ros/policy.py``'s ``PolicyDecision`` shape
(``allowed`` / ``requires_approval`` / ``risk`` / ``reason``) and extends
it with the settings + live-state checks the spec asks for around arm
motion. This module never asks the model anything, and the model can
never override what it returns -- ``agent/skills.py`` calls
``resolve_skill`` before dispatching any skill, unconditionally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .models import RobotStateSnapshot
from .settings.models import AgentSettings

# Skills that specifically require arm-motion authority. agent/skills.py
# owns the full registry; this only needs to know which of those names the
# arm-specific checks below apply to.
#
# Two tiers, matching sdk_client.Robot's actual architecture (confirmed by
# reading it, not assumed): low-level Cartesian control
# (extend_arm_forward/move_upper_body_joint, wrapped by llm_client's
# "reach_forward" tool) goes over the /arm_sdk (or /low_cmd) DDS channel,
# so it gates on real backend availability. High-level canned actions
# (wave/high-five/etc, wrapped as "gesture", and release_arm()/
# release_arms(), wrapped as "release_arms") go through a separate,
# always-on G1ArmActionClient service -- they still require arm-motion
# authority and a commandable arm, but have no /arm_sdk or /low_cmd
# backend to be "available" or not.
LOW_LEVEL_ARM_SKILLS = {"move_arm_demo", "reach_forward"}
HIGH_LEVEL_ARM_SKILLS = {"gesture", "wave", "high_wave", "release_arms"}
HAND_SKILLS = {"grab", "release"}
ARM_SKILLS = LOW_LEVEL_ARM_SKILLS | HIGH_LEVEL_ARM_SKILLS | HAND_SKILLS
LOW_LEVEL_ARM_FAULTS = ("lowstate",)
HAND_FAULTS = ("left_hand_state", "right_hand_state")


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_approval: bool
    risk: str
    reason: str


AvailabilityProbe = Callable[[], bool]


def _default_unavailable() -> bool:
    """Conservative default when no real availability probe is wired in.

    Phase 1 has no live ROS/DDS connection in this sandbox to check
    ``/arm_sdk`` or ``/low_cmd`` liveness against (no rclpy here to verify
    it) -- defaulting a missing probe to "unavailable" means its absence
    can only make the resolver *more* conservative, never silently
    permissive. Wiring a real probe (e.g. attempting
    ``arm_sdk.ArmSdk(...).resync()``) is a TODO for the ROS-integration
    phase.
    """
    return False


class CapabilityResolver:
    def __init__(
        self,
        *,
        arm_sdk_available: AvailabilityProbe = _default_unavailable,
        low_cmd_available: AvailabilityProbe = _default_unavailable,
    ) -> None:
        self._arm_sdk_available = arm_sdk_available
        self._low_cmd_available = low_cmd_available

    @staticmethod
    def _matching_faults(robot_state: RobotStateSnapshot, prefixes: tuple[str, ...]) -> list[str]:
        return [
            fault
            for fault in robot_state.active_faults
            if any(fault == prefix or fault.startswith(f"{prefix}[") for prefix in prefixes)
        ]

    def resolve_arm_motion(
        self, *, settings: AgentSettings, robot_state: RobotStateSnapshot
    ) -> PolicyDecision:
        """Ground a query_capability(target=arm) / move_arm request against reality."""
        if not settings.motion.allow_arm_motion:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                risk="low",
                reason="Arm motion is disabled in the current settings (motion.allow_arm_motion=false).",
            )

        if robot_state.arm_control_state == "released":
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                risk="low",
                reason="The arm is currently in a released/uncommandable state.",
            )

        arm_faults = self._matching_faults(robot_state, LOW_LEVEL_ARM_FAULTS)
        if arm_faults:
            return PolicyDecision(
                allowed=False,
                requires_approval=True,
                risk="high",
                reason=f"Arm-relevant faults present: {', '.join(arm_faults)}.",
            )

        arm_sdk_permitted = settings.motion.allow_arm_sdk
        low_cmd_permitted = settings.motion.allow_low_cmd

        if arm_sdk_permitted and self._arm_sdk_available():
            return PolicyDecision(
                allowed=True,
                requires_approval=False,
                risk="low",
                reason="Arm motion is currently available through /arm_sdk.",
            )
        if low_cmd_permitted and self._low_cmd_available():
            return PolicyDecision(
                allowed=True,
                requires_approval=True,
                risk="high",
                reason=(
                    "Arm motion is only available through the lower-level /low_cmd "
                    "backend, which requires operator approval."
                ),
            )
        if arm_sdk_permitted:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                risk="low",
                reason="/arm_sdk is permitted by settings but is not currently available.",
            )
        if low_cmd_permitted:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                risk="low",
                reason="/low_cmd is permitted by settings but is not currently available.",
            )
        return PolicyDecision(
            allowed=False,
            requires_approval=False,
            risk="low",
            reason="Neither /arm_sdk nor /low_cmd is permitted by the current settings.",
        )

    def resolve_high_level_arm_action(
        self, *, settings: AgentSettings, robot_state: RobotStateSnapshot
    ) -> PolicyDecision:
        """Gate a canned HL arm action (gesture / release_arms).

        Unlike ``resolve_arm_motion``, this does not check ``/arm_sdk`` or
        ``/low_cmd`` availability: HL actions go through
        ``G1ArmActionClient``, a separate always-on service, not either of
        those DDS channels (see the ``ARM_SKILLS`` comment above).
        """
        if not settings.motion.allow_arm_motion:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                risk="low",
                reason="Arm motion is disabled in the current settings (motion.allow_arm_motion=false).",
            )
        if robot_state.arm_control_state == "released":
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                risk="low",
                reason="The arm is currently in a released/uncommandable state.",
            )
        return PolicyDecision(
            allowed=True,
            requires_approval=False,
            risk="low",
            reason="High-level arm action is available.",
        )

    def resolve_hand_action(
        self, *, settings: AgentSettings, robot_state: RobotStateSnapshot
    ) -> PolicyDecision:
        """Gate gripper actions against hand-state availability."""
        if not settings.motion.allow_arm_motion:
            return PolicyDecision(
                allowed=False,
                requires_approval=False,
                risk="low",
                reason="Hand motion is disabled because motion.allow_arm_motion=false.",
            )
        hand_faults = self._matching_faults(robot_state, HAND_FAULTS)
        if hand_faults:
            return PolicyDecision(
                allowed=False,
                requires_approval=True,
                risk="high",
                reason=f"Hand-state faults present: {', '.join(hand_faults)}.",
            )
        return PolicyDecision(
            allowed=True,
            requires_approval=False,
            risk="low",
            reason="Gripper action is available.",
        )

    def resolve_skill(
        self, skill_name: str, *, settings: AgentSettings, robot_state: RobotStateSnapshot
    ) -> PolicyDecision:
        """Entry point ``agent/skills.py`` calls before dispatching any skill."""
        if skill_name in LOW_LEVEL_ARM_SKILLS:
            return self.resolve_arm_motion(settings=settings, robot_state=robot_state)
        if skill_name in HIGH_LEVEL_ARM_SKILLS:
            return self.resolve_high_level_arm_action(settings=settings, robot_state=robot_state)
        if skill_name in HAND_SKILLS:
            return self.resolve_hand_action(settings=settings, robot_state=robot_state)
        # Non-arm skills (announce, navigate, base move, stop, ...) are not
        # gated here beyond the settings toggles their own handlers already
        # read -- see the nav/ll checks scene_executor.STEP_HANDLERS
        # performs, reused as-is by agent/skills.py.
        return PolicyDecision(
            allowed=True,
            requires_approval=False,
            risk="low",
            reason="No additional capability gate defined for this skill.",
        )
