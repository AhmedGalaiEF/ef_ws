from __future__ import annotations

from dataclasses import dataclass


SAFE_ACTIONS = frozenset(
    {
        "get_state",
        "get_pose",
        "stop",
        "balanced_stand",
        "hand_open",
        "hand_close",
        "walk_distance",
        "turn_angle",
        "start_slam",
        "stop_slam",
    }
)

HIGH_RISK_ACTIONS = frozenset(
    {
        "walk_distance",
        "turn_angle",
        "start_slam",
        "stop_slam",
    }
)

CRITICAL_ACTIONS = frozenset({"run_script", "run_shell", "firmware_update"})


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_approval: bool
    risk: str
    reason: str


def classify_action(action: str) -> PolicyDecision:
    normalized = str(action).strip().lower()
    if normalized in CRITICAL_ACTIONS:
        return PolicyDecision(
            allowed=False,
            requires_approval=False,
            risk="critical",
            reason="Arbitrary code execution and firmware changes are denied by policy.",
        )
    if normalized not in SAFE_ACTIONS:
        return PolicyDecision(
            allowed=False,
            requires_approval=False,
            risk="unknown",
            reason="Action is not in the approved command allowlist.",
        )
    if normalized in HIGH_RISK_ACTIONS:
        return PolicyDecision(
            allowed=True,
            requires_approval=True,
            risk="high",
            reason="Motion or SLAM actions require operator approval.",
        )
    return PolicyDecision(
        allowed=True,
        requires_approval=False,
        risk="low",
        reason="Allowed low-risk action.",
    )
