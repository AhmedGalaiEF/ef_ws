"""Lifecycle state machine and startup classification (spec sections 8, 12).

The critical invariant this module protects: an unexpected process
restart must never be classified the same way as a deliberate sleep. The
*only* signal that produces ``agent_wake`` is a persisted checkpoint whose
``lifecycle_state`` was explicitly written as ``sleeping`` by the
deliberate-sleep sequence (``agent/skills.py``'s ``request_sleep``
skill) before the process exited. A bare reboot, a crash, or a
``systemctl restart`` all land on a checkpoint in some other state (most
commonly ``awake``), which classifies as ``agent_restart`` instead --
matching spec section 12's "a reboot alone must not be interpreted as
deliberate sleep".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import EventType, LifecycleState, RuntimeCheckpoint

ALLOWED_TRANSITIONS: dict[LifecycleState, set[LifecycleState]] = {
    LifecycleState.FIRST_BOOT: {LifecycleState.AWAKE},
    LifecycleState.RESTART_RECOVERY: {LifecycleState.AWAKE},
    LifecycleState.WAKING: {LifecycleState.AWAKE},
    LifecycleState.AWAKE: {
        LifecycleState.AWAKE,
        LifecycleState.MAINTENANCE,
        LifecycleState.PRE_SLEEP,
    },
    LifecycleState.MAINTENANCE: {
        LifecycleState.MAINTENANCE,
        LifecycleState.AWAKE,
        LifecycleState.PRE_SLEEP,
    },
    LifecycleState.PRE_SLEEP: {
        LifecycleState.SLEEPING,
        LifecycleState.AWAKE,  # sleep aborted (e.g. permission denied mid-sequence)
    },
    LifecycleState.SLEEPING: set(),  # only left via a new process (classify_startup)
}


class InvalidLifecycleTransition(RuntimeError):
    pass


def classify_startup(checkpoint: Optional[RuntimeCheckpoint]) -> tuple[EventType, LifecycleState]:
    """Decide the first event of this runtime from the persisted checkpoint.

    Returns ``(event_type, entry_lifecycle_state)``. The entry lifecycle
    state is a transitional one (``first_boot`` / ``restart_recovery`` /
    ``waking``); the scheduler moves it to ``awake`` once the first
    cognitive call for that event has completed (spec sections 4, 7, 11).
    """
    if checkpoint is None:
        return EventType.AGENT_FIRST_BOOT, LifecycleState.FIRST_BOOT
    if checkpoint.lifecycle_state == LifecycleState.SLEEPING:
        return EventType.AGENT_WAKE, LifecycleState.WAKING
    return EventType.AGENT_RESTART, LifecycleState.RESTART_RECOVERY


@dataclass
class LifecycleController:
    """Tracks the live lifecycle state within one running process."""

    state: LifecycleState
    history: list[LifecycleState] = field(default_factory=list)

    def transition(self, new_state: LifecycleState) -> None:
        allowed = ALLOWED_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidLifecycleTransition(
                f"{self.state.value} -> {new_state.value} is not a permitted lifecycle transition "
                f"(allowed: {sorted(s.value for s in allowed) or 'none'})"
            )
        self.history.append(self.state)
        self.state = new_state
