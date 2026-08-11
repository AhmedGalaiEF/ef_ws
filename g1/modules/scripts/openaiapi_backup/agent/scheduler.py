"""Event-driven cognitive scheduler (spec section 5, 13).

Owns two things only: the pending-event queue, and the
``previous_cognitive_timestamp`` / ``elapsed_since_last_cognition_s``
bookkeeping every planner call needs. It does not itself decide *whether*
an event is worth a model call -- that is the planner's job, and
``no_action`` is an entirely legitimate response to most periodic ticks
(the scheduler just makes sure a tick becomes due at a sensible cadence,
honoring the previous decision's ``next_tick_s`` when one was given).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import EventType, RuntimeCheckpoint

DEFAULT_TICK_INTERVAL_S = 30.0


@dataclass
class QueuedEvent:
    event_type: EventType
    payload: dict = field(default_factory=dict)


class CognitiveScheduler:
    def __init__(self, *, default_tick_interval_s: float = DEFAULT_TICK_INTERVAL_S) -> None:
        self.default_tick_interval_s = default_tick_interval_s
        self._queue: list[QueuedEvent] = []
        self.last_cognitive_timestamp: Optional[float] = None
        self._next_tick_due_at: Optional[float] = None

    # -- seeding -----------------------------------------------------------

    def seed_from_checkpoint(self, checkpoint: Optional[RuntimeCheckpoint]) -> None:
        """Restore ``previous_cognitive_timestamp`` after a restart/wake.

        Deliberately does *not* touch ``_next_tick_due_at`` -- the caller
        still needs to enqueue the appropriate agent_restart/agent_wake
        event first; the next periodic tick is scheduled only once that
        turn completes, via ``record_cognition``.
        """
        if checkpoint is not None:
            self.last_cognitive_timestamp = checkpoint.last_cognitive_timestamp

    # -- queue ---------------------------------------------------------

    def enqueue(self, event_type: EventType, **payload) -> None:
        self._queue.append(QueuedEvent(event_type, payload))

    def has_pending(self) -> bool:
        return bool(self._queue)

    def pop_next(self) -> Optional[QueuedEvent]:
        if not self._queue:
            return None
        return self._queue.pop(0)

    # -- timing --------------------------------------------------------

    def elapsed_since_last_cognition(self, now: float) -> Optional[float]:
        if self.last_cognitive_timestamp is None:
            return None
        return max(0.0, now - self.last_cognitive_timestamp)

    def record_cognition(self, now: float, next_tick_s: Optional[float] = None) -> None:
        """Call after every completed cognitive turn (spec section 5)."""
        self.last_cognitive_timestamp = now
        interval = next_tick_s if next_tick_s and next_tick_s > 0 else self.default_tick_interval_s
        self._next_tick_due_at = now + interval

    def maybe_enqueue_tick(self, now: float) -> bool:
        """Enqueue a COGNITIVE_TICK if due and nothing else is already pending.

        Returns True if a tick was enqueued.
        """
        if self.has_pending():
            return False
        if self._next_tick_due_at is None or now < self._next_tick_due_at:
            return False
        self.enqueue(EventType.COGNITIVE_TICK)
        return True
