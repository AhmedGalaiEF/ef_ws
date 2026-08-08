from __future__ import annotations

import sys
from pathlib import Path

import pytest

pydantic = pytest.importorskip("pydantic")
if not pydantic.VERSION.startswith("2"):
    pytest.skip(f"agent package requires pydantic>=2 (found {pydantic.VERSION})", allow_module_level=True)

AGENT_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "g1" / "modules" / "scripts"
if str(AGENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_SCRIPTS_DIR))

from agent.models import EventType, LifecycleState, RuntimeCheckpoint  # noqa: E402
from agent.scheduler import CognitiveScheduler  # noqa: E402


def test_elapsed_is_none_before_any_cognition() -> None:
    scheduler = CognitiveScheduler()
    assert scheduler.elapsed_since_last_cognition(1000.0) is None


def test_elapsed_is_computed_after_record_cognition() -> None:
    scheduler = CognitiveScheduler()
    scheduler.record_cognition(1000.0)
    assert scheduler.elapsed_since_last_cognition(1004.81) == pytest.approx(4.81)


def test_seed_from_checkpoint_restores_previous_timestamp() -> None:
    scheduler = CognitiveScheduler()
    checkpoint = RuntimeCheckpoint(last_cognitive_timestamp=500.0, lifecycle_state=LifecycleState.AWAKE)
    scheduler.seed_from_checkpoint(checkpoint)
    assert scheduler.elapsed_since_last_cognition(510.0) == pytest.approx(10.0)


def test_seed_from_checkpoint_handles_missing_checkpoint() -> None:
    scheduler = CognitiveScheduler()
    scheduler.seed_from_checkpoint(None)
    assert scheduler.last_cognitive_timestamp is None


def test_record_cognition_honors_next_tick_s_from_decision() -> None:
    scheduler = CognitiveScheduler(default_tick_interval_s=30.0)
    scheduler.record_cognition(1000.0, next_tick_s=5.0)
    assert not scheduler.maybe_enqueue_tick(1004.0)
    assert scheduler.maybe_enqueue_tick(1005.0)


def test_record_cognition_falls_back_to_default_interval() -> None:
    scheduler = CognitiveScheduler(default_tick_interval_s=30.0)
    scheduler.record_cognition(1000.0, next_tick_s=None)
    assert not scheduler.maybe_enqueue_tick(1010.0)
    assert scheduler.maybe_enqueue_tick(1030.0)


def test_no_action_tick_scheduling_does_not_stack_events() -> None:
    """Most periodic ticks should legitimately produce no_action (spec section 5) --
    the scheduler itself must not enqueue a second tick while one is pending."""
    scheduler = CognitiveScheduler(default_tick_interval_s=1.0)
    scheduler.record_cognition(0.0)
    assert scheduler.maybe_enqueue_tick(1.0)
    assert scheduler.has_pending()
    # A second call before the pending tick is drained must not double-enqueue.
    assert not scheduler.maybe_enqueue_tick(2.0)
    event = scheduler.pop_next()
    assert event is not None
    assert event.event_type == EventType.COGNITIVE_TICK
    assert scheduler.pop_next() is None
