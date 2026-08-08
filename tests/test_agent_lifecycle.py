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

from agent.checkpoint import CheckpointStore  # noqa: E402
from agent.lifecycle import (  # noqa: E402
    ALLOWED_TRANSITIONS,
    InvalidLifecycleTransition,
    LifecycleController,
    classify_startup,
)
from agent.models import EventType, IntentType, LifecycleState, RuntimeCheckpoint  # noqa: E402


def test_first_boot_when_no_checkpoint_exists() -> None:
    event, entry_state = classify_startup(None)
    assert event == EventType.AGENT_FIRST_BOOT
    assert entry_state == LifecycleState.FIRST_BOOT


def test_restart_when_checkpoint_is_awake() -> None:
    checkpoint = RuntimeCheckpoint(lifecycle_state=LifecycleState.AWAKE)
    event, entry_state = classify_startup(checkpoint)
    assert event == EventType.AGENT_RESTART
    assert entry_state == LifecycleState.RESTART_RECOVERY


def test_wake_only_when_checkpoint_is_sleeping() -> None:
    checkpoint = RuntimeCheckpoint(lifecycle_state=LifecycleState.SLEEPING, sleep_reason="battery_conservation")
    event, entry_state = classify_startup(checkpoint)
    assert event == EventType.AGENT_WAKE
    assert entry_state == LifecycleState.WAKING


@pytest.mark.parametrize(
    "stuck_state",
    [LifecycleState.AWAKE, LifecycleState.MAINTENANCE, LifecycleState.PRE_SLEEP, LifecycleState.FIRST_BOOT],
)
def test_crash_or_reboot_never_classified_as_deliberate_sleep(stuck_state: LifecycleState) -> None:
    """A bare reboot/crash must never be mistaken for a deliberate sleep (spec section 12)."""
    checkpoint = RuntimeCheckpoint(lifecycle_state=stuck_state)
    event, _entry_state = classify_startup(checkpoint)
    assert event == EventType.AGENT_RESTART


def test_lifecycle_controller_allows_documented_transitions() -> None:
    controller = LifecycleController(state=LifecycleState.FIRST_BOOT)
    controller.transition(LifecycleState.AWAKE)
    assert controller.state == LifecycleState.AWAKE
    controller.transition(LifecycleState.MAINTENANCE)
    controller.transition(LifecycleState.AWAKE)
    controller.transition(LifecycleState.PRE_SLEEP)
    controller.transition(LifecycleState.SLEEPING)
    assert controller.history == [
        LifecycleState.FIRST_BOOT,
        LifecycleState.AWAKE,
        LifecycleState.MAINTENANCE,
        LifecycleState.AWAKE,
        LifecycleState.PRE_SLEEP,
    ]


def test_lifecycle_controller_rejects_undocumented_transitions() -> None:
    controller = LifecycleController(state=LifecycleState.AWAKE)
    with pytest.raises(InvalidLifecycleTransition):
        controller.transition(LifecycleState.SLEEPING)  # must go through PRE_SLEEP first


def test_sleeping_state_has_no_outgoing_live_transitions() -> None:
    """Leaving SLEEPING only happens via a new process (classify_startup)."""
    assert ALLOWED_TRANSITIONS[LifecycleState.SLEEPING] == set()


def test_checkpoint_roundtrip_atomic_write_and_load(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoint.json")
    assert store.load() is None  # nothing written yet

    checkpoint = RuntimeCheckpoint(
        last_cognitive_timestamp=time.time(),
        lifecycle_state=LifecycleState.AWAKE,
        last_event_type=EventType.USER_MESSAGE,
        last_decision=IntentType.CONVERSATION,
    )
    store.save(checkpoint)

    reloaded = store.load()
    assert reloaded is not None
    assert reloaded.lifecycle_state == LifecycleState.AWAKE
    assert reloaded.last_decision == IntentType.CONVERSATION
    # No stray temp file left behind after the atomic rename.
    assert list(tmp_path.iterdir()) == [tmp_path / "checkpoint.json"]


def test_corrupt_checkpoint_is_treated_as_missing(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = CheckpointStore(path)
    assert store.load() is None
