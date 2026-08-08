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

from agent.models import (  # noqa: E402
    EventType,
    IntentType,
    LifecycleState,
    PlannerDecision,
    PlannerInput,
    RobotStateSnapshot,
)
from agent.planner import MockPlanner, OpenAIPlanner, PlannerError  # noqa: E402


def _input(event: EventType, user_text: str | None = None) -> PlannerInput:
    return PlannerInput(
        event=event,
        timestamp=time.time(),
        input_source="chat" if user_text is not None else None,
        user_text=user_text,
        robot_state=RobotStateSnapshot(timestamp=time.time(), source="mock"),
        lifecycle_state=LifecycleState.AWAKE,
    )


def test_mock_planner_no_action_on_periodic_tick() -> None:
    planner = MockPlanner()
    decision = planner.decide(_input(EventType.COGNITIVE_TICK))
    assert decision.intent == IntentType.NO_ACTION


def test_mock_planner_no_action_on_empty_user_text() -> None:
    planner = MockPlanner()
    decision = planner.decide(_input(EventType.USER_MESSAGE, user_text="   "))
    assert decision.intent == IntentType.NO_ACTION


def test_mock_planner_default_conversation_reply() -> None:
    planner = MockPlanner()
    decision = planner.decide(_input(EventType.USER_MESSAGE, user_text="Hello, how are you?"))
    assert decision.intent == IntentType.CONVERSATION
    assert "Hello, how are you?" in (decision.response_text or "")


def test_mock_planner_query_capability_for_arm_question() -> None:
    """Matches the spec's worked example: 'Can you move your arm?' -> query_capability(target=arm)."""
    planner = MockPlanner()
    decision = planner.decide(_input(EventType.USER_MESSAGE, user_text="Can you move your arm?"))
    assert decision.intent == IntentType.QUERY_CAPABILITY
    assert decision.target == "arm"
    # The runtime, not the model, fills in the grounded yes/no answer.
    assert decision.response_text is None


def test_mock_planner_move_arm_for_explicit_action_request() -> None:
    decision = MockPlanner().decide(_input(EventType.USER_MESSAGE, user_text="raise your right arm"))
    assert decision.intent == IntentType.MOVE_ARM
    assert "reach_forward" in decision.requested_skills
    assert decision.intent_announcement is not None


def test_mock_planner_request_sleep_keyword() -> None:
    decision = MockPlanner().decide(_input(EventType.USER_MESSAGE, user_text="please go to sleep now"))
    assert decision.intent == IntentType.REQUEST_SLEEP


def test_mock_planner_request_charge_keyword() -> None:
    decision = MockPlanner().decide(_input(EventType.USER_MESSAGE, user_text="your battery seems low, go charge"))
    assert decision.intent == IntentType.REQUEST_CHARGE


def test_mock_planner_records_every_call() -> None:
    planner = MockPlanner()
    planner.decide(_input(EventType.COGNITIVE_TICK))
    planner.decide(_input(EventType.USER_MESSAGE, user_text="hi"))
    assert len(planner.calls) == 2


def test_mock_planner_decide_fn_override_takes_priority() -> None:
    scripted = PlannerDecision(intent=IntentType.MAINTENANCE, response_text="scripted")
    planner = MockPlanner(decide_fn=lambda _pi: scripted)
    decision = planner.decide(_input(EventType.USER_MESSAGE, user_text="anything"))
    assert decision is scripted


def test_openai_planner_fails_fast_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(PlannerError):
        OpenAIPlanner()
