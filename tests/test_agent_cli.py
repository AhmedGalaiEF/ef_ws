from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Optional

import pytest

pydantic = pytest.importorskip("pydantic")
if not pydantic.VERSION.startswith("2"):
    pytest.skip(f"agent package requires pydantic>=2 (found {pydantic.VERSION})", allow_module_level=True)

AGENT_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "g1" / "modules" / "scripts"
if str(AGENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_SCRIPTS_DIR))

from agent.checkpoint import CheckpointStore  # noqa: E402
from agent.cli.router import G1Agent  # noqa: E402
from agent.knowledge.sdk_wrapper_knowledge import SdkWrapperKnowledge  # noqa: E402
from agent.memory.manager import MemoryManager  # noqa: E402
from agent.models import (  # noqa: E402
    EventType,
    IntentAnnouncement,
    IntentType,
    LifecycleState,
    PlannerDecision,
    PlannerInput,
    RuntimeCheckpoint,
)
from agent.planner import MockPlanner  # noqa: E402
from agent.settings.manager import SettingsManager  # noqa: E402
from agent.skills import build_offline_registry  # noqa: E402
from agent.state import MockRobotStateSource  # noqa: E402


def _build_agent(
    tmp_path: Path,
    *,
    decide_fn: Optional[Callable[[PlannerInput], PlannerDecision]] = None,
    auto_confirm: bool = True,
) -> tuple[G1Agent, MockPlanner]:
    planner = MockPlanner(decide_fn=decide_fn)
    agent = G1Agent(
        planner=planner,
        skills=build_offline_registry(),
        state_source=MockRobotStateSource(),
        settings=SettingsManager(tmp_path / "settings.json"),
        memory=MemoryManager(base_dir=tmp_path / "memory"),
        checkpoint_store=CheckpointStore(tmp_path / "checkpoint.json"),
        auto_confirm=auto_confirm,
    )
    return agent, planner


# -- first boot / restart / wake -------------------------------------------


def test_first_boot_is_exactly_one_cognitive_call(tmp_path: Path) -> None:
    agent, planner = _build_agent(tmp_path)
    assert agent.boot_event == EventType.AGENT_FIRST_BOOT

    decision = agent.boot()

    assert decision.intent == IntentType.NO_ACTION
    assert len(planner.calls) == 1
    assert planner.calls[0].event == EventType.AGENT_FIRST_BOOT
    assert agent.lifecycle.state == LifecycleState.AWAKE


def test_restart_classified_when_checkpoint_left_awake(tmp_path: Path) -> None:
    agent, _planner = _build_agent(tmp_path)
    agent.boot()

    agent2, planner2 = _build_agent(tmp_path)
    assert agent2.boot_event == EventType.AGENT_RESTART
    agent2.boot()
    assert planner2.calls[0].event == EventType.AGENT_RESTART
    assert planner2.calls[0].elapsed_since_last_cognition_s is not None


def test_crash_or_reboot_is_never_classified_as_deliberate_sleep(tmp_path: Path) -> None:
    checkpoint_store = CheckpointStore(tmp_path / "checkpoint.json")
    checkpoint_store.save(
        RuntimeCheckpoint(last_cognitive_timestamp=time.time(), lifecycle_state=LifecycleState.AWAKE)
    )
    agent, _planner = _build_agent(tmp_path)
    assert agent.boot_event == EventType.AGENT_RESTART


# -- human input handling ---------------------------------------------------


def test_first_human_message_preserved_verbatim(tmp_path: Path) -> None:
    agent, planner = _build_agent(tmp_path)
    agent.boot()
    text = "Hello,   how ARE you??  "
    agent.handle_chat(text)
    assert planner.calls[-1].user_text == text


def test_audio_asr_disabled_produces_no_conversational_event(tmp_path: Path) -> None:
    agent, planner = _build_agent(tmp_path)
    agent.boot()
    agent.settings.set("audio.asr_enabled", False)
    calls_before = len(planner.calls)

    outcome = agent.handle_audio_msg("hello")
    assert outcome is None
    assert len(planner.calls) == calls_before  # no planner call was made at all


def test_chat_keeps_working_when_asr_is_disabled(tmp_path: Path) -> None:
    agent, planner = _build_agent(tmp_path)
    agent.boot()
    agent.settings.set("audio.asr_enabled", False)
    calls_before = len(planner.calls)

    outcome = agent.handle_chat("I can still communicate through text.")
    assert outcome.decision.intent == IntentType.CONVERSATION
    assert len(planner.calls) == calls_before + 1


def test_no_action_decision_dispatches_no_skills(tmp_path: Path) -> None:
    agent, _planner = _build_agent(tmp_path)
    agent.boot()
    robot_state = agent.state_source.read()
    planner_input = agent._build_planner_input(
        event=EventType.COGNITIVE_TICK,
        timestamp=time.time(),
        user_text=None,
        input_source=None,
        robot_state=robot_state,
    )
    outcome = agent._execute_decision(PlannerDecision(intent=IntentType.NO_ACTION), planner_input)
    assert outcome.skill_outcomes == []


# -- grounded capability answers ---------------------------------------------


def test_capability_denial_grounded_when_motion_disabled_by_settings(tmp_path: Path) -> None:
    def decide_fn(planner_input: PlannerInput) -> PlannerDecision:
        if planner_input.event == EventType.USER_MESSAGE:
            return PlannerDecision(intent=IntentType.QUERY_CAPABILITY, target="arm")
        return PlannerDecision(intent=IntentType.NO_ACTION)

    agent, _planner = _build_agent(tmp_path, decide_fn=decide_fn)
    agent.boot()
    agent.settings.set("motion.allow_arm_motion", False)

    outcome = agent.handle_chat("Can you move your arm?")
    assert outcome.grounded_response is not None
    assert outcome.grounded_response.startswith("No.")
    assert "allow_arm_motion" in outcome.grounded_response


def test_capability_query_grounded_answer_is_not_the_models_own_text(tmp_path: Path) -> None:
    """The model's response_text for query_capability must be overridden by the
    runtime's grounded answer -- the model must not hallucinate hardware state."""

    def decide_fn(planner_input: PlannerInput) -> PlannerDecision:
        if planner_input.event == EventType.USER_MESSAGE:
            return PlannerDecision(
                intent=IntentType.QUERY_CAPABILITY, target="arm", response_text="Sure, no problem!"
            )
        return PlannerDecision(intent=IntentType.NO_ACTION)

    agent, _planner = _build_agent(tmp_path, decide_fn=decide_fn)
    agent.boot()
    outcome = agent.handle_chat("Can you move your arm?")
    assert outcome.grounded_response != "Sure, no problem!"
    assert outcome.grounded_response.startswith(("Yes.", "No."))


# -- intent announcements: all four combinations -----------------------------


@pytest.mark.parametrize(
    "audio_enabled,gesture_enabled",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_intent_announcement_four_combinations(tmp_path: Path, audio_enabled: bool, gesture_enabled: bool) -> None:
    def decide_fn(planner_input: PlannerInput) -> PlannerDecision:
        if planner_input.event == EventType.USER_MESSAGE:
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                intent_announcement=IntentAnnouncement(speech="doing the thing", gesture="gesture"),
            )
        return PlannerDecision(intent=IntentType.NO_ACTION)

    agent, _planner = _build_agent(tmp_path, decide_fn=decide_fn)
    agent.boot()
    agent.settings.set("announcements.audio_enabled", audio_enabled)
    agent.settings.set("announcements.gesture_enabled", gesture_enabled)

    outcome = agent.handle_chat("do the thing")
    assert outcome.announcement is not None
    assert outcome.announcement.spoke == audio_enabled
    assert outcome.announcement.gestured == gesture_enabled


# -- per-skill auto/confirm/disabled -----------------------------------------


def test_disabled_skill_is_denied(tmp_path: Path) -> None:
    def decide_fn(planner_input: PlannerInput) -> PlannerDecision:
        if planner_input.event == EventType.USER_MESSAGE:
            return PlannerDecision(intent=IntentType.EXECUTE_TASK, requested_skills=["release_arms"])
        return PlannerDecision(intent=IntentType.NO_ACTION)

    agent, _planner = _build_agent(tmp_path, decide_fn=decide_fn)
    agent.boot()
    agent.settings.set_skill_mode("release_arms", "disabled")

    outcome = agent.handle_chat("release your arms")
    name, skill_outcome = outcome.skill_outcomes[0]
    assert name == "release_arms"
    assert skill_outcome.status == "denied"
    assert "disabled" in skill_outcome.policy.reason


def test_auto_mode_skill_executes_without_prompting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def decide_fn(planner_input: PlannerInput) -> PlannerDecision:
        if planner_input.event == EventType.USER_MESSAGE:
            return PlannerDecision(intent=IntentType.EXECUTE_TASK, requested_skills=["release_arms"])
        return PlannerDecision(intent=IntentType.NO_ACTION)

    agent, _planner = _build_agent(tmp_path, decide_fn=decide_fn, auto_confirm=False)
    agent.boot()
    agent.settings.set_skill_mode("release_arms", "auto")

    def _fail_if_prompted(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("an 'auto' skill must not prompt for confirmation")

    monkeypatch.setattr("builtins.input", _fail_if_prompted)

    outcome = agent.handle_chat("release your arms")
    name, skill_outcome = outcome.skill_outcomes[0]
    assert skill_outcome.status == "executed"


def test_confirm_mode_denied_when_operator_declines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def decide_fn(planner_input: PlannerInput) -> PlannerDecision:
        if planner_input.event == EventType.USER_MESSAGE:
            return PlannerDecision(intent=IntentType.EXECUTE_TASK, requested_skills=["release_arms"])
        return PlannerDecision(intent=IntentType.NO_ACTION)

    agent, _planner = _build_agent(tmp_path, decide_fn=decide_fn, auto_confirm=False)
    agent.boot()
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")

    outcome = agent.handle_chat("release your arms")
    name, skill_outcome = outcome.skill_outcomes[0]
    assert skill_outcome.status == "denied"
    assert skill_outcome.result is not None and "denied by operator" in skill_outcome.result.message


# -- deliberate sleep ---------------------------------------------------------


def test_request_sleep_persists_checkpoint_without_real_shutdown(tmp_path: Path) -> None:
    def decide_fn(planner_input: PlannerInput) -> PlannerDecision:
        if planner_input.event == EventType.USER_MESSAGE:
            return PlannerDecision(
                intent=IntentType.REQUEST_SLEEP,
                target="operator_request",
                requested_skills=["request_sleep"],
            )
        return PlannerDecision(intent=IntentType.NO_ACTION)

    agent, _planner = _build_agent(tmp_path, decide_fn=decide_fn)
    agent.boot()

    outcome = agent.handle_chat("please sleep now")

    assert agent.lifecycle.state == LifecycleState.SLEEPING
    name, skill_outcome = outcome.skill_outcomes[0]
    assert name == "request_sleep"
    assert skill_outcome.status == "executed"
    assert skill_outcome.result is not None
    assert skill_outcome.result.detail["shutdown_issued"] is False

    reloaded = agent.checkpoint_store.load()
    assert reloaded is not None
    assert reloaded.lifecycle_state == LifecycleState.SLEEPING


def test_wake_after_deliberate_sleep_restores_elapsed_time(tmp_path: Path) -> None:
    def decide_fn(planner_input: PlannerInput) -> PlannerDecision:
        if planner_input.event == EventType.USER_MESSAGE:
            return PlannerDecision(intent=IntentType.REQUEST_SLEEP, requested_skills=["request_sleep"])
        return PlannerDecision(intent=IntentType.NO_ACTION)

    agent, _planner = _build_agent(tmp_path, decide_fn=decide_fn)
    agent.boot()
    agent.handle_chat("go to sleep")
    assert agent.lifecycle.state == LifecycleState.SLEEPING

    # Simulate the Jetson being powered back on: a brand-new process reads
    # the same checkpoint file.
    agent2, planner2 = _build_agent(tmp_path)
    assert agent2.boot_event == EventType.AGENT_WAKE
    agent2.boot()
    assert planner2.calls[0].event == EventType.AGENT_WAKE
    assert planner2.calls[0].runtime.get("sleep_reason") is not None


# -- provenance ----------------------------------------------------------------


def test_robot_state_snapshot_is_tagged_as_live_mock_source() -> None:
    state = MockRobotStateSource().read()
    assert state.source == "mock"


def test_sdk_wrapper_knowledge_is_tagged_implementation_and_fallible() -> None:
    knowledge = SdkWrapperKnowledge()
    refs = knowledge.search("arm", top_k=3)
    if not refs:
        pytest.skip("sdk_wrapper_v3.py was not found in this checkout")
    assert all(ref.source_type == "implementation" for ref in refs)
    assert all(ref.trust == "low" for ref in refs)
    assert all("FALLIBLE" in ref.note for ref in refs)
