"""``G1Agent`` orchestrator + the deterministic ``/settings`` CLI namespace
(spec sections 5, 14, 16, 17, 31).

``/settings`` (and ``/status``, ``/memory``, ``/tools``, ``/help``) never
touch the planner -- they are plain deterministic code, per spec section
14. ``/chat`` and ``/audio_msg`` are the two paths that construct a
``PlannerInput`` and call ``planner.decide()``.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from ..announcements import announce
from ..active_learning import ActiveLearningManager, LearningQuestionRecord
from ..activity import ActivityManager
from ..asr import AsrRuntime
from ..attention import AttentionManager
from ..capabilities import CapabilityResolver, PolicyDecision
from ..checkpoint import CheckpointStore
from ..expressive_motion import ExpressiveMotionController
from ..learning import LearningManager
from ..lifecycle import LifecycleController, classify_startup
from ..llctl import LlctlAdapter
from ..memory.manager import MemoryManager, MemoryProposalError
from ..monitor import MonitorEventBus
from ..navigation import NavigationAdapter
from ..models import (
    CapabilityStatus,
    EventType,
    IntentAnnouncement,
    IntentType,
    LifecycleState,
    PlannerDecision,
    PlannerInput,
    RobotStateSnapshot,
    RuntimeCheckpoint,
)
from ..outcomes import OutcomeEvaluator, SkillOutcome
from ..planner import Planner
from ..reset import ResetManager, ResetResult
from ..scheduler import CognitiveScheduler
from ..semantic_state import SemanticState, SemanticStateTracker
from ..self_model import SelfModelStore
from ..settings.manager import InvalidSettingError, SettingsManager
from ..settings.models import SkillMode
from ..skills import (
    SkillInvocationOutcome,
    SkillRegistry,
    SkillResult,
    invoke_with_capability_check,
    resolve_and_maybe_invoke,
)
from ..state import RobotStateSource, build_robot_state
from ..tacit import TacitKnowledgeService
from ..tools import ToolContext, build_default_tool_registry
from ..visual_observation import VisualObservationTracker

try:
    from ..knowledge.sdk_wrapper_knowledge import SdkWrapperKnowledge
except Exception:  # pragma: no cover - always importable (stdlib-only), kept defensive
    SdkWrapperKnowledge = None  # type: ignore[assignment]


@dataclass
class TurnOutcome:
    decision: PlannerDecision
    grounded_response: Optional[str]
    skill_outcomes: list[tuple[str, SkillInvocationOutcome]]
    announcement: Any = None
    skill_evaluations: list[SkillOutcome] | None = None
    learning_question: Optional[LearningQuestionRecord] = None

    def __post_init__(self) -> None:
        if self.skill_evaluations is None:
            self.skill_evaluations = []


class G1Agent:
    """Ties scheduler + lifecycle + checkpoint + settings + memory + skills
    + planner into the one coherent agent-level loop the spec asks for."""

    def __init__(
        self,
        *,
        planner: Planner,
        skills: SkillRegistry,
        state_source: RobotStateSource,
        settings: Optional[SettingsManager] = None,
        memory: Optional[MemoryManager] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        sdk_knowledge: "Optional[SdkWrapperKnowledge]" = None,
        document_rag: Optional[Any] = None,
        resolver: Optional[CapabilityResolver] = None,
        auto_confirm: bool = False,
        robot: Optional[Any] = None,
    ) -> None:
        self.planner = planner
        self.skills = skills
        self.state_source = state_source
        self.settings = settings or SettingsManager()
        self.memory = memory or MemoryManager()
        self.checkpoint_store = checkpoint_store or CheckpointStore()
        self.sdk_knowledge = sdk_knowledge
        self.document_rag = document_rag
        self.resolver = resolver or CapabilityResolver()
        self.robot = robot
        self.scheduler = CognitiveScheduler()
        self.monitor_bus = MonitorEventBus(max_events=self.settings.effective().monitor.event_buffer_size)
        self.asr_runtime = AsrRuntime(monitor=self.monitor_bus)
        self.expressive_motion = ExpressiveMotionController(robot=robot, monitor=self.monitor_bus)
        self.activity = ActivityManager(robot=robot, monitor=self.monitor_bus, expressive_motion=self.expressive_motion)
        self.active_learning = ActiveLearningManager(monitor=self.monitor_bus)
        self.navigation = NavigationAdapter(robot=robot)
        self.llctl = LlctlAdapter()
        self.visual_observations = VisualObservationTracker()
        self.semantic_tracker = SemanticStateTracker()
        self.attention = AttentionManager()
        self.outcome_evaluator = OutcomeEvaluator()
        self.learning = LearningManager(self.memory, monitor=self.monitor_bus)
        self_model_settings = self.settings.effective().self_model
        self.self_model = SelfModelStore.from_memory_base(
            base_dir=self.learning.base_dir,
            robot_id=self_model_settings.robot_id or None,
            monitor=self.monitor_bus,
        )
        self.reset_manager = ResetManager(agent=self, monitor=self.monitor_bus)
        self.tacit = TacitKnowledgeService(agent=self)
        self._mutation_lock = threading.RLock()
        self.tool_registry = build_default_tool_registry(self)
        self._last_tool_context: Optional[ToolContext] = None
        # Non-interactive confirmation policy for tests/scripted runs: when
        # True, a "needs_confirmation" skill is treated as approved without
        # prompting on stdin. The CLI REPL leaves this False and prompts.
        self.auto_confirm = auto_confirm

        previous_checkpoint = self.checkpoint_store.load()
        boot_event, entry_state = classify_startup(previous_checkpoint)
        self.lifecycle = LifecycleController(state=entry_state)
        self.scheduler.seed_from_checkpoint(previous_checkpoint)
        self._previous_checkpoint = previous_checkpoint
        self._boot_event = boot_event
        self._booted = False
        self._vision_answerer: Any = None
        self._boot_time = time.time()
        self._cognition_count = 0
        self._last_semantic_state: SemanticState = self.semantic_tracker.current
        self._current_objectives: list[dict[str, Any]] = []
        self._behavior_owner: str = "none"
        self._last_learning_question_shown: Optional[LearningQuestionRecord] = None

    @property
    def boot_event(self) -> EventType:
        return self._boot_event

    def _update_semantic_state(
        self,
        robot_state: RobotStateSnapshot,
        *,
        interaction: str = "alone",
        task: str = "idle",
    ) -> tuple[SemanticState, list[str]]:
        semantic_state, changes = self.semantic_tracker.update(
            robot_state,
            lifecycle=self.lifecycle.state.value,
            interaction=interaction,
            task=task,
        )
        self._last_semantic_state = semantic_state
        return semantic_state, changes

    # -- boot -------------------------------------------------------------

    def boot(self) -> PlannerDecision:
        """Run the first cognitive turn: agent_first_boot / _restart / _wake."""
        now = time.time()
        robot_state = build_robot_state(self.state_source)
        semantic_state, semantic_changes = self._update_semantic_state(robot_state, interaction="alone", task="idle")
        self.monitor_bus.emit(
            "event",
            "event_received",
            f"boot {self._boot_event.value}",
            metadata={"semantic_changes": semantic_changes},
        )
        runtime: dict[str, Any] = {"platform": "Unitree G1", "reason": self._boot_event.value}
        prev = self._previous_checkpoint
        if self._boot_event == EventType.AGENT_RESTART and prev is not None:
            runtime.update(
                previous_lifecycle_state=prev.lifecycle_state.value,
                previous_task=prev.active_task,
                previous_scenario=prev.active_scenario,
                previous_robot_state_summary=prev.last_robot_state_summary,
            )
        elif self._boot_event == EventType.AGENT_WAKE and prev is not None:
            runtime.update(
                sleep_reason=prev.sleep_reason,
                sleep_timestamp=prev.sleep_timestamp,
                pre_sleep_state=prev.last_robot_state_summary,
                memory_restored=True,
            )

        settings = self.settings.effective()
        with self.activity.activity("thinking", settings=settings, reason=self._boot_event.value):
            planner_input = self._build_planner_input(
                event=self._boot_event,
                timestamp=now,
                user_text=None,
                input_source="system",
                robot_state=robot_state,
                runtime=runtime,
                semantic_state=semantic_state,
            )
            decision = self.planner.decide(planner_input)
        self.monitor_bus.emit("planner", "planner_decision", f"intent={decision.intent.value} target={decision.target}")
        # Reach AWAKE before persisting the checkpoint, so a boot turn's
        # checkpoint always records "awake", not a transitional state.
        self.lifecycle.transition(LifecycleState.AWAKE)
        semantic_state.lifecycle = self.lifecycle.state.value
        self._last_semantic_state = semantic_state
        self._after_turn(planner_input, decision)
        self._record_boot_memory(robot_state)
        self.monitor_bus.emit("lifecycle", "lifecycle_transition", f"entered {self.lifecycle.state.value}")
        self._booted = True
        return decision

    # -- /chat and /audio_msg ----------------------------------------------

    def handle_chat(self, text: str) -> TurnOutcome:
        """``/chat`` -- always active, independent of every audio setting."""
        return self._handle_user_text(text, input_source="chat")

    def handle_cli_text(self, text: str) -> TurnOutcome:
        """Bare REPL text; may answer a pending learning question."""
        settings = self.settings.effective()
        if self.active_learning.should_consume_text_as_answer(text, settings=settings):
            return self._handle_learning_answer(text)
        return self.handle_chat(text)

    def handle_audio_msg(self, text: str) -> Optional[TurnOutcome]:
        """``/audio_msg`` -- gated by ``audio.asr_enabled`` (spec section 15).

        Returns ``None`` (no conversational model event constructed at
        all) when ASR is disabled; ``/chat`` is unaffected either way.
        ``audio.audio_to_state_enabled`` is a separate toggle for a live
        ambient-audio-derived state pipeline, which this phase does not
        implement (there is no continuous audio stream here, only this
        explicit transcript command) -- it intentionally does not gate
        ``/audio_msg``.
        """
        effective = self.settings.effective()
        if not (effective.audio.asr_enabled and effective.asr.enabled):
            return None
        return self._handle_user_text(text, input_source="audio")

    def handle_cognitive_tick(self) -> TurnOutcome:
        """Run one periodic mostly-idle cognition turn."""
        self._last_learning_question_shown = None
        now = time.time()
        settings = self.settings.effective()
        robot_state = build_robot_state(self.state_source)
        semantic_state, semantic_changes = self._update_semantic_state(robot_state, interaction="alone", task="idle")
        attention = self.attention.decide(
            event_type=EventType.COGNITIVE_TICK.value,
            semantic_state=semantic_state,
            semantic_changes=semantic_changes,
            settings=settings,
            self_model=self.self_model,
        )
        self.monitor_bus.emit(
            "attention",
            "attention_decision",
            f"priority=P{attention.priority} action={attention.action} reason={attention.reason_code}",
            metadata={"event_summary": attention.event_summary, "semantic_changes": semantic_changes},
        )
        if attention.action in {"ignore", "record", "aggregate"}:
            self.scheduler.record_cognition(now, settings.cognition.periodic_interval_s)
            return TurnOutcome(
                decision=PlannerDecision(intent=IntentType.NO_ACTION),
                grounded_response=None,
                skill_outcomes=[],
            )
        maintenance_result = None
        if attention.reason_code == "maintenance_due" and settings.learning.enabled:
            maintenance_result = self.learning.consolidate(settings=settings)
        self.monitor_bus.emit("cognition", "cognition_started", f"tick reason={attention.reason_code}")
        with self.activity.activity("thinking", settings=settings, reason=f"tick:{attention.reason_code}"):
            planner_input = self._build_planner_input(
                event=EventType.COGNITIVE_TICK,
                timestamp=now,
                user_text=None,
                input_source="system",
                robot_state=robot_state,
                runtime={"reason": "periodic_tick", "attention": attention.model_dump(), "maintenance": maintenance_result},
                semantic_state=semantic_state,
            )
            decision = self.planner.decide(planner_input)
        self.monitor_bus.emit("planner", "planner_decision", f"intent={decision.intent.value} target={decision.target}")
        self._after_turn(planner_input, decision, announce_maintenance=False)
        return self._execute_decision(decision, planner_input)

    def _handle_user_text(self, text: str, *, input_source: str) -> TurnOutcome:
        now = time.time()
        self._last_learning_question_shown = None
        event = EventType.ASR_MESSAGE if input_source == "audio" else EventType.USER_MESSAGE
        robot_state = build_robot_state(self.state_source)
        semantic_state, semantic_changes = self._update_semantic_state(robot_state, interaction="user_engaged", task="idle")
        attention = self.attention.decide(
            event_type=event.value,
            semantic_state=semantic_state,
            semantic_changes=semantic_changes,
            settings=self.settings.effective(),
            event_summary=text[:120],
            self_model=self.self_model,
        )
        self.monitor_bus.emit("event", "event_received", f"{input_source}: {text[:120]}")
        self.monitor_bus.emit(
            "attention",
            "attention_decision",
            f"priority=P{attention.priority} action={attention.action} reason={attention.reason_code}",
        )
        if self._is_thanks_intent(text):
            self.expressive_motion.run_background(
                "thanking",
                settings=self.settings.effective(),
                reason="dialogue_thanking",
            )
        age_response = self._maybe_handle_age_query(text)
        if age_response is not None:
            decision = PlannerDecision(intent=IntentType.CONVERSATION, response_text=age_response)
            planner_input = self._build_planner_input(
                event=event, timestamp=now, user_text=text, input_source=input_source, robot_state=robot_state
            )
            self._after_turn(planner_input, decision)
            return self._execute_decision(decision, planner_input)
        rag_response = self._maybe_handle_rag_help_query(text)
        if rag_response is not None:
            decision = PlannerDecision(intent=IntentType.CONVERSATION, response_text=rag_response)
            planner_input = self._build_planner_input(
                event=event, timestamp=now, user_text=text, input_source=input_source, robot_state=robot_state
            )
            self._after_turn(planner_input, decision)
            return self._execute_decision(decision, planner_input)
        thought_response = self._maybe_handle_thought_query(text, robot_state)
        if thought_response is not None:
            decision = PlannerDecision(intent=IntentType.CONVERSATION, response_text=thought_response)
            planner_input = self._build_planner_input(
                event=event, timestamp=now, user_text=text, input_source=input_source, robot_state=robot_state
            )
            self._after_turn(planner_input, decision)
            return self._execute_decision(decision, planner_input)
        learned_response = self._maybe_handle_learned_query(text)
        if learned_response is not None:
            decision = PlannerDecision(intent=IntentType.CONVERSATION, response_text=learned_response)
            planner_input = self._build_planner_input(
                event=event, timestamp=now, user_text=text, input_source=input_source, robot_state=robot_state
            )
            self._after_turn(planner_input, decision)
            return self._execute_decision(decision, planner_input)
        memory_response = self._maybe_handle_memory_query(text, robot_state)
        if memory_response is not None:
            decision = PlannerDecision(intent=IntentType.CONVERSATION, response_text=memory_response)
            planner_input = self._build_planner_input(
                event=event, timestamp=now, user_text=text, input_source=input_source, robot_state=robot_state
            )
            self._after_turn(planner_input, decision)
            return self._execute_decision(decision, planner_input)
        vision_response = self._maybe_handle_vision_query(text)
        if vision_response is not None:
            decision = PlannerDecision(
                intent=IntentType.QUERY_STATE,
                target="vision",
                response_text=vision_response,
            )
            planner_input = self._build_planner_input(
                event=event, timestamp=now, user_text=text, input_source=input_source, robot_state=robot_state
            )
            self._after_turn(planner_input, decision)
            return self._execute_decision(decision, planner_input)
        with self.activity.activity("thinking", settings=self.settings.effective(), reason=f"user:{input_source}"):
            planner_input = self._build_planner_input(
                event=event, timestamp=now, user_text=text, input_source=input_source, robot_state=robot_state
            )
            decision = self.planner.decide(planner_input)
        decision = self._normalize_decision(decision, planner_input)
        decision = self._apply_command_fallbacks(decision, planner_input)
        self._after_turn(planner_input, decision)
        return self._execute_decision(decision, planner_input)

    def _handle_learning_answer(self, text: str) -> TurnOutcome:
        now = time.time()
        settings = self.settings.effective()
        record, proposal = self.active_learning.answer(text, settings=settings)
        robot_state = build_robot_state(self.state_source)
        runtime = {
            "reason": "learning_answer",
            "related_question_id": None if record is None else record.id,
            "learning_answer_provenance": None if proposal is None else proposal.content,
        }
        if proposal is not None:
            try:
                self.memory.apply_proposal(proposal)
                self.monitor_bus.emit("memory", "memory_proposed", f"learning answer stored as {proposal.kind}")
            except MemoryProposalError as exc:
                self.monitor_bus.emit("memory", "memory_rejected", str(exc))
        with self.activity.activity("thinking", settings=settings, reason="learning_answer"):
            planner_input = self._build_planner_input(
                event=EventType.LEARNING_ANSWER,
                timestamp=now,
                user_text=text,
                input_source="learning_answer",
                robot_state=robot_state,
                runtime=runtime,
            )
            decision = self.planner.decide(planner_input)
        self._after_turn(planner_input, decision)
        outcome = self._execute_decision(decision, planner_input)
        if outcome.grounded_response is None and record is not None:
            outcome.grounded_response = "Thanks. I recorded that as operator-provided learning evidence."
        return outcome

    def _maybe_handle_memory_query(self, text: str, robot_state: RobotStateSnapshot) -> Optional[str]:
        lowered = text.strip().lower()
        if not any(term in lowered for term in ("remember", "memory", "erinner", "gedächtnis", "gedaechtnis")):
            return None
        settings = self.settings.effective()
        max_entries = max(1, min(10, int(getattr(settings.response, "memory_max_entries", 3))))
        bio = self.memory.autobiography.summary(max_entries=max_entries) or "(empty)"
        total_entries = len(self.memory.autobiography.all())
        reply_language = self._effective_reply_language(self.settings.effective())
        if reply_language == "de":
            return (
                f"Mein Gedächtnissystem ist verfügbar. Ich zeige die letzten {max_entries} von {total_entries} autobiografischen Einträgen:\n"
                f"{bio}\n"
                f"Aktueller Zustand: Haltung={robot_state.posture}, Stabilität={robot_state.stability}."
            )
        return (
            f"My memory system is available. Showing the last {max_entries} of {total_entries} autobiographical entries:\n"
            f"{bio}\n"
            f"Current state summary: posture={robot_state.posture}, stability={robot_state.stability}."
        )

    def _maybe_handle_learned_query(self, text: str) -> Optional[str]:
        lowered = text.strip().lower()
        if not any(
            phrase in lowered
            for phrase in (
                "what have you learned",
                "what did you learn",
                "what have u learned",
                "was hast du gelernt",
                "was hast du gelernt?",
                "was hast du bisher gelernt",
            )
        ):
            return None
        reply_language = self._effective_reply_language(self.settings.effective())
        learned = self.learning.learned.all()
        procedures = self.memory.procedural.all()
        episodes = self.memory.episodic.all()
        self_records = self.self_model.model.skills.records
        learned_skills = [
            name for name, record in sorted(self_records.items())
            if int(record.attempts) > 0
        ]
        if reply_language == "de":
            lines = [
                "Ich kann nur belegtes Lernen berichten:",
                f"- Episoden: {len(episodes)}",
                f"- Gelernte semantische Claims: {len(learned)}",
                f"- Prozedurale/tazite Regeln: {len(procedures)}",
                f"- Skill-Statistiken im Selbstmodell: {len(learned_skills)}",
            ]
            if learned_skills:
                lines.append("- Beobachtete Skills: " + ", ".join(learned_skills[:8]))
            if not learned and not procedures:
                lines.append("Es gibt noch keine validierten semantischen oder prozeduralen Lernergebnisse.")
            return "\n".join(lines)
        lines = [
            "I can report only grounded learned state:",
            f"- Episodes: {len(episodes)}",
            f"- Learned semantic claims: {len(learned)}",
            f"- Procedural/tacit rules: {len(procedures)}",
            f"- Skill statistics in self-model: {len(learned_skills)}",
        ]
        if learned_skills:
            lines.append("- Observed skills: " + ", ".join(learned_skills[:8]))
        if not learned and not procedures:
            lines.append("There are no validated semantic or procedural learned items yet.")
        return "\n".join(lines)

    @staticmethod
    def _is_thanks_intent(text: str) -> bool:
        lowered = text.strip().lower()
        return any(token in lowered for token in ("thank you", "thanks", "danke", "vielen dank"))

    def _maybe_handle_age_query(self, text: str) -> Optional[str]:
        lowered = text.strip().lower()
        if not ("how old are you" in lowered or "what is your age" in lowered or "wie alt bist du" in lowered):
            return None
        uptime_s = max(0.0, time.time() - self._boot_time)
        reply_language = self._effective_reply_language(self.settings.effective())
        if reply_language == "de":
            return f"Ich bin seit {uptime_s:.0f} Sekunden in diesem Prozess wach und hatte {self._cognition_count} Kognitionsdurchläufe."
        return f"I have been awake in this process for {uptime_s:.0f} seconds and have completed {self._cognition_count} cognition iterations."

    def _maybe_handle_thought_query(self, text: str, robot_state: RobotStateSnapshot) -> Optional[str]:
        lowered = text.strip().lower()
        triggers = (
            "what are you thinking",
            "what do you think about",
            "what are your cognition",
            "cognition iteration",
            "woran denkst du",
            "was denkst du",
            "kognitions",
        )
        if not any(trigger in lowered for trigger in triggers):
            return None
        settings = self.settings.effective()
        uptime_s = max(0.0, time.time() - self._boot_time)
        faults = ", ".join(robot_state.active_faults[:5]) if robot_state.active_faults else "none"
        if self._effective_reply_language(settings) == "de":
            return (
                f"Ich laufe meistens im Leerlauf und prüfe alle Zustände. "
                f"Seit dem Start bin ich {uptime_s:.0f} Sekunden wach, hatte {self._cognition_count} "
                f"Kognitionsdurchläufe und beobachte aktuell: Haltung={robot_state.posture}, "
                f"Stabilität={robot_state.stability}, Fehler={faults}."
            )
        return (
            f"I am mostly idle and monitoring state. Since this process started I have been awake "
            f"for {uptime_s:.0f} seconds, completed {self._cognition_count} cognition iterations, "
            f"and I am currently tracking posture={robot_state.posture}, stability={robot_state.stability}, "
            f"faults={faults}."
        )

    def _maybe_handle_rag_help_query(self, text: str) -> Optional[str]:
        lowered = text.strip().lower()
        if not ("rag" in lowered or "knowledge" in lowered and "configured" in lowered or "add files" in lowered):
            return None
        return (
            "RAG loads the automatic SDK notes from `agent/knowledge/default_sdk_knowledge.md`, plus any "
            "extra startup files passed with repeated `--knowledge-file <path>` arguments. Those files are "
            "loaded by `agent.knowledge.document_rag.DocumentRAG`, which wraps the existing "
            "`ollama_ai/nav_bot.py` keyword retriever. Source-code SDK wrapper snippets are also searched via "
            "`agent/knowledge/sdk_wrapper_knowledge.py`. Conversation memory is separate under `~/.g1_agent/` "
            "unless overridden with `G1_AGENT_MEMORY_DIR` / `G1_AGENT_AUTOBIOGRAPHY`. "
            "For robot-learned facts, the planner can emit validated memory proposals; automatic self-extension "
            "of documentary RAG files is not enabled yet because it needs operator review to avoid polluting the knowledge base."
        )

    def handle_vision_question(self, question: str) -> TurnOutcome:
        now = time.time()
        robot_state = build_robot_state(self.state_source)
        response = self._answer_vision_question(question)
        decision = PlannerDecision(intent=IntentType.QUERY_STATE, target="vision", response_text=response)
        planner_input = self._build_planner_input(
            event=EventType.USER_MESSAGE,
            timestamp=now,
            user_text=question,
            input_source="chat",
            robot_state=robot_state,
        )
        self._after_turn(planner_input, decision)
        return self._execute_decision(decision, planner_input)

    def _maybe_handle_vision_query(self, text: str) -> Optional[str]:
        lowered = text.strip().lower()
        triggers = (
            "what do you see",
            "what can you see",
            "what's in front",
            "what is in front",
            "describe what you see",
            "look in front",
            "camera",
            "was siehst du",
            "was kannst du sehen",
            "was ist vor dir",
            "was ist vor mir",
            "beschreibe was du siehst",
            "kamera",
        )
        if not any(trigger in lowered for trigger in triggers):
            return None
        return self._answer_vision_question(text)

    def _answer_vision_question(self, question: str) -> str:
        settings = self.settings.effective()
        reply_language = self._effective_reply_language(settings)
        german = str(reply_language) == "de"
        if not settings.vision.rgbd_enabled:
            if german:
                return (
                    "RGB-D-Sehen ist deaktiviert. Aktiviere vision.rgbd_enabled in /settings-ui "
                    "oder mit /settings set vision.rgbd_enabled true."
                )
            return (
                "RGB-D vision input is disabled. Enable vision.rgbd_enabled in /settings-ui "
                "or run /settings set vision.rgbd_enabled true."
            )
        try:
            if self._vision_answerer is None:
                from ..vision import OpenAIVisionAnswerer

                self._vision_answerer = OpenAIVisionAnswerer()
            answer = self._vision_answerer.answer(
                settings=settings.vision,
                question=question,
                reply_language=str(reply_language),
            )
            obs = self.visual_observations.observe_from_answer(
                answer=answer,
                model=str(settings.vision.openai_model),
                confidence=0.6,
            )
            self.monitor_bus.emit(
                "vision",
                "visual_observation",
                (obs.scene_summary or "")[:160],
                metadata=obs.model_dump(),
            )
            if self.visual_observations.should_wake_cognition(obs):
                self.monitor_bus.emit(
                    "attention",
                    "attention_decision",
                    "priority=P4 action=record reason=visual_semantic_change",
                    metadata={"notable_changes": list(obs.notable_changes)},
                )
            return answer
        except Exception as exc:
            if german:
                return f"Ich kann das RGB-D-Bild gerade nicht auswerten: {exc}"
            return f"I cannot analyze the RGB-D image right now: {exc}"

    # -- planner input construction ----------------------------------------

    def _build_planner_input(
        self,
        *,
        event: EventType,
        timestamp: float,
        user_text: Optional[str],
        input_source: Optional[str],
        robot_state: RobotStateSnapshot,
        runtime: Optional[dict[str, Any]] = None,
        semantic_state: Optional[SemanticState] = None,
    ) -> PlannerInput:
        settings = self.settings.effective()
        self.monitor_bus.resize(settings.monitor.event_buffer_size)
        semantic_state = semantic_state or self._last_semantic_state
        runtime_payload = dict(runtime or {})
        runtime_payload.setdefault("reply_language", self._effective_reply_language(settings))
        runtime_payload.setdefault("cognition_count", self._cognition_count)
        runtime_payload.setdefault("agent_uptime_s", max(0.0, time.time() - self._boot_time))
        runtime_payload.setdefault("semantic_state", semantic_state.model_dump())
        runtime_payload.setdefault("self", self.self_model.summary())
        query = user_text or ""

        memory_refs = {"episodic": [], "semantic": [], "procedural": []}

        sdk_refs = []
        doc_refs = []
        tool_context = self._make_tool_context(
            settings=settings,
            robot_state=robot_state,
            event=event.value,
            profile=self._tool_profile_for(event=event, user_text=user_text, runtime=runtime_payload),
        )
        self._last_tool_context = tool_context
        if hasattr(self.planner, "set_tool_context"):
            try:
                self.planner.set_tool_context(registry=self.tool_registry, context=tool_context)
            except Exception:
                pass

        arm_policy = self.resolver.resolve_arm_motion(settings=settings, robot_state=robot_state)
        capability_summary = {
            "arm_motion": CapabilityStatus(available=arm_policy.allowed, reason=arm_policy.reason),
        }
        planner_robot_state = self._compact_planner_robot_state(robot_state)

        return PlannerInput(
            event=event,
            timestamp=timestamp,
            previous_cognitive_timestamp=self.scheduler.last_cognitive_timestamp,
            elapsed_since_last_cognition_s=self.scheduler.elapsed_since_last_cognition(timestamp),
            input_source=input_source,
            user_text=user_text,
            robot_state=planner_robot_state,
            lifecycle_state=self.lifecycle.state,
            available_skills=self.skills.names(),
            capability_summary=capability_summary,
            settings=settings.as_flat_dict(),
            documentary_rag=doc_refs,
            sdk_wrapper_knowledge=sdk_refs,
            episodic_memory=memory_refs["episodic"],
            semantic_memory=memory_refs["semantic"],
            procedural_memory=memory_refs["procedural"],
            autobiography_summary=self.memory.autobiography.summary(
                max_entries=max(1, min(10, int(getattr(settings.response, "memory_max_entries", 3))))
            ),
            available_tools=[
                row["name"]
                for row in self.tool_registry.available_for(tool_context)
                if row.get("availability") == "available"
            ],
            runtime=runtime_payload,
        )

    def _make_tool_context(
        self,
        *,
        settings: Any,
        robot_state: RobotStateSnapshot,
        event: str,
        profile: str,
    ) -> ToolContext:
        return ToolContext(agent=self, settings=settings, robot_state=robot_state, profile=profile, event=event)

    @staticmethod
    def _compact_planner_robot_state(robot_state: RobotStateSnapshot) -> RobotStateSnapshot:
        data = robot_state.model_dump()
        lowstate = data.get("lowstate") or {}
        if lowstate:
            data["lowstate"] = {
                "timestamp": lowstate.get("timestamp"),
                "joint_count": lowstate.get("joint_count"),
                "imu": lowstate.get("imu"),
                "source": lowstate.get("source"),
            }
        return RobotStateSnapshot(**data)

    @staticmethod
    def _tool_profile_for(*, event: EventType, user_text: Optional[str], runtime: Optional[dict[str, Any]]) -> str:
        text = (user_text or "").lower()
        if event in {EventType.TASK_FAILED, EventType.ANOMALY}:
            return "diagnostic"
        if any(token in text for token in ("why", "debug", "fault", "error", "didn't", "diagnostic")):
            return "diagnostic"
        if any(token in text for token in ("navigate", "slam", "map", "relocate")):
            return "navigation"
        if any(token in text for token in ("arm", "grab", "wave", "gesture", "hand")):
            return "manipulation"
        return "social"

    # -- post-turn bookkeeping ----------------------------------------------

    def _after_turn(
        self,
        planner_input: PlannerInput,
        decision: PlannerDecision,
        *,
        announce_maintenance: bool = True,
    ) -> None:
        self._cognition_count += 1
        self.monitor_bus.emit(
            "planner",
            "planner_decision",
            f"intent={decision.intent.value} target={decision.target}",
            metadata={"event": planner_input.event.value},
        )
        self.scheduler.record_cognition(planner_input.timestamp, decision.next_tick_s)
        checkpoint = RuntimeCheckpoint(
            last_cognitive_timestamp=planner_input.timestamp,
            lifecycle_state=self.lifecycle.state,
            last_event_type=planner_input.event,
            last_decision=decision.intent,
            active_skill=(decision.requested_skills[0] if decision.requested_skills else None),
            last_robot_state_summary=planner_input.robot_state.model_dump(),
            self_model_version=self.self_model.model.version,
        )
        self.checkpoint_store.save(checkpoint)
        self._previous_checkpoint = checkpoint

        if decision.memory_proposal is not None:
            try:
                self.memory.apply_proposal(decision.memory_proposal)
                self.monitor_bus.emit("memory", "memory_proposed", f"{decision.memory_proposal.kind} proposal accepted")
            except MemoryProposalError as exc:
                self.monitor_bus.emit("memory", "memory_rejected", str(exc))
                print(f"[memory] rejected proposal: {exc}")
        if decision.learning_question is not None or decision.intent == IntentType.ASK_USER_TO_LEARN:
            record = self.active_learning.consider(
                decision.learning_question,
                settings=self.settings.effective(),
                interaction_state=self._active_learning_interaction_state(),
            )
            self._last_learning_question_shown = record
            if record is not None:
                self.activity.set("listening", reason="active_learning_question_shown")
        if decision.maintenance_proposal is not None and announce_maintenance:
            self.monitor_bus.emit("maintenance", "maintenance_proposed", decision.maintenance_proposal.description)
            print(
                "[maintenance] proposal recorded (no automated consolidation job in this "
                f"phase): {decision.maintenance_proposal.description}"
            )

    def _record_boot_memory(self, robot_state: RobotStateSnapshot) -> None:
        lowstate = "available" if robot_state.lowstate else "unavailable"
        battery = (
            f"{robot_state.battery_pct:.0f}%"
            if robot_state.battery_pct is not None
            else "unavailable"
        )
        faults = ", ".join(robot_state.active_faults) if robot_state.active_faults else "none"
        summary = (
            f"Boot event {self._boot_event.value}; lifecycle entered awake; "
            f"posture={robot_state.posture}; stability={robot_state.stability}; "
            f"battery={battery}; lowstate={lowstate}; active_faults={faults}."
        )
        try:
            self.memory.autobiography.append(summary)
        except Exception as exc:
            print(f"[memory] could not record boot memory: {exc}")

    # -- decision execution ---------------------------------------------------

    def _execute_decision(self, decision: PlannerDecision, planner_input: PlannerInput) -> TurnOutcome:
        settings = self.settings.effective()
        robot_state = planner_input.robot_state
        outcome = TurnOutcome(decision=decision, grounded_response=decision.response_text, skill_outcomes=[])
        outcome.learning_question = self._last_learning_question_shown
        before_semantic = self._last_semantic_state

        if decision.intent == IntentType.QUERY_CAPABILITY and (decision.target or "").strip().lower() == "arm":
            policy = self.resolver.resolve_arm_motion(settings=settings, robot_state=robot_state)
            outcome.grounded_response = self._phrase_capability_answer(policy)
            outcome.grounded_response = self._limit_response(outcome.grounded_response, settings=settings)
            self._speak_grounded_response(outcome.grounded_response, settings=settings, robot_state=robot_state)
            return outcome
        if decision.intent == IntentType.QUERY_CAPABILITY:
            target = (decision.target or planner_input.user_text or "").strip().lower()
            grounded = self._maybe_describe_runtime_capability(target, robot_state)
            if grounded is not None:
                outcome.grounded_response = self._limit_response(grounded, settings=settings)
                self._speak_grounded_response(outcome.grounded_response, settings=settings, robot_state=robot_state)
                return outcome

        if decision.intent == IntentType.QUERY_STATE:
            target = (decision.target or "").strip().lower()
            if "battery" in target or "charge" in target:
                outcome.grounded_response = self._describe_battery(robot_state)
            else:
                outcome.grounded_response = decision.response_text or self._describe_state(robot_state)
            outcome.grounded_response = self._limit_response(outcome.grounded_response, settings=settings)
            self._speak_grounded_response(outcome.grounded_response, settings=settings, robot_state=robot_state)
            return outcome

        if decision.intent in (IntentType.NO_ACTION, IntentType.CONVERSATION, IntentType.MAINTENANCE, IntentType.ASK_USER_TO_LEARN):
            outcome.grounded_response = self._limit_response(outcome.grounded_response, settings=settings)
            self._speak_grounded_response(outcome.grounded_response, settings=settings, robot_state=robot_state)
            return outcome

        outcome.announcement = announce(
            decision.intent_announcement,
            registry=self.skills,
            resolver=self.resolver,
            settings=settings,
            robot_state=robot_state,
            is_denial=False,
        )

        for skill_name in decision.requested_skills:
            skill_kwargs = self._skill_kwargs(skill_name, decision, planner_input, settings)
            invocation_id = f"{skill_name}_{int(time.time() * 1000)}"
            started_at = datetime.now(timezone.utc)
            self._current_objectives = [
                {"summary": f"execute {skill_name}", "priority": "P3"},
                {"summary": "respond to CLI user", "priority": "P1"},
            ]
            self.monitor_bus.emit("skill", "skill_started", f"{skill_name} started", references=[invocation_id])
            skill_outcome = resolve_and_maybe_invoke(
                self.skills,
                self.resolver,
                skill_name,
                settings=settings,
                robot_state=robot_state,
                **skill_kwargs,
            )
            if skill_outcome.status == "needs_confirmation":
                approved = self.auto_confirm or self._confirm_prompt(skill_name, skill_outcome.policy)
                if approved:
                    skill_outcome = resolve_and_maybe_invoke(
                        self.skills,
                        self.resolver,
                        skill_name,
                        settings=settings,
                        robot_state=robot_state,
                        confirmed=True,
                        **skill_kwargs,
                    )
                else:
                    skill_outcome = SkillInvocationOutcome(
                        policy=PolicyDecision(
                            allowed=False,
                            requires_approval=True,
                            risk=skill_outcome.policy.risk,
                            reason="Denied by operator (confirmation declined).",
                        ),
                        skill_mode=skill_outcome.skill_mode,
                        status="denied",
                        result=SkillResult(ok=False, message="denied by operator"),
                    )
            if skill_outcome.status == "denied":
                announce(
                    IntentAnnouncement(speech=f"I can't do that: {skill_outcome.policy.reason}"),
                    registry=self.skills,
                    resolver=self.resolver,
                    settings=settings,
                    robot_state=robot_state,
                    is_denial=True,
                )
            outcome.skill_outcomes.append((skill_name, skill_outcome))
            completed_at = datetime.now(timezone.utc)
            try:
                observed_robot_state = build_robot_state(self.state_source)
                after_semantic, _ = self._update_semantic_state(
                    observed_robot_state,
                    interaction="user_engaged",
                    task="completed" if skill_outcome.status == "executed" else "failed",
                )
            except Exception:
                after_semantic = self._last_semantic_state
            evaluation = self.outcome_evaluator.evaluate(
                skill_id=skill_name,
                invocation_id=invocation_id,
                invocation_outcome=skill_outcome,
                before=before_semantic,
                after=after_semantic,
                started_at=started_at,
                completed_at=completed_at,
            )
            outcome.skill_evaluations.append(evaluation)
            event_name = "skill_completed" if evaluation.goal_reached else "skill_failed"
            self.monitor_bus.emit(
                "skill",
                event_name,
                f"{skill_name} goal_reached={evaluation.goal_reached} failure={evaluation.failure_type}",
                references=[invocation_id],
                metadata={"outcome": evaluation.model_dump(mode="json")},
            )
            episode_id = self.learning.record_skill_outcome(
                evaluation,
                before=before_semantic,
                after=after_semantic,
                settings=settings,
            )
            if settings.self_model.enabled and settings.self_model.update_from_skill_outcomes:
                self.self_model.update_from_skill_outcome(
                    evaluation,
                    episode_id=episode_id,
                    before=before_semantic,
                    after=after_semantic,
                )
            before_semantic = after_semantic

        if decision.intent == IntentType.REQUEST_SLEEP and any(
            o.status == "executed" and o.result and o.result.ok for _, o in outcome.skill_outcomes
        ):
            self._enter_sleep(decision)

        outcome.grounded_response = self._limit_response(outcome.grounded_response, settings=settings)
        self._current_objectives = []
        outcome.learning_question = self._last_learning_question_shown
        return outcome

    def _active_learning_interaction_state(self) -> str:
        if self._current_objectives:
            return "task"
        if self._last_semantic_state.task not in {"idle", "", None}:
            return "task"
        return "idle"

    def _skill_kwargs(
        self,
        skill_name: str,
        decision: PlannerDecision,
        planner_input: PlannerInput,
        settings: Any,
    ) -> dict[str, Any]:
        if skill_name == "reach_forward":
            text = (planner_input.user_text or "").lower()
            arm = "left" if "left" in text else "right"
            kwargs = {"arm": arm}
            kwargs.update(self._learned_skill_kwargs(skill_name, settings))
            return kwargs
        if skill_name in {"turn_left", "turn_right"}:
            return {"degrees": self._extract_turn_degrees(planner_input.user_text or "")}
        if skill_name == "grab":
            text = (planner_input.user_text or "").lower()
            prompt = decision.target or self._extract_grab_prompt(planner_input.user_text or "")
            if prompt == "grab":
                prompt = self._extract_grab_prompt(planner_input.user_text or "")
            arm = "auto"
            if "left" in text:
                arm = "left"
            elif "right" in text:
                arm = "right"
            kwargs = {"prompt": prompt, "agent_settings": settings, "arm": arm}
            kwargs.update(self._learned_skill_kwargs(skill_name, settings))
            return kwargs
        if skill_name == "save_current_nav_point":
            return {"name": self._extract_save_point_name(planner_input.user_text or "")}
        if skill_name == "navigate_named_point":
            return {"name": decision.target or self._extract_go_to_point_name(planner_input.user_text or ""), "auto_relocate": True}
        if skill_name in {"thinking_motion", "explain_motion", "thanking_motion"}:
            return {"agent_settings": settings, "reason": "user_request"}
        return self._learned_skill_kwargs(skill_name, settings)

    def _learned_skill_kwargs(self, skill_name: str, settings: Any) -> dict[str, Any]:
        automatic_max = int(getattr(settings.learning, "automatic_level_max", 1))
        if automatic_max < 3:
            return {}
        self_kwargs = self.self_model.learned_skill_kwargs(skill_name)
        if self_kwargs:
            self.monitor_bus.emit(
                "self",
                "self_model_procedure_applied",
                f"applying {skill_name} from self-model {self_kwargs}",
            )
            return self_kwargs
        for adaptation in self.memory.procedural.all():
            if adaptation.skill != skill_name:
                continue
            pre_pose = adaptation.recommended_parameters.get("pre_pose")
            if pre_pose:
                self.self_model.apply_procedural_adaptation(adaptation)
                self.monitor_bus.emit(
                    "learning",
                    "procedural_adaptation_validated",
                    f"applying {skill_name} pre_pose={pre_pose}",
                    references=list(adaptation.derived_from),
                )
                return {"learned_pre_pose": pre_pose}
        return {}

    def _apply_command_fallbacks(
        self, decision: PlannerDecision, planner_input: PlannerInput
    ) -> PlannerDecision:
        """Deterministic recovery for simple robot commands the planner missed."""
        if decision.requested_skills:
            return decision
        if planner_input.event not in (EventType.USER_MESSAGE, EventType.ASR_MESSAGE):
            return decision
        text = (planner_input.user_text or "").strip().lower()
        direct_skill_aliases = {
            "step_forward": ("step_forward", "I'll step forward.", "Ich gehe einen Schritt nach vorne."),
            "step_back": ("step_back", "I'll step backward.", "Ich gehe einen Schritt zurück."),
            "turn_left": ("turn_left", "I'll turn left.", "Ich drehe mich nach links."),
            "turn_right": ("turn_right", "I'll turn right.", "Ich drehe mich nach rechts."),
            "start_mapping": ("start_mapping", "I'll start SLAM mapping.", "Ich starte das SLAM-Mapping."),
            "save_map": ("save_map", "I'll save the SLAM map.", "Ich speichere die SLAM-Karte."),
            "start_relocation": ("start_relocation", "I'll start SLAM relocation.", "Ich starte die SLAM-Relokalisierung."),
            "relocate": ("start_relocation", "I'll start SLAM relocation.", "Ich starte die SLAM-Relokalisierung."),
            "add_current_nav_pose": ("add_current_nav_pose", "I'll add the current pose as a navigation task.", "Ich füge die aktuelle Pose als Navigationsziel hinzu."),
            "navigate_selected_pose": ("navigate_selected_pose", "I'll navigate to the selected pose.", "Ich navigiere zur ausgewählten Pose."),
            "execute_nav_tasks": ("execute_nav_tasks", "I'll execute the queued navigation tasks.", "Ich führe die Navigationsaufgaben aus."),
            "pause_navigation": ("pause_navigation", "I'll pause navigation.", "Ich pausiere die Navigation."),
            "resume_navigation": ("resume_navigation", "I'll resume navigation.", "Ich setze die Navigation fort."),
            "stop_slam": ("stop_slam", "I'll stop SLAM.", "Ich stoppe SLAM."),
            "slam_status": ("slam_status", "I'll check SLAM status.", "Ich prüfe den SLAM-Status."),
            "slam_preflight": ("slam_preflight", "I'll check SLAM mapping prerequisites.", "Ich prüfe die SLAM-Mapping-Voraussetzungen."),
            "list_nav_points": ("list_nav_points", "I'll list saved SLAM points.", "Ich liste gespeicherte SLAM-Punkte auf."),
            "clear_nav_points": ("clear_nav_points", "I'll clear saved SLAM points.", "Ich lösche gespeicherte SLAM-Punkte."),
            "release_arms": ("release_arms", "I'll release arm control authority.", "Ich gebe die Armsteuerung frei."),
            "wave": ("wave", "I'll try a wave.", "Ich versuche zu winken."),
            "face_wave": ("face_wave", "I'll try a face wave.", "Ich versuche vor dem Gesicht zu winken."),
            "high_wave": ("high_wave", "I'll try a high wave.", "Ich versuche hoch zu winken."),
            "clap": ("clap", "I'll clap.", "Ich klatsche."),
            "left_kiss": ("left_kiss", "I'll do a left-kiss gesture.", "Ich mache die Left-Kiss-Geste."),
            "shake_hand": ("shake_hand", "I'll offer a handshake.", "Ich biete einen Handschlag an."),
            "squat": ("squat", "I'll enter squat mode.", "Ich gehe in den Squat-Modus."),
            "prepare": ("prepare", "I'll enter prepare mode.", "Ich gehe in den Prepare-Modus."),
            "damp": ("damp", "I'll enter damp mode.", "Ich gehe in den Damp-Modus."),
            "zero_torque": ("zero_torque", "I'll enter zero-torque mode.", "Ich gehe in den Zero-Torque-Modus."),
            "dev_mode": ("dev_mode", "I'll enter developer mode.", "Ich gehe in den Developer-Modus."),
            "exit_dev_mode": ("exit_dev_mode", "I'll exit developer mode and re-enable ai_sport.", "Ich verlasse den Developer-Modus und aktiviere ai_sport wieder."),
            "walk_mode": ("walk_mode", "I'll enter walk mode.", "Ich gehe in den Walk-Modus."),
            "walking_mode": ("walk_mode", "I'll enter walk mode.", "Ich gehe in den Walk-Modus."),
            "run_mode": ("run_mode", "I'll enter run mode.", "Ich gehe in den Run-Modus."),
            "running_mode": ("run_mode", "I'll enter run mode.", "Ich gehe in den Run-Modus."),
            "thinking": ("thinking_motion", "I'll play the thinking expressive motion.", "Ich spiele die Thinking-Ausdrucksbewegung."),
            "explain": ("explain_motion", "I'll play the explain expressive motion.", "Ich spiele die Explain-Ausdrucksbewegung."),
            "thanking": ("thanking_motion", "I'll play the thanking expressive motion.", "Ich spiele die Thanking-Ausdrucksbewegung."),
        }
        if text in direct_skill_aliases:
            skill_name, english, german_text = direct_skill_aliases[text]
            if skill_name in self.skills.skills:
                german = self._effective_reply_language(self.settings.effective()) == "de"
                return PlannerDecision(
                    intent=IntentType.EXECUTE_TASK,
                    target=skill_name,
                    response_text=german_text if german else english,
                    requested_skills=[skill_name],
                )
        wants_extend_arm = (
            "extend arm" in text
            or "extend the arm" in text
            or "extend right arm" in text
            or "extend the right arm" in text
            or "extend left arm" in text
            or "extend the left arm" in text
            or "arm forward" in text
            or "arm nach vorne" in text
            or "hand nach vorne" in text
            or "rechte hand nach vorne" in text
            or "rechten arm nach vorne" in text
            or "rechter arm nach vorne" in text
            or "linke hand nach vorne" in text
            or "linken arm nach vorne" in text
            or "linker arm nach vorne" in text
            or "nach vorne bringen" in text
        )
        if wants_extend_arm and "reach_forward" in self.skills.skills:
            side = "left" if ("left" in text or "linke" in text or "linken" in text or "linker" in text) else "right"
            german = self._effective_reply_language(self.settings.effective()) == "de"
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target=f"{side}_arm",
                response_text=(
                    ("Ich strecke meinen rechten Arm nach vorne." if side == "right" else "Ich strecke meinen linken Arm nach vorne.")
                    if german
                    else f"I'll extend my {side} arm forward."
                ),
                requested_skills=["reach_forward"],
            )
        wants_release_arms = (
            "release arm" in text
            or "release arms" in text
            or "release the arm" in text
            or "release the arms" in text
            or "release your arm" in text
            or "release you right arm" in text
            or "release you left arm" in text
            or "free arm" in text
            or "freigeben" in text
        )
        if wants_release_arms and "release_arms" in self.skills.skills:
            german = self._effective_reply_language(self.settings.effective()) == "de"
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="release_arms",
                response_text="Ich gebe die Armsteuerung frei." if german else "I'll release arm control authority.",
                requested_skills=["release_arms"],
            )
        slam_skill = self._match_slam_command(text)
        if slam_skill and slam_skill in self.skills.skills:
            german = self._effective_reply_language(self.settings.effective()) == "de"
            responses = {
                "start_mapping": ("I'll start SLAM mapping.", "Ich starte das SLAM-Mapping."),
                "save_map": ("I'll save the SLAM map.", "Ich speichere die SLAM-Karte."),
                "start_relocation": ("I'll start SLAM relocation.", "Ich starte die SLAM-Relokalisierung."),
                "save_current_nav_point": ("I'll save the current SLAM point.", "Ich speichere den aktuellen SLAM-Punkt."),
                "list_nav_points": ("I'll list saved SLAM points.", "Ich liste gespeicherte SLAM-Punkte auf."),
                "clear_nav_points": ("I'll clear saved SLAM points.", "Ich lösche gespeicherte SLAM-Punkte."),
                "navigate_named_point": ("I'll navigate to the saved SLAM point.", "Ich navigiere zum gespeicherten SLAM-Punkt."),
                "add_current_nav_pose": ("I'll add the current pose as a navigation task.", "Ich füge die aktuelle Pose als Navigationsziel hinzu."),
                "navigate_selected_pose": ("I'll navigate to the selected pose.", "Ich navigiere zur ausgewählten Pose."),
                "execute_nav_tasks": ("I'll execute the queued navigation tasks.", "Ich führe die Navigationsaufgaben aus."),
                "pause_navigation": ("I'll pause navigation.", "Ich pausiere die Navigation."),
                "resume_navigation": ("I'll resume navigation.", "Ich setze die Navigation fort."),
                "stop_slam": ("I'll stop SLAM.", "Ich stoppe SLAM."),
                "slam_status": ("I'll check SLAM status.", "Ich prüfe den SLAM-Status."),
                "slam_preflight": ("I'll check SLAM mapping prerequisites.", "Ich prüfe die SLAM-Mapping-Voraussetzungen."),
            }
            english, german_text = responses[slam_skill]
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target=slam_skill,
                response_text=german_text if german else english,
                requested_skills=[slam_skill],
            )
        point_name = self._extract_save_point_name(planner_input.user_text or "")
        if point_name and "save_current_nav_point" in self.skills.skills:
            german = self._effective_reply_language(self.settings.effective()) == "de"
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target=point_name,
                response_text=(f"Ich speichere den aktuellen SLAM-Punkt als {point_name}." if german else f"I'll save the current SLAM point as {point_name}."),
                requested_skills=["save_current_nav_point"],
            )
        go_to_name = self._extract_go_to_point_name(planner_input.user_text or "")
        if go_to_name and "navigate_named_point" in self.skills.skills:
            german = self._effective_reply_language(self.settings.effective()) == "de"
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target=go_to_name,
                response_text=(f"Ich navigiere zum gespeicherten Punkt {go_to_name}." if german else f"I'll navigate to saved point {go_to_name}."),
                requested_skills=["navigate_named_point"],
            )
        wants_step_forward = (
            "move forward" in text
            or "step forward" in text
            or "go forward" in text
            or "walk forward" in text
            or "forward" == text
            or "geh nach vorne" in text
            or "gehe nach vorne" in text
            or "vorwärts" in text
        )
        wants_step_back = (
            "move backward" in text
            or "move back" in text
            or "step backward" in text
            or "step back" in text
            or "go backward" in text
            or "go back" in text
            or "walk backward" in text
            or "backward" == text
            or "back" == text
            or "geh zurück" in text
            or "gehe zurück" in text
            or "rückwärts" in text
        )
        if wants_step_forward and "step_forward" in self.skills.skills:
            german = self._effective_reply_language(self.settings.effective()) == "de"
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="step_forward",
                response_text="Ich gehe einen Schritt nach vorne." if german else "I'll step forward.",
                requested_skills=["step_forward"],
            )
        if wants_step_back and "step_back" in self.skills.skills:
            german = self._effective_reply_language(self.settings.effective()) == "de"
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="step_back",
                response_text="Ich gehe einen Schritt zurück." if german else "I'll step backward.",
                requested_skills=["step_back"],
            )
        wants_turn_around = (
            "turn around" in text
            or "turnaround" in text
            or "turn round" in text
            or "tunr around" in text
            or "rotate 180" in text
            or "180" in text and ("turn" in text or "rotate" in text)
            or "umdrehen" in text
        )
        wants_turn_left = (
            "turn left" in text
            or "turn the robot" in text and "left" in text
            or "rotate left" in text
            or "dreh links" in text
        )
        wants_turn_right = (
            "turn right" in text
            or "turn the robot" in text and "right" in text
            or "rotate right" in text
            or "dreh rechts" in text
        )
        if wants_turn_around and not wants_turn_left and not wants_turn_right:
            wants_turn_left = True
        if wants_turn_left and "turn_left" in self.skills.skills:
            german = self._effective_reply_language(self.settings.effective()) == "de"
            degrees = self._extract_turn_degrees(planner_input.user_text or "")
            response = f"I'll turn left about {degrees:.0f} degrees."
            german_response = f"Ich drehe mich etwa {degrees:.0f} Grad nach links."
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="turn_left",
                response_text=german_response if german else response,
                requested_skills=["turn_left"],
            )
        if wants_turn_right and "turn_right" in self.skills.skills:
            german = self._effective_reply_language(self.settings.effective()) == "de"
            degrees = self._extract_turn_degrees(planner_input.user_text or "")
            response = f"I'll turn right about {degrees:.0f} degrees."
            german_response = f"Ich drehe mich etwa {degrees:.0f} Grad nach rechts."
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="turn_right",
                response_text=german_response if german else response,
                requested_skills=["turn_right"],
            )
        wants_grab = any(word in text for word in ("grab", "grba", "grasp", "pick up", "greif", "greife", "nimm"))
        if wants_grab and "grab" in self.skills.skills:
            german = self._effective_reply_language(self.settings.effective()) == "de"
            prompt = self._extract_grab_prompt(planner_input.user_text or "")
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target=prompt,
                response_text=(
                    f"Ich versuche, {prompt} zu greifen."
                    if german
                    else f"I'll try to grab {prompt}."
                ),
                requested_skills=["grab"],
                intent_announcement=IntentAnnouncement(
                    speech=(
                        f"Ich versuche, {prompt} zu greifen."
                        if german
                        else f"I'll try to grab {prompt}."
                    )
                ),
            )
        phrase_skills = (
            (
                "face_wave",
                ("face wave", "wave face", "wink face", "gesicht wink"),
                "I'll try a face wave.",
                "Ich versuche vor dem Gesicht zu winken.",
            ),
            ("high_wave", ("high wave", "big wave", "wave high", "hoch wink"), "I'll try a high wave.", "Ich versuche hoch zu winken."),
            ("clap", ("clap", "klatsch"), "I'll clap.", "Ich klatsche."),
            ("left_kiss", ("left kiss", "left hand kiss", "lefthand kiss", "linker kuss"), "I'll do a left-kiss gesture.", "Ich mache die Left-Kiss-Geste."),
            ("shake_hand", ("shake hand", "handshake", "shake my hand", "handschlag"), "I'll offer a handshake.", "Ich biete einen Handschlag an."),
            ("squat", ("squat", "hocke"), "I'll enter squat mode.", "Ich gehe in den Squat-Modus."),
            ("prepare", ("prepare mode", "prepare", "bereit machen"), "I'll enter prepare mode.", "Ich gehe in den Prepare-Modus."),
            ("damp", ("damp mode", "damp", "dämpfen", "daempfen"), "I'll enter damp mode.", "Ich gehe in den Damp-Modus."),
            ("zero_torque", ("zero torque", "zero-torque", "zero_torque", "nullmoment"), "I'll enter zero-torque mode.", "Ich gehe in den Zero-Torque-Modus."),
            ("exit_dev_mode", ("exit dev mode", "leave dev mode", "exit developer mode", "leave developer mode", "ai sport on", "enable ai_sport", "enable ai sport"), "I'll exit developer mode and re-enable ai_sport.", "Ich verlasse den Developer-Modus und aktiviere ai_sport wieder."),
            ("dev_mode", ("dev mode", "developer mode", "entwickler modus"), "I'll enter developer mode.", "Ich gehe in den Developer-Modus."),
            ("walk_mode", ("walk mode", "walking mode", "set walk mode", "gehmodus"), "I'll enter walk mode.", "Ich gehe in den Walk-Modus."),
            ("run_mode", ("run mode", "running mode", "set run mode", "rennmodus"), "I'll enter run mode.", "Ich gehe in den Run-Modus."),
            ("thinking_motion", ("thinking motion", "denk bewegung"), "I'll play the thinking expressive motion.", "Ich spiele die Thinking-Ausdrucksbewegung."),
            ("explain_motion", ("explain motion", "erklär bewegung", "erklaer bewegung"), "I'll play the explain expressive motion.", "Ich spiele die Explain-Ausdrucksbewegung."),
            ("thanking_motion", ("thanking motion", "thank motion", "dank bewegung"), "I'll play the thanking expressive motion.", "Ich spiele die Thanking-Ausdrucksbewegung."),
        )
        for skill_name, phrases, english, german_text in phrase_skills:
            if skill_name in self.skills.skills and any(phrase in text for phrase in phrases):
                german = self._effective_reply_language(self.settings.effective()) == "de"
                return PlannerDecision(
                    intent=IntentType.EXECUTE_TASK,
                    target=skill_name,
                    response_text=german_text if german else english,
                    requested_skills=[skill_name],
                    intent_announcement=IntentAnnouncement(speech=german_text if german else english),
                )
        wants_wave = "wave" in text or "wink" in text
        if not wants_wave:
            return decision
        if "high_wave" in self.skills.skills and ("high" in text or "big" in text or "hoch" in text):
            german = self._effective_reply_language(self.settings.effective()) == "de"
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="high_wave",
                response_text="Ich versuche hoch zu winken." if german else "I'll try a high wave.",
                requested_skills=["high_wave"],
                intent_announcement=IntentAnnouncement(speech="Ich winke hoch." if german else "I'll wave high."),
            )
        if "wave" in self.skills.skills:
            german = self._effective_reply_language(self.settings.effective()) == "de"
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="wave",
                response_text="Ich versuche zu winken." if german else "I'll try a wave.",
                requested_skills=["wave"],
                intent_announcement=IntentAnnouncement(speech="Ich winke." if german else "I'll wave."),
            )
        return decision

    def _normalize_decision(self, decision: PlannerDecision, planner_input: PlannerInput) -> PlannerDecision:
        """Correct common planner/tool mismatches before capability dispatch."""
        if planner_input.event not in (EventType.USER_MESSAGE, EventType.ASR_MESSAGE):
            return decision
        text = (planner_input.user_text or "").strip().lower()
        skills = list(decision.requested_skills)

        wants_exit_dev_mode = (
            "exit dev mode" in text
            or "leave dev mode" in text
            or "exit developer mode" in text
            or "leave developer mode" in text
            or "enable ai_sport" in text
            or "enable ai sport" in text
            or "ai sport on" in text
        )
        if wants_exit_dev_mode and "exit_dev_mode" in self.skills.skills:
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="exit_dev_mode",
                response_text="I'll exit developer mode and re-enable ai_sport.",
                requested_skills=["exit_dev_mode"],
                intent_announcement=decision.intent_announcement,
            )
        wants_save_mapping = (
            "stop mapping" in text
            or "finish mapping" in text
            or "end mapping" in text
            or "save map" in text
            or "save the map" in text
        )
        if wants_save_mapping and "save_map" in self.skills.skills:
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="save_map",
                response_text="I'll save the SLAM map.",
                requested_skills=["save_map"],
                intent_announcement=decision.intent_announcement,
            )

        if (
            decision.intent == IntentType.MOVE_ARM
            or "extend arm" in text
            or "extend the arm" in text
            or "extend right arm" in text
            or "extend the right arm" in text
            or "extend left arm" in text
            or "extend the left arm" in text
            or "arm forward" in text
        ):
            if "reach_forward" in self.skills.skills and (not skills or "move" in skills):
                side = "left" if "left" in text else "right"
                return PlannerDecision(
                    intent=IntentType.EXECUTE_TASK,
                    target=f"{side}_arm",
                    response_text=f"I'll extend my {side} arm forward.",
                    requested_skills=["reach_forward"],
                    intent_announcement=decision.intent_announcement,
                )

        wants_release_arms = (
            "release arm" in text
            or "release arms" in text
            or "release the arm" in text
            or "release the arms" in text
            or "release your arm" in text
            or "release you right arm" in text
            or "release you left arm" in text
            or "free arm" in text
            or "freigeben" in text
        )
        if wants_release_arms and "release_arms" in self.skills.skills:
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="release_arms",
                response_text="I'll release arm control authority.",
                requested_skills=["release_arms"],
                intent_announcement=decision.intent_announcement,
            )
        slam_skill = self._match_slam_command(text)
        if slam_skill and slam_skill in self.skills.skills and (not skills or "move" in skills):
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target=slam_skill,
                response_text={
                    "start_mapping": "I'll start SLAM mapping.",
                    "save_map": "I'll save the SLAM map.",
                    "start_relocation": "I'll start SLAM relocation.",
                    "save_current_nav_point": "I'll save the current SLAM point.",
                    "list_nav_points": "I'll list saved SLAM points.",
                    "clear_nav_points": "I'll clear saved SLAM points.",
                    "navigate_named_point": "I'll navigate to the saved SLAM point.",
                    "add_current_nav_pose": "I'll add the current pose as a navigation task.",
                    "navigate_selected_pose": "I'll navigate to the selected pose.",
                    "execute_nav_tasks": "I'll execute the queued navigation tasks.",
                    "pause_navigation": "I'll pause navigation.",
                    "resume_navigation": "I'll resume navigation.",
                    "stop_slam": "I'll stop SLAM.",
                    "slam_status": "I'll check SLAM status.",
                    "slam_preflight": "I'll check SLAM mapping prerequisites.",
                }[slam_skill],
                requested_skills=[slam_skill],
                intent_announcement=decision.intent_announcement,
            )
        wants_step_forward = (
            "move forward" in text
            or "step forward" in text
            or "go forward" in text
            or "walk forward" in text
            or "forward" == text
            or "geh nach vorne" in text
            or "gehe nach vorne" in text
            or "vorwärts" in text
        )
        wants_step_back = (
            "move backward" in text
            or "move back" in text
            or "step backward" in text
            or "step back" in text
            or "go backward" in text
            or "go back" in text
            or "walk backward" in text
            or "backward" == text
            or "back" == text
            or "geh zurück" in text
            or "gehe zurück" in text
            or "rückwärts" in text
        )
        if wants_step_forward and "step_forward" in self.skills.skills and (not skills or "move" in skills):
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="step_forward",
                response_text="I'll step forward.",
                requested_skills=["step_forward"],
                intent_announcement=decision.intent_announcement,
            )
        if wants_step_back and "step_back" in self.skills.skills and (not skills or "move" in skills):
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="step_back",
                response_text="I'll step backward.",
                requested_skills=["step_back"],
                intent_announcement=decision.intent_announcement,
            )
        wants_turn_around = (
            "turn around" in text
            or "turnaround" in text
            or "turn round" in text
            or "tunr around" in text
            or "rotate 180" in text
            or "180" in text and ("turn" in text or "rotate" in text)
            or "umdrehen" in text
        )
        wants_turn_left = (
            "turn left" in text
            or "turn the robot" in text and "left" in text
            or "rotate left" in text
            or "dreh links" in text
        )
        wants_turn_right = (
            "turn right" in text
            or "turn the robot" in text and "right" in text
            or "rotate right" in text
            or "dreh rechts" in text
        )
        if wants_turn_around and not wants_turn_left and not wants_turn_right:
            wants_turn_left = True
        if wants_turn_left and "turn_left" in self.skills.skills and (not skills or "move" in skills):
            degrees = self._extract_turn_degrees(planner_input.user_text or "")
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="turn_left",
                response_text=f"I'll turn left about {degrees:.0f} degrees.",
                requested_skills=["turn_left"],
                intent_announcement=decision.intent_announcement,
            )
        if wants_turn_right and "turn_right" in self.skills.skills and (not skills or "move" in skills):
            degrees = self._extract_turn_degrees(planner_input.user_text or "")
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="turn_right",
                response_text=f"I'll turn right about {degrees:.0f} degrees.",
                requested_skills=["turn_right"],
                intent_announcement=decision.intent_announcement,
            )
        return decision

    @staticmethod
    def _match_slam_command(text: str) -> Optional[str]:
        if "slam status" in text or "mapping status" in text or "map status" in text:
            return "slam_status"
        if "slam preflight" in text or "mapping preflight" in text or "check slam" in text or "check mapping" in text:
            return "slam_preflight"
        if "start mapping" in text or "begin mapping" in text or "mapping starten" in text:
            return "start_mapping"
        if "save map" in text or "end mapping" in text or "finish mapping" in text or "stop mapping" in text or "karte speichern" in text:
            return "save_map"
        if "start relocation" in text or "start localization" in text or "localize on map" in text or text in {"relocate", "localize", "relocalize", "init pose"}:
            return "start_relocation"
        if "list points" in text or "list saved points" in text or "saved points" in text:
            return "list_nav_points"
        if "clear points" in text or "clear saved points" in text:
            return "clear_nav_points"
        if "add current pose" in text or "add current location" in text or "save current pose" in text:
            return "add_current_nav_pose"
        if "go to selected" in text or "navigate selected" in text:
            return "navigate_selected_pose"
        if "execute nav" in text or "execute task" in text or "run nav task" in text:
            return "execute_nav_tasks"
        if "pause navigation" in text or "pause nav" in text:
            return "pause_navigation"
        if "resume navigation" in text or "resume nav" in text:
            return "resume_navigation"
        if "stop slam" in text or "close slam" in text or "slam stoppen" in text:
            return "stop_slam"
        return None

    @classmethod
    def _extract_save_point_name(cls, text: str) -> str:
        lowered = text.strip().lower()
        patterns = (
            r"(?:add|save|mark|remember)\s+(?:the\s+)?current\s+(?:slam\s+)?point\s+(?:as|called|named)\s+(.+)$",
            r"(?:add|save|mark|remember)\s+(?:this\s+)?(?:place|location|position)\s+(?:as|called|named)\s+(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                return cls._clean_point_name(match.group(1))
        return ""

    @classmethod
    def _extract_go_to_point_name(cls, text: str) -> str:
        lowered = text.strip().lower()
        patterns = (
            r"^(?:go|navigate|drive|walk)\s+to\s+(.+)$",
            r"^take\s+me\s+to\s+(.+)$",
            r"^go\s+to\s+point\s+(.+)$",
            r"^navigate\s+to\s+point\s+(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                name = cls._clean_point_name(match.group(1))
                blocked = {"selected", "selected pose", "current", "current pose"}
                return "" if name in blocked else name
        return ""

    @staticmethod
    def _clean_point_name(text: str) -> str:
        cleaned = str(text).strip().lower()
        cleaned = re.sub(r"^(call it|name it|save it as|save as|called|named)\s+", "", cleaned).strip()
        cleaned = re.sub(r"[^a-z0-9 _-]+", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_")
        return cleaned

    @staticmethod
    def _extract_turn_degrees(text: str) -> float:
        lowered = text.lower()
        if "turn around" in lowered or "turnaround" in lowered or "turn round" in lowered or "tunr around" in lowered or "umdrehen" in lowered:
            return 180.0
        match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|grad)\b", lowered)
        if match:
            try:
                return max(5.0, min(180.0, float(match.group(1))))
            except ValueError:
                pass
        if "180" in lowered:
            return 180.0
        if "90" in lowered:
            return 90.0
        return 28.6

    @staticmethod
    def _extract_grab_prompt(text: str) -> str:
        cleaned = text.strip()
        prefixes = (
            "please grab ",
            "please grba ",
            "the grab ",
            "the grba ",
            "grab ",
            "grba ",
            "please grasp ",
            "grasp ",
            "pick up ",
            "greife ",
            "greif ",
            "nimm ",
        )
        while True:
            lowered = cleaned.lower()
            for prefix in prefixes:
                if lowered.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break
            else:
                break
        lowered = cleaned.lower()
        for marker in (" grab ", " grba ", " grasp ", " pick up ", " greif ", " greife ", " nimm "):
            idx = lowered.rfind(marker)
            if idx >= 0:
                cleaned = cleaned[idx + len(marker):].strip()
                break
        return cleaned or "the object"

    def _confirm_prompt(self, skill_name: str, policy: PolicyDecision) -> bool:
        try:
            reply = input(f"  [confirm] execute skill '{skill_name}'? ({policy.reason}) [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return reply in ("y", "yes", "j", "ja")

    def _enter_sleep(self, decision: PlannerDecision) -> None:
        self.lifecycle.transition(LifecycleState.PRE_SLEEP)
        self.lifecycle.transition(LifecycleState.SLEEPING)
        checkpoint = RuntimeCheckpoint(
            last_cognitive_timestamp=self.scheduler.last_cognitive_timestamp,
            lifecycle_state=LifecycleState.SLEEPING,
            last_event_type=EventType.USER_MESSAGE,
            last_decision=decision.intent,
            sleep_reason=decision.target or "unspecified",
            sleep_timestamp=time.time(),
        )
        self.checkpoint_store.save(checkpoint)
        self._previous_checkpoint = checkpoint
        print(
            "[lifecycle] checkpoint written with lifecycle_state=sleeping. Phase 1 does NOT "
            "issue a real OS shutdown -- see the request_sleep skill result message and the "
            "plan's deferred-TODO list."
        )

    def _speak_grounded_response(
        self,
        text: Optional[str],
        *,
        settings: Any,
        robot_state: RobotStateSnapshot,
    ) -> None:
        """Best-effort TTS for normal printed responses."""
        if not text or not settings.announcements.audio_enabled:
            return
        if len(text) >= int(getattr(settings.expressive_motion, "explain_minimum_speech_chars", 80)):
            self.monitor_bus.emit(
                "expressive",
                "long_explanation_started",
                f"speech chars={len(text)}",
            )
            self.expressive_motion.run_background(
                "explain",
                settings=settings,
                reason="long_explanation_started",
            )
        if "announce" not in self.skills.skills:
            print("[warn] announcements.audio_enabled=true but no 'announce' skill is available")
            return
        decision, result = invoke_with_capability_check(
            self.skills,
            self.resolver,
            "announce",
            settings=settings,
            robot_state=robot_state,
            text=text,
            language=self._effective_tts_language(settings),
            voice_model=settings.announcements.tts_voice_model or None,
            speaker=settings.announcements.tts_speaker,
        )
        if not decision.allowed:
            print(f"[warn] speech denied: {decision.reason}")
        elif result is not None and not result.ok:
            print(f"[warn] speech failed: {result.message}")
        elif len(text) >= int(getattr(settings.expressive_motion, "explain_minimum_speech_chars", 80)):
            self.monitor_bus.emit("expressive", "long_explanation_completed", "speech/explain event completed")

    @staticmethod
    def _effective_tts_language(settings: Any) -> Optional[str]:
        explicit = str(getattr(settings.announcements, "tts_language", "") or "").strip()
        if explicit:
            return explicit
        reply = G1Agent._effective_reply_language(settings)
        return reply if reply in {"en", "de"} else None

    @staticmethod
    def _effective_reply_language(settings: Any) -> str:
        reply = getattr(settings.interface.reply_language, "value", settings.interface.reply_language)
        if str(reply) in {"en", "de"}:
            return str(reply)
        command = getattr(settings.interface.command_language, "value", settings.interface.command_language)
        if str(command) in {"en", "de"}:
            return str(command)
        return "auto"

    @staticmethod
    def _limit_response(text: Optional[str], *, settings: Any) -> Optional[str]:
        if not text:
            return text
        try:
            max_chars = int(getattr(settings.response, "max_chars", 700))
        except Exception:
            max_chars = 700
        max_chars = max(120, min(4000, max_chars))
        if len(text) <= max_chars:
            return text
        suffix = "\n...[response truncated; raise response.max_chars in /settings-ui to show more]"
        return text[: max(0, max_chars - len(suffix))].rstrip() + suffix

    @staticmethod
    def _phrase_capability_answer(policy: PolicyDecision) -> str:
        prefix = "Yes." if policy.allowed else "No."
        return f"{prefix} {policy.reason}"

    def _maybe_describe_runtime_capability(self, target: str, robot_state: RobotStateSnapshot) -> Optional[str]:
        lowered = str(target).lower()
        if "navigate" in lowered or "navigation" in lowered:
            nav = self.navigation.snapshot()
            available = "start_mapping" in self.skills.skills or "navigate_named_point" in self.skills.skills
            return (
                ("Yes" if available else "No")
                + f". Navigation/SLAM backend={available}; slam={nav.slam}; localization={nav.localization}; last_error={nav.last_error or 'none'}."
            )
        if "run" in lowered:
            policy = self.resolver.resolve_skill("run_mode", settings=self.settings.effective(), robot_state=robot_state)
            return self._phrase_capability_answer(policy)
        if "walk" in lowered:
            policy = self.resolver.resolve_skill("walk_mode", settings=self.settings.effective(), robot_state=robot_state)
            return self._phrase_capability_answer(policy)
        if "see" in lowered or "camera" in lowered or "vision" in lowered:
            settings = self.settings.effective()
            return (
                ("Yes" if settings.vision.rgbd_enabled else "No")
                + f". RGB-D vision enabled={settings.vision.rgbd_enabled}; source=tcp://{settings.vision.rgbd_host}:{settings.vision.rgbd_port}; model={settings.vision.openai_model}."
            )
        if "hear" in lowered or "audio" in lowered or "asr" in lowered:
            settings = self.settings.effective()
            available = settings.audio.input_enabled and settings.audio.asr_enabled and settings.asr.enabled
            return (
                ("Yes" if available else "No")
                + f". Speech input enabled={available}; input topic=/audio_msg; /chat text input remains available."
            )
        return None

    @staticmethod
    def _describe_state(state: RobotStateSnapshot) -> str:
        return (
            f"posture={state.posture}, stability={state.stability}, "
            f"arm_control_state={state.arm_control_state}, "
            f"faults={state.active_faults or ['none']} (source={state.source})"
        )

    @staticmethod
    def _describe_battery(state: RobotStateSnapshot) -> str:
        if state.battery_pct is None:
            if state.battery:
                source = state.battery.get("source") or "unknown source"
                fields = state.battery.get("available_fields") or []
                field_text = f"; available fields: {', '.join(fields[:12])}" if fields else ""
                error = state.battery.get("error")
                error_text = f"; {error}" if error else ""
                return f"Battery percentage is unavailable from {source}{field_text}{error_text}."
            return "Battery percentage is unavailable: no lowstate or BMS battery packet has been received yet."
        charging = ""
        if state.charging is not None:
            charging = " and appears to be charging" if state.charging else " and does not appear to be charging"
        source = ""
        if state.battery and state.battery.get("source"):
            source = f" (source: {state.battery.get('source')})"
        return f"Battery is at {state.battery_pct:.0f}%{charging}{source}."

    def monitor_snapshot(self, *, panel: str = "overview") -> dict[str, Any]:
        settings = self.settings.effective()
        now = time.time()
        next_due = self.scheduler.next_tick_due_at
        memory_stats = self.learning.memory_stats(settings=settings)
        disk_stats = self.learning.disk_stats(settings=settings)
        learned = self.learning.learned.all()
        capability_lines = {
            "audio_response": settings.announcements.audio_enabled,
            "gesture_response": settings.announcements.gesture_enabled,
            "asr": settings.audio.asr_enabled and settings.audio.input_enabled and settings.asr.enabled,
            "arm_motion": settings.motion.allow_arm_motion,
            "arm_sdk": settings.motion.allow_arm_sdk,
            "low_cmd": settings.motion.allow_low_cmd,
            "locomotion_mode_change": settings.motion.allow_locomotion_mode_change,
            "navigation": any(name in self.skills.skills for name in ("start_mapping", "navigate_named_point", "step_forward")),
        }
        nav_snapshot = self.navigation.snapshot().as_dict()
        asr_snapshot = self.asr_runtime.snapshot()
        vision_snapshot = self.visual_observations.snapshot()
        llctl_snapshot = self.llctl.snapshot(settings)
        expressive_snapshot = self.expressive_motion.runtime.snapshot()
        active_learning_snapshot = self.active_learning.snapshot(settings=settings)
        activity_snapshot = self.activity.snapshot()
        tool_context = self._make_tool_context(
            settings=settings,
            robot_state=build_robot_state(self.state_source),
            event="monitor",
            profile="diagnostic" if panel in {"tools", "events"} else "social",
        )
        tool_snapshot = self.tool_registry.snapshot(tool_context)
        return {
            "panel": panel,
            "lifecycle": self.lifecycle.state.value,
            "model": type(self.planner).__name__,
            "last_cognition_age_s": self.scheduler.elapsed_since_last_cognition(now),
            "next_scheduled_check_s": None if next_due is None else max(0.0, next_due - now),
            "attention_queue": self.scheduler.queue_size,
            "semantic_state": self._last_semantic_state.model_dump(),
            "self": self.self_model.summary(),
            "objectives": list(self._current_objectives)
            or [{"summary": "respond to CLI user", "priority": "P1"} if self._last_semantic_state.interaction == "user_engaged" else {"summary": "mostly idle state monitoring", "priority": "P5"}],
            "events": [event.model_dump() for event in self.monitor_bus.recent(80)],
            "learning": {
                "candidate_claims": sum(1 for claim in learned if claim.status == "candidate"),
                "active_claims": sum(1 for claim in learned if claim.status == "active"),
                "contested_claims": sum(1 for claim in learned if claim.status == "contested"),
                "deprecated_claims": sum(1 for claim in learned if claim.status == "deprecated"),
                "procedural_rules": len(self.memory.procedural.all()),
                "latest": [claim.model_dump() for claim in learned[-5:]],
            },
            "memory": memory_stats,
            "disk": disk_stats,
            "tools": capability_lines,
            "navigation": nav_snapshot,
            "asr": asr_snapshot,
            "vision": vision_snapshot,
            "llctl": llctl_snapshot,
            "expressive": expressive_snapshot,
            "active_learning": active_learning_snapshot,
            "activity": activity_snapshot,
            "tooling": tool_snapshot,
        }

    def navigation_snapshot(self) -> dict[str, Any]:
        return self.navigation.snapshot().as_dict()

    def navigation_action(self, name: str, **kwargs: Any) -> str:
        result = self.navigation.action(name, **kwargs)
        self.monitor_bus.emit("navigation", "navigation_action", f"{name}: {result[:160]}")
        return result

    def asr_snapshot(self) -> dict[str, Any]:
        self.asr_runtime.update_settings(self.settings.effective())
        return self.asr_runtime.snapshot()

    def llctl_snapshot(self) -> dict[str, Any]:
        return self.llctl.snapshot(self.settings.effective())

    def llctl_enable(self) -> str:
        result = self.llctl.enable_session(self.settings.effective())
        self.monitor_bus.emit("llctl", "llctl_session", result)
        return result

    def llctl_disable(self) -> str:
        result = self.llctl.disable_session()
        self.monitor_bus.emit("llctl", "llctl_session", result)
        return result

    def llctl_command(self, args: list[str]) -> str:
        if not args:
            return (
                "LLCTL commands:\n"
                "/llctl enable | disable | status\n"
                "/llctl select <joint_id|joint_name>\n"
                "/llctl backend arm_sdk|lowcmd\n"
                "/llctl joint <joint> q <rad> [dq <rad/s>] [kp <gain>] [kd|kq <gain>] [tau <Nm>] [ramp <s>] [backend arm_sdk|lowcmd]\n"
                "/llctl ee left|right <x> <y> <z> <roll> <pitch> <yaw>\n"
                "/llctl ee left|right [x <m> y <m> z <m> roll <rad> pitch <rad> yaw <rad>]\n"
                "/llctl ee left|right [dx <m>] [dy <m>] [dz <m>] [droll <rad>] [dpitch <rad>] [dyaw <rad>]\n"
                "/llctl release_arms"
            )
        cmd = args[0].strip().lower()
        settings = self.settings.effective()
        if cmd == "enable":
            return self.llctl_enable()
        if cmd in {"disable", "off"}:
            return self.llctl_disable()
        if cmd == "status":
            return "\n".join(f"{key} = {value}" for key, value in sorted(self.llctl_snapshot().items()))
        if cmd == "select" and len(args) >= 2:
            result = self.llctl.select_joint(settings, joint=args[1])
            self.monitor_bus.emit("llctl", "llctl_select_joint", result)
            return result
        if cmd == "backend" and len(args) >= 2:
            result = self.llctl.set_backend(settings, backend=args[1])
            self.monitor_bus.emit("llctl", "llctl_backend", result)
            return result
        if cmd == "release_arms":
            result = self.llctl.release_arms(settings)
            self.monitor_bus.emit("llctl", "llctl_release_arms", result)
            return result
        if cmd == "joint" and len(args) >= 2:
            try:
                params = self._parse_key_value_args(args[2:])
                result = self.llctl.command_joint(
                    settings,
                    joint=args[1],
                    q=float(params.get("q", params.get("target", 0.0))),
                    dq=float(params.get("dq", 0.0)),
                    kp=float(params.get("kp", 30.0)),
                    kd=float(params.get("kd", params.get("kq", 1.5))),
                    tau=float(params.get("tau", 0.0)),
                    ramp_s=float(params.get("ramp", params.get("ramp_s", 0.6))),
                    backend=str(params.get("backend", "arm_sdk")),
                )
            except Exception as exc:
                result = f"invalid joint command: {exc}"
            self.monitor_bus.emit("llctl", "llctl_joint_command", result)
            return result
        if cmd == "ee" and len(args) >= 2:
            try:
                tail = args[2:]
                if len(tail) == 6 and all(self._looks_float(item) for item in tail):
                    result = self.llctl.command_ee_target(
                        settings,
                        side=args[1],
                        x=float(tail[0]),
                        y=float(tail[1]),
                        z=float(tail[2]),
                        roll=float(tail[3]),
                        pitch=float(tail[4]),
                        yaw=float(tail[5]),
                    )
                else:
                    params = self._parse_key_value_args(tail)
                    if {"x", "y", "z"} <= set(params):
                        result = self.llctl.command_ee_target(
                            settings,
                            side=args[1],
                            x=float(params["x"]),
                            y=float(params["y"]),
                            z=float(params["z"]),
                            roll=float(params.get("roll", params.get("r", 0.0))),
                            pitch=float(params.get("pitch", params.get("p", 0.0))),
                            yaw=float(params.get("yaw", params.get("yw", 0.0))),
                        )
                    else:
                        result = self.llctl.command_ee_delta(
                            settings,
                            side=args[1],
                            dx=float(params.get("dx", 0.0)),
                            dy=float(params.get("dy", 0.0)),
                            dz=float(params.get("dz", 0.0)),
                            droll=float(params.get("droll", params.get("dr", 0.0))),
                            dpitch=float(params.get("dpitch", params.get("dp", 0.0))),
                            dyaw=float(params.get("dyaw", params.get("dyw", 0.0))),
                        )
            except Exception as exc:
                result = f"invalid EE command: {exc}"
            self.monitor_bus.emit("llctl", "llctl_ee_command", result)
            return result
        return (
            "usage:\n"
            "/llctl enable | disable | status\n"
            "/llctl select <joint_id|joint_name>\n"
            "/llctl backend arm_sdk|lowcmd\n"
            "/llctl joint <joint> q <rad> [dq <rad/s>] [kp <gain>] [kd|kq <gain>] [tau <Nm>] [ramp <s>] [backend arm_sdk|lowcmd]\n"
            "/llctl ee left|right <x> <y> <z> <roll> <pitch> <yaw>\n"
            "/llctl ee left|right [x <m> y <m> z <m> roll <rad> pitch <rad> yaw <rad>]\n"
            "/llctl ee left|right [dx <m>] [dy <m>] [dz <m>] [droll <rad>] [dpitch <rad>] [dyaw <rad>]\n"
            "/llctl release_arms"
        )

    @staticmethod
    def _parse_key_value_args(args: list[str]) -> dict[str, str]:
        if len(args) % 2 != 0:
            raise ValueError("expected key/value pairs")
        out: dict[str, str] = {}
        for idx in range(0, len(args), 2):
            out[args[idx].strip().lower()] = args[idx + 1]
        return out

    @staticmethod
    def _looks_float(value: str) -> bool:
        try:
            float(value)
            return True
        except Exception:
            return False

    def tool_snapshot(self, *, profile: str = "social", include_unavailable: bool = True) -> dict[str, Any]:
        settings = self.settings.effective()
        context = self._make_tool_context(
            settings=settings,
            robot_state=build_robot_state(self.state_source),
            event="cli_tools",
            profile=profile,
        )
        snapshot = self.tool_registry.snapshot(context)
        if not include_unavailable:
            snapshot["tools"] = [row for row in snapshot["tools"] if row.get("availability") == "available"]
        return snapshot

    def _capability_tool_summary(self, robot_state: RobotStateSnapshot) -> dict[str, Any]:
        settings = self.settings.effective()
        return {
            "arm_motion": self.resolver.resolve_arm_motion(settings=settings, robot_state=robot_state).__dict__,
            "high_level_arm_action": self.resolver.resolve_high_level_arm_action(settings=settings, robot_state=robot_state).__dict__,
            "hand_action": self.resolver.resolve_hand_action(settings=settings, robot_state=robot_state).__dict__,
            "navigation": {
                "available": any(name in self.skills.skills for name in ("start_mapping", "navigate_named_point", "step_forward")),
                "source": self.navigation.snapshot().as_dict().get("planner_status"),
            },
            "asr": {
                "available": settings.audio.input_enabled and settings.audio.asr_enabled and settings.asr.enabled,
                "input_topic": "/audio_msg",
            },
            "vision": {
                "available": settings.vision.rgbd_enabled,
                "source": f"tcp://{settings.vision.rgbd_host}:{settings.vision.rgbd_port}",
                "model": settings.vision.openai_model,
            },
        }

    def reset(self, scope: str) -> ResetResult:
        with self._mutation_lock:
            return self.reset_manager.reset(scope)

    def cmd_reset(self, scope: str) -> str:
        result = self.reset(scope)
        if result.ok and result.scope in {"runtime", "full"}:
            boot_decision = self.boot()
            return f"{result.message}\nboot event={self._boot_event.value} decision={boot_decision.intent.value}"
        return result.message

    def reset_backups(self) -> str:
        backups = self.reset_manager.list_backups()
        if not backups:
            return "(no reset backups)"
        return "\n".join(
            f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(item['created_at']))}  {item['name']}  {item['path']}"
            for item in backups
        )

    def tacit_lines(self, *, panel: str = "recent", item_id: Optional[str] = None) -> list[str]:
        if not self.settings.effective().tacit.ui_enabled:
            return ["tacit UI is disabled by tacit.ui_enabled=false"]
        return self.tacit.render_lines(panel=panel, item_id=item_id)

    def self_lines(self, *, panel: str = "summary") -> list[str]:
        panel = (panel or "summary").strip().lower()
        model = self.self_model.model
        summary = self.self_model.summary()
        if panel == "summary":
            lines = [
                "G1 SELF MODEL",
                f"Robot ID             {model.robot_id}",
                f"Platform             {model.platform}",
                f"Version              {model.version}",
                f"Last updated         {model.updated_at}",
                f"Overall confidence   {summary.get('overall_confidence'):.2f}",
                "",
                "BODY",
            ]
            body_notes = summary.get("notable_body_facts") or []
            lines.extend(f"- {note}" for note in body_notes[:8])
            if not body_notes:
                lines.append("(no learned body traits)")
            lines.extend(["", "CAPABILITIES"])
            for skill, info in sorted((summary.get("skill_confidence") or {}).items()):
                lines.append(f"{skill:<20} reliability {float(info.get('success_rate') or 0):.2f} confidence {float(info.get('confidence') or 0):.2f}")
            if not summary.get("skill_confidence"):
                lines.append("(no skill experience yet)")
            lines.extend(["", "ENERGY"])
            energy = summary.get("energy") or {}
            lines.append(f"Model calibrated      {energy.get('calibrated')}")
            lines.append(f"Mean prediction error {energy.get('mean_prediction_error_pct')}")
            lines.extend(["", "PREFERENCES"])
            prefs = summary.get("important_learned_preferences") or []
            lines.extend(f"- {pref}" for pref in prefs[:8])
            if not prefs:
                lines.append("(none)")
            lines.extend(["", "COMMITMENTS"])
            commitments = summary.get("current_commitments") or []
            lines.extend(f"- {item}" for item in commitments[:8])
            if not commitments:
                lines.append("(none)")
            return lines
        if panel == "body":
            body = model.body
            lines = ["SELF / BODY", f"Confidence            {body.confidence:.2f}", f"Hardware revision     {body.hardware_revision or 'unknown'}", "", "Learned constraints"]
            lines.extend(f"- {item.status} {item.confidence:.2f}: {item.description}" for item in body.learned_constraints)
            return lines if len(lines) > 4 else lines + ["(none)"]
        if panel == "capabilities":
            lines = ["SELF / CAPABILITIES"]
            for name, estimate in sorted(model.capabilities.estimates.items()):
                lines.append(f"{name:<20} p={estimate.success_probability} confidence={estimate.confidence:.2f} status={estimate.status}")
                if estimate.failure_modes:
                    lines.append(f"  failures: {', '.join(estimate.failure_modes[:5])}")
            return lines if len(lines) > 1 else lines + ["(none)"]
        if panel == "skills":
            lines = ["SELF / SKILLS"]
            for name, record in sorted(model.skills.records.items()):
                lines.append(f"{name:<20} attempts={record.attempts} success={record.success_rate} confidence={record.confidence:.2f}")
                if record.active_procedures:
                    lines.append(f"  active procedures: {', '.join(record.active_procedures)}")
                if record.common_failure_modes:
                    lines.append(f"  failure modes: {', '.join(record.common_failure_modes[:5])}")
            return lines if len(lines) > 1 else lines + ["(none)"]
        if panel == "energy":
            energy = model.energy
            lines = [
                "SELF / ENERGY",
                f"Calibrated            {energy.calibrated}",
                f"Observations          {energy.observations}",
                f"Mean prediction error {energy.mean_prediction_error_pct}",
                f"Confidence            {energy.confidence:.2f}",
                "Task costs",
            ]
            lines.extend(f"- {task}: {cost:.2f}%" for task, cost in sorted(energy.task_cost_pct.items()))
            return lines
        if panel == "preferences":
            lines = ["SELF / PREFERENCES"]
            lines.extend(f"- {pref.status} {pref.confidence:.2f} {pref.domain}: {pref.preferred_option}" for pref in model.preferences.preferences)
            return lines if len(lines) > 1 else lines + ["(none)"]
        if panel == "commitments":
            lines = ["SELF / COMMITMENTS"]
            lines.extend(f"- P{item.priority} {item.state}: {item.description}" for item in model.commitments.commitments)
            return lines if len(lines) > 1 else lines + ["(none)"]
        if panel == "relationships":
            lines = ["SELF / RELATIONSHIPS"]
            lines.extend(f"- {item.label}: {item.preferred_name or 'unnamed'} trust={item.trust}" for item in model.relationships.records)
            return lines if len(lines) > 1 else lines + ["(none)"]
        if panel == "history":
            lines = ["SELF / HISTORY"]
            for item in model.history[-20:]:
                lines.append(f"v{item.version} <- v{item.previous_version} {item.timestamp} {','.join(item.domains_changed)}: {item.reason}")
            return lines if len(lines) > 1 else lines + ["(none)"]
        return ["usage: /self summary|body|capabilities|skills|energy|preferences|commitments|relationships|history|invalidate <target>"]

    def cmd_self(self, args: list[str]) -> str:
        if args and args[0] == "invalidate" and len(args) >= 2:
            target = " ".join(args[1:])
            self.self_model.invalidate(target, reason="operator hardware invalidation")
            return f"self-model invalidated learned entries related to {target!r}"
        panel = args[0] if args else "summary"
        return "\n".join(self.self_lines(panel=panel))

    # -- deterministic /settings /status /memory /tools namespaces ------------

    def cmd_settings(self, args: list[str]) -> str:
        if not args or args[0] == "show":
            flat = self.settings.as_flat_dict()
            lines = [f"{key} = {self._format_setting_value(value)}" for key, value in sorted(flat.items())]
            return "\n".join(lines) if lines else "(no settings)"
        if args[0] == "get" and len(args) == 2:
            try:
                return f"{args[1]} = {self._format_setting_value(self.settings.get(args[1]))}"
            except InvalidSettingError as exc:
                return f"error: {exc}"
        if args[0] == "set" and len(args) == 3:
            key, raw_value = args[1], args[2]
            value = self._coerce_setting_value(raw_value)
            try:
                self.settings.set(key, value)
            except InvalidSettingError as exc:
                return f"error: {exc}"
            return f"{key} = {self._format_setting_value(self.settings.get(key))}"
        if args[0] == "skill" and len(args) == 3:
            skill_name, mode = args[1], args[2]
            try:
                self.settings.set_skill_mode(skill_name, mode)
            except ValueError as exc:
                return f"error: invalid mode {mode!r}: {exc}"
            return f"skills.{skill_name} = {self.settings.get_skill_mode(skill_name).value}"
        if args[0] == "skills":
            modes = {name: self.settings.get_skill_mode(name).value for name in self.skills.names()}
            return "\n".join(f"{name}: {mode}" for name, mode in sorted(modes.items()))
        return "usage: /settings show | get <key> | set <key> <value> | skill <name> <auto|confirm|disabled> | skills"

    @staticmethod
    def _format_setting_value(value: Any) -> Any:
        return getattr(value, "value", value)

    @staticmethod
    def _coerce_setting_value(raw: str) -> Any:
        lowered = raw.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        return raw

    def cmd_status(self) -> str:
        robot_state = build_robot_state(self.state_source)
        now = time.time()
        settings = self.settings.effective()
        lines = [
            f"lifecycle_state = {self.lifecycle.state.value}",
            f"last_boot_event = {self._boot_event.value}",
            f"agent_uptime_s = {max(0.0, now - self._boot_time):.1f}",
            f"cognition_iterations = {self._cognition_count}",
            f"previous_cognitive_timestamp = {self.scheduler.last_cognitive_timestamp}",
            f"elapsed_since_last_cognition_s = {self.scheduler.elapsed_since_last_cognition(now)}",
            f"robot_state = {robot_state.model_dump()}",
            (
                "vision = "
                f"enabled={settings.vision.rgbd_enabled}, "
                f"rgbd=tcp://{settings.vision.rgbd_host}:{settings.vision.rgbd_port}, "
                f"topic={settings.vision.rgbd_topic!r}, "
                f"model={settings.vision.openai_model}"
            ),
            f"skills_backend = {self.skills.backend_label}",
        ]
        return "\n".join(lines)

    def cmd_faults(self) -> str:
        robot_state = build_robot_state(self.state_source)
        lines: list[str] = []
        if robot_state.active_faults:
            lines.append("Active faults affecting robot state:")
        else:
            lines.append("No active robot-state faults are currently reported.")
        for fault in robot_state.active_faults:
            lines.append(f"- {fault}: {self._fault_hint(fault)}")
        diagnostics = [fault for fault in robot_state.stale_sensor_topics if fault not in robot_state.active_faults]
        if diagnostics:
            lines.append("Stale watched-topic diagnostics:")
            for fault in diagnostics:
                lines.append(f"- {fault}: {self._fault_hint(fault)}")
        return "\n".join(lines)

    @staticmethod
    def _fault_hint(fault: str) -> str:
        if fault == "lidar_map":
            return "No fresh lidar map state. Check SLAM/lidar map publisher and DDS interface/domain."
        if fault.startswith("lidar_cloud"):
            return "No fresh point cloud on this lidar topic. Check the lidar driver/SLAM pipeline and topic name."
        if fault == "odom":
            return "No fresh odometry. Check /odommodestate (unitree_go/msg/SportModeState) and SLAM odometry publishers."
        if fault == "lowstate":
            return "No fresh /lowstate (unitree_hg/msg/LowState). Check ROS/DDS interface/domain and Unitree lowstate publication before lowstate-dependent actions."
        if fault == "left_hand_state":
            return "No fresh rt/dex3/left/state. Check Dex3 left hand power, topic publisher, and hand SDK connection."
        if fault == "right_hand_state":
            return "No fresh rt/dex3/right/state. Check Dex3 right hand power, topic publisher, and hand SDK connection."
        if fault == "lidar_imu":
            return "No fresh /utlidar/imu_livox_mid360. Check lidar IMU publisher."
        if fault == "slam_odom":
            return "No fresh SLAM odometry. Start/check the SLAM odom publisher."
        return "No specific hint known; inspect robot_state.sensor_timestamps and DDS topic publication."

    def cmd_memory(self, args: list[str]) -> str:
        if args and args[0] == "pin" and len(args) == 2:
            self.learning.pin(args[1])
            return f"pinned memory {args[1]}"
        if args and args[0] == "unpin" and len(args) == 2:
            self.learning.unpin(args[1])
            return f"unpinned memory {args[1]}"
        if not args or args[0] == "show":
            episodes = self.memory.episodic.all()
            claims = self.memory.semantic.all()
            adaptations = self.memory.procedural.all()
            learned = self.learning.learned.all()
            bio = self.memory.autobiography_summary() or "(empty)"
            return (
                f"episodic: {len(episodes)} entries\n"
                f"semantic: {len(claims)} claims\n"
                f"procedural: {len(adaptations)} adaptations\n"
                f"learned_semantic: {len(learned)} claims\n"
                f"pinned: {len(self.learning.pinned_ids())}\n"
                f"autobiography:\n{bio}"
            )
        if args[0] == "search" and len(args) >= 2:
            query = " ".join(args[1:])
            refs = self.memory.retrieve(query)
            lines = []
            for kind, items in refs.items():
                lines.append(f"-- {kind} --")
                lines.extend(f"  {ref.text}" for ref in items)
            return "\n".join(lines) if lines else "(no matches)"
        return "usage: /memory show | search <query> | pin <id> | unpin <id>"

    def cmd_tools(self) -> str:
        descriptions = self.skills.describe()
        lines = []
        for name in sorted(descriptions):
            mode = self.settings.get_skill_mode(name).value
            lines.append(f"{name} [{mode}]: {descriptions[name]}")
        return "\n".join(lines)

    def cmd_tooling(self, args: list[str]) -> str:
        profile = args[1] if len(args) > 1 else "social"
        if args and args[0] == "mcp":
            snapshot = self.tool_snapshot(profile="diagnostic")
            lines = ["MCP"]
            for name, health in sorted((snapshot.get("mcp") or {}).items()):
                lines.append(
                    f"{name:<16} connected={health.get('connected')} tools={health.get('available_tools')} "
                    f"last_error={health.get('last_error') or 'none'}"
                )
            return "\n".join(lines)
        if args and args[0] == "show" and len(args) >= 2:
            snapshot = self.tool_snapshot(profile="diagnostic")
            for row in snapshot.get("tools") or []:
                if row.get("name") == args[1]:
                    return "\n".join(f"{key} = {value}" for key, value in sorted(row.items()))
            return f"tool {args[1]!r} not found"
        if args and args[0] in {"available", "actions", "diagnostics", "memory", "knowledge", "observation"}:
            snapshot = self.tool_snapshot(profile=profile)
            rows = snapshot.get("tools") or []
            if args[0] == "available":
                rows = [row for row in rows if row.get("availability") == "available"]
            elif args[0] == "actions":
                rows = [row for row in rows if row.get("category") == "action"]
            elif args[0] == "diagnostics":
                rows = [row for row in rows if row.get("category") == "diagnostic"]
            else:
                rows = [row for row in rows if row.get("category") == args[0]]
            return self._format_tool_rows(rows)
        snapshot = self.tool_snapshot(profile="social")
        return self._format_tool_rows(snapshot.get("tools") or [])

    @staticmethod
    def _format_tool_rows(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "(no tools)"
        lines = ["TOOLS"]
        current_category = None
        for row in sorted(rows, key=lambda r: (r.get("category", ""), r.get("name", ""))):
            category = row.get("category", "")
            if category != current_category:
                current_category = category
                lines.append("")
                lines.append(str(category).upper())
            mark = "✓" if row.get("availability") == "available" else "✗"
            suffix = "" if row.get("availability") == "available" else f"  {row.get('availability')}"
            if row.get("operator_only"):
                suffix = "  operator_only"
            lines.append(f"{mark} {row.get('name'):<32} {row.get('risk_level'):<6}{suffix}")
        return "\n".join(lines)

    def cmd_help(self) -> str:
        language_value = self.settings.get("interface.command_language")
        language = getattr(language_value, "value", str(language_value))
        english = (
            "/chat <text>                converse (always active)\n"
            "/audio_msg <text>           simulated ASR transcript (gated by audio.asr_enabled)\n"
            "/settings show|get|set|skill|skills\n"
            "/settings-ui                interactive settings editor (arrow keys)\n"
            "/monitor [panel]            live read-only agent monitor\n"
            "/tacit [panel]              read-only learned/tacit knowledge view\n"
            "/self [panel]               read-only persistent functional self-model\n"
            "/reset [scope|backups]      reset cognitive continuity by explicit scope\n"
            "/navigation [panel|action]   SLAM/navigation monitor and supported actions\n"
            "/asr                         microphone/ASR monitor\n"
            "/llctl [enable|disable]      operator low-level-control panel\n"
            "/llctl-ui                    interactive low-level-control TUI\n"
            "/vision <question>          answer from RGB-D camera via OpenAI vision\n"
            "/status                     lifecycle + robot state snapshot\n"
            "/faults                     explain stale sensor topics and fixes\n"
            "/memory show|search <query>|pin <id>|unpin <id>\n"
            "/tools                      list skills and their auto/confirm/disabled mode\n"
            "/tools available|mcp|show   inspect cognitive tool/MCP availability\n"
            "/help\n"
            "/exit"
        )
        german = (
            "/chat <text>                Unterhaltung (immer aktiv)\n"
            "/audio_msg <text>           simuliertes ASR-Transkript\n"
            "/einstellungen              Einstellungen anzeigen\n"
            "/einstellungen-ui           interaktiver Einstellungseditor (Pfeiltasten)\n"
            "/tacit [panel]              gelernte/tazite Wissensansicht\n"
            "/self [panel]               funktionales Selbstmodell anzeigen\n"
            "/reset [scope|backups]      kognitive Kontinuität nach Bereich zurücksetzen\n"
            "/monitor [bereich]          Live-Monitor des Agenten (nur lesen)\n"
            "/navigation                 SLAM-/Navigationsmonitor\n"
            "/asr                         Mikrofon-/ASR-Monitor\n"
            "/llctl [enable|disable]      Bediener-Panel für Low-Level-Control\n"
            "/llctl-ui                    interaktive Low-Level-Control-TUI\n"
            "/sehen <frage>              Antwort von der RGB-D-Kamera über OpenAI Vision\n"
            "/status                     Lebenszyklus und Roboterzustand\n"
            "/fehler                     Sensor-/Topic-Diagnose und Hinweise\n"
            "/speicher                   Gedächtnis anzeigen\n"
            "/werkzeuge                  verfügbare Fähigkeiten und Modi anzeigen\n"
            "/werkzeuge available|mcp|show kognitive Tool-/MCP-Verfügbarkeit anzeigen\n"
            "/hilfe\n"
            "/ende"
        )
        if str(language) == "de":
            return german
        if str(language) == "both":
            return english + "\n\nDeutsch:\n" + german
        return english
