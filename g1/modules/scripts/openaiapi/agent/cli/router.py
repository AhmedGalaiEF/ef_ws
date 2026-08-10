"""``G1Agent`` orchestrator + the deterministic ``/settings`` CLI namespace
(spec sections 5, 14, 16, 17, 31).

``/settings`` (and ``/status``, ``/memory``, ``/tools``, ``/help``) never
touch the planner -- they are plain deterministic code, per spec section
14. ``/chat`` and ``/audio_msg`` are the two paths that construct a
``PlannerInput`` and call ``planner.decide()``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from ..announcements import announce
from ..capabilities import CapabilityResolver, PolicyDecision
from ..checkpoint import CheckpointStore
from ..lifecycle import LifecycleController, classify_startup
from ..memory.manager import MemoryManager, MemoryProposalError
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
from ..planner import Planner
from ..scheduler import CognitiveScheduler
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
        self.scheduler = CognitiveScheduler()
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

    @property
    def boot_event(self) -> EventType:
        return self._boot_event

    # -- boot -------------------------------------------------------------

    def boot(self) -> PlannerDecision:
        """Run the first cognitive turn: agent_first_boot / _restart / _wake."""
        now = time.time()
        robot_state = build_robot_state(self.state_source)
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

        planner_input = self._build_planner_input(
            event=self._boot_event,
            timestamp=now,
            user_text=None,
            input_source="system",
            robot_state=robot_state,
            runtime=runtime,
        )
        decision = self.planner.decide(planner_input)
        # Reach AWAKE before persisting the checkpoint, so a boot turn's
        # checkpoint always records "awake", not a transitional state.
        self.lifecycle.transition(LifecycleState.AWAKE)
        self._after_turn(planner_input, decision)
        self._record_boot_memory(robot_state)
        self._booted = True
        return decision

    # -- /chat and /audio_msg ----------------------------------------------

    def handle_chat(self, text: str) -> TurnOutcome:
        """``/chat`` -- always active, independent of every audio setting."""
        return self._handle_user_text(text, input_source="chat")

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
        if not self.settings.effective().audio.asr_enabled:
            return None
        return self._handle_user_text(text, input_source="audio")

    def handle_cognitive_tick(self) -> TurnOutcome:
        """Run one periodic mostly-idle cognition turn."""
        now = time.time()
        robot_state = build_robot_state(self.state_source)
        planner_input = self._build_planner_input(
            event=EventType.COGNITIVE_TICK,
            timestamp=now,
            user_text=None,
            input_source="system",
            robot_state=robot_state,
            runtime={"reason": "periodic_tick"},
        )
        decision = self.planner.decide(planner_input)
        self._after_turn(planner_input, decision, announce_maintenance=False)
        return self._execute_decision(decision, planner_input)

    def _handle_user_text(self, text: str, *, input_source: str) -> TurnOutcome:
        now = time.time()
        event = EventType.ASR_MESSAGE if input_source == "audio" else EventType.USER_MESSAGE
        robot_state = build_robot_state(self.state_source)
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
        planner_input = self._build_planner_input(
            event=event, timestamp=now, user_text=text, input_source=input_source, robot_state=robot_state
        )
        decision = self.planner.decide(planner_input)
        decision = self._normalize_decision(decision, planner_input)
        decision = self._apply_command_fallbacks(decision, planner_input)
        self._after_turn(planner_input, decision)
        return self._execute_decision(decision, planner_input)

    def _maybe_handle_memory_query(self, text: str, robot_state: RobotStateSnapshot) -> Optional[str]:
        lowered = text.strip().lower()
        if not any(term in lowered for term in ("remember", "memory", "erinner", "gedächtnis", "gedaechtnis")):
            return None
        bio = self.memory.autobiography_summary() or "(empty)"
        reply_language = getattr(self.settings.effective().interface.reply_language, "value", "auto")
        if reply_language == "de":
            return (
                "Mein Gedächtnissystem ist verfügbar. Aktuelles autobiografisches Gedächtnis:\n"
                f"{bio}\n"
                f"Aktueller Zustand: Haltung={robot_state.posture}, "
                f"Stabilität={robot_state.stability}, aktive_Fehler={robot_state.active_faults or ['keine']}. "
                "Veraltete Hand-State-Topics betreffen nur das Greifer-Feedback; sie beeinflussen mein Gedächtnis nicht."
            )
        return (
            "My memory system is available. Current autobiographical memory:\n"
            f"{bio}\n"
            f"Current state summary: posture={robot_state.posture}, "
            f"stability={robot_state.stability}, active_faults={robot_state.active_faults or ['none']}. "
            "Stale hand-state topics affect gripper feedback only; they do not affect memory."
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
        reply_language = getattr(settings.interface.reply_language, "value", "auto")
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
            return self._vision_answerer.answer(
                settings=settings.vision,
                question=question,
                reply_language=str(reply_language),
            )
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
    ) -> PlannerInput:
        settings = self.settings.effective()
        runtime_payload = dict(runtime or {})
        runtime_payload.setdefault("reply_language", getattr(settings.interface.reply_language, "value", settings.interface.reply_language))
        query = user_text or ""

        memory_refs = self.memory.retrieve(query) if query else {"episodic": [], "semantic": [], "procedural": []}

        sdk_refs = []
        if self.sdk_knowledge is not None:
            try:
                sdk_refs = self.sdk_knowledge.search(query)
            except Exception:
                sdk_refs = []

        doc_refs = []
        if self.document_rag is not None and query:
            try:
                doc_refs = self.document_rag.search(query)
            except Exception:
                doc_refs = []

        arm_policy = self.resolver.resolve_arm_motion(settings=settings, robot_state=robot_state)
        capability_summary = {
            "arm_motion": CapabilityStatus(available=arm_policy.allowed, reason=arm_policy.reason),
        }

        return PlannerInput(
            event=event,
            timestamp=timestamp,
            previous_cognitive_timestamp=self.scheduler.last_cognitive_timestamp,
            elapsed_since_last_cognition_s=self.scheduler.elapsed_since_last_cognition(timestamp),
            input_source=input_source,
            user_text=user_text,
            robot_state=robot_state,
            lifecycle_state=self.lifecycle.state,
            available_skills=self.skills.names(),
            capability_summary=capability_summary,
            settings=settings.as_flat_dict(),
            documentary_rag=doc_refs,
            sdk_wrapper_knowledge=sdk_refs,
            episodic_memory=memory_refs["episodic"],
            semantic_memory=memory_refs["semantic"],
            procedural_memory=memory_refs["procedural"],
            autobiography_summary=self.memory.autobiography_summary(),
            available_tools=[],
            runtime=runtime_payload,
        )

    # -- post-turn bookkeeping ----------------------------------------------

    def _after_turn(
        self,
        planner_input: PlannerInput,
        decision: PlannerDecision,
        *,
        announce_maintenance: bool = True,
    ) -> None:
        self.scheduler.record_cognition(planner_input.timestamp, decision.next_tick_s)
        checkpoint = RuntimeCheckpoint(
            last_cognitive_timestamp=planner_input.timestamp,
            lifecycle_state=self.lifecycle.state,
            last_event_type=planner_input.event,
            last_decision=decision.intent,
            active_skill=(decision.requested_skills[0] if decision.requested_skills else None),
            last_robot_state_summary=planner_input.robot_state.model_dump(),
        )
        self.checkpoint_store.save(checkpoint)
        self._previous_checkpoint = checkpoint

        if decision.memory_proposal is not None:
            try:
                self.memory.apply_proposal(decision.memory_proposal)
            except MemoryProposalError as exc:
                print(f"[memory] rejected proposal: {exc}")
        if decision.maintenance_proposal is not None and announce_maintenance:
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

        if decision.intent == IntentType.QUERY_CAPABILITY and (decision.target or "").strip().lower() == "arm":
            policy = self.resolver.resolve_arm_motion(settings=settings, robot_state=robot_state)
            outcome.grounded_response = self._phrase_capability_answer(policy)
            self._speak_grounded_response(outcome.grounded_response, settings=settings, robot_state=robot_state)
            return outcome

        if decision.intent == IntentType.QUERY_STATE:
            target = (decision.target or "").strip().lower()
            if "battery" in target or "charge" in target:
                outcome.grounded_response = self._describe_battery(robot_state)
            else:
                outcome.grounded_response = decision.response_text or self._describe_state(robot_state)
            self._speak_grounded_response(outcome.grounded_response, settings=settings, robot_state=robot_state)
            return outcome

        if decision.intent in (IntentType.NO_ACTION, IntentType.CONVERSATION, IntentType.MAINTENANCE):
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

        if decision.intent == IntentType.REQUEST_SLEEP and any(
            o.status == "executed" and o.result and o.result.ok for _, o in outcome.skill_outcomes
        ):
            self._enter_sleep(decision)

        return outcome

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
            return {"arm": arm}
        if skill_name != "grab":
            return {}
        prompt = decision.target or self._extract_grab_prompt(planner_input.user_text or "")
        if prompt == "grab":
            prompt = self._extract_grab_prompt(planner_input.user_text or "")
        return {"prompt": prompt, "agent_settings": settings}

    def _apply_command_fallbacks(
        self, decision: PlannerDecision, planner_input: PlannerInput
    ) -> PlannerDecision:
        """Deterministic recovery for simple robot commands the planner missed."""
        if decision.requested_skills:
            return decision
        if planner_input.event not in (EventType.USER_MESSAGE, EventType.ASR_MESSAGE):
            return decision
        text = (planner_input.user_text or "").strip().lower()
        wants_extend_arm = (
            "extend arm" in text
            or "extend right arm" in text
            or "extend left arm" in text
            or "arm forward" in text
        )
        if wants_extend_arm and "reach_forward" in self.skills.skills:
            side = "left" if "left" in text else "right"
            german = getattr(self.settings.effective().interface.reply_language, "value", "auto") == "de"
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
            german = getattr(self.settings.effective().interface.reply_language, "value", "auto") == "de"
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="release_arms",
                response_text="Ich gebe die Armsteuerung frei." if german else "I'll release arm control authority.",
                requested_skills=["release_arms"],
            )
        wants_grab = any(word in text for word in ("grab", "grba", "grasp", "pick up", "greif", "greife", "nimm"))
        if wants_grab and "grab" in self.skills.skills:
            german = getattr(self.settings.effective().interface.reply_language, "value", "auto") == "de"
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
        wants_wave = "wave" in text or "wink" in text
        if not wants_wave:
            return decision
        if "high_wave" in self.skills.skills and ("high" in text or "big" in text or "hoch" in text):
            german = getattr(self.settings.effective().interface.reply_language, "value", "auto") == "de"
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="high_wave",
                response_text="Ich versuche hoch zu winken." if german else "I'll try a high wave.",
                requested_skills=["high_wave"],
                intent_announcement=IntentAnnouncement(speech="Ich winke hoch." if german else "I'll wave high."),
            )
        if "wave" in self.skills.skills:
            german = getattr(self.settings.effective().interface.reply_language, "value", "auto") == "de"
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

        if (
            decision.intent == IntentType.MOVE_ARM
            or "extend arm" in text
            or "extend right arm" in text
            or "extend left arm" in text
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
        return decision

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
        return reply in ("y", "yes")

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
            language=settings.announcements.tts_language or None,
            voice_model=settings.announcements.tts_voice_model or None,
            speaker=settings.announcements.tts_speaker,
        )
        if not decision.allowed:
            print(f"[warn] speech denied: {decision.reason}")
        elif result is not None and not result.ok:
            print(f"[warn] speech failed: {result.message}")

    @staticmethod
    def _phrase_capability_answer(policy: PolicyDecision) -> str:
        prefix = "Yes." if policy.allowed else "No."
        return f"{prefix} {policy.reason}"

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
            return "Battery percentage is unavailable: rt/lowstate.bms_state.soc is not present in the current robot state."
        charging = ""
        if state.charging is not None:
            charging = " and appears to be charging" if state.charging else " and does not appear to be charging"
        return f"Battery is at {state.battery_pct:.0f}%{charging}."

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
            return "No fresh rt/utlidar/map_state. Start/check the lidar or SLAM mapping publisher; verify DDS iface/domain."
        if fault.startswith("lidar_cloud"):
            return "No fresh point cloud on this lidar topic. Check the lidar driver/SLAM pipeline and topic name."
        if fault == "odom":
            return "No fresh rt/odom. Check odometry publisher or use the sport-state odom fallback if navigation does not require rt/odom."
        if fault == "lowstate":
            return "No fresh rt/lowstate. Check DDS interface/domain and Unitree lowstate publication before arm/lowstate-dependent actions."
        if fault == "left_hand_state":
            return "No fresh rt/dex3/left/state. Check Dex3 left hand power, topic publisher, and hand SDK connection."
        if fault == "right_hand_state":
            return "No fresh rt/dex3/right/state. Check Dex3 right hand power, topic publisher, and hand SDK connection."
        if fault == "lidar_imu":
            return "No fresh rt/utlidar/imu_livox_mid360. Check lidar IMU publisher."
        if fault == "slam_odom":
            return "No fresh SLAM odometry. Start/check the SLAM odom publisher."
        return "No specific hint known; inspect robot_state.sensor_timestamps and DDS topic publication."

    def cmd_memory(self, args: list[str]) -> str:
        if not args or args[0] == "show":
            episodes = self.memory.episodic.all()
            claims = self.memory.semantic.all()
            adaptations = self.memory.procedural.all()
            bio = self.memory.autobiography_summary() or "(empty)"
            return (
                f"episodic: {len(episodes)} entries\n"
                f"semantic: {len(claims)} claims\n"
                f"procedural: {len(adaptations)} adaptations\n"
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
        return "usage: /memory show | search <query>"

    def cmd_tools(self) -> str:
        descriptions = self.skills.describe()
        lines = []
        for name in sorted(descriptions):
            mode = self.settings.get_skill_mode(name).value
            lines.append(f"{name} [{mode}]: {descriptions[name]}")
        return "\n".join(lines)

    def cmd_help(self) -> str:
        language_value = self.settings.get("interface.command_language")
        language = getattr(language_value, "value", str(language_value))
        english = (
            "/chat <text>                converse (always active)\n"
            "/audio_msg <text>           simulated ASR transcript (gated by audio.asr_enabled)\n"
            "/settings show|get|set|skill|skills\n"
            "/settings-ui                interactive settings editor (arrow keys)\n"
            "/vision <question>          answer from RGB-D camera via OpenAI vision\n"
            "/status                     lifecycle + robot state snapshot\n"
            "/faults                     explain stale sensor topics and fixes\n"
            "/memory show|search <query>\n"
            "/tools                      list skills and their auto/confirm/disabled mode\n"
            "/help\n"
            "/exit"
        )
        german = (
            "/chat <text>                Unterhaltung (immer aktiv)\n"
            "/audio_msg <text>           simuliertes ASR-Transkript\n"
            "/einstellungen              Einstellungen anzeigen\n"
            "/einstellungen-ui           interaktiver Einstellungseditor (Pfeiltasten)\n"
            "/sehen <frage>              Antwort von der RGB-D-Kamera über OpenAI Vision\n"
            "/status                     Lebenszyklus und Roboterzustand\n"
            "/fehler                     Sensor-/Topic-Diagnose und Hinweise\n"
            "/speicher                   Gedächtnis anzeigen\n"
            "/werkzeuge                  verfügbare Fähigkeiten und Modi anzeigen\n"
            "/hilfe\n"
            "/ende"
        )
        if str(language) == "de":
            return german
        if str(language) == "both":
            return english + "\n\nDeutsch:\n" + german
        return english
