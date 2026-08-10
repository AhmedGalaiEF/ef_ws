"""Validated skill registry (spec sections 2, 18, 27).

Wraps -- does not reimplement -- the deterministic skill surfaces that
already exist in the repo:

  * ``llm_client/robot_tools.py: build_robot_tools(robot)`` -- move,
    reach_forward, grab, release. Only needs a live ``sdk_client.Robot``.
  * ``ollama_ai/scene_executor.py: STEP_HANDLERS`` -- announce, navigate,
    move, gesture, think_gesture, listen, rag_answer, vision_detect,
    grasp (upstream stub), hand_open, hand_close, release_arms, stop.
    These need a fully composed ``scene_executor.SceneContext`` (its own
    ``NavState``/``Speaker``/``MotionPlayer``/``KnowledgeRetriever``, each
    bound to real ROS/DDS resources) -- Phase 1 accepts one via dependency
    injection (``scene_ctx=``) if the caller has built one, rather than
    fabricating a fake context here that couldn't be verified against the
    real contract.
  * ``ai_control/robot_backend.py: MockRobotBackend`` -- the existing
    no-hardware stand-in, reused as-is for the offline registry so tests
    and ``--no-robot`` CLI runs exercise the *same* planner -> capability
    -> skill dispatch path real hardware would.

Two skills are genuinely new (spec sections 9/10/29) and have no upstream
equivalent: ``request_sleep`` and ``request_charge``. Both are
deterministic and explicitly do **not** perform a real OS shutdown or
docking maneuver in this phase -- see their handlers below.

The planner (``agent/planner.py``) can only ever name a skill from this
registry's ``names()``; there is no code path anywhere in this package
from a planner decision to a raw q/dq/kp/kd/tau value or a ``/low_cmd``
packet (spec section 2).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .capabilities import CapabilityResolver, PolicyDecision
from .models import RobotStateSnapshot
from .settings.models import AgentSettings, SkillMode


@dataclass
class SkillResult:
    ok: bool
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Skill:
    name: str
    description: str
    handler: Callable[..., SkillResult]
    source: str  # provenance: which existing module this wraps, for auditability


@dataclass
class SkillRegistry:
    skills: dict[str, Skill]
    backend_label: str
    unavailable: list[str] = field(default_factory=list)

    def names(self) -> list[str]:
        return sorted(self.skills)

    def describe(self) -> dict[str, str]:
        return {name: skill.description for name, skill in self.skills.items()}

    def invoke(self, skill_name: str, **kwargs: Any) -> SkillResult:
        # NOTE: the dispatch parameter is named `skill_name`, not `name` --
        # several real skills (gesture, hand_open, ...) take their own
        # `name`/`hand` kwargs, and a `name` dispatch parameter here would
        # collide with a skill's `name=` kwarg (caught by the test suite:
        # invoking "gesture" with name="gesture" raised "multiple values
        # for argument 'name'" before this was renamed).
        skill = self.skills.get(skill_name)
        if skill is None:
            return SkillResult(ok=False, message=f"unknown skill '{skill_name}'")
        try:
            return skill.handler(**kwargs)
        except Exception as exc:
            return SkillResult(ok=False, message=f"skill '{skill_name}' raised: {exc}")


class SkillUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# sys.path bootstrap -- mirrors ollama_ai/scene_executor.py's own convention
# (this codebase uses flat, sys.path-injected module names throughout
# rather than dotted cross-package imports; see the Phase 1 plan's
# "Package layout" note for why this file follows the same pattern).
# ---------------------------------------------------------------------------

def _bootstrap_repo_paths() -> dict[str, Path]:
    here = Path(__file__).resolve()
    ef_ws_root = next(
        (parent for parent in here.parents if (parent / "dev" / "ai_control").exists()),
        here.parents[4],
    )
    modules_dir = next(
        (parent / "g1" / "modules" for parent in (ef_ws_root, *ef_ws_root.parents) if (parent / "g1" / "modules").exists()),
        ef_ws_root / "g1" / "modules",
    )
    scripts_dir = modules_dir / "scripts"
    g1_dir = modules_dir.parent
    ollama_ai_dir = scripts_dir / "ollama_ai"
    wbc_dir = g1_dir / "WBC"

    paths = {
        "ef_ws_root": ef_ws_root,
        "modules_dir": modules_dir,
        "scripts_dir": scripts_dir,
        "ollama_ai_dir": ollama_ai_dir,
        "wbc_dir": wbc_dir,
    }
    for path in paths.values():
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return paths


# ---------------------------------------------------------------------------
# New skills with no upstream equivalent (spec sections 9, 10, 29).
# ---------------------------------------------------------------------------

def _skill_request_sleep(**kwargs: Any) -> SkillResult:
    reason = str(kwargs.get("reason", "unspecified"))
    return SkillResult(
        ok=True,
        message=(
            f"Deliberate-sleep sequence validated for reason={reason!r}. "
            "The lifecycle controller will write the runtime checkpoint with "
            "lifecycle_state=sleeping. Phase 1 does NOT issue a real OS shutdown "
            "command -- see agent/lifecycle.py and the plan's deferred-TODO list "
            "for wiring an actual validated shutdown on the target Jetson."
        ),
        detail={"reason": reason, "shutdown_issued": False},
    )


def _skill_request_charge(**kwargs: Any) -> SkillResult:
    return SkillResult(
        ok=False,
        message=(
            "request_charge is a deterministic stub in this phase: no charging-dock "
            "detection, navigation target, or charge-state verification exists yet "
            "anywhere in this repo to act on."
        ),
        detail={"stub": True},
    )


def _new_skills() -> dict[str, Skill]:
    return {
        "request_sleep": Skill(
            name="request_sleep",
            description="Validate and record a deliberate-sleep request (does not power off hardware in this phase).",
            handler=_skill_request_sleep,
            source="agent.skills (new, spec sections 9/10)",
        ),
        "request_charge": Skill(
            name="request_charge",
            description="Request docking/charging. Deterministic stub -- always denies, no dock integration exists yet.",
            handler=_skill_request_charge,
            source="agent.skills (new, spec section 29)",
        ),
    }


# ---------------------------------------------------------------------------
# Offline registry: ai_control.robot_backend.MockRobotBackend, reused as-is.
# ---------------------------------------------------------------------------

def build_offline_registry() -> SkillRegistry:
    """No hardware, no Unitree SDK required. Backs tests and the CLI's default mode."""
    _bootstrap_repo_paths()
    # dev/ai_control has no top-level package name of its own on sys.path;
    # it's reached as dev.ai_control, same as tests/test_topic_csv_monitor.py's
    # `from dev.topic_csv_monitor import ...` -- `dev` is an implicit
    # namespace package rooted at ef_ws_root, which _bootstrap_repo_paths
    # already added to sys.path.
    from dev.ai_control.robot_backend import MockRobotBackend  # pure stdlib, no SDK needed

    backend = MockRobotBackend()

    def _wrap(fn: Callable[..., str], name: str) -> Callable[..., SkillResult]:
        def _handler(**kwargs: Any) -> SkillResult:
            message = fn(**kwargs)
            return SkillResult(ok=True, message=message, detail={"backend": "mock", "skill": name})

        return _handler

    skills: dict[str, Skill] = {
        "move": Skill("move", "Drive the base for a duration at (vx, vy, vyaw).", _wrap(backend.move, "move"), "ai_control.robot_backend.MockRobotBackend"),
        "step_back": Skill("step_back", "Step backward a short distance.", lambda **_kwargs: SkillResult(ok=True, message=backend.move(vx=-0.25, vy=0.0, vyaw=0.0, duration=1.0), detail={"backend": "mock", "skill": "step_back"}), "ai_control.robot_backend.MockRobotBackend"),
        "turn_left": Skill("turn_left", "Turn left in place by a small angle.", lambda **_kwargs: SkillResult(ok=True, message=backend.move(vx=0.0, vy=0.0, vyaw=0.5, duration=1.0), detail={"backend": "mock", "skill": "turn_left"}), "ai_control.robot_backend.MockRobotBackend"),
        "turn_right": Skill("turn_right", "Turn right in place by a small angle.", lambda **_kwargs: SkillResult(ok=True, message=backend.move(vx=0.0, vy=0.0, vyaw=-0.5, duration=1.0), detail={"backend": "mock", "skill": "turn_right"}), "ai_control.robot_backend.MockRobotBackend"),
        "navigate_to": Skill("navigate_to", "Navigate to an (x, y, yaw) pose.", _wrap(backend.navigate_to, "navigate_to"), "ai_control.robot_backend.MockRobotBackend"),
        "stop": Skill("stop", "Stop all base motion.", _wrap(backend.stop, "stop"), "ai_control.robot_backend.MockRobotBackend"),
        "hand_open": Skill("hand_open", "Open the given hand.", _wrap(backend.hand_open, "hand_open"), "ai_control.robot_backend.MockRobotBackend"),
        "hand_close": Skill("hand_close", "Close the given hand.", _wrap(backend.hand_close, "hand_close"), "ai_control.robot_backend.MockRobotBackend"),
        "reach_forward": Skill(
            "reach_forward",
            "Extend the chosen arm forward.",
            lambda **kwargs: SkillResult(
                ok=True,
                message=f"mock reached {str(kwargs.get('arm') or 'right')} arm forward",
                detail={"backend": "mock", "skill": "reach_forward"},
            ),
            "ai_control.robot_backend.MockRobotBackend",
        ),
        "grab": Skill(
            "grab",
            "Prompt-based RGB-D grab placeholder; live mode uses OpenAI vision + hand_pose_navigation IK.",
            lambda **kwargs: SkillResult(
                ok=False,
                message=(
                    "prompt-based grab requires --robot with live RGB-D input and OpenAI vision; "
                    f"requested prompt={str(kwargs.get('prompt') or 'object')!r}"
                ),
                detail={"backend": "mock", "skill": "grab"},
            ),
            "ai_control.robot_backend.MockRobotBackend",
        ),
        "gesture": Skill("gesture", "Play a named high-level arm gesture.", _wrap(backend.gesture, "gesture"), "ai_control.robot_backend.MockRobotBackend"),
        "wave": Skill("wave", "Wave using the face-wave high-level arm gesture.", lambda **_kwargs: SkillResult(ok=True, message=backend.gesture("face wave"), detail={"backend": "mock", "skill": "wave"}), "ai_control.robot_backend.MockRobotBackend"),
        "high_wave": Skill("high_wave", "Wave high using the high-wave high-level arm gesture.", lambda **_kwargs: SkillResult(ok=True, message=backend.gesture("high wave"), detail={"backend": "mock", "skill": "high_wave"}), "ai_control.robot_backend.MockRobotBackend"),
        "release_arms": Skill("release_arms", "Release arm control authority.", _wrap(backend.release_arms, "release_arms"), "ai_control.robot_backend.MockRobotBackend"),
        "announce": Skill("announce", "Speak text through the robot's speaker.", _wrap(backend.say, "announce"), "ai_control.robot_backend.MockRobotBackend"),
    }
    skills.update(_new_skills())
    return SkillRegistry(skills, backend_label="offline (ai_control.robot_backend.MockRobotBackend)")


# ---------------------------------------------------------------------------
# Live registry: real llm_client.robot_tools + optional scene_executor steps.
# ---------------------------------------------------------------------------

def _wrap_tool(fn: Callable[..., str], name: str) -> Callable[..., SkillResult]:
    def _handler(**kwargs: Any) -> SkillResult:
        message = fn(**kwargs)
        ok = not str(message).startswith("error:")
        return SkillResult(ok=ok, message=str(message), detail={"backend": "llm_client.robot_tools", "skill": name})

    return _handler


def _wrap_step(step_fn: Callable[[Any, dict], str], ctx: Any, step_type: str) -> Callable[..., SkillResult]:
    def _handler(**kwargs: Any) -> SkillResult:
        step = {"type": step_type, **kwargs}
        try:
            message = step_fn(ctx, step)
        except Exception as exc:  # scene_executor.StepError or anything unexpected
            return SkillResult(ok=False, message=str(exc), detail={"backend": "scene_executor", "skill": step_type})
        return SkillResult(ok=True, message=message, detail={"backend": "scene_executor", "skill": step_type})

    return _handler


def build_live_registry(
    *,
    robot: Optional[Any] = None,
    scene_ctx: Optional[Any] = None,
    iface: str = "eth0",
    domain_id: int = 0,
) -> SkillRegistry:
    """Bind real skills. Requires the Unitree SDK2 Python stack and, for
    ``robot``, a live DDS connection -- neither is available in this dev
    sandbox, so this path is only exercised on the actual deployment
    target. Pass ``robot`` (a ``sdk_client.Robot`` instance) for the
    ``llm_client.robot_tools`` skills, and/or a pre-built
    ``scene_executor.SceneContext`` as ``scene_ctx`` for the
    ``STEP_HANDLERS`` skills; at least one is required.
    """
    _bootstrap_repo_paths()
    skills: dict[str, Skill] = {}
    unavailable: list[str] = []

    if robot is not None:
        try:
            from llm_client.robot_tools import build_robot_tools
        except Exception as exc:
            unavailable.append(f"llm_client.robot_tools unavailable: {exc}")
        else:
            tools, _schemas = build_robot_tools(robot)
            for tool_name, fn in tools.items():
                skills[tool_name] = Skill(
                    name=tool_name,
                    description=f"llm_client robot tool '{tool_name}' (see robot_tools.py for its JSON schema).",
                    handler=_wrap_tool(fn, tool_name),
                    source="llm_client.robot_tools.build_robot_tools",
                )
            try:
                from agent.vision_grab import OpenAIVisionGrabber
            except Exception as exc:
                unavailable.append(f"prompt vision grab unavailable: {exc}")
            else:
                vision_grabber = OpenAIVisionGrabber(robot=robot, iface=iface, domain_id=domain_id)

                def _vision_grab(**kwargs: Any) -> SkillResult:
                    settings = kwargs.get("agent_settings")
                    if settings is None or not hasattr(settings, "vision"):
                        return SkillResult(ok=False, message="vision settings were not provided to grab skill")
                    prompt = str(kwargs.get("prompt") or kwargs.get("object") or kwargs.get("target") or "object")
                    arm = str(kwargs.get("arm") or "auto")
                    message = vision_grabber.grab(settings=settings.vision, prompt=prompt, arm=arm)
                    return SkillResult(
                        ok=True,
                        message=message,
                        detail={
                            "backend": "agent.vision_grab.OpenAIVisionGrabber",
                            "skill": "grab",
                            "prompt": prompt,
                            "arm": arm,
                        },
                    )

                skills["grab"] = Skill(
                    name="grab",
                    description=(
                        "Prompt-based RGB-D grab: localize an object with OpenAI vision, "
                        "then move the end effector toward it with hand_pose_navigation IK."
                    ),
                    handler=_vision_grab,
                    source="agent.vision_grab.OpenAIVisionGrabber",
                )
            if "move" in tools:
                skills["step_back"] = Skill(
                    name="step_back",
                    description="Step backward a short distance using llm_client.robot_tools.move().",
                    handler=lambda **_kwargs: _wrap_tool(tools["move"], "step_back")(
                        direction="backward", distance_m=0.3, speed_mps=0.2
                    ),
                    source="llm_client.robot_tools.build_robot_tools",
                )
            if hasattr(robot, "move_for"):
                def _turn(direction: str) -> SkillResult:
                    side = str(direction).strip().lower()
                    if side not in {"left", "right"}:
                        return SkillResult(ok=False, message=f"turn direction must be left or right, got {direction!r}")
                    vyaw = 0.5 if side == "left" else -0.5
                    duration_s = 1.0
                    robot.move_for(duration_s, vx=0.0, vy=0.0, vyaw=vyaw)
                    return SkillResult(
                        ok=True,
                        message=f"turned {side} in place for {duration_s:.1f}s at vyaw={vyaw:+.2f}rad/s",
                        detail={"backend": "sdk_client.Robot.move_for", "skill": f"turn_{side}", "vyaw": vyaw},
                    )

                skills["turn_left"] = Skill(
                    name="turn_left",
                    description="Turn left in place using sdk_client.Robot.move_for().",
                    handler=lambda **_kwargs: _turn("left"),
                    source="sdk_client.Robot.move_for",
                )
                skills["turn_right"] = Skill(
                    name="turn_right",
                    description="Turn right in place using sdk_client.Robot.move_for().",
                    handler=lambda **_kwargs: _turn("right"),
                    source="sdk_client.Robot.move_for",
                )
        if hasattr(robot, "say"):
            def _announce(**kwargs: Any) -> SkillResult:
                text = str(kwargs.get("text", "")).strip()
                if not text:
                    return SkillResult(ok=False, message="no announcement text provided")
                language = kwargs.get("language") or None
                voice_model = kwargs.get("voice_model") or None
                speaker_raw = kwargs.get("speaker")
                speaker = None
                if speaker_raw is not None:
                    speaker_value = int(speaker_raw)
                    speaker = speaker_value if speaker_value >= 0 else None
                code = robot.say(text, language=language, voice_model=voice_model, speaker=speaker)
                return SkillResult(
                    ok=True,
                    message=f"spoke through sdk_client.Robot.say() -> code {code}",
                    detail={
                        "backend": "sdk_client.Robot.say",
                        "skill": "announce",
                        "code": code,
                        "language": language,
                        "voice_model": voice_model,
                        "speaker": speaker,
                    },
                )

            skills["announce"] = Skill(
                name="announce",
                description="Speak text through sdk_client.Robot.say().",
                handler=_announce,
                source="sdk_client.Robot.say",
            )

        if hasattr(robot, "execute_arm_action"):
            def _run_arm_action(action_name: str) -> SkillResult:
                code = robot.execute_arm_action(action_name)
                return SkillResult(
                    ok=True,
                    message=f"executed high-level arm action {action_name!r} -> code {code}",
                    detail={"backend": "sdk_client.Robot.execute_arm_action", "skill": action_name, "code": code},
                )

            def _gesture(**kwargs: Any) -> SkillResult:
                name = str(kwargs.get("name", "face wave")).strip()
                if not name:
                    return SkillResult(ok=False, message="no gesture name provided")
                return _run_arm_action(name)

            skills["gesture"] = Skill(
                name="gesture",
                description="Play a named high-level arm gesture through sdk_client.Robot.execute_arm_action().",
                handler=_gesture,
                source="sdk_client.Robot.execute_arm_action",
            )
            skills["wave"] = Skill(
                name="wave",
                description="Wave using the SDK 'face wave' high-level arm action.",
                handler=lambda **_kwargs: _run_arm_action("face wave"),
                source="sdk_client.Robot.execute_arm_action",
            )
            skills["high_wave"] = Skill(
                name="high_wave",
                description="Wave high using the SDK 'high wave' high-level arm action.",
                handler=lambda **_kwargs: _run_arm_action("high wave"),
                source="sdk_client.Robot.execute_arm_action",
            )
        if hasattr(robot, "release_arms"):
            def _release_arms(**kwargs: Any) -> SkillResult:
                duration_s = float(kwargs.get("duration_s", 0.5))
                try:
                    result = robot.release_arms(duration_s=duration_s)
                except TypeError:
                    result = robot.release_arms()
                return SkillResult(
                    ok=True,
                    message=f"released arm control authority: {result}",
                    detail={"backend": "sdk_client.Robot.release_arms", "skill": "release_arms"},
                )

            skills["release_arms"] = Skill(
                name="release_arms",
                description="Release arm control authority through sdk_client.Robot.release_arms().",
                handler=_release_arms,
                source="sdk_client.Robot.release_arms",
            )

    if scene_ctx is not None:
        try:
            import scene_executor
        except Exception as exc:
            unavailable.append(f"scene_executor unavailable: {exc}")
        else:
            for step_type, step_fn in scene_executor.STEP_HANDLERS.items():
                skills[f"scene.{step_type}"] = Skill(
                    name=f"scene.{step_type}",
                    description=f"scene_executor step handler '{step_type}'.",
                    handler=_wrap_step(step_fn, scene_ctx, step_type),
                    source="ollama_ai.scene_executor.STEP_HANDLERS",
                )

    if robot is None and scene_ctx is None:
        raise SkillUnavailable(
            "build_live_registry requires at least a live sdk_client.Robot instance "
            "(for llm_client.robot_tools) or a pre-built scene_executor.SceneContext "
            "(for STEP_HANDLERS); neither was provided."
        )
    if not skills:
        raise SkillUnavailable("; ".join(unavailable) or "no live skills could be bound")

    skills.update(_new_skills())
    sources = sorted({skill.source for skill in skills.values()})
    return SkillRegistry(skills, backend_label="live (" + ", ".join(sources) + ")", unavailable=unavailable)


# ---------------------------------------------------------------------------
# Capability-gated invocation. Two entry points:
#   * invoke_with_capability_check -- capability-only, always executes if
#     allowed. Used for low-stakes announcement accompaniment (speech,
#     gesture-as-acknowledgement) where asking "confirm?" before speaking
#     a sentence would be absurd -- see agent/announcements.py.
#   * resolve_and_maybe_invoke -- capability + the per-skill auto/confirm/
#     disabled setting (settings.skills). This is the path planner-
#     requested action skills go through (agent/cli/router.py).
# ---------------------------------------------------------------------------

def invoke_with_capability_check(
    registry: SkillRegistry,
    resolver: CapabilityResolver,
    skill_name: str,
    *,
    settings: AgentSettings,
    robot_state: RobotStateSnapshot,
    **kwargs: Any,
) -> tuple[PolicyDecision, Optional[SkillResult]]:
    """Resolve, then (if and only if allowed) dispatch. Never the reverse order."""
    decision = resolver.resolve_skill(skill_name, settings=settings, robot_state=robot_state)
    if not decision.allowed:
        return decision, None
    if decision.requires_approval:
        # Phase 1 has no human-approval round trip wired in (that is
        # g1_approval_ros's job, and it's ROS-only) -- a high-risk action
        # that would require approval is denied with a clear reason rather
        # than silently executed. Wiring resolve_skill's "requires_approval"
        # results through to g1_approval_ros's /g1/command_request is a
        # documented TODO for the ROS-integration phase.
        denial = PolicyDecision(
            allowed=False,
            requires_approval=True,
            risk=decision.risk,
            reason=decision.reason + " (approval workflow not wired in this phase; denying rather than executing unapproved.)",
        )
        return denial, None
    result = registry.invoke(skill_name, **kwargs)
    return decision, result


@dataclass
class SkillInvocationOutcome:
    policy: PolicyDecision
    skill_mode: SkillMode
    status: str  # "executed" | "denied" | "needs_confirmation"
    result: Optional[SkillResult] = None


def resolve_and_maybe_invoke(
    registry: SkillRegistry,
    resolver: CapabilityResolver,
    skill_name: str,
    *,
    settings: AgentSettings,
    robot_state: RobotStateSnapshot,
    confirmed: bool = False,
    **kwargs: Any,
) -> SkillInvocationOutcome:
    """Resolve capability, then apply the per-skill auto/confirm/disabled setting.

    A skill set to ``confirm`` (the default) returns
    ``status="needs_confirmation"`` without executing anything; the caller
    (``cli/router.py``) prompts the operator and calls this again with
    ``confirmed=True`` to actually dispatch -- the same y/N pattern
    ``llm_client/cli.py``'s ``--confirm-tools`` and
    ``scene_executor.py``'s per-step "Press Enter to run" already use, now
    persisted per skill instead of being an all-or-nothing CLI flag.

    A capability decision that itself demands approval (high-risk /
    ``requires_approval``) is a floor: it can raise an ``auto`` skill's
    effective mode up to ``confirm``, but a ``confirm``/``disabled``
    setting is never *weakened* by settings.skills -- and ``disabled``
    always wins.
    """
    decision = resolver.resolve_skill(skill_name, settings=settings, robot_state=robot_state)
    configured_mode = settings.get_skill_mode(skill_name)

    if not decision.allowed:
        return SkillInvocationOutcome(policy=decision, skill_mode=configured_mode, status="denied")

    effective_mode = configured_mode
    if decision.requires_approval and effective_mode == SkillMode.AUTO:
        effective_mode = SkillMode.CONFIRM

    if effective_mode == SkillMode.DISABLED:
        denial = PolicyDecision(
            allowed=False,
            requires_approval=decision.requires_approval,
            risk=decision.risk,
            reason=f"Skill '{skill_name}' is disabled (settings.skills mode=disabled).",
        )
        return SkillInvocationOutcome(policy=denial, skill_mode=effective_mode, status="denied")

    if effective_mode == SkillMode.CONFIRM and not confirmed:
        return SkillInvocationOutcome(policy=decision, skill_mode=effective_mode, status="needs_confirmation")

    result = registry.invoke(skill_name, **kwargs)
    return SkillInvocationOutcome(policy=decision, skill_mode=effective_mode, status="executed", result=result)
