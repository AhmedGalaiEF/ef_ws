"""Planner abstraction (spec section 2).

``decision = planner.decide(planner_input)`` is the only contract the rest
of the system depends on -- nothing outside this file imports an
OpenAI-specific type. ``OpenAIPlanner`` is one implementation, built on
``llm_client/chat.py``'s existing HTTP transport rather than a second HTTP
client: that module's ``extra_body`` passthrough already lets a caller
attach OpenAI's structured-JSON-output ``response_format`` without any
change to ``chat.py`` itself. ``MockPlanner`` is a second, fully offline
implementation that backs every test in this package and the CLI's
default (no API key, no network) mode.
"""
from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

from .identity import SYSTEM_IDENTITY
from .models import EventType, IntentAnnouncement, IntentType, PlannerDecision, PlannerInput


class Planner(ABC):
    @abstractmethod
    def decide(self, planner_input: PlannerInput) -> PlannerDecision:
        raise NotImplementedError


class PlannerError(RuntimeError):
    pass


def _bootstrap_llm_client_path() -> None:
    here = Path(__file__).resolve()
    scripts_dir = next(
        (
            parent
            for parent in here.parents
            if (parent / "llm_client").exists() and (parent / "sdk_client.py").exists() is False
        ),
        None,
    )
    if scripts_dir is None:
        modules_dir = next((parent for parent in here.parents if (parent / "sdk_client.py").exists()), None)
        scripts_dir = modules_dir / "scripts" if modules_dir is not None else here.parents[1]
    if scripts_dir.exists() and str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


class OpenAIPlanner(Planner):
    """Structured-output planner over an OpenAI-compatible chat endpoint."""

    DEFAULT_BASE = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base: str = DEFAULT_BASE,
        api_key_env: str = "OPENAI_API_KEY",
        max_retries: int = 1,
    ) -> None:
        _bootstrap_llm_client_path()
        try:
            from llm_client import chat as chat_module
        except Exception as exc:
            raise PlannerError(f"llm_client.chat is unavailable: {exc}") from exc
        try:
            import llm_client.secrets  # noqa: F401  (side effect: fills os.environ)
        except Exception:
            pass

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise PlannerError(
                f"{api_key_env} is not set; OpenAIPlanner cannot authenticate. "
                "Use MockPlanner for offline/test use."
            )
        # Same auth pattern as llm_client/cli.py: a module-level auth
        # header hook. This is a real limitation carried over from
        # chat.py's design (the hook is a module global, so the most
        # recently constructed OpenAIPlanner "wins" if more than one is
        # created against different keys/providers in one process) --
        # not something this file should silently paper over by forking a
        # second transport.
        chat_module.dnabot_auth = SimpleNamespace(
            get_auth_header=lambda: {"Authorization": f"Bearer {api_key}"}
        )
        self._chat_module = chat_module
        self._model = model
        self._base = base
        self._max_retries = max(0, max_retries)
        self._schema = PlannerDecision.model_json_schema()

    def decide(self, planner_input: PlannerInput) -> PlannerDecision:
        messages = [
            {"role": "system", "content": SYSTEM_IDENTITY},
            {"role": "user", "content": planner_input.model_dump_json()},
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "planner_decision", "schema": self._schema, "strict": False},
        }

        attempts_left = self._max_retries + 1
        last_error: Optional[Exception] = None
        while attempts_left > 0:
            attempts_left -= 1
            content = self._chat_module.send_chat_with_tool_usage_loop(
                model_key=self._model,
                messages=messages,
                base=self._base,
                extra_body={"response_format": response_format},
            )
            try:
                return PlannerDecision.model_validate_json(content)
            except Exception as exc:
                last_error = exc
                if attempts_left <= 0:
                    break
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"That reply was not valid JSON for a PlannerDecision ({exc}). "
                            "Reply again with ONLY the corrected JSON object, no prose."
                        ),
                    }
                )
        raise PlannerError(f"planner did not return a valid PlannerDecision after retries: {last_error}")


class MockPlanner(Planner):
    """Deterministic, offline planner. Backs every test and the CLI's default mode.

    Accepts an optional ``decide_fn`` override for tests that need a
    specific scripted decision; otherwise falls back to a small set of
    keyword rules, good enough to drive the example CLI session end to
    end with no network access or API key.
    """

    def __init__(self, decide_fn: Optional[Callable[[PlannerInput], PlannerDecision]] = None) -> None:
        self._decide_fn = decide_fn
        self.calls: list[PlannerInput] = []

    def decide(self, planner_input: PlannerInput) -> PlannerDecision:
        self.calls.append(planner_input)
        if self._decide_fn is not None:
            return self._decide_fn(planner_input)
        return self._default_rule(planner_input)

    @staticmethod
    def _default_rule(planner_input: PlannerInput) -> PlannerDecision:
        if planner_input.event not in (EventType.USER_MESSAGE, EventType.ASR_MESSAGE):
            return PlannerDecision(intent=IntentType.NO_ACTION, next_tick_s=30.0)

        text = (planner_input.user_text or "").strip().lower()
        if not text:
            return PlannerDecision(intent=IntentType.NO_ACTION)

        if "who are you" in text or "what are you" in text:
            return PlannerDecision(
                intent=IntentType.CONVERSATION,
                response_text=(
                    "I am the local cognitive CLI for a Unitree G1 robot. "
                    "Right now I am using the offline MockPlanner because OPENAI_API_KEY is not set."
                ),
            )
        if "remember" in text or "memory" in text:
            bio = planner_input.autobiography_summary or "No autobiographical memory has been recorded yet."
            return PlannerDecision(
                intent=IntentType.CONVERSATION,
                response_text=f"My current autobiographical memory is:\n{bio}",
            )
        if "sleep" in text:
            return PlannerDecision(
                intent=IntentType.REQUEST_SLEEP,
                target="operator_request",
                response_text="Understood, I'll get ready to sleep.",
                requested_skills=["request_sleep"],
            )
        if "battery" in text:
            return PlannerDecision(
                intent=IntentType.QUERY_STATE,
                target="battery",
            )
        if "charge" in text:
            return PlannerDecision(
                intent=IntentType.REQUEST_CHARGE,
                response_text="Let me see about charging.",
                requested_skills=["request_charge"],
            )
        if "step back" in text or "move back" in text or "back up" in text:
            return PlannerDecision(
                intent=IntentType.EXECUTE_TASK,
                target="step_back",
                response_text="I'll step back.",
                requested_skills=["step_back"],
            )
        if "arm" in text and ("move" in text or "can you" in text or "raise" in text):
            wants_action = "raise" in text or "please raise" in text
            if wants_action:
                return PlannerDecision(
                    intent=IntentType.MOVE_ARM,
                    target="right_arm",
                    response_text="I'll try moving my arm.",
                    requested_skills=["reach_forward"],
                    intent_announcement=IntentAnnouncement(
                        speech="I'll raise my right arm.", gesture="gesture"
                    ),
                )
            return PlannerDecision(
                intent=IntentType.QUERY_CAPABILITY,
                target="arm",
                response_text=None,  # the runtime fills in the grounded answer, not the model
            )
        return PlannerDecision(
            intent=IntentType.CONVERSATION,
            response_text=f"Hello! You said: {planner_input.user_text}",
        )
