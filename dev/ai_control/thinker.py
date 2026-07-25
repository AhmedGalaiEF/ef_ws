from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_control import ollama_client, tools
from ai_control.config import AIConfig

_SYSTEM_PROMPT_TEMPLATE = """You are the reasoning core of a robot control assistant.
You can converse normally, and you can propose exactly one tool call per turn when
the user's request calls for robot action. A separate execution layer runs the tool
only after the human operator explicitly confirms it -- so never say the action has
already happened; describe it as proposed/pending.

Available tools:
{tools_block}

Respond with ONLY a JSON object, no prose, no markdown fences, in this exact shape:
{{"response": "<what you say to the user>", "tool_call": null}}
or, when a tool call is warranted:
{{"response": "<what you say to the user>", "tool_call": {{"name": "<tool name>", "args": {{...}}}}}}
"""


@dataclass
class ThinkResult:
    response: str
    tool_call: dict[str, Any] | None
    raw: str


def _system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(tools_block=tools.prompt_block())


def think(
    history: list[dict[str, str]],
    user_text: str,
    cfg: AIConfig,
    *,
    vision_context: str | None = None,
) -> ThinkResult:
    messages = [{"role": "system", "content": _system_prompt()}, *history]
    if vision_context:
        messages.append({"role": "system", "content": f"Latest camera observation: {vision_context}"})
    messages.append({"role": "user", "content": user_text})

    raw = ollama_client.chat(
        cfg.ollama_host,
        cfg.thinker_model,
        messages,
        timeout_s=cfg.request_timeout_s,
    )

    try:
        parsed = ollama_client.extract_json_object(raw)
    except ollama_client.OllamaError:
        # Model ignored the format instructions -- fall back to plain text, no tool call.
        return ThinkResult(response=raw.strip(), tool_call=None, raw=raw)

    response = str(parsed.get("response", "")).strip()
    tool_call = parsed.get("tool_call")
    if tool_call is not None and not (isinstance(tool_call, dict) and "name" in tool_call):
        tool_call = None
    return ThinkResult(response=response or raw.strip(), tool_call=tool_call, raw=raw)
