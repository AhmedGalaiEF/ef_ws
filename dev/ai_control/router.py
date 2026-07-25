from __future__ import annotations

from dataclasses import dataclass

from ai_control import ollama_client
from ai_control.config import AIConfig

INTENTS = ("chat", "navigation", "endeffector", "gesture", "speaker", "vision_query")

_SYSTEM_PROMPT = f"""You are a fast intent router in front of a robot control assistant.
Classify the user's message into exactly one intent:
- chat: general conversation, questions, or anything not below.
- navigation: asks the robot to move, turn, walk, stop, or go somewhere.
- endeffector: asks the robot to open/close/grip with a hand.
- gesture: asks for an expressive gesture (wave, clap, hug, kiss, high five, etc).
- speaker: asks the robot to say/announce/speak something out loud.
- vision_query: asks what the robot currently sees / to look at / describe the scene.

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"intent": "<one of {', '.join(INTENTS)}>"}}
"""


@dataclass
class RouteResult:
    intent: str
    raw: str


def classify(user_text: str, cfg: AIConfig) -> RouteResult:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    raw = ollama_client.chat(
        cfg.ollama_host,
        cfg.router_model,
        messages,
        timeout_s=cfg.request_timeout_s,
        options={"temperature": 0.0},
    )
    try:
        parsed = ollama_client.extract_json_object(raw)
        intent = str(parsed.get("intent", "chat")).strip().lower()
    except ollama_client.OllamaError:
        intent = "chat"
    if intent not in INTENTS:
        intent = "chat"
    return RouteResult(intent=intent, raw=raw)
