from __future__ import annotations

import base64

from ai_control import ollama_client
from ai_control.config import AIConfig

_SYSTEM_PROMPT = (
    "You are the vision component of a robot control assistant. "
    "Describe only what is visible in the image, concisely and factually. "
    "Do not invent objects or actions you cannot see."
)


def describe(image_jpeg: bytes, question: str, cfg: AIConfig) -> str:
    image_b64 = base64.b64encode(image_jpeg).decode("ascii")
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return ollama_client.chat(
        cfg.ollama_host,
        cfg.vision_model,
        messages,
        images_b64=[image_b64],
        timeout_s=cfg.request_timeout_s,
    )
