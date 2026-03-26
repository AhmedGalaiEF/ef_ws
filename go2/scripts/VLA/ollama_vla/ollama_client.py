from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatMessage:
    role: str
    content: str
    images: Optional[List[str]] = None


class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout_sec: float = 90.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "stream": stream,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    **({"images": m.images} if m.images else {}),
                }
                for m in messages
            ],
        }
        if options:
            payload["options"] = options

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise OllamaError(f"ollama request failed: {exc}") from exc

        data = json.loads(raw)
        if "error" in data:
            raise OllamaError(str(data["error"]))
        return data

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt, images=images),
        ]
        response = self.chat(messages=messages)
        content = response.get("message", {}).get("content", "").strip()
        return extract_json_object(content)


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise OllamaError("empty model response")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = JSON_BLOCK_RE.search(text)
    if not match:
        raise OllamaError(f"no JSON object found in model response: {text[:200]}")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise OllamaError("model returned JSON but not an object")
    return parsed

