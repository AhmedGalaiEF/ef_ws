from __future__ import annotations

import json
import re
import time
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
        except TimeoutError as exc:
            raise OllamaError(f"ollama request timed out after {self.timeout_sec:.1f}s") from exc
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


class DryRunOllamaChatClient(OllamaChatClient):
    def __init__(self, model: str = "dry-run", timeout_sec: float = 0.0):
        super().__init__(base_url="dry-run://local", model=model, timeout_sec=timeout_sec)

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _ = (stream, options)
        content = self._generate_content(messages)
        return {
            "model": self.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": content},
            "done": True,
        }

    def _generate_content(self, messages: List[ChatMessage]) -> str:
        system_prompt = next((m.content for m in messages if m.role == "system"), "")
        user_prompt = next((m.content for m in messages if m.role == "user"), "")

        if "planner agent" in system_prompt:
            return json.dumps(
                {
                    "perception_prompt": "Describe immediate obstacles, free space, and whether motion is safe.",
                    "world_model": "Dry-run fallback with no live planner inference.",
                    "reasoning_brief": "No live model is available, so the robot should remain stopped.",
                    "suggested_actions": [
                        {"name": "stop_move", "args": {}, "duration_sec": 0.0}
                    ],
                }
            )

        if "actor agent" in system_prompt:
            return json.dumps(
                {
                    "commands": [
                        {"name": "stop_move", "args": {}, "duration_sec": 0.0}
                    ]
                }
            )

        if "perception agent" in system_prompt:
            return json.dumps(
                {
                    "summary": "Dry-run mode: no model inference; scene analysis unavailable.",
                    "hazards": ["Perception is simulated in dry-run mode."],
                    "free_space": {
                        "forward_clear": False,
                        "left_clear": False,
                        "right_clear": False,
                        "confidence": 0.0,
                    },
                    "targets": [],
                    "recommended_cautions": ["Hold position until a live model is available."],
                }
            )

        return json.dumps(
            {
                "message": "Dry-run fallback response",
                "user_prompt": user_prompt,
            }
        )


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
