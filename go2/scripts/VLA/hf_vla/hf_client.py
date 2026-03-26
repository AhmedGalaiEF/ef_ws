from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ollama_vla.ollama_client import extract_json_object


class HuggingFaceError(RuntimeError):
    pass


@dataclass
class ChatMessage:
    role: str
    content: str
    images: Optional[List[str]] = None


class HuggingFaceChatClient:
    def __init__(
        self,
        api_url: str,
        model: str,
        api_token: str = "",
        timeout_sec: float = 90.0,
        temperature: float = 0.1,
    ):
        self.api_url = api_url
        self.model = model
        self.api_token = api_token or os.environ.get("HF_TOKEN", "")
        self.timeout_sec = timeout_sec
        self.temperature = temperature

    def chat(
        self,
        messages: List[ChatMessage],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.api_token:
            raise HuggingFaceError(
                "missing Hugging Face API token; pass --hf-token or set HF_TOKEN"
            )

        payload = {
            "model": self.model,
            "messages": [self._format_message(message) for message in messages],
            "temperature": self.temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except TimeoutError as exc:
            raise HuggingFaceError(
                f"hugging face request timed out after {self.timeout_sec:.1f}s"
            ) from exc
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise HuggingFaceError(
                f"hugging face request failed: HTTP {exc.code}: {details[:300]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise HuggingFaceError(f"hugging face request failed: {exc}") from exc

        data = json.loads(raw)
        if "error" in data:
            raise HuggingFaceError(str(data["error"]))
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
        try:
            response = self.chat(messages=messages, response_format={"type": "json_object"})
        except HuggingFaceError as exc:
            error_text = str(exc).lower()
            if "response_format" not in error_text and "json_object" not in error_text:
                raise
            response = self.chat(messages=messages)
        content = self._extract_content(response)
        return extract_json_object(content)

    def _format_message(self, message: ChatMessage) -> Dict[str, Any]:
        if not message.images:
            return {"role": message.role, "content": message.content}

        content: List[Dict[str, Any]] = [{"type": "text", "text": message.content}]
        for image_b64 in message.images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )
        return {"role": message.role, "content": content}

    def _extract_content(self, response: Dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            raise HuggingFaceError("empty model response")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            return "\n".join(part for part in text_parts if part).strip()
        raise HuggingFaceError("unsupported response content format")


class DryRunHuggingFaceChatClient(HuggingFaceChatClient):
    def __init__(self, model: str = "dry-run", timeout_sec: float = 0.0):
        super().__init__(
            api_url="dry-run://local",
            model=model,
            api_token="dry-run",
            timeout_sec=timeout_sec,
            temperature=0.0,
        )

    def chat(
        self,
        messages: List[ChatMessage],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _ = response_format
        content = self._generate_content(messages)
        return {
            "id": "dry-run",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
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
