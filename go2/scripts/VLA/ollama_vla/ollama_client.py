from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


class OllamaError(RuntimeError):
    pass


@dataclass
class ChatMessage:
    role: str
    content: str
    images: Optional[List[str]] = None


class OllamaChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_sec: float = 90.0,
        default_options: Optional[Dict[str, Any]] = None,
        keep_alive: str = "10m",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec
        self.default_options = dict(default_options or {})
        self.keep_alive = keep_alive

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        response_format: Optional[str] = None,
        think: Optional[bool] = None,
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
        merged_options = dict(self.default_options)
        if options:
            merged_options.update(options)
        if merged_options:
            payload["options"] = merged_options
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        if response_format:
            payload["format"] = response_format
        if think is not None:
            payload["think"] = think

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
            raise OllamaError(
                f"ollama request timed out after {self.timeout_sec:.1f}s; "
                "the model may still be loading or generating"
            ) from exc
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise OllamaError(
                f"ollama request failed: HTTP {exc.code}: {details[:300]}"
            ) from exc
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
        response = self.chat(messages=messages, response_format="json", think=False)
        content = response.get("message", {}).get("content", "").strip()
        if not content:
            fallback_response = self.chat(messages=messages, think=False)
            fallback_content = fallback_response.get("message", {}).get("content", "").strip()
            if fallback_content:
                return extract_json_object(fallback_content)
        try:
            return extract_json_object(content)
        except OllamaError as exc:
            if response.get("done_reason") != "length":
                raise

            base_predict = int(self.default_options.get("num_predict", 128) or 128)
            retry_options = {"num_predict": max(base_predict * 2, 192)}
            retry_response = self.chat(
                messages=messages,
                options=retry_options,
                response_format="json",
                think=False,
            )
            retry_content = retry_response.get("message", {}).get("content", "").strip()
            try:
                return extract_json_object(retry_content)
            except OllamaError:
                raise exc


class DryRunOllamaChatClient(OllamaChatClient):
    def __init__(self, model: str = "dry-run", timeout_sec: float = 0.0):
        super().__init__(
            base_url="dry-run://local",
            model=model,
            timeout_sec=timeout_sec,
            default_options={},
            keep_alive="",
        )

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        response_format: Optional[str] = None,
        think: Optional[bool] = None,
    ) -> Dict[str, Any]:
        _ = (stream, options, response_format, think)
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


class FallbackOllamaChatClient(OllamaChatClient):
    def __init__(
        self,
        primary: OllamaChatClient,
        fallback: OllamaChatClient,
        on_error: Optional[Callable[[OllamaError], None]] = None,
    ):
        super().__init__(
            base_url=primary.base_url,
            model=primary.model,
            timeout_sec=primary.timeout_sec,
            default_options=primary.default_options,
            keep_alive=primary.keep_alive,
        )
        self._primary = primary
        self._fallback = fallback
        self._on_error = on_error
        self._using_fallback = False

    def chat(
        self,
        messages: List[ChatMessage],
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        response_format: Optional[str] = None,
        think: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if self._using_fallback:
            return self._fallback.chat(
                messages=messages,
                stream=stream,
                options=options,
                response_format=response_format,
                think=think,
            )

        try:
            return self._primary.chat(
                messages=messages,
                stream=stream,
                options=options,
                response_format=response_format,
                think=think,
            )
        except OllamaError as exc:
            self._using_fallback = True
            if self._on_error is not None:
                self._on_error(exc)
            return self._fallback.chat(
                messages=messages,
                stream=stream,
                options=options,
                response_format=response_format,
                think=think,
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

    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    match = JSON_BLOCK_RE.search(text)
    if not match:
        raise OllamaError(f"no JSON object found in model response: {text[:200]}")

    raise OllamaError(f"model returned malformed JSON: {match.group(0)[:200]}")


def _json_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    start = -1
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : index + 1])
                start = -1
    return candidates
