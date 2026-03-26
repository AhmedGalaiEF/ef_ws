from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

from ollama_vla.ollama_client import extract_json_object


class HuggingFaceError(RuntimeError):
    pass


@dataclass
class ChatMessage:
    role: str
    content: str
    images: Optional[List[str]] = None


class HuggingFaceChatClient:
    _PREFERRED_TEXT_MODELS = (
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "google/gemma-2-2b-it",
    )
    _PREFERRED_VISION_MODELS = (
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "CohereLabs/aya-vision-32b:cohere",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "zai-org/GLM-4.5V",
    )

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
        self._text_model_override: Optional[str] = None
        self._vision_model_override: Optional[str] = None
        self._available_models_cache: Optional[List[str]] = None

    def chat(
        self,
        messages: List[ChatMessage],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.api_token:
            raise HuggingFaceError(
                "missing Hugging Face API token; pass --hf-token or set HF_TOKEN"
            )

        has_images = any(message.images for message in messages)
        model = self._model_for_request(has_images=has_images)
        payload = {
            "model": model,
            "messages": [self._format_message(message) for message in messages],
            "temperature": self.temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        try:
            data = self._post_json(self.api_url, payload)
        except TimeoutError as exc:
            raise HuggingFaceError(
                f"hugging face request timed out after {self.timeout_sec:.1f}s"
            ) from exc
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            if exc.code == 400 and self._is_model_not_supported_error(details):
                candidates = self._candidate_models(
                    requested_model=model,
                    has_images=has_images,
                )
                errors: List[str] = []
                for retry_model in candidates:
                    if retry_model == model:
                        continue
                    payload["model"] = retry_model
                    try:
                        data = self._post_json(self.api_url, payload)
                        self._remember_model_override(has_images=has_images, model=retry_model)
                        break
                    except TimeoutError as retry_exc:
                        raise HuggingFaceError(
                            f"hugging face request timed out after {self.timeout_sec:.1f}s"
                        ) from retry_exc
                    except urllib.error.HTTPError as retry_exc:
                        retry_details = retry_exc.read().decode("utf-8", errors="replace")
                        errors.append(
                            f"{retry_model} -> HTTP {retry_exc.code}: {retry_details[:120]}"
                        )
                        continue
                    except urllib.error.URLError as retry_exc:
                        errors.append(f"{retry_model} -> {retry_exc}")
                        continue
                else:
                    raise HuggingFaceError(
                        self._build_model_not_supported_message(model, errors)
                    ) from exc
            else:
                raise HuggingFaceError(
                    f"hugging face request failed: HTTP {exc.code}: {details[:300]}"
                ) from exc
        except urllib.error.URLError as exc:
            raise HuggingFaceError(f"hugging face request failed: {exc}") from exc

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

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _get_json(self, url: str) -> Dict[str, Any]:
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.api_token}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _model_for_request(self, has_images: bool) -> str:
        if has_images and self._vision_model_override:
            return self._vision_model_override
        if not has_images and self._text_model_override:
            return self._text_model_override
        return self.model

    def _remember_model_override(self, has_images: bool, model: str) -> None:
        if has_images:
            self._vision_model_override = model
        else:
            self._text_model_override = model

    def _candidate_models(
        self,
        requested_model: str,
        has_images: bool,
    ) -> List[str]:
        available_models = self._available_models()
        if not available_models:
            return []

        ordered: List[str] = []
        seen = set()

        def add(model_id: str) -> None:
            if model_id not in seen:
                seen.add(model_id)
                ordered.append(model_id)

        for candidate in self._matching_model_ids(requested_model, available_models):
            add(candidate)

        preferred_models = (
            self._PREFERRED_VISION_MODELS if has_images else self._PREFERRED_TEXT_MODELS
        )
        for preferred in preferred_models:
            for candidate in self._matching_model_ids(preferred, available_models):
                add(candidate)

        for candidate in available_models:
            if has_images and self._looks_like_vision_model(candidate):
                add(candidate)
            if not has_images and self._looks_like_text_chat_model(candidate):
                add(candidate)
        return ordered

    def _available_models(self) -> List[str]:
        if self._available_models_cache is not None:
            return self._available_models_cache

        models_url = self._models_url()
        try:
            payload = self._get_json(models_url)
        except Exception:
            self._available_models_cache = []
            return self._available_models_cache

        data = payload.get("data", [])
        models: List[str] = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    model_id = item.get("id")
                    if isinstance(model_id, str) and model_id:
                        models.append(model_id)
        self._available_models_cache = models
        return self._available_models_cache

    def _models_url(self) -> str:
        parsed = urlsplit(self.api_url)
        path = parsed.path
        if path.endswith("/chat/completions"):
            path = path[: -len("/chat/completions")] + "/models"
        elif path.endswith("/completions"):
            path = path[: -len("/completions")] + "/models"
        else:
            path = "/v1/models"
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    def _build_model_not_supported_message(
        self,
        model: str,
        candidate_errors: Optional[List[str]] = None,
    ) -> str:
        available_models = self._available_models()
        visible = ", ".join(available_models[:8])
        if len(available_models) > 8:
            visible += ", ..."
        if visible:
            visible = f" Available models for this token: {visible}."
        retry_summary = ""
        if candidate_errors:
            retry_summary = " Retry attempts: " + " | ".join(candidate_errors[:4]) + "."
        return (
            f"configured Hugging Face model '{model}' is not available through your enabled "
            "providers. Pass --model with a router-supported model id or provider-suffixed id "
            f"such as 'model:provider'.{visible}{retry_summary}"
        )

    def _matching_model_ids(self, requested_model: str, available_models: List[str]) -> List[str]:
        matches: List[str] = []
        if requested_model in available_models:
            matches.append(requested_model)
        requested_base = self._base_model_id(requested_model)
        for candidate in available_models:
            if candidate not in matches and self._base_model_id(candidate) == requested_base:
                matches.append(candidate)
        return matches

    def _is_model_not_supported_error(self, details: str) -> bool:
        text = details.lower()
        return "model_not_supported" in text or "not supported by any provider" in text

    def _looks_like_vision_model(self, model_id: str) -> bool:
        normalized = self._base_model_id(model_id).lower()
        markers = ("vision", "-vl", "/vl", "vl-", "llava", "multimodal", "pixtral", "glm-4.5v")
        return any(marker in normalized for marker in markers)

    def _looks_like_text_chat_model(self, model_id: str) -> bool:
        normalized = self._base_model_id(model_id).lower()
        blocked = (
            "embed",
            "embedding",
            "rerank",
            "rank",
            "whisper",
            "asr",
            "speech",
            "tts",
            "transcrib",
            "moderation",
        )
        return not any(marker in normalized for marker in blocked)

    def _base_model_id(self, model_id: str) -> str:
        return model_id.split(":", 1)[0]


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


class FallbackHuggingFaceChatClient(HuggingFaceChatClient):
    def __init__(
        self,
        primary: HuggingFaceChatClient,
        fallback: HuggingFaceChatClient,
        on_error: Optional[Callable[[HuggingFaceError], None]] = None,
    ):
        super().__init__(
            api_url=primary.api_url,
            model=primary.model,
            api_token=primary.api_token,
            timeout_sec=primary.timeout_sec,
            temperature=primary.temperature,
        )
        self._primary = primary
        self._fallback = fallback
        self._on_error = on_error
        self._using_fallback = False

    def chat(
        self,
        messages: List[ChatMessage],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._using_fallback:
            return self._fallback.chat(messages=messages, response_format=response_format)

        try:
            return self._primary.chat(messages=messages, response_format=response_format)
        except HuggingFaceError as exc:
            self._using_fallback = True
            if self._on_error is not None:
                self._on_error(exc)
            return self._fallback.chat(messages=messages, response_format=response_format)
