from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OllamaError(RuntimeError):
    pass


def chat(
    host: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    images_b64: list[str] | None = None,
    timeout_s: float = 60.0,
    options: dict[str, Any] | None = None,
) -> str:
    """Call Ollama's /api/chat with a non-streamed request and return the reply text.

    `images_b64`, if given, is attached to the *last* message (Ollama's
    convention for vision models: base64-encoded image bytes per message).
    """
    payload_messages = [dict(message) for message in messages]
    if images_b64:
        if not payload_messages:
            raise OllamaError("images_b64 given but there are no messages to attach them to.")
        payload_messages[-1]["images"] = list(images_b64)

    body = {
        "model": model,
        "messages": payload_messages,
        "stream": False,
    }
    if options:
        body["options"] = options

    request = urllib.request.Request(
        url=f"{host.rstrip('/')}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            raise OllamaError(
                f"Ollama could not find model {model!r} at {host}. "
                f"Run `ollama pull {model}` or choose an installed model with the CLI flags. "
                f"Underlying response: {detail[:200]}"
            ) from exc
        raise OllamaError(
            f"Ollama HTTP error {exc.code} at {host} (model={model!r}). "
            f"Underlying response: {detail[:200]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Could not reach Ollama at {host} (model={model!r}). "
            f"Is `ollama serve` running and is the model pulled? Underlying error: {exc}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Ollama returned non-JSON response: {raw[:200]!r}") from exc

    try:
        return str(data["message"]["content"])
    except (KeyError, TypeError) as exc:
        raise OllamaError(f"Unexpected Ollama response shape: {data!r}") from exc


def extract_json_object(text: str) -> dict[str, Any]:
    """Best-effort extraction of a single top-level JSON object from model output.

    Models routinely wrap JSON in prose or ```json fences despite instructions
    not to -- this strips the outermost braces and parses that.
    """
    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            candidate, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(candidate, dict):
            return candidate
        last_error = json.JSONDecodeError("JSON value is not an object", text, start + end)

    if last_error is None:
        raise OllamaError(f"No JSON object found in model output: {text!r}")
    raise OllamaError(f"Failed to parse JSON object from model output: {text!r}") from last_error
