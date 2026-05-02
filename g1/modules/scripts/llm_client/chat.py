"""LLM chat client with tool-use support."""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests


# Fill these in before calling send_chat_with_tool_usage.
DNABOT_BASE: Optional[str] = None
dnabot_auth: Any = None
_extract_reasoning_content_from_gpt_oss: Optional[Callable[[Dict[str, Any]], Any]] = None


def send_chat_with_tool_usage(
    model_key: str,
    messages: List[Dict[str, Any]],
    *,
    base: Optional[str] = None,
    endpoint: str = "/chat/completions",
    headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    tools: Optional[Dict[str, Callable[..., Any]]] = None,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    if base is None:
        base = DNABOT_BASE
    if base is None:
        raise RuntimeError("base URL is not set. Set DNABOT_BASE or pass base=...")
    url = f"{base.rstrip('/')}{endpoint}"

    req_headers = {"Content-Type": "application/json"}
    if dnabot_auth is not None and hasattr(dnabot_auth, "get_auth_header"):
        req_headers.update(dnabot_auth.get_auth_header())
    if headers:
        req_headers.update(headers)

    def _post_once() -> Dict[str, Any]:
        body = {
            "model": model_key,
            "messages": messages,
            "stream": False,
        }
        if tool_schemas:
            body["tools"] = tool_schemas
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if extra_body:
            body.update(extra_body)

        resp = requests.post(url, json=body, headers=req_headers, timeout=300)
        if not resp.ok:
            try:
                msg = resp.json().get("error", {}).get("message", resp.reason)
            except Exception:
                msg = resp.reason
            raise RuntimeError(f"Request failed: {msg}")
        if _extract_reasoning_content_from_gpt_oss is not None:
            print(_extract_reasoning_content_from_gpt_oss(resp.json()))
        return resp.json()

    result = _post_once()
    choice = (result.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    tool_calls = msg.get("tool_calls", [])

    if not tool_calls:
        return msg.get("content", ""), result

    if not tools:
        raise RuntimeError("Model requested tools but none were provided.")

    for call in tool_calls:
        fn = call["function"]["name"]
        args = json.loads(call["function"].get("arguments", "{}"))
        if fn not in tools:
            raise RuntimeError(f"Unknown tool: {fn}")
        output = tools[fn](**args)
        if not isinstance(output, str):
            output = json.dumps(output, ensure_ascii=False)

        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "name": fn,
            "content": output,
        })

    messages.append({"role": "assistant", "tool_calls": tool_calls})

    result = _post_once()
    choice = (result.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    return msg.get("content", ""), result


def send_chat_with_tool_usage_loop(
    model_key: str,
    messages: List[Dict[str, Any]],
    *,
    max_iterations: int = 10,
    on_tool_call: Optional[Callable[[str, Dict[str, Any], str], None]] = None,
    confirm_tool_call: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    base: Optional[str] = None,
    endpoint: str = "/chat/completions",
    headers: Optional[Dict[str, str]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    tools: Optional[Dict[str, Callable[..., Any]]] = None,
    tool_schemas: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> str:
    """Multi-step tool-use loop. Mutates ``messages`` in place.

    Loops until either the model returns a response without tool_calls,
    or ``max_iterations`` is reached. ``on_tool_call(name, args, output)``
    is invoked after each tool execution if provided — useful for the CLI
    to show tool activity.
    """
    if base is None:
        base = DNABOT_BASE
    if base is None:
        raise RuntimeError("base URL is not set. Set DNABOT_BASE or pass base=...")
    url = f"{base.rstrip('/')}{endpoint}"

    req_headers = {"Content-Type": "application/json"}
    if dnabot_auth is not None and hasattr(dnabot_auth, "get_auth_header"):
        req_headers.update(dnabot_auth.get_auth_header())
    if headers:
        req_headers.update(headers)

    def _post() -> Dict[str, Any]:
        body = {
            "model": model_key,
            "messages": messages,
            "stream": False,
        }
        if tool_schemas:
            body["tools"] = tool_schemas
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if extra_body:
            body.update(extra_body)
        resp = requests.post(url, json=body, headers=req_headers, timeout=300)
        if not resp.ok:
            try:
                err = resp.json().get("error", {}).get("message", resp.reason)
            except Exception:
                err = resp.reason
            raise RuntimeError(f"Request failed: {err}")
        return resp.json()

    for _ in range(max_iterations):
        result = _post()
        choice = (result.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            return msg.get("content", "")

        if not tools:
            raise RuntimeError("Model requested tools but none were provided.")

        # Order matters: assistant message with tool_calls FIRST, then tool results.
        messages.append({
            "role": "assistant",
            "content": msg.get("content", "") or None,
            "tool_calls": tool_calls,
        })

        for call in tool_calls:
            fn = call["function"]["name"]
            args = json.loads(call["function"].get("arguments", "{}"))
            approved = True
            if confirm_tool_call is not None:
                try:
                    approved = bool(confirm_tool_call(fn, args))
                except Exception as exc:
                    approved = False
                    output = f"error: confirmation hook raised {exc!r}; treated as deny"
            if not approved:
                output = f"denied: user blocked execution of tool '{fn}'"
            elif fn in tools:
                try:
                    output = tools[fn](**args)
                except Exception as e:
                    output = f"error: {e}"
            else:
                output = f"error: unknown tool '{fn}'"
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": fn,
                "content": output,
            })
            if on_tool_call is not None:
                try:
                    on_tool_call(fn, args, output)
                except Exception:
                    pass

    raise RuntimeError(f"max_iterations={max_iterations} reached without final answer")
