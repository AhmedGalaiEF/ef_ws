#!/usr/bin/env python3
from __future__ import annotations
from std_msgs.msg import String
from rclpy.node import Node
import rclpy

import argparse
from html.parser import HTMLParser
import json
import os
import re
import subprocess
import string
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


DEFAULT_SYSTEM_PROMPT = (
    "You are the voice of a Unitree humanoid robot. Chat naturally with nearby people. "
    "Reply in no more than 25 words. "
    "Do not mention that you are a language model. Do not use markdown or hidden reasoning."
)
COMPANY_WEB_TOOL_PROMPT = (
    "You may use the company_web_request tool for questions about EF Robotics, its robots, "
    "or the Humanoid Academy. Use only facts from the returned page text for company-specific answers. "
    "If the requested detail is not in the returned pages, say you do not know. "
    "Do not mention tool calls."
)
COMPANY_WEB_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "company_web_request",
        "description": (
            "Fetch public EF Robotics web pages. Only these URLs are allowed: "
            "https://www.ef-robotics.de/en, "
            "https://www.ef-robotics.de/lieferroboter/*, "
            "https://www.ef-robotics.de/reinigungsroboter/*, "
            "https://www.ef-robotics.de/humanoideroboter/*, "
            "and https://www.ef-robotics.de/humanoid-academy."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The exact EF Robotics URL to fetch.",
                }
            },
            "required": ["url"],
        },
    },
}
ALLOWED_COMPANY_URL_EXACT_PATHS = {
    "/en",
    "/lieferroboter",
    "/reinigungsroboter",
    "/humanoideroboter",
    "/humanoid-academy",
}
ALLOWED_COMPANY_URL_PREFIXES = (
    "/lieferroboter/",
    "/reinigungsroboter/",
    "/humanoideroboter/",
)
FILLER_TEXTS = {
    "ah",
    "eh",
    "er",
    "hmm",
    "hm",
    "mm",
    "uh",
    "um",
    "嗯",
    "呃",
    "啊",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen to robot ASR, answer with Ollama qwen3.5:9b, and speak replies through the robot."
    )
    parser.add_argument("--topic", default="/audio_msg", help="ROS 2 ASR topic to subscribe to.")
    parser.add_argument("--out", default="/tmp/robot_chat.jsonl",
                        help="JSONL file for chat events.")
    parser.add_argument("--text-out", default="/tmp/robot_chat.txt",
                        help="Plain text transcript output file.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL.")
    parser.add_argument("--model", default="qwen3.5:9b", help="Ollama model name.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt for the robot persona.")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Ollama sampling temperature.")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Ollama request timeout in seconds.")
    parser.add_argument("--max-history", type=int, default=4,
                        help="Maximum user/assistant messages to keep.")
    parser.add_argument("--num-predict", type=int, default=48,
                        help="Maximum tokens to generate per reply.")
    parser.add_argument("--num-ctx", type=int, default=1024, help="Ollama context window size.")
    parser.add_argument("--keep-alive", default="15m",
                        help="How long Ollama should keep the model loaded.")
    parser.add_argument("--num-thread", type=int, default=None,
                        help="Optional Ollama CPU thread count.")
    parser.add_argument("--company-web-timeout", type=float, default=8.0,
                        help="Timeout for EF Robotics web requests.")
    parser.add_argument(
        "--company-web-max-chars",
        type=int,
        default=5000,
        help="Maximum page text characters returned to Ollama per EF Robotics web request.",
    )
    parser.add_argument(
        "--company-web-tool-rounds",
        type=int,
        default=3,
        help="Maximum Ollama tool-call rounds per user message.",
    )
    parser.add_argument("--no-warmup", action="store_true",
                        help="Do not preload the model at startup.")
    parser.add_argument("--iface", default="eth0", help="DDS interface for robot audio playback.")
    parser.add_argument("--domain-id", type=int, default=0,
                        help="DDS domain ID for robot audio playback.")
    parser.add_argument("--volume", type=int, default=None, help="Optional playback volume 0-100.")
    parser.add_argument("--tts-language", default=None,
                        help="Optional Piper language, for example en, de, fr, es, ar.")
    parser.add_argument("--startup-speech", default="chat mode activated",
                        help="Text to speak when chat mode starts.")
    parser.add_argument("--no-startup-speech", action="store_true",
                        help="Do not speak the startup phrase.")
    parser.add_argument("--headlight-color", default="#123456",
                        help="Headlight color to set when chat mode starts.")
    parser.add_argument("--headlight-intensity", type=int, default=100,
                        help="Startup headlight intensity 0-100.")
    parser.add_argument("--no-headlight", action="store_true",
                        help="Do not change the headlight on startup.")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Ignore ASR below this confidence.")
    parser.add_argument("--answer-fillers", action="store_true",
                        help="Answer short filler utterances like um or hmm.")
    parser.add_argument("--no-reply", action="store_true",
                        help="Generate and save replies; do not speak them.")
    parser.add_argument(
        "--post-speak-ignore-s",
        type=float,
        default=1.5,
        help="Ignore likely echo/self-ASR for this many seconds after speaking.",
    )
    return parser.parse_args()


def decode_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {"raw": raw}


def clean_reply(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    # Qwen reasoning models can emit hidden-thought style sections. Never speak them.
    while "<think>" in text and "</think>" in text:
        before, rest = text.split("<think>", 1)
        _hidden, after = rest.split("</think>", 1)
        text = (before + after).strip()
    return " ".join(text.split())


class TextExtractingHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag in {"p", "br", "li", "div", "section", "article", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        text = " ".join(part.strip() for part in self._parts if part.strip())
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        return re.sub(r"\n\s*\n+", "\n", text).strip()


class RobotChat(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("robot_chat")
        self.args = args
        self.out_path = Path(args.out).expanduser()
        self.text_out_path = Path(args.text_out).expanduser()
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.text_out_path.parent.mkdir(parents=True, exist_ok=True)

        system_prompt = f"{args.system_prompt} {COMPANY_WEB_TOOL_PROMPT}"
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self.last_index: int | None = None
        self.last_text: str | None = None
        self.last_reply: str | None = None
        self.last_reply_ts = 0.0
        self.create_subscription(String, args.topic, self.on_audio_msg, 10)
        self.get_logger().info(
            f"Chatting from {args.topic} with Ollama model={args.model}; saving to {self.out_path}"
        )
        self._run_startup_actions()
        if not args.no_warmup:
            self._warm_up_ollama()

    def on_audio_msg(self, msg: String) -> None:
        payload = decode_payload(str(msg.data))
        text = str(payload.get("text", "")).strip()
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        index = self._payload_index(payload)
        received_at = time.time()

        if not self._should_answer(text, confidence, index, received_at):
            return

        self.get_logger().info(f'Heard: "{text}" confidence={confidence:.2f} index={index}')
        try:
            reply = self._ask_ollama(text)
        except Exception as exc:
            self.get_logger().error(f"Ollama request failed: {exc}")
            self._save_event(received_at, payload, text, None, error=str(exc))
            return

        self._save_event(received_at, payload, text, reply)
        self.get_logger().info(f'Reply: "{reply}"')

        if reply and not self.args.no_reply:
            code = self._speak_once(reply)
            self.get_logger().info(f"robot_say_once exited {code}")

        self.last_index = index
        self.last_text = text
        self.last_reply = reply
        self.last_reply_ts = time.time()

    def _should_answer(self, text: str, confidence: float, index: int | None, received_at: float) -> bool:
        if not text or confidence < float(self.args.min_confidence):
            return False
        if not self.args.answer_fillers and self._is_filler(text):
            return False
        if not any(char.isalnum() for char in text):
            return False
        if index is not None and index == self.last_index:
            return False
        if index is None and text == self.last_text and (received_at - self.last_reply_ts) < 2.0:
            return False
        if (received_at - self.last_reply_ts) < float(self.args.post_speak_ignore_s):
            return False
        if self.last_reply and self._similar(text, self.last_reply) >= 0.82:
            return False
        return True

    @staticmethod
    def _payload_index(payload: dict[str, Any]) -> int | None:
        try:
            value = payload.get("index")
            return int(value) if value is not None else None
        except Exception:
            return None

    @staticmethod
    def _similar(left: str, right: str) -> float:
        return SequenceMatcher(None, left.lower().strip(), right.lower().strip()).ratio()

    @staticmethod
    def _is_filler(text: str) -> bool:
        normalized = text.strip().lower().strip(string.punctuation + "，。！？、；：")
        return normalized in FILLER_TEXTS

    def _ask_ollama(self, user_text: str) -> str:
        max_history = max(1, int(self.args.max_history))
        messages = self.messages + [{"role": "user", "content": user_text}]
        del messages[1: max(1, len(messages) - max_history)]

        started = time.time()
        result = self._chat_with_company_web_tool(messages)
        self.get_logger().info(f"Ollama replied in {time.time() - started:.1f}s")

        reply = clean_reply(str(result.get("message", {}).get("content", "")))
        if not reply:
            reply = "I heard you, but I am not sure how to answer that yet."
        messages.append({"role": "assistant", "content": reply})
        self.messages = self._conversation_history(messages)
        return reply

    @staticmethod
    def _conversation_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                history.append({"role": "system", "content": str(message.get("content", ""))})
            elif role in {"user", "assistant"} and not message.get("tool_calls"):
                history.append({"role": str(role), "content": str(message.get("content", ""))})
        return history

    def _chat_with_company_web_tool(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for _ in range(max(1, int(self.args.company_web_tool_rounds))):
            body = self._ollama_chat_body(messages, tools=[COMPANY_WEB_TOOL_SCHEMA])
            result = self._post_ollama_chat(body, timeout=float(self.args.timeout))
            message = result.get("message", {})
            tool_calls = self._ollama_tool_calls(message)
            if not tool_calls:
                return result

            assistant_message: dict[str, Any] = {
                "role": "assistant", "content": message.get("content", "") or ""}
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            messages.append(assistant_message)

            for call in tool_calls:
                name, arguments = self._tool_call_name_and_arguments(call)
                if name != "company_web_request":
                    output = f"error: unknown tool '{name}'"
                else:
                    try:
                        output = self._company_web_request(**arguments)
                    except Exception as exc:
                        output = f"error: {exc}"
                self.get_logger().info(f"Tool {name}({arguments}) -> {len(output)} chars")
                tool_message: dict[str, Any] = {"role": "tool", "content": output}
                if call.get("id"):
                    tool_message["tool_call_id"] = call.get("id")
                messages.append(tool_message)

        raise RuntimeError(
            f"company_web_tool_rounds={self.args.company_web_tool_rounds} reached without final answer"
        )

    def _ollama_chat_body(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": str(self.args.model),
            "messages": messages,
            "stream": False,
            "keep_alive": str(self.args.keep_alive),
            "think": False,
            "options": {
                "temperature": float(self.args.temperature),
                "num_predict": int(self.args.num_predict),
                "num_ctx": int(self.args.num_ctx),
            },
        }
        if tools:
            body["tools"] = tools
        if self.args.num_thread is not None:
            body["options"]["num_thread"] = int(self.args.num_thread)
        return body

    @staticmethod
    def _ollama_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
        tool_calls = message.get("tool_calls") or []
        return tool_calls if isinstance(tool_calls, list) else []

    @staticmethod
    def _tool_call_name_and_arguments(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(function.get("name") or call.get("name") or "")
        raw_arguments = function.get("arguments", call.get("arguments", {}))
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
        elif isinstance(raw_arguments, dict):
            arguments = raw_arguments
        else:
            arguments = {}
        return name, arguments

    def _company_web_request(self, url: str) -> str:
        normalized_url = self._normalize_company_url(url)
        request = urllib.request.Request(
            normalized_url,
            headers={
                "User-Agent": "EF-Robotics-RobotChat/1.0",
                "Accept": "text/html,application/xhtml+xml",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(self.args.company_web_timeout)) as response:
                final_url = response.geturl()
                if self._normalize_company_url(final_url) != normalized_url and not self._is_allowed_company_url(final_url):
                    return f"error: blocked redirect to non-allowlisted URL {final_url}"
                content_type = response.headers.get("Content-Type", "")
                raw = response.read(1_500_000)
        except urllib.error.HTTPError as exc:
            return f"error: HTTP {exc.code} for {normalized_url}"
        except urllib.error.URLError as exc:
            return f"error: web request failed for {normalized_url}: {exc.reason}"

        charset = "utf-8"
        match = re.search(r"charset=([^;\s]+)", content_type, flags=re.I)
        if match:
            charset = match.group(1).strip("\"'")
        html = raw.decode(charset, errors="replace")
        parser = TextExtractingHTMLParser()
        parser.feed(html)
        text = parser.text()
        max_chars = max(500, int(self.args.company_web_max_chars))
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
        return f"source: {normalized_url}\n{text or 'No readable text found.'}"

    def _normalize_company_url(self, url: str) -> str:
        parsed = urllib.parse.urlparse(str(url).strip())
        if not parsed.scheme:
            parsed = urllib.parse.urlparse(
                f"https://www.ef-robotics.de/{str(url).strip().lstrip('/')}")
        normalized = parsed._replace(fragment="", query="")
        path = normalized.path.rstrip("/") or "/"
        normalized = normalized._replace(path=path)
        url_text = urllib.parse.urlunparse(normalized)
        if not self._is_allowed_company_url(url_text):
            raise ValueError(f"blocked non-allowlisted EF Robotics URL: {url}")
        return url_text

    @staticmethod
    def _is_allowed_company_url(url: str) -> bool:
        parsed = urllib.parse.urlparse(str(url).strip())
        path = parsed.path.rstrip("/") or "/"
        if parsed.scheme != "https" or parsed.netloc.lower() != "www.ef-robotics.de":
            return False
        if path in ALLOWED_COMPANY_URL_EXACT_PATHS:
            return True
        return any(path.startswith(prefix) for prefix in ALLOWED_COMPANY_URL_PREFIXES)

    def _warm_up_ollama(self) -> None:
        self.get_logger().info(f"Preloading Ollama model={self.args.model}")
        started = time.time()
        body = {
            "model": str(self.args.model),
            "messages": [
                {"role": "system", "content": "Answer with one short word."},
                {"role": "user", "content": "Ready?"},
            ],
            "stream": False,
            "keep_alive": str(self.args.keep_alive),
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": 2,
                "num_ctx": int(self.args.num_ctx),
            },
        }
        if self.args.num_thread is not None:
            body["options"]["num_thread"] = int(self.args.num_thread)
        try:
            self._post_ollama_chat(body, timeout=float(self.args.timeout))
        except Exception as exc:
            self.get_logger().warning(f"Ollama warm-up failed: {exc}")
            return
        self.get_logger().info(f"Ollama warm-up finished in {time.time() - started:.1f}s")

    def _post_ollama_chat(self, body: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{str(self.args.ollama_url).rstrip('/')}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    def _run_startup_actions(self) -> None:
        if not self.args.no_headlight:
            try:
                code = self._set_headlight_once(
                    str(self.args.headlight_color), int(self.args.headlight_intensity))
                self.get_logger().info(
                    f"Headlight set color={self.args.headlight_color} intensity={self.args.headlight_intensity} code={code}"
                )
            except Exception as exc:
                self.get_logger().warning(f"Headlight startup action failed: {exc}")

        startup_speech = str(self.args.startup_speech).strip()
        if startup_speech and not self.args.no_startup_speech:
            self.last_reply = startup_speech
            self.last_reply_ts = time.time()
            code = self._speak_once(startup_speech)
            self.last_reply_ts = time.time()
            self.get_logger().info(f'Startup speech "{startup_speech}" exited {code}')

    def _set_headlight_once(self, color: str, intensity: int) -> int:
        script = Path(__file__).with_name("robot_headlight_once.py")
        command = [
            sys.executable,
            str(script),
            "--color",
            str(color),
            "--intensity",
            str(max(0, min(100, int(intensity)))),
            "--iface",
            str(self.args.iface),
            "--domain-id",
            str(int(self.args.domain_id)),
        ]
        env = os.environ.copy()
        env.setdefault("CYCLONEDDS_HOME", "/home/unitree/cyclonedds_ws/install/cyclonedds")
        env.setdefault("CYCLONEDDS_URI", "/home/unitree/cyclonedds_ws/cyclonedds.xml")
        proc = subprocess.run(command, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.stdout.strip():
            self.get_logger().info(proc.stdout.strip())
        return int(proc.returncode)

    def _save_event(
        self,
        received_at: float,
        payload: dict[str, Any],
        text: str,
        reply: str | None,
        *,
        error: str | None = None,
    ) -> None:
        record = {
            "received_at": received_at,
            "topic": self.args.topic,
            "model": self.args.model,
            "payload": payload,
            "text": text,
            "reply": reply,
        }
        if error:
            record["error"] = error
        with self.out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with self.text_out_path.open("a", encoding="utf-8") as f:
            f.write(f"user: {text}\n")
            if reply:
                f.write(f"robot: {reply}\n")
            if error:
                f.write(f"error: {error}\n")

    def _speak_once(self, text: str) -> int:
        script = Path(__file__).with_name("robot_say_once.py")
        command = [
            sys.executable,
            str(script),
            text,
            "--iface",
            str(self.args.iface),
            "--domain-id",
            str(int(self.args.domain_id)),
        ]
        if self.args.volume is not None:
            command.extend(["--volume", str(int(self.args.volume))])
        if self.args.tts_language:
            command.extend(["--language", str(self.args.tts_language)])
        env = os.environ.copy()
        env.setdefault("CYCLONEDDS_HOME", "/home/unitree/cyclonedds_ws/install/cyclonedds")
        env.setdefault("CYCLONEDDS_URI", "/home/unitree/cyclonedds_ws/cyclonedds.xml")
        proc = subprocess.run(command, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.stdout.strip():
            self.get_logger().info(proc.stdout.strip())
        return int(proc.returncode)


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = RobotChat(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
