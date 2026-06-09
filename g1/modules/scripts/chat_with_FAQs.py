#!/usr/bin/env python3
from __future__ import annotations
from chat import DEFAULT_SYSTEM_PROMPT, RobotChat, clean_reply

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


FAQ_SYSTEM_PROMPT = (
    "Use the FAQ context when it is relevant. "
    "For questions about the FAQ topic, answer only from the FAQ context. "
    "If the FAQ context does not contain the answer, say you do not know yet. "
    "Keep the spoken answer natural and concise."
)
DEFAULT_FAQ_PATH = Path(__file__).with_name("FAQs.md")
WORD_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)
STOP_WORDS = {
    "a",
    "about",
    "and",
    "are",
    "as",
    "at",
    "be",
    "can",
    "could",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "our",
    "please",
    "tell",
    "that",
    "the",
    "this",
    "to",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class FAQEntry:
    title: str
    text: str
    source: str
    tokens: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen to robot ASR, retrieve FAQ context, answer with Ollama, and speak replies."
    )
    parser.add_argument("--topic", default="/audio_msg", help="ROS 2 ASR topic to subscribe to.")
    parser.add_argument("--out", default="/tmp/robot_chat_faqs.jsonl",
                        help="JSONL file for chat events.")
    parser.add_argument("--text-out", default="/tmp/robot_chat_faqs.txt",
                        help="Plain text transcript output file.")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434", help="Ollama base URL.")
    parser.add_argument("--model", default="qwen3.5:9b", help="Ollama model name.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT,
                        help="System prompt for the robot persona.")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Ollama sampling temperature.")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Ollama request timeout in seconds.")
    parser.add_argument("--max-history", type=int, default=4,
                        help="Maximum user/assistant messages to keep.")
    parser.add_argument("--num-predict", type=int, default=64,
                        help="Maximum tokens to generate per reply.")
    parser.add_argument("--num-ctx", type=int, default=2048, help="Ollama context window size.")
    parser.add_argument("--keep-alive", default="15m",
                        help="How long Ollama should keep the model loaded.")
    parser.add_argument("--num-thread", type=int, default=None,
                        help="Optional Ollama CPU thread count.")
    parser.add_argument(
        "--faq-file",
        action="append",
        default=None,
        help="FAQ file to retrieve from. Repeat for multiple files. Supports md, txt, json, jsonl, and csv.",
    )
    parser.add_argument("--faq-top-k", type=int, default=3,
                        help="Number of FAQ chunks to pass to the model.")
    parser.add_argument(
        "--faq-min-score",
        type=float,
        default=0.08,
        help="Minimum lexical retrieval score before FAQ context is used.",
    )
    parser.add_argument(
        "--faq-max-chars",
        type=int,
        default=1800,
        help="Maximum FAQ context characters sent to Ollama for one answer.",
    )
    parser.add_argument("--no-warmup", action="store_true",
                        help="Do not preload the model at startup.")
    parser.add_argument("--iface", default="eth0", help="DDS interface for robot audio playback.")
    parser.add_argument("--domain-id", type=int, default=0,
                        help="DDS domain ID for robot audio playback.")
    parser.add_argument("--volume", type=int, default=None, help="Optional playback volume 0-100.")
    parser.add_argument("--tts-language", default=None,
                        help="Optional Piper language, for example en, de, fr, es, ar.")
    parser.add_argument("--startup-speech", default="FAQ chat mode activated",
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


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (match.group(0).lower() for match in WORD_RE.finditer(text))
        if len(token) > 1 and token not in STOP_WORDS
    )


def entry_from_parts(title: str, body: str, source: str) -> FAQEntry | None:
    title = " ".join(str(title).split())
    body = " ".join(str(body).split())
    text = f"{title}\n{body}".strip() if title else body
    tokens = tokenize(text)
    if not text or not tokens:
        return None
    return FAQEntry(title=title or "FAQ", text=text, source=source, tokens=tokens)


def load_faq_entries(paths: list[Path]) -> list[FAQEntry]:
    entries: list[FAQEntry] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".json":
            entries.extend(load_json_faq(path))
        elif suffix == ".jsonl":
            entries.extend(load_jsonl_faq(path))
        elif suffix == ".csv":
            entries.extend(load_csv_faq(path))
        else:
            entries.extend(load_text_faq(path))
    return entries


def load_json_faq(path: Path) -> list[FAQEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("faqs", data) if isinstance(data, dict) else data
    if isinstance(rows, dict):
        rows = [{"question": key, "answer": value} for key, value in rows.items()]
    return entries_from_rows(rows if isinstance(rows, list) else [], str(path))


def load_jsonl_faq(path: Path) -> list[FAQEntry]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return entries_from_rows(rows, str(path))


def load_csv_faq(path: Path) -> list[FAQEntry]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return entries_from_rows(list(csv.DictReader(handle)), str(path))


def entries_from_rows(rows: list[Any], source: str) -> list[FAQEntry]:
    entries: list[FAQEntry] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            entry = entry_from_parts(f"FAQ {index}", str(row), source)
        else:
            title = row.get("question") or row.get("q") or row.get(
                "title") or row.get("heading") or f"FAQ {index}"
            answer = row.get("answer") or row.get("a") or row.get(
                "content") or row.get("text") or ""
            entry = entry_from_parts(str(title), str(answer), source)
        if entry:
            entries.append(entry)
    return entries


def load_text_faq(path: Path) -> list[FAQEntry]:
    text = path.read_text(encoding="utf-8")
    blocks = split_markdown_or_text(text)
    entries: list[FAQEntry] = []
    for index, block in enumerate(blocks, start=1):
        lines = [line.strip(" \t#") for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        title = lines[0]
        body = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]
        entry = entry_from_parts(title, body, str(path))
        if entry:
            entries.append(entry)
    if entries:
        return entries
    entry = entry_from_parts(path.stem, text, str(path))
    return [entry] if entry else []


def split_markdown_or_text(text: str) -> list[str]:
    heading_blocks = re.split(r"(?m)^(?=#{1,3}\s+)", text)
    blocks = [block.strip() for block in heading_blocks if block.strip()]
    if len(blocks) > 1:
        return blocks
    return [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]


class FAQRetriever:
    def __init__(self, entries: list[FAQEntry]) -> None:
        self.entries = entries
        doc_count = max(1, len(entries))
        document_frequency: dict[str, int] = {}
        for entry in entries:
            for token in set(entry.tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.idf = {
            token: math.log((doc_count + 1) / (frequency + 0.5)) + 1.0
            for token, frequency in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int, min_score: float) -> list[tuple[FAQEntry, float]]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_set = set(query_tokens)
        scored: list[tuple[FAQEntry, float]] = []
        for entry in self.entries:
            entry_counts: dict[str, int] = {}
            for token in entry.tokens:
                entry_counts[token] = entry_counts.get(token, 0) + 1
            score = 0.0
            for token in query_set:
                if token in entry_counts:
                    score += self.idf.get(token, 1.0) * (1.0 + math.log(entry_counts[token]))
            if query.strip().lower() in entry.text.lower():
                score += 2.0
            norm = math.sqrt(max(1, len(query_set)) * max(1, len(set(entry.tokens))))
            score = score / norm
            if score >= min_score:
                scored.append((entry, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, top_k)]


class FAQRobotChat(RobotChat):
    def __init__(self, args: argparse.Namespace) -> None:
        faq_paths = [Path(item).expanduser() for item in (args.faq_file or [])]
        if not faq_paths and DEFAULT_FAQ_PATH.exists():
            faq_paths = [DEFAULT_FAQ_PATH]
        missing = [path for path in faq_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("FAQ file not found: " + ", ".join(str(path)
                                    for path in missing))
        self.faq_entries = load_faq_entries(faq_paths) if faq_paths else []
        self.retriever = FAQRetriever(self.faq_entries)
        super().__init__(args)
        if self.faq_entries:
            self.get_logger().info(
                f"Loaded {len(self.faq_entries)} FAQ entries from {len(faq_paths)} file(s)")
        else:
            self.get_logger().warning(
                f"No FAQ entries loaded. Pass --faq-file or create {DEFAULT_FAQ_PATH}."
            )

    def _ask_ollama(self, user_text: str) -> str:
        max_history = max(1, int(self.args.max_history))
        history = self.messages[1:][-max_history:]
        retrieved = self.retriever.search(
            user_text,
            top_k=int(self.args.faq_top_k),
            min_score=float(self.args.faq_min_score),
        )
        faq_context = self._format_faq_context(retrieved)
        messages: list[dict[str, str]] = [self.messages[0], *history]
        if faq_context:
            messages.append(
                {"role": "system", "content": f"{FAQ_SYSTEM_PROMPT}\n\nFAQ context:\n{faq_context}"})
        else:
            messages.append({"role": "system", "content": FAQ_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": user_text})

        body = {
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
        if self.args.num_thread is not None:
            body["options"]["num_thread"] = int(self.args.num_thread)

        result = self._post_ollama_chat(body, timeout=float(self.args.timeout))
        reply = clean_reply(str(result.get("message", {}).get("content", "")))
        if not reply:
            reply = "I do not know yet."
        self.messages = [self.messages[0], *history,
                         {"role": "user", "content": user_text}, {"role": "assistant", "content": reply}]
        if retrieved:
            sources = ", ".join(f"{entry.source}:{score:.2f}" for entry, score in retrieved)
            self.get_logger().info(f"FAQ RAG sources: {sources}")
        return reply

    def _format_faq_context(self, retrieved: list[tuple[FAQEntry, float]]) -> str:
        max_chars = max(200, int(self.args.faq_max_chars))
        parts: list[str] = []
        total = 0
        for index, (entry, score) in enumerate(retrieved, start=1):
            chunk = f"[{index}] source={entry.source} score={score:.2f}\n{entry.text}"
            remaining = max_chars - total
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk[:remaining].rsplit(" ", 1)[0].strip()
            parts.append(chunk)
            total += len(chunk) + 2
        return "\n\n".join(parts)


def main() -> int:
    args = parse_args()
    import rclpy

    rclpy.init()
    node = FAQRobotChat(args)
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
