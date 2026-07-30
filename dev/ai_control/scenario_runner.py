from __future__ import annotations

import re
import string
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict

from ai_control import ollama_client, thinker, vision
from ai_control.config import AIConfig
from ai_control.robot_backend import RobotBackend


DispatchFn = Callable[[Dict[str, Any], RobotBackend], str]


@dataclass
class ScenarioContext:
    history: list[dict[str, str]]
    answers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioStep:
    index: int
    text: str


def parse_scenario_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[\.)]\s+", "", line)
        if line:
            lines.append(line)
    return lines


class ScenarioRunner:
    def __init__(
        self,
        cfg: AIConfig,
        backend: RobotBackend,
        dispatch: DispatchFn,
        knowledge_files: list[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.backend = backend
        self.dispatch = dispatch
        self.knowledge = _KnowledgeBase(knowledge_files or [])

    def run(self, lines: list[str], history: list[dict[str, str]]) -> None:
        steps = [ScenarioStep(index=index, text=line) for index, line in enumerate(lines, start=1)]
        if not steps:
            print("assistant> No scenario steps found.")
            return
        print(f"assistant> Parsed {len(steps)} scenario steps.")
        answer = input("  Execute this scenario now? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("  - scenario declined")
            return

        ctx = ScenarioContext(history=history)
        for step in steps:
            print(f"  [scenario {step.index}/{len(steps)}] {step.text}")
            self._run_step(step.text, ctx)

    def _run_step(self, text: str, ctx: ScenarioContext) -> None:
        low = _normalize(text)

        if _is_question_wait(low):
            question = input("host question> ").strip()
            ctx.answers["last_question"] = question
            ctx.history.append({"role": "user", "content": question})
            return

        if _is_object_selection_wait(low):
            selected = input("selected object> ").strip()
            ctx.answers["selected_object"] = selected
            return

        if _is_generic_wait(low):
            observed = input("host> ").strip()
            ctx.answers["last_host_prompt"] = observed
            ctx.history.append({"role": "user", "content": observed})
            return

        if low.startswith("announce"):
            self._say(_after_colon(text) or text)
            return

        if low.startswith("ask "):
            self._say(_spoken_question(text))
            return

        if "answer question" in low or "ansewer question" in low:
            self._answer_last_question(text, ctx)
            return

        if "what objects" in low or "objects do you see" in low or "name 2 objects" in low or "name two objects" in low:
            self._vision_answer(text, ctx)
            return

        if "look for" in low and ("object" in low or "<" in text):
            self._vision_answer(text, ctx)
            return

        if "go to" in low or "go to '" in low:
            point = _extract_point_name(text)
            if point:
                self._dispatch({"name": "navbot_command", "args": {"text": f"go to {point}"}})
            else:
                print("assistant> [scenario pause] I could not find the destination point name.")
            return

        if "greet" in low or "wave" in low:
            self._dispatch({"name": "gesture", "args": {"name": "high wave"}})
            if "introduce yourself" in low:
                self._say(_limit_words("Hello, I am the EF Robotics humanoid assistant.", _word_limit(text)))
            return

        if "thinking gesture" in low or "thunking gesture" in low or "let me think" in low:
            self._dispatch({"name": "gesture", "args": {"name": "right hand up"}})
            if "announce" in low:
                self._say(_after_colon(text) or "let me think")
            return

        if "introduce ef robotics" in low:
            self._say(_limit_words("EF Robotics builds practical humanoid robot systems for real-world assistance.", _word_limit(text)))
            return

        if low.startswith("grab ") or "grab <selected object>" in low or "grab selected object" in low:
            selected = ctx.answers.get("selected_object") or _extract_angle_value(text) or "the selected object"
            print(f"assistant> [scenario note] No grasp planner is available; closing the right hand for {selected!r}.")
            self._dispatch({"name": "hand_close", "args": {"hand": "right"}})
            return

        if "open right hand" in low:
            self._dispatch({"name": "hand_open", "args": {"hand": "right"}})
            return

        if "open left hand" in low:
            self._dispatch({"name": "hand_open", "args": {"hand": "left"}})
            return

        if "step back" in low:
            self._dispatch({"name": "move", "args": {"vx": -0.2, "vy": 0.0, "vyaw": 0.0, "duration": 1.0}})
            return

        if "release arms" in low or "release arm" in low:
            self._dispatch({"name": "release_arms", "args": {}})
            return

        print(f"assistant> [scenario pause] Unsupported step, handle manually: {text}")
        input("  Press Enter to continue scenario...")

    def _say(self, text: str) -> None:
        message = _clean_spoken(text)
        if not message:
            return
        print(f"assistant> {message}")
        self._dispatch({"name": "say", "args": {"text": message}})

    def _answer_last_question(self, text: str, ctx: ScenarioContext) -> None:
        question = ctx.answers.get("last_question") or input("host question> ").strip()
        ctx.answers["last_question"] = question
        limit = _word_limit(text)
        prompt = question
        context = self.knowledge.context_for(question)
        if limit:
            prompt = f"Answer in less than {limit} words: {question}"
        if context:
            prompt = (
                "Use this local knowledge context when relevant. "
                "If the context does not answer the question, say so briefly.\n\n"
                f"{context}\n\nQuestion: {prompt}"
            )
        try:
            result = thinker.think(ctx.history, prompt, self.cfg)
            answer = result.response
        except ollama_client.OllamaError as exc:
            answer = f"I cannot answer right now: {exc}"
        answer = _limit_words(answer, limit)
        ctx.history.append({"role": "assistant", "content": answer})
        self._say(answer)

    def _vision_answer(self, text: str, ctx: ScenarioContext) -> None:
        frame = self.backend.capture_frame()
        if frame is None:
            answer = "No camera frame is available."
        else:
            try:
                answer = vision.describe(frame, text, self.cfg)
            except ollama_client.OllamaError as exc:
                answer = f"Vision model error: {exc}"
        ctx.answers["last_vision"] = answer
        self._say(_limit_words(answer, _word_limit(text)))

    def _dispatch(self, tool_call: dict[str, Any]) -> None:
        outcome = self.dispatch(tool_call, self.backend)
        if "failed:" in outcome.lower() or "declined" in outcome.lower():
            answer = input("  Continue scenario after this outcome? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                raise KeyboardInterrupt


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().strip(string.punctuation + "，。！？、；：").split())


def _after_colon(text: str) -> str:
    return text.split(":", 1)[1].strip() if ":" in text else ""


def _clean_spoken(text: str) -> str:
    return text.strip().strip("\"'")


def _spoken_question(text: str) -> str:
    cleaned = re.sub(r"^ask\s+", "", text.strip(), flags=re.IGNORECASE).strip()
    return cleaned if cleaned.endswith("?") else f"{cleaned}?"


def _word_limit(text: str) -> int | None:
    match = re.search(r"(?:less than|under|max(?:imum)? of?)\s+(\d+)\s+words?", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _limit_words(text: str, limit: int | None) -> str:
    if not limit or limit <= 0:
        return text.strip()
    words = text.strip().split()
    if len(words) < limit:
        return text.strip()
    return " ".join(words[: max(1, limit - 1)]).rstrip(" ,;:") + "."


def _extract_point_name(text: str) -> str:
    angle_value = _extract_angle_value(text)
    if angle_value:
        return re.sub(r"^point\s+", "", angle_value, flags=re.IGNORECASE).strip()
    match = re.search(r"go\s+to\s+['\"]?([^'\"]+)['\"]?$", text, flags=re.IGNORECASE)
    if not match:
        return ""
    point = match.group(1).strip()
    point = re.sub(r"^point\s+", "", point, flags=re.IGNORECASE).strip()
    return point


def _extract_angle_value(text: str) -> str:
    match = re.search(r"<([^>]+)>", text)
    return match.group(1).strip() if match else ""


def _is_question_wait(low: str) -> bool:
    return "listen to question" in low or ("wait" in low and "question" in low and "object" not in low)


def _is_object_selection_wait(low: str) -> bool:
    return "which object" in low or "selected object" in low or ("wait" in low and "object to grab" in low)


def _is_generic_wait(low: str) -> bool:
    return low.startswith("wait ") or low.startswith("what for host") or low.startswith("wait for host")


class _KnowledgeBase:
    def __init__(self, paths: list[str]) -> None:
        self.entries: list[tuple[str, str]] = []
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            if not path.is_file():
                print(f"  ! knowledge file not found: {path}")
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                print(f"  ! failed to read knowledge file {path}: {exc}")
                continue
            self.entries.extend(_entries_from_file(path, text))

    def context_for(self, query: str, *, limit: int = 3, max_chars: int = 2200) -> str:
        query_tokens = set(_tokens(query))
        if not query_tokens or not self.entries:
            return ""
        scored: list[tuple[int, str, str]] = []
        for title, text in self.entries:
            score = len(query_tokens.intersection(_tokens(text)))
            if score:
                scored.append((score, title, text))
        scored.sort(key=lambda item: item[0], reverse=True)
        chunks: list[str] = []
        remaining = max_chars
        for _score, title, text in scored[:limit]:
            chunk = f"{title}\n{text}".strip()
            if len(chunk) > remaining:
                chunk = chunk[:remaining].rsplit(" ", 1)[0]
            if chunk:
                chunks.append(chunk)
                remaining -= len(chunk)
            if remaining <= 0:
                break
        return "\n\n".join(chunks)


def _entries_from_file(path: Path, text: str) -> list[tuple[str, str]]:
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return [(path.name, text)]
        return _entries_from_json(path.name, value)
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    return [(f"{path.name} #{index}", block) for index, block in enumerate(blocks, start=1)]


def _entries_from_json(title: str, value: Any) -> list[tuple[str, str]]:
    if isinstance(value, list):
        return [(f"{title} #{index}", _flatten_json(item)) for index, item in enumerate(value, start=1)]
    if isinstance(value, dict):
        return [(str(key), _flatten_json(child)) for key, child in value.items()]
    return [(title, str(value))]


def _flatten_json(value: Any) -> str:
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_flatten_json(child)}" for key, child in value.items())
    if isinstance(value, list):
        return "\n".join(_flatten_json(child) for child in value)
    return str(value)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2)
