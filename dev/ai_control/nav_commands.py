from __future__ import annotations

import re
import string
from dataclasses import dataclass
from difflib import SequenceMatcher


NUMBER_WORDS = {
    "oh": "0",
    "zero": "0",
    "one": "1",
    "won": "1",
    "two": "2",
    "too": "2",
    "to": "2",
    "three": "3",
    "four": "4",
    "for": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "ate": "8",
    "nine": "9",
}


@dataclass(frozen=True)
class NavCommand:
    intent: str
    command_text: str
    response: str


def compact_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def normalize_text(text: str) -> str:
    return compact_text(text).lower().strip(string.punctuation + "，。！？、；：")


def clean_point_name(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^(call it|name it|save it as|save as|called|named)\s+", "", text).strip()
    text = re.sub(r"[^a-z0-9 _-]+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -_")
    tokens = [NUMBER_WORDS.get(token, token) for token in text.split()]
    return " ".join(tokens)


def similar_to_any(text: str, phrases: tuple[str, ...], threshold: float = 0.82) -> bool:
    low = normalize_text(text)
    return any(SequenceMatcher(None, low, phrase).ratio() >= threshold for phrase in phrases)


def parse_nav_command(text: str) -> NavCommand | None:
    """Recognize the same text command shapes accepted by nav_bot.py.

    The returned command_text is intended for /model_api/navbot_command, where
    nav_bot.py still owns the SLAM state machine, point file, and execution.
    """
    original = compact_text(text)
    low = normalize_text(original)
    if not low:
        return None

    inline_name = _extract_add_point_name(low)
    if inline_name is not None:
        command = f"save current point as {inline_name}"
        return NavCommand("add_current_point", command, f"Send nav bot command: {command}")

    if _wants_add_current_point(low):
        return NavCommand("ask_point_name", original, "Send nav bot command: save the current point")

    if _wants_start_mapping(low):
        return NavCommand("start_mapping", original, "Send nav bot command: start mapping")

    if _wants_stop_mapping(low):
        return NavCommand("stop_mapping", original, "Send nav bot command: stop mapping")

    if _wants_relocate(low):
        return NavCommand("relocate", original, "Send nav bot command: relocate")

    if _wants_resume(low):
        return NavCommand("resume_navigation", original, "Send nav bot command: resume navigation")

    if _wants_pause_or_stop(low):
        return NavCommand("pause_navigation", original, "Send nav bot command: stop navigation")

    if _wants_close_slam(low):
        return NavCommand("close_slam", original, "Send nav bot command: stop SLAM")

    point_name = _extract_go_to_name(low)
    if point_name:
        command = f"go to {point_name}"
        return NavCommand("go_to_point", command, f"Send nav bot command: {command}")

    if "list" in low and "point" in low:
        return NavCommand("list_points", original, "Send nav bot command: list points")

    if _wants_clear_points(low):
        return NavCommand("clear_points", original, "Send nav bot command: clear points")

    if "status" in low and any(word in low for word in ("nav", "navigation", "slam", "map", "mapping", "point")):
        return NavCommand("status", original, "Send nav bot command: navigation status")

    return None


def _wants_clear_points(low: str) -> bool:
    if "point" not in low:
        return False
    return any(phrase in low for phrase in ("clear", "erase", "reset", "delete all", "forget all", "remove all"))


def _wants_start_mapping(low: str) -> bool:
    phrases = ("start mapping", "begin mapping", "create map", "make a map")
    return any(phrase in low for phrase in phrases) or similar_to_any(low, phrases)


def _wants_stop_mapping(low: str) -> bool:
    phrases = ("stop mapping", "finish mapping", "end mapping", "save map", "save the map")
    return any(phrase in low for phrase in phrases) or similar_to_any(low, phrases)


def _wants_relocate(low: str) -> bool:
    phrases = ("relocate", "localize", "relocalize", "init pose")
    return any(word in low for word in phrases) or similar_to_any(low, phrases, threshold=0.68)


def _wants_add_current_point(low: str) -> bool:
    if "current" not in low or "point" not in low:
        return False
    return any(word in low for word in ("add", "at", "save", "mark", "remember"))


def _extract_add_point_name(low: str) -> str | None:
    match = re.search(r"(?:add|save|mark|remember)\s+(?:the\s+)?current\s+point\s+(?:as|called|named)\s+(.+)$", low)
    return clean_point_name(match.group(1)) if match else None


def _wants_pause_or_stop(low: str) -> bool:
    return low in {"stop", "cancel", "halt"} or any(
        phrase in low for phrase in ("stop navigation", "pause navigation", "cancel navigation", "hold position")
    )


def _wants_resume(low: str) -> bool:
    return any(phrase in low for phrase in ("resume navigation", "continue navigation", "keep going"))


def _wants_close_slam(low: str) -> bool:
    return any(phrase in low for phrase in ("stop slam", "close slam", "shutdown slam", "shut down slam"))


def _extract_go_to_name(low: str) -> str | None:
    patterns = (
        r"^(?:go|navigate|drive|walk)\s+to\s+(.+)$",
        r"^take\s+me\s+to\s+(.+)$",
        r"^go\s+to\s+point\s+(.+)$",
        r"^navigate\s+to\s+point\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, low)
        if match:
            return clean_point_name(match.group(1))
    return None
