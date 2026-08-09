"""Autobiographical memory (spec section 23).

A compact, append-only log of meaningful events -- not the full event
history (that's episodic memory / engineering logs). ``summary()`` renders
a bounded amount of recent entries as the ``autobiography_summary`` string
handed to the planner, per spec section 23's "retrieve relevant
autobiography context rather than injecting the entire history into every
model call".
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_AUTOBIOGRAPHY_PATH = Path(
    os.environ.get("G1_AGENT_AUTOBIOGRAPHY", "~/.g1_agent/autobiography.jsonl")
).expanduser()


@dataclass
class AutobiographyEntry:
    timestamp: float
    summary: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "AutobiographyEntry":
        return cls(**json.loads(line))


class AutobiographyStore:
    def __init__(self, path: Path | str = DEFAULT_AUTOBIOGRAPHY_PATH) -> None:
        self.path = Path(path).expanduser()

    def append(self, summary: str, *, timestamp: float | None = None) -> None:
        entry = AutobiographyEntry(timestamp=timestamp if timestamp is not None else time.time(), summary=summary)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.to_json() + "\n")

    def all(self) -> list[AutobiographyEntry]:
        if not self.path.exists():
            return []
        entries: list[AutobiographyEntry] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(AutobiographyEntry.from_json(line))
            except Exception:
                continue
        return entries

    def summary(self, *, max_entries: int = 10) -> str | None:
        entries = sorted(self.all(), key=lambda e: e.timestamp, reverse=True)[:max_entries]
        if not entries:
            return None
        lines = []
        for entry in reversed(entries):  # oldest-first for a readable narrative
            date = time.strftime("%Y-%m-%d", time.localtime(entry.timestamp))
            lines.append(f"{date}: {entry.summary}")
        return "\n".join(lines)
