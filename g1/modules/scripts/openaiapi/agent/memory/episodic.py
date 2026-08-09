"""Episodic memory (spec section 23).

Phase 1 store: append-only JSON Lines, one experience per line. No vector
index and no consolidation job yet (both are explicit deferred TODOs in
the plan) -- ``search`` here is a naive keyword/recency ranking, enough to
prove the write -> retrieve -> feed-to-planner contract end-to-end without
pretending to be more than it is.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_EPISODIC_PATH = Path(
    os.environ.get("G1_AGENT_EPISODIC", "~/.g1_agent/episodic.jsonl")
).expanduser()


@dataclass
class Episode:
    id: str
    timestamp: float
    goal: str
    initial_state: dict[str, Any] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    outcome: str = ""
    anomalies: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "Episode":
        return cls(**json.loads(line))


def new_episode(
    goal: str,
    *,
    initial_state: dict[str, Any] | None = None,
    actions: list[str] | None = None,
    observations: list[str] | None = None,
    outcome: str = "",
    anomalies: list[str] | None = None,
) -> Episode:
    return Episode(
        id=uuid.uuid4().hex,
        timestamp=time.time(),
        goal=goal,
        initial_state=initial_state or {},
        actions=actions or [],
        observations=observations or [],
        outcome=outcome,
        anomalies=anomalies or [],
    )


class EpisodicStore:
    def __init__(self, path: Path | str = DEFAULT_EPISODIC_PATH) -> None:
        self.path = Path(path).expanduser()

    def append(self, episode: Episode) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(episode.to_json() + "\n")

    def all(self) -> list[Episode]:
        if not self.path.exists():
            return []
        episodes: list[Episode] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(Episode.from_json(line))
            except Exception:
                continue  # a corrupt line is skipped, not fatal to the whole store
        return episodes

    def search(self, query: str, *, top_k: int = 5) -> list[Episode]:
        terms = {term.lower() for term in query.split() if term}
        if not terms:
            return sorted(self.all(), key=lambda ep: ep.timestamp, reverse=True)[:top_k]
        scored: list[tuple[int, float, Episode]] = []
        for ep in self.all():
            haystack = " ".join(
                [ep.goal, ep.outcome, " ".join(ep.actions), " ".join(ep.observations)]
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                scored.append((score, ep.timestamp, ep))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [ep for _, _, ep in scored[:top_k]]
