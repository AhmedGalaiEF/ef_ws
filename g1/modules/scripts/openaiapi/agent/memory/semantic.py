"""Semantic / empirical memory (spec sections 23, 25).

Consolidated claims with a confidence value and provenance
(``supporting_episodes``). Phase 1 stores these as a flat JSON list (small
volume expected relative to episodic) with exact-text dedup on write --
real clustering/consolidation from repeated episodic patterns is a
deferred TODO (spec section 24's "repeated pattern -> semantic empirical
claim" pipeline is not automated yet).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SEMANTIC_PATH = Path(
    os.environ.get("G1_AGENT_SEMANTIC", "~/.g1_agent/semantic.json")
).expanduser()


@dataclass
class SemanticClaim:
    claim: str
    confidence: float = 0.5
    supporting_episodes: list[str] = field(default_factory=list)


class SemanticStore:
    def __init__(self, path: Path | str = DEFAULT_SEMANTIC_PATH) -> None:
        self.path = Path(path).expanduser()

    def _load_raw(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_raw(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(items, indent=2, sort_keys=True), encoding="utf-8")

    def all(self) -> list[SemanticClaim]:
        return [SemanticClaim(**item) for item in self._load_raw()]

    def upsert(self, claim: SemanticClaim) -> None:
        """Exact-text dedup: a repeated claim strengthens confidence and
        merges supporting_episodes rather than duplicating the entry."""
        items = self._load_raw()
        for item in items:
            if item["claim"] == claim.claim:
                item["confidence"] = max(float(item["confidence"]), float(claim.confidence))
                merged = set(item.get("supporting_episodes", [])) | set(claim.supporting_episodes)
                item["supporting_episodes"] = sorted(merged)
                self._save_raw(items)
                return
        items.append(asdict(claim))
        self._save_raw(items)

    def search(self, query: str, *, top_k: int = 5) -> list[SemanticClaim]:
        terms = {term.lower() for term in query.split() if term}
        claims = self.all()
        if not terms:
            return sorted(claims, key=lambda c: c.confidence, reverse=True)[:top_k]
        scored = [
            (sum(1 for term in terms if term in claim.claim.lower()), claim.confidence, claim)
            for claim in claims
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [claim for _, _, claim in scored[:top_k]]
