"""Procedural / tacit memory (spec sections 23, 24, 25).

Validated behavioral adaptations -- ``{skill, condition, recommended_parameters,
confidence, derived_from}``. This store only ever holds the *proposal*/
*advisory* form: a recommended-parameters record the planner may read and
factor into a decision. It is never wired to actually rewrite a
controller's live gains or safety-relevant parameters (spec section 24's
hard rule) -- promoting one of these into an actual skill/controller
parameter, if that is ever done, is a separate, explicitly deterministic,
out-of-band change to that skill's code, not something this store or the
model can do by itself.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import atomic_write_json

DEFAULT_PROCEDURAL_PATH = Path(
    os.environ.get("G1_AGENT_PROCEDURAL", "~/.g1_agent/procedural.json")
).expanduser()


@dataclass
class ProceduralAdaptation:
    skill: str
    condition: dict[str, Any] = field(default_factory=dict)
    recommended_parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    derived_from: list[str] = field(default_factory=list)


class ProceduralStore:
    def __init__(self, path: Path | str = DEFAULT_PROCEDURAL_PATH) -> None:
        self.path = Path(path).expanduser()

    def _load_raw(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_raw(self, items: list[dict[str, Any]]) -> None:
        atomic_write_json(self.path, items)

    def all(self) -> list[ProceduralAdaptation]:
        return [ProceduralAdaptation(**item) for item in self._load_raw()]

    def add(self, adaptation: ProceduralAdaptation) -> None:
        items = self._load_raw()
        items.append(asdict(adaptation))
        self._save_raw(items)

    def for_skill(self, skill: str) -> list[ProceduralAdaptation]:
        return [a for a in self.all() if a.skill == skill]
