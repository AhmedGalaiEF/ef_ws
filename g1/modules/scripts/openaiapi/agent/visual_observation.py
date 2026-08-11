"""Semantic visual observations for cognition and monitor state."""
from __future__ import annotations

import re
import time
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class VisualEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    confidence: Optional[float] = None
    location: Optional[str] = None


class VisualObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: float = Field(default_factory=time.time)
    scene_summary: Optional[str] = None
    people: List[VisualEntity] = Field(default_factory=list)
    objects: List[VisualEntity] = Field(default_factory=list)
    spatial_relations: List[str] = Field(default_factory=list)
    notable_changes: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    source_type: str = "vision_observation"
    model: str = ""


class VisualObservationTracker:
    def __init__(self) -> None:
        self.latest: Optional[VisualObservation] = None
        self._last_signature: str = ""
        self._last_event_at: float = 0.0

    def observe_from_answer(self, *, answer: str, model: str, confidence: Optional[float] = 0.6) -> VisualObservation:
        objects = [VisualEntity(label=label, confidence=confidence) for label in self._extract_objects(answer)]
        people = []
        lowered = answer.lower()
        if any(word in lowered for word in ("person", "people", "human", "user", "man", "woman", "person")):
            people.append(VisualEntity(label="person", confidence=confidence))
        obs = VisualObservation(
            scene_summary=answer.strip(),
            people=people,
            objects=objects,
            confidence=confidence,
            model=model,
        )
        previous = self.latest
        self.latest = obs
        changes = self._changes(previous, obs)
        obs.notable_changes.extend(changes)
        return obs

    def should_wake_cognition(self, obs: VisualObservation, *, cooldown_s: float = 10.0) -> bool:
        signature = self._signature(obs)
        now = time.time()
        if signature == self._last_signature and now - self._last_event_at < cooldown_s:
            return False
        if obs.notable_changes:
            self._last_signature = signature
            self._last_event_at = now
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        obs = self.latest
        if obs is None:
            return {
                "person_count": 0,
                "important_objects": [],
                "last_semantic_visual_change": "",
                "vision_confidence": None,
                "last_observation_age_s": None,
                "model": "",
            }
        return {
            "person_count": len(obs.people),
            "important_objects": [item.label for item in obs.objects[:8]],
            "last_semantic_visual_change": ", ".join(obs.notable_changes[:4]),
            "vision_confidence": obs.confidence,
            "last_observation_age_s": max(0.0, time.time() - obs.timestamp),
            "model": obs.model,
            "scene_summary": obs.scene_summary,
        }

    @staticmethod
    def _extract_objects(answer: str) -> list[str]:
        lowered = answer.lower()
        candidates = [
            "cup",
            "coffee cup",
            "table",
            "chair",
            "laptop",
            "phone",
            "door",
            "trash can",
            "paper",
            "bottle",
            "person",
        ]
        found = []
        for label in candidates:
            if re.search(r"\b" + re.escape(label) + r"s?\b", lowered) and label not in found:
                found.append(label)
        return found

    @staticmethod
    def _signature(obs: VisualObservation) -> str:
        objects = ",".join(sorted(item.label for item in obs.objects))
        people = str(len(obs.people))
        return f"p={people};o={objects}"

    def _changes(self, previous: Optional[VisualObservation], current: VisualObservation) -> List[str]:
        if previous is None:
            changes = []
            if current.people:
                changes.append("person detected")
            if current.objects:
                changes.append("objects detected: " + ", ".join(item.label for item in current.objects[:4]))
            return changes
        old_objects = {item.label for item in previous.objects}
        new_objects = {item.label for item in current.objects}
        changes = []
        added = sorted(new_objects - old_objects)
        removed = sorted(old_objects - new_objects)
        if len(current.people) > len(previous.people):
            changes.append("new person detected")
        if added:
            changes.append("new objects: " + ", ".join(added[:4]))
        if removed:
            changes.append("objects disappeared: " + ", ".join(removed[:4]))
        return changes
