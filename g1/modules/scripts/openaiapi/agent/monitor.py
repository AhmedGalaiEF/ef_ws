"""Structured monitor event stream for the agent runtime.

The monitor intentionally records public runtime facts only: observed
state summaries, planner decisions, skill events, memory activity and
learning activity. It is not a place for hidden model reasoning.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional
import time

from pydantic import BaseModel, ConfigDict, Field


class MonitorEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: float = Field(default_factory=time.time)
    category: str
    event: str
    summary: str
    references: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def line(self) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"{stamp}  {self.category.upper():<9} {self.event:<28} {self.summary}"


class MonitorEventBus:
    def __init__(self, *, max_events: int = 1000) -> None:
        self._events: deque[MonitorEvent] = deque(maxlen=max(10, int(max_events)))

    @property
    def max_events(self) -> int:
        return int(self._events.maxlen or 0)

    def resize(self, max_events: int) -> None:
        max_events = max(10, int(max_events))
        if max_events == self.max_events:
            return
        self._events = deque(list(self._events)[-max_events:], maxlen=max_events)

    def clear(self) -> None:
        self._events.clear()

    def emit(
        self,
        category: str,
        event: str,
        summary: str,
        *,
        references: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MonitorEvent:
        item = MonitorEvent(
            category=str(category),
            event=str(event),
            summary=str(summary),
            references=list(references or []),
            metadata=dict(metadata or {}),
        )
        self._events.append(item)
        return item

    def recent(self, limit: int = 50) -> List[MonitorEvent]:
        return list(self._events)[-max(0, int(limit)):]

    def counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for event in self._events:
            counts[event.category] = counts.get(event.category, 0) + 1
        return counts

    def snapshot(self, *, limit: int = 80) -> Dict[str, Any]:
        return {
            "event_buffer_size": self.max_events,
            "events": [event.model_dump() for event in self.recent(limit)],
            "counts": self.counts(),
        }
