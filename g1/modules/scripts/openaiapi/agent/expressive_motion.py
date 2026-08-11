"""Expressive CSV motion adapter.

These motions visually accompany interaction; they are not evidence of
the model's internal reasoning or emotional state.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .monitor import MonitorEventBus
from .settings.models import AgentSettings
from .skills import SkillResult


@dataclass
class ExpressiveMotionRuntime:
    current_kind: Optional[str] = None
    current_file: Optional[str] = None
    reason: Optional[str] = None
    started_at: Optional[float] = None
    last_completed_at: Optional[float] = None
    last_error: Optional[str] = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_expressive_motion": self.current_kind,
            "motion_source_file": self.current_file,
            "reason": self.reason,
            "started_at": self.started_at,
            "last_completed_at": self.last_completed_at,
            "last_error": self.last_error,
        }


class ExpressiveMotionController:
    def __init__(
        self,
        *,
        robot: Any = None,
        monitor: Optional[MonitorEventBus] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.robot = robot
        self.monitor = monitor
        self.rng = rng or random.Random()
        self.runtime = ExpressiveMotionRuntime()
        self._last_by_kind: dict[str, float] = {}
        self._lock = threading.RLock()

    def select_motion_file(self, kind: str, directory: str) -> Path:
        root = Path(directory).expanduser()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"expressive motion directory does not exist: {root}")
        matches = sorted(path for path in root.glob(f"{kind}_*.csv") if path.is_file())
        if not matches:
            raise FileNotFoundError(f"no {kind}_*.csv files found in {root}")
        choice = self.rng.choice(matches)
        if not choice.exists() or not choice.is_file():
            raise FileNotFoundError(f"selected expressive motion file is missing: {choice}")
        if not choice.stat().st_size:
            raise ValueError(f"selected expressive motion file is empty: {choice}")
        return choice

    def can_run(self, kind: str, settings: AgentSettings) -> tuple[bool, str]:
        if self.robot is None or not hasattr(self.robot, "repeat"):
            return False, "Robot.repeat() is unavailable."
        expressive = settings.expressive_motion
        if not expressive.enabled:
            return False, "expressive_motion.enabled=false."
        kind_settings = getattr(expressive, kind, None)
        if kind_settings is None:
            return False, f"unknown expressive motion kind {kind!r}."
        if not kind_settings.enabled:
            return False, f"expressive_motion.{kind}.enabled=false."
        if not settings.announcements.gesture_enabled:
            return False, "announcements.gesture_enabled=false."
        elapsed = time.time() - self._last_by_kind.get(kind, 0.0)
        if elapsed < float(kind_settings.cooldown_s):
            return False, f"expressive motion cooldown active ({elapsed:.1f}s elapsed)."
        if self.runtime.current_kind:
            return False, f"expressive motion already running: {self.runtime.current_kind}."
        return True, "allowed"

    def run(self, kind: str, *, settings: AgentSettings, reason: str = "manual") -> SkillResult:
        allowed, reason_text = self.can_run(kind, settings)
        if not allowed:
            return SkillResult(ok=False, message=reason_text, detail={"kind": kind, "reason": reason})
        try:
            motion_file = self.select_motion_file(kind, settings.expressive_motion.motion_directory)
        except Exception as exc:
            self._emit("expressive_motion_failed", f"{kind}: {exc}", kind=kind, reason=reason)
            with self._lock:
                self.runtime.last_error = str(exc)
            return SkillResult(ok=False, message=str(exc), detail={"kind": kind, "reason": reason})

        self._emit(
            "expressive_motion_selected",
            f"{kind} selected {motion_file.name}",
            kind=kind,
            reason=reason,
            file=str(motion_file),
        )
        with self._lock:
            self.runtime.current_kind = kind
            self.runtime.current_file = str(motion_file)
            self.runtime.reason = reason
            self.runtime.started_at = time.time()
            self.runtime.last_error = None
        self._emit(
            "expressive_motion_started",
            f"{kind} started",
            kind=kind,
            reason=reason,
            file=str(motion_file),
        )
        try:
            result = self.robot.repeat(motion_file=str(motion_file))
        except Exception as exc:
            with self._lock:
                self.runtime.last_error = str(exc)
                self.runtime.current_kind = None
                self.runtime.current_file = None
                self.runtime.reason = None
                self.runtime.last_completed_at = time.time()
            self._emit("expressive_motion_failed", f"{kind}: {exc}", kind=kind, reason=reason, file=str(motion_file))
            return SkillResult(ok=False, message=f"{kind} expressive motion failed: {exc}", detail={"kind": kind})
        finally:
            self._last_by_kind[kind] = time.time()

        with self._lock:
            self.runtime.current_kind = None
            self.runtime.current_file = None
            self.runtime.reason = None
            self.runtime.last_completed_at = time.time()
        self._emit(
            "expressive_motion_completed",
            f"{kind} completed",
            kind=kind,
            reason=reason,
            file=str(motion_file),
        )
        return SkillResult(
            ok=True,
            message=f"{kind} expressive motion executed from {motion_file.name}",
            detail={"kind": kind, "motion_file": str(motion_file), "repeat_result": result},
        )

    def run_background(self, kind: str, *, settings: AgentSettings, reason: str = "event") -> None:
        allowed, _ = self.can_run(kind, settings)
        if not allowed:
            return
        thread = threading.Thread(
            target=lambda: self.run(kind, settings=settings, reason=reason),
            name=f"g1-expressive-{kind}",
            daemon=True,
        )
        thread.start()

    def _emit(self, event: str, summary: str, **metadata: Any) -> None:
        if self.monitor is not None:
            self.monitor.emit("expressive", event, summary, metadata=metadata)
