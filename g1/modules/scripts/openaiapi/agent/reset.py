"""Cognitive-state reset manager.

``/reset`` is deliberately scoped to agent-owned cognitive continuity. It
does not touch source code, static/documentary RAG, robot controllers,
motion CSVs, SDK wrappers, or certified safety configuration.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from .lifecycle import LifecycleController
from .models import EventType, LifecycleState
from .monitor import MonitorEventBus
from .scheduler import CognitiveScheduler
from .semantic_state import SemanticStateTracker


ResetScope = Literal["runtime", "conversation", "learned", "autobiography", "full"]


@dataclass
class ResetResult:
    scope: str
    ok: bool
    message: str
    cleared: list[str] = field(default_factory=list)
    preserved: list[str] = field(default_factory=list)
    backup_dir: Optional[str] = None


class ResetManager:
    SCOPES = {"runtime", "conversation", "learned", "autobiography", "full"}
    DESTRUCTIVE_SCOPES = {"learned", "autobiography", "full"}

    def __init__(self, *, agent: Any, monitor: Optional[MonitorEventBus] = None) -> None:
        self.agent = agent
        self.monitor = monitor
        self._lock = threading.RLock()
        self.base_dir = self._discover_base_dir()
        self.backup_root = self.base_dir / "reset_backups"
        self.audit_path = self.base_dir / "reset_audit.jsonl"

    @staticmethod
    def confirmation_phrase(scope: str) -> str:
        return f"RESET {scope.strip().upper()}"

    @classmethod
    def confirmation_matches(cls, scope: str, text: str) -> bool:
        return text.strip() == cls.confirmation_phrase(scope)

    def requires_confirmation(self, scope: str, settings: Any) -> bool:
        return scope in self.DESTRUCTIVE_SCOPES and bool(settings.reset.require_confirmation)

    def list_backups(self) -> list[dict[str, Any]]:
        if not self.backup_root.exists():
            return []
        rows = []
        for path in sorted(self.backup_root.iterdir(), reverse=True):
            if not path.is_dir():
                continue
            rows.append({"name": path.name, "path": str(path), "created_at": path.stat().st_mtime})
        return rows

    def reset(self, scope: str, *, initiator: str = "cli_operator") -> ResetResult:
        scope = scope.strip().lower()
        if scope not in self.SCOPES:
            return ResetResult(scope=scope, ok=False, message=f"unknown reset scope {scope!r}")
        settings = self.agent.settings.effective()
        with self._lock:
            self._emit("reset_requested", scope, metadata={"scope": scope, "initiator": initiator})
            self._emit("reset_started", scope, metadata={"scope": scope, "initiator": initiator})
            backup_dir = None
            if bool(settings.reset.create_backup) and scope in self.DESTRUCTIVE_SCOPES:
                backup_dir = self._create_backup(scope)
            touched: list[tuple[Path, str]] = []
            result = ResetResult(scope=scope, ok=False, message="", backup_dir=None if backup_dir is None else str(backup_dir))
            try:
                if scope == "runtime":
                    touched.extend(self._reset_runtime())
                elif scope == "conversation":
                    touched.extend(self._reset_conversation())
                elif scope == "learned":
                    touched.extend(self._reset_learned())
                elif scope == "autobiography":
                    touched.extend(self._reset_autobiography())
                elif scope == "full":
                    touched.extend(self._reset_full(settings=settings))
                result.ok = True
                result.cleared = sorted({label for _, label in touched})
                result.preserved = self._preserved_for(scope, settings=settings)
                result.message = self._result_message(scope, result)
                self._append_audit(scope=scope, initiator=initiator, ok=True, backup_dir=backup_dir)
                self._emit("reset_completed", result.message, metadata=result.__dict__)
                return result
            except Exception as exc:
                if backup_dir is not None:
                    self._restore_backup(backup_dir)
                result.ok = False
                result.message = f"reset {scope} failed: {exc}"
                self._append_audit(scope=scope, initiator=initiator, ok=False, backup_dir=backup_dir, error=str(exc))
                self._emit("reset_failed", result.message, metadata={"scope": scope, "error": str(exc)})
                return result

    def _discover_base_dir(self) -> Path:
        memory = getattr(self.agent, "memory", None)
        episodic = getattr(memory, "episodic", None)
        path = getattr(episodic, "path", None)
        if path is not None:
            return Path(path).expanduser().parent
        return Path(os.environ.get("G1_AGENT_MEMORY_DIR", "~/.g1_agent/memory")).expanduser()

    def _owned_files(self) -> dict[str, Path]:
        agent = self.agent
        memory = agent.memory
        files: dict[str, Path] = {
            "runtime_checkpoint": Path(agent.checkpoint_store.path).expanduser(),
            "episodic_memory": Path(memory.episodic.path).expanduser(),
            "semantic_memory": Path(memory.semantic.path).expanduser(),
            "procedural_tacit_memory": Path(memory.procedural.path).expanduser(),
            "autobiography": Path(memory.autobiography.path).expanduser(),
            "learned_empirical_memory": Path(agent.learning.learned.path).expanduser(),
            "skill_statistics": Path(agent.learning.stats_store.path).expanduser(),
            "pinned_memory_ids": Path(agent.learning.pins_path).expanduser(),
            "self_model": Path(agent.self_model.path).expanduser(),
        }
        return files

    def _create_backup(self, scope: str) -> Path:
        stamp = time.strftime("%Y-%m-%dT%H%M%S", time.localtime())
        destination = self.backup_root / f"{stamp}_{scope}"
        destination.mkdir(parents=True, exist_ok=False)
        manifest = {"scope": scope, "created_at": time.time(), "files": {}}
        for label, path in self._owned_files().items():
            if not path.exists():
                continue
            target = destination / f"{label}{path.suffix or '.data'}"
            shutil.copy2(path, target)
            manifest["files"][label] = {"source": str(path), "backup": target.name}
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return destination

    def _restore_backup(self, backup_dir: Path) -> None:
        manifest_path = backup_dir / "manifest.json"
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for info in manifest.get("files", {}).values():
            source = Path(info["source"]).expanduser()
            backup = backup_dir / info["backup"]
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, source)

    def _reset_runtime(self) -> list[tuple[Path, str]]:
        checkpoint = self._owned_files()["runtime_checkpoint"]
        self._remove_file(checkpoint)
        self.agent.scheduler = CognitiveScheduler()
        self.agent.lifecycle = LifecycleController(state=LifecycleState.FIRST_BOOT)
        self.agent._previous_checkpoint = None
        self.agent._boot_event = EventType.AGENT_FIRST_BOOT
        self.agent._booted = False
        self.agent._boot_time = time.time()
        self.agent._cognition_count = 0
        self.agent._current_objectives = []
        self.agent._last_learning_question_shown = None
        self.agent.semantic_tracker = SemanticStateTracker()
        self.agent._last_semantic_state = self.agent.semantic_tracker.current
        if hasattr(self.agent.active_learning, "pending"):
            self.agent.active_learning.pending = None
        if hasattr(self.agent.monitor_bus, "clear"):
            self.agent.monitor_bus.clear()
        return [(checkpoint, "runtime_checkpoint"), (checkpoint, "pending_runtime_state")]

    def _reset_conversation(self) -> list[tuple[Path, str]]:
        self.agent._current_objectives = []
        self.agent._last_learning_question_shown = None
        if hasattr(self.agent.active_learning, "pending"):
            self.agent.active_learning.pending = None
        return [(self.base_dir / "conversation_runtime", "conversation_runtime")]

    def _reset_learned(self) -> list[tuple[Path, str]]:
        files = self._owned_files()
        touched = [
            (files["semantic_memory"], "semantic_memory"),
            (files["procedural_tacit_memory"], "procedural_tacit_memory"),
            (files["learned_empirical_memory"], "learned_empirical_memory"),
            (files["skill_statistics"], "skill_statistics"),
            (files["pinned_memory_ids"], "pinned_memory_ids"),
        ]
        self._write_json_list(files["semantic_memory"])
        self._write_json_list(files["procedural_tacit_memory"])
        self._write_json_list(files["learned_empirical_memory"])
        self._write_json_object(files["skill_statistics"])
        self._write_json_list(files["pinned_memory_ids"])
        if bool(getattr(getattr(self.agent.settings.effective(), "self_model", object()), "reset_learned_components_on_reset_learned", True)):
            self.agent.self_model.reset_to_baseline(reason="reset learned")
            touched.append((files["self_model"], "self_model_learned_components"))
        if self.monitor is not None:
            self.monitor.emit("memory", "learned_memory_cleared", "semantic/procedural learned memory cleared")
        return touched

    def _reset_autobiography(self) -> list[tuple[Path, str]]:
        path = self._owned_files()["autobiography"]
        self._write_text(path, "")
        return [(path, "autobiography")]

    def _reset_full(self, *, settings: Any) -> list[tuple[Path, str]]:
        touched: list[tuple[Path, str]] = []
        touched.extend(self._reset_runtime())
        touched.extend(self._reset_learned())
        files = self._owned_files()
        self._write_text(files["episodic_memory"], "")
        self._write_text(files["autobiography"], "")
        touched.extend([(files["episodic_memory"], "episodic_memory"), (files["autobiography"], "autobiography")])
        self.agent.active_learning.pending = None
        self.agent.active_learning.history.clear()
        self.agent.active_learning.last_autonomous_learning_question_timestamp = None
        if not bool(settings.reset.full_preserve_settings):
            self.agent.settings.reset_to_defaults()
            touched.append((self.base_dir / "settings", "settings"))
        self.agent._boot_event = EventType.AGENT_FIRST_BOOT
        self.agent._previous_checkpoint = None
        self.agent._booted = False
        self.agent.self_model.reset_to_baseline(reason="reset full")
        return touched

    def _preserved_for(self, scope: str, *, settings: Any) -> list[str]:
        preserved = [
            "application source code",
            "sdk_wrapper.py/source knowledge",
            "official/documentary RAG",
            "static engineering knowledge",
            "robot safety configuration",
            "installed tools",
            "ROS integration",
            "motion CSV files",
            "live physical robot state",
            "reset audit log",
        ]
        if scope != "full" or bool(settings.reset.full_preserve_settings):
            preserved.append("settings")
        if scope in {"runtime", "conversation", "autobiography"}:
            preserved.extend(["episodic memory", "semantic learned memory", "procedural/tacit memory"])
            preserved.append("persistent self-model")
        if scope in {"runtime", "conversation", "learned", "autobiography"}:
            preserved.append("autobiography" if scope != "autobiography" else "episodic/semantic/procedural evidence")
        return preserved

    def _result_message(self, scope: str, result: ResetResult) -> str:
        backup = f"; backup={result.backup_dir}" if result.backup_dir else ""
        return f"reset {scope} completed; cleared={', '.join(result.cleared) or 'nothing'}{backup}"

    def _append_audit(
        self,
        *,
        scope: str,
        initiator: str,
        ok: bool,
        backup_dir: Optional[Path],
        error: Optional[str] = None,
    ) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.time(),
            "event": "agent_reset",
            "scope": scope,
            "initiator": initiator,
            "ok": ok,
            "backup_dir": None if backup_dir is None else str(backup_dir),
            "error": error,
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _emit(self, event: str, summary: str, *, metadata: Optional[dict[str, Any]] = None) -> None:
        if self.monitor is not None:
            self.monitor.emit("reset", event, summary, metadata=metadata or {})

    @staticmethod
    def _atomic_replace(path: Path, data: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def _write_json_list(cls, path: Path) -> None:
        cls._atomic_replace(path, "[]\n")

    @classmethod
    def _write_json_object(cls, path: Path) -> None:
        cls._atomic_replace(path, "{}\n")

    @classmethod
    def _write_text(cls, path: Path, data: str) -> None:
        cls._atomic_replace(path, data)

    @staticmethod
    def _remove_file(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
