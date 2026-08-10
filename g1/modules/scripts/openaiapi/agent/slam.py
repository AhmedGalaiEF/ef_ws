"""Lazy SLAM skill backend reusing modules/scripts/slam_web_app.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class SlamBackendError(RuntimeError):
    pass


class SlamBackend:
    def __init__(self, *, iface: str = "eth0", domain_id: int = 0, map_path: str = "/home/unitree/test.pcd") -> None:
        self.iface = iface
        self.domain_id = int(domain_id)
        self.map_path = map_path
        self._state: Any = None

    def _ensure_state(self) -> Any:
        if self._state is not None:
            return self._state
        scripts_dir = Path(__file__).resolve().parents[2]
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        try:
            from slam_web_app import DEFAULT_TOPICS, SlamWebState
        except Exception as exc:
            raise SlamBackendError(f"slam_web_app backend unavailable: {exc}") from exc
        self._state = SlamWebState(self.iface, self.domain_id, DEFAULT_TOPICS, self.map_path)
        return self._state

    def start_mapping(self, slam_type: str = "indoor") -> str:
        return self._format(self._ensure_state().start_mapping(slam_type))

    def save_map(self, path: str | None = None) -> str:
        return self._format(self._ensure_state().save_map(path or self.map_path))

    def start_relocation(self, path: str | None = None) -> str:
        return self._format(self._ensure_state().relocate(path or self.map_path))

    def add_current_pose(self) -> str:
        return self._format(self._ensure_state().add_current_pose())

    def go_to_selected_pose(self) -> str:
        return self._format(self._ensure_state().go_to_selected_pose())

    def execute_tasks(self) -> str:
        return self._format(self._ensure_state().execute_tasks())

    def pause(self) -> str:
        return self._format(self._ensure_state().pause())

    def resume(self) -> str:
        return self._format(self._ensure_state().resume())

    def stop_slam(self) -> str:
        return self._format(self._ensure_state().stop_slam())

    def status(self) -> str:
        state = self._ensure_state()
        status = state.status()
        compact = {
            "slam_running": status.get("slam_running"),
            "relocation_ready": status.get("relocation_ready"),
            "pose": status.get("pose"),
            "last_action": status.get("last_action"),
            "fresh_topics": [
                row["name"]
                for row in status.get("topics", [])
                if row.get("fresh") and int(row.get("count") or 0) > 0
            ],
        }
        return json.dumps(compact, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _format(result: dict[str, Any]) -> str:
        label = result.get("label", "slam")
        ok = bool(result.get("ok"))
        raw = result.get("raw")
        return f"{label} {'ok' if ok else 'failed'}: {raw}"
