from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AuditLogger:
    def __init__(self, log_path: str) -> None:
        self._path = Path(log_path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {"event_type": event_type, **payload}
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
