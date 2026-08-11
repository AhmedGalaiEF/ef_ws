"""Atomic runtime checkpoint persistence (spec section 6).

The checkpoint is lifecycle/runtime continuity only -- it is deliberately
kept separate from episodic/semantic/procedural memory, autobiography, and
engineering/audit logs (each of those has its own store under
``agent/memory/``). Writes are atomic: write to a temp file in the same
directory, fsync it, then ``os.replace`` it over the target, so a crash
mid-write can never leave a half-written checkpoint behind.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from .models import RuntimeCheckpoint

DEFAULT_CHECKPOINT_PATH = Path(
    os.environ.get("G1_AGENT_CHECKPOINT", "~/.g1_agent/checkpoint.json")
).expanduser()


class CheckpointStore:
    def __init__(self, path: Path | str = DEFAULT_CHECKPOINT_PATH) -> None:
        self.path = Path(path).expanduser()

    def load(self) -> Optional[RuntimeCheckpoint]:
        """Return the persisted checkpoint, or None if none exists / is unreadable.

        An unreadable (corrupt) checkpoint is treated the same as "missing"
        rather than raising -- startup classification (agent/lifecycle.py)
        falls back to agent_first_boot in that case, which is the safe
        default: it never claims a deliberate sleep or a known prior state
        that it cannot actually verify.
        """
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        try:
            return RuntimeCheckpoint.model_validate(raw)
        except Exception:
            return None

    def save(self, checkpoint: RuntimeCheckpoint) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = checkpoint.model_dump_json(indent=2)

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            # If os.replace already succeeded, tmp_name no longer exists and
            # this is a no-op; if we raised before replace, clean up.
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

        # Best-effort: fsync the containing directory too, so the rename
        # itself is durable across a crash immediately after this call.
        try:
            dir_fd = os.open(str(self.path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
