from __future__ import annotations

import shutil
import subprocess
import time
from typing import Optional


class LocalSpeechAnnouncer:
    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._command = self._detect_command() if enabled else None
        self._last_message = ""
        self._last_spoken_at = 0.0
        self._process: Optional[subprocess.Popen[bytes]] = None

    def available(self) -> bool:
        return self._command is not None

    def announce(self, message: str) -> None:
        if not self._enabled or self._command is None:
            return

        cleaned = " ".join(message.split()).strip()
        if not cleaned or cleaned == self._last_message:
            return

        now = time.time()
        if now - self._last_spoken_at < 1.0:
            return

        self._stop_current()
        self._process = subprocess.Popen(
            self._command + [cleaned],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._last_message = cleaned
        self._last_spoken_at = now

    def close(self) -> None:
        self._stop_current()

    def _stop_current(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=0.5)
        self._process = None

    def _detect_command(self) -> Optional[list[str]]:
        for command in ("spd-say", "espeak-ng", "espeak", "say"):
            if shutil.which(command):
                return [command]
        return None
