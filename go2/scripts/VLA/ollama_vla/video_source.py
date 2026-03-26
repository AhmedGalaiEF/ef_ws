from __future__ import annotations

import base64
import threading
import time
from dataclasses import dataclass
from typing import Optional

from unitree_sdk2py.go2.video.video_client import VideoClient


@dataclass
class VideoFrame:
    jpeg_bytes: bytes
    timestamp: float

    def to_base64(self) -> str:
        return base64.b64encode(self.jpeg_bytes).decode("ascii")


class Go2VideoSource:
    def __init__(self, timeout_sec: float = 3.0, fps: float = 30.0):
        self._client = VideoClient()
        self._timeout_sec = timeout_sec
        self._frame_period_sec = 1.0 / max(fps, 1.0)
        self._lock = threading.Lock()
        self._last_frame: Optional[VideoFrame] = None
        self._last_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._client.SetTimeout(self._timeout_sec)
        self._client.Init()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def poll(self) -> Optional[VideoFrame]:
        code, data = self._client.GetImageSample()
        if code != 0:
            with self._lock:
                self._last_error = f"GetImageSample failed with code {code}"
            return None
        jpeg_bytes = bytes(data)
        if not jpeg_bytes:
            with self._lock:
                self._last_error = "GetImageSample returned empty frame"
            return None

        frame = VideoFrame(jpeg_bytes=jpeg_bytes, timestamp=time.time())
        with self._lock:
            self._last_frame = frame
            self._last_error = None
        return frame

    def latest(self) -> Optional[VideoFrame]:
        with self._lock:
            return self._last_frame

    def latest_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll()
            self._stop.wait(self._frame_period_sec)


class RgbVisualizer:
    def __init__(self, video_source: Go2VideoSource, window_name: str = "Go2 RGB"):
        self._video_source = video_source
        self._window_name = window_name
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cv2 = None
        self._np = None

    def start(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "RGB visualization requires opencv-python and numpy in the runtime environment"
            ) from exc
        self._cv2 = cv2
        self._np = np
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        cv2 = self._cv2
        np = self._np

        while not self._stop.is_set():
            frame = self._video_source.latest()
            if frame is not None:
                array = np.frombuffer(frame.jpeg_bytes, dtype=np.uint8)
                image = cv2.imdecode(array, cv2.IMREAD_COLOR)
                if image is not None:
                    cv2.imshow(self._window_name, image)
            cv2.waitKey(1)
            self._stop.wait(1.0 / 30.0)

        cv2.destroyWindow(self._window_name)
