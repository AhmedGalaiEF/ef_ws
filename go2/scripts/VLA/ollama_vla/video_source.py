from __future__ import annotations

import base64
import multiprocessing
import os
import sysconfig
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full
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
        self._started = False
        self._stop = multiprocessing.Event()
        self._queue: Optional[multiprocessing.Queue[bytes]] = None
        self._process: Optional[multiprocessing.Process] = None
        self._thread: Optional[threading.Thread] = None
        self._last_timestamp = 0.0

    def start(self) -> None:
        self._queue = multiprocessing.Queue(maxsize=2)
        self._process = multiprocessing.Process(
            target=_visualizer_process,
            args=(self._queue, self._stop, self._window_name),
            daemon=True,
        )
        self._process.start()
        self._thread = threading.Thread(target=self._feed_frames, daemon=True)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._process is not None:
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)
        self._started = False

    def render_latest(self) -> None:
        if not self._started:
            return
        self._publish_latest_frame()

    def pump_events(self) -> None:
        if not self._started:
            return
        self._publish_latest_frame()

    def _feed_frames(self) -> None:
        while not self._stop.is_set():
            self._publish_latest_frame()
            self._stop.wait(1.0 / 30.0)

    def _publish_latest_frame(self) -> None:
        if self._queue is None:
            return
        frame = self._video_source.latest()
        if frame is None or frame.timestamp <= self._last_timestamp:
            return
        self._last_timestamp = frame.timestamp
        try:
            self._queue.put_nowait(frame.jpeg_bytes)
        except Full:
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(frame.jpeg_bytes)
            except Full:
                pass


def _configure_qt_font_dir() -> None:
    if os.environ.get("QT_QPA_FONTDIR"):
        return

    candidates = (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
    )
    for candidate in candidates:
        if candidate.is_dir():
            os.environ["QT_QPA_FONTDIR"] = str(candidate)
            return


def _install_cv2_qt_fonts(cv2_module: object) -> None:
    cv2_path = Path(getattr(cv2_module, "__file__", "")).resolve().parent
    qt_fonts_dir = cv2_path / "qt" / "fonts"
    qt_fonts_dir.mkdir(parents=True, exist_ok=True)

    for source_dir in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
    ):
        if not source_dir.is_dir():
            continue
        for font_file in source_dir.glob("*.ttf"):
            target = qt_fonts_dir / font_file.name
            if target.exists():
                continue
            try:
                target.symlink_to(font_file)
            except Exception:
                try:
                    target.write_bytes(font_file.read_bytes())
                except Exception:
                    pass


def _install_qt_fonts_before_cv2_import() -> None:
    purelib = Path(sysconfig.get_paths().get("purelib", ""))
    if not purelib:
        return
    qt_fonts_dir = purelib / "cv2" / "qt" / "fonts"
    qt_fonts_dir.mkdir(parents=True, exist_ok=True)
    for source_dir in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
    ):
        if not source_dir.is_dir():
            continue
        for font_file in source_dir.glob("*.ttf"):
            target = qt_fonts_dir / font_file.name
            if target.exists():
                continue
            try:
                target.symlink_to(font_file)
            except Exception:
                try:
                    target.write_bytes(font_file.read_bytes())
                except Exception:
                    pass


def _visualizer_process(
    frame_queue: multiprocessing.Queue[bytes],
    stop_event: multiprocessing.Event,
    window_name: str,
) -> None:
    _configure_qt_font_dir()
    _install_qt_fonts_before_cv2_import()
    import cv2
    import numpy as np

    _install_cv2_qt_fonts(cv2)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    latest_image = None
    while not stop_event.is_set():
        jpeg_bytes = None
        try:
            jpeg_bytes = frame_queue.get(timeout=0.03)
            while True:
                try:
                    jpeg_bytes = frame_queue.get_nowait()
                except Empty:
                    break
        except Empty:
            pass

        if jpeg_bytes is not None:
            array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            latest_image = cv2.imdecode(array, cv2.IMREAD_COLOR)

        if latest_image is not None:
            cv2.imshow(window_name, latest_image)
        cv2.waitKey(1)

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)
