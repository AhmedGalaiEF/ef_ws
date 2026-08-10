"""OpenAI vision answerer for the RGB-D stream used by recognition_app_v3.py."""
from __future__ import annotations

import base64
import os
import statistics
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional


class VisionError(RuntimeError):
    pass


def _bootstrap_scripts_path() -> None:
    here = Path(__file__).resolve()
    scripts_dir = next((parent for parent in here.parents if (parent / "llm_client").exists()), None)
    if scripts_dir is None:
        scripts_dir = here.parents[1]
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


class ZmqRgbdSource:
    """Persistent RGB-D ZMQ frame source compatible with recognition_app_v3.py."""

    def __init__(self, host: str, port: int, topic: str = "") -> None:
        try:
            import zmq  # type: ignore
        except Exception as exc:
            raise VisionError(f"pyzmq is unavailable: {exc}") from exc

        self.host = str(host)
        self.port = int(port)
        self.topic = str(topic)
        self._zmq = zmq
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, self.topic.encode("utf-8"))
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.setsockopt(zmq.RCVHWM, 2)
        self._socket.connect(f"tcp://{self.host}:{self.port}")

    def get_rgbd(self, timeout: float = 2.0) -> dict[str, Any]:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:
            raise VisionError(f"OpenCV/numpy are unavailable for RGB-D decoding: {exc}") from exc

        deadline = time.time() + max(0.2, float(timeout))
        last_error = ""
        while time.time() < deadline:
            try:
                parts = self._socket.recv_multipart()
                while True:
                    try:
                        parts = self._socket.recv_multipart(flags=self._zmq.NOBLOCK)
                    except self._zmq.Again:
                        break
            except self._zmq.Again:
                continue
            except Exception as exc:
                last_error = str(exc)
                continue

            if len(parts) >= 4:
                parts = parts[-3:]
            if len(parts) < 2:
                last_error = f"expected RGB-D multipart frame, got {len(parts)} part(s)"
                continue

            rgb_jpeg = bytes(parts[0])
            depth_png = bytes(parts[1])
            depth_scale = 0.001
            if len(parts) >= 3 and len(parts[2]) >= 4:
                try:
                    depth_scale = float(struct.unpack("f", parts[2][:4])[0])
                except Exception:
                    depth_scale = 0.001

            rgb = cv2.imdecode(np.frombuffer(rgb_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            depth_raw = cv2.imdecode(np.frombuffer(depth_png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if rgb is None:
                last_error = "failed to decode RGB JPEG"
                continue
            if depth_raw is None:
                last_error = "failed to decode depth PNG"
                continue
            if depth_raw.ndim == 3:
                depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
            depth_m = depth_raw.astype("float32") * float(depth_scale)
            return {
                "source": f"zmq://{self.host}:{self.port}",
                "rgb_jpeg": rgb_jpeg,
                "rgb_bgr": rgb,
                "depth_m": depth_m,
                "depth_scale_m_per_unit": float(depth_scale),
                "valid_depth_fraction": float((depth_raw > 0).mean()) if depth_raw.size else 0.0,
                "shape": tuple(int(x) for x in rgb.shape[:2]),
            }

        detail = f" Last error: {last_error}" if last_error else ""
        raise VisionError(
            f"No RGB-D frames received from tcp://{self.host}:{self.port} within {timeout:.1f}s.{detail}"
        )


class OpenAIVisionAnswerer:
    def __init__(self) -> None:
        self._source: Optional[ZmqRgbdSource] = None
        self._source_key: Optional[tuple[str, int, str]] = None

    def answer(self, *, settings: Any, question: str, reply_language: str = "auto") -> str:
        frame = self._get_source(settings).get_rgbd(timeout=settings.rgbd_timeout_s)
        image_b64 = base64.b64encode(frame["rgb_jpeg"]).decode("ascii")
        depth_context = self._depth_context(frame)
        language_instruction = {
            "de": "Antworte auf Deutsch.",
            "en": "Answer in English.",
        }.get(str(reply_language), "Answer in the user's language when clear; otherwise use English.")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the robot's vision layer. Answer only from the current RGB image "
                    "and the supplied depth summary. Be concise and mention uncertainty."
                ),
            },
            {"role": "system", "content": language_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{question}\n\nDepth/context: {depth_context}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            },
        ]
        return self._send(settings.openai_model, messages).strip()

    def _get_source(self, settings: Any) -> ZmqRgbdSource:
        key = (str(settings.rgbd_host), int(settings.rgbd_port), str(settings.rgbd_topic))
        if self._source is None or self._source_key != key:
            self._source = ZmqRgbdSource(*key)
            self._source_key = key
        return self._source

    @staticmethod
    def _depth_context(frame: dict[str, Any]) -> str:
        parts = [
            f"source={frame.get('source')}",
            f"valid_depth_fraction={frame.get('valid_depth_fraction', 0.0):.2f}",
        ]
        try:
            depth = frame["depth_m"]
            height, width = depth.shape[:2]
            y0, y1 = int(height * 0.4), int(height * 0.6)
            x0, x1 = int(width * 0.4), int(width * 0.6)
            center = depth[y0:y1, x0:x1]
            values = [float(v) for v in center.reshape(-1) if float(v) > 0.0]
            if values:
                sample = values[:: max(1, len(values) // 2000)]
                parts.append(f"center_depth_median_m={statistics.median(sample):.2f}")
        except Exception:
            pass
        return ", ".join(parts)

    @staticmethod
    def _send(model: str, messages: list[dict[str, Any]]) -> str:
        _bootstrap_scripts_path()
        try:
            from llm_client import chat as chat_module
        except Exception as exc:
            raise VisionError(f"llm_client.chat is unavailable: {exc}") from exc
        try:
            import llm_client.secrets  # noqa: F401
        except Exception:
            pass

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise VisionError("OPENAI_API_KEY is not set; OpenAI vision cannot authenticate.")
        chat_module.dnabot_auth = SimpleNamespace(
            get_auth_header=lambda: {"Authorization": f"Bearer {api_key}"}
        )
        content, _ = chat_module.send_chat_with_tool_usage(
            model_key=str(model),
            messages=messages,
            base="https://api.openai.com/v1",
            extra_body={"max_tokens": 300},
        )
        return content
