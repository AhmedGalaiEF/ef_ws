"""
Open-vocabulary, natural-language-prompted object detection.
================================================================================
Wraps Ultralytics YOLO-World: give it a free-text description ("the red mug",
"soda can, bottle") and it returns labeled 2-D boxes, no fixed class list and
no fine-tuning. This is the "vision model" leg of the recognition layer — it
feeds candidate objects into segmentation.py (mask + PCA pose) the same way
target_detector.py feeds ArUco tags in.

Kept as an optional dependency on purpose: importing this module must not
require ultralytics/torch to be installed, matching the philosophy already
stated in hand_pose_navigation/__init__.py. Construct a VisionDetector and
check `.available` before calling `.detect()`.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class VisionDetection:
    box_xyxy: Tuple[int, int, int, int]
    label: str
    score: float


# A default open-vocab class list used when no NL prompt has been set yet.
_DEFAULT_CLASSES = [
    "phone", "cell phone", "smartphone", "mobile phone",
    "cup", "mug", "bottle", "can", "bowl", "box", "ball",
]
_ALIASES = {
    "phone": ["phone", "cell phone", "smartphone", "mobile phone"],
}


def parse_prompt_to_classes(text: str) -> List[str]:
    """Turn a free-text description into a class list for YOLO-World.

    YOLO-World matches against short noun phrases, not full sentences, so
    this splits on common separators/connectives and drops filler words
    rather than attempting real NLP. "the red mug on the left" -> ["red mug"]
    (articles/positional filler stripped, comma/or/and-separated phrases
    kept as-is otherwise).
    """
    text = text.strip()
    if not text:
        return list(_DEFAULT_CLASSES)

    parts = re.split(r",|\band\b|\bor\b", text, flags=re.IGNORECASE)
    filler = re.compile(
        r"^\s*(the|a|an|that|this|on the (left|right)|"
        r"in front|near me|please)\s*|"
        r"\s*(on the (left|right)|in front|near me)\s*$",
        re.IGNORECASE,
    )
    classes = []
    for part in parts:
        cleaned = filler.sub(" ", part).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned:
            classes.extend(_ALIASES.get(cleaned.lower(), [cleaned]))
    deduped = []
    for cls in classes:
        if cls not in deduped:
            deduped.append(cls)
    return deduped or list(_DEFAULT_CLASSES)


class VisionDetector:
    """
    NL-prompted 2-D object detector.

    Args:
        model_name:   an Ultralytics YOLO-World checkpoint id/path, e.g.
                       "yolov8s-world.pt" (auto-downloaded on first use).
        conf:         minimum confidence to keep a detection.
        device:       "cpu" or "cuda:0"; auto-detected if left as "auto".
    """

    def __init__(
        self,
        model_name: str = "yolov8s-world.pt",
        conf: float = 0.15,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.conf = conf
        self.device = device
        self.available = False
        self.error = ""
        self._model = None
        self._classes: List[str] = list(_DEFAULT_CLASSES)
        self._lock = threading.RLock()
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            from ultralytics import YOLOWorld  # type: ignore
        except Exception as exc:
            self.error = (
                f"ultralytics not available ({exc}); NL object detection is "
                "disabled. Install with: pip install ultralytics"
            )
            return
        try:
            device = None if self.device == "auto" else self.device
            self._model = YOLOWorld(self.model_name)
            if device is not None:
                self._model.to(device)
            self._model.set_classes(self._classes)
            self.available = True
        except Exception as exc:
            self.error = f"Failed to load {self.model_name}: {exc}"
            self._model = None

    # ------------------------------------------------------------------
    def set_prompt(self, text: str) -> List[str]:
        """Update the open-vocab class list from a free-text description.

        Returns the parsed class list so the caller can show it in the UI.
        """
        classes = parse_prompt_to_classes(text)
        with self._lock:
            self._classes = classes
            if self._model is not None:
                self._model.set_classes(classes)
        return classes

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    # ------------------------------------------------------------------
    def detect(self, rgb_bgr: np.ndarray) -> List[VisionDetection]:
        """Run detection on a single BGR frame. Returns [] if unavailable."""
        if not self.available or self._model is None:
            return []
        try:
            with self._lock:
                results = self._model.predict(
                    rgb_bgr, conf=self.conf, verbose=False,
                )
        except Exception as exc:
            self.error = f"Inference failed: {exc}"
            return []

        detections: List[VisionDetection] = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        names = result.names if hasattr(result, "names") else {}
        for xyxy, conf_t, cls_t in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
        ):
            cls_i = int(cls_t)
            if isinstance(names, dict):
                label = names.get(cls_i, str(cls_i))
            elif isinstance(names, (list, tuple)) and 0 <= cls_i < len(names):
                label = names[cls_i]
            else:
                label = str(cls_i)
            detections.append(
                VisionDetection(
                    box_xyxy=(
                        int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]),
                    ),
                    label=str(label),
                    score=float(conf_t),
                )
            )
        return detections
