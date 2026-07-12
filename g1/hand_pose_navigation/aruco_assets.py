"""
Printable ArUco marker/board assets for the recognition layer.
================================================================================
Two marker roles, same dictionary (DICT_4X4_50) as target_detector.py's
existing default so nothing else has to change:

  Object tags  — ids 0-7,  6 cm square. Stick one on any object you want a
                 millimeter-accurate grasp pose for (no vision model needed
                 for that object at all; solvePnP gives full 6-DoF directly).
                 id=0 matches TargetDetector's pre-existing default aruco_id,
                 so old single-target configs still refer to the same tag.
  Hand tag     — id 49 (last id in the 50-marker dictionary, reserved so it
                 never collides with an object tag), 4 cm square. Stick on
                 the Dex3 palm/back-of-hand. Used to sanity-check / calibrate
                 the camera-to-base_link transform against forward kinematics
                 — not required for grasping itself (FK already gives hand
                 pose for free), but lets you see in the UI whether your
                 camera extrinsic is still trustworthy.

Printing matters: the marker must come out at the *exact* physical size
below, or every solvePnP pose will be scaled wrong. Print at 100% / "actual
size" — disable any printer "fit to page" / "shrink to fit" option.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50

OBJECT_MARKER_IDS: List[int] = list(range(0, 8))
OBJECT_MARKER_SIZE_M: float = 0.06

HAND_MARKER_ID: int = 49
HAND_MARKER_SIZE_M: float = 0.04

_MM_PER_M = 1000.0
_PRINT_DPI = 300
_PX_PER_MM = _PRINT_DPI / 25.4


@dataclass
class PrintableMarker:
    marker_id: int
    size_m: float
    role: str  # "object" | "hand"
    png_bytes: bytes

    @property
    def size_mm(self) -> float:
        return self.size_m * _MM_PER_M

    def data_uri(self) -> str:
        return to_data_uri(self.png_bytes)


# ---------------------------------------------------------------------------
def generate_marker_png(marker_id: int, size_m: float, border_bits: int = 1) -> bytes:
    """Render a single ArUco marker at exact print resolution (300 DPI)."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    side_px = max(64, int(round(size_m * _MM_PER_M * _PX_PER_MM)))
    marker_img = cv2.aruco.generateImageMarker(
        dictionary, int(marker_id), side_px, borderBits=border_bits,
    )

    # Pad with a white margin + a printed caption so the sheet is
    # self-documenting once it's off the printer.
    margin = max(20, side_px // 8)
    caption_h = max(30, side_px // 6)
    canvas_h = side_px + 2 * margin + caption_h
    canvas_w = side_px + 2 * margin
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    canvas[margin:margin + side_px, margin:margin + side_px] = marker_img

    caption = f"id={marker_id}  {size_m * _MM_PER_M:.0f}mm"
    cv2.putText(
        canvas, caption, (margin, canvas_h - margin // 2),
        cv2.FONT_HERSHEY_SIMPLEX, side_px / 600.0, (0,), 2, cv2.LINE_AA,
    )

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Failed to encode marker PNG")
    return bytes(buf)


def generate_printable_sheet_png(markers: List[Tuple[int, float, str]]) -> bytes:
    """Lay out several (marker_id, size_m, role) markers on one printable
    sheet with per-marker captions. Caller still prints at 100% scale.
    """
    tiles = []
    for marker_id, size_m, role in markers:
        png = generate_marker_png(marker_id, size_m)
        tile = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        role_caption = f"[{role}]"
        cv2.putText(
            tile, role_caption, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,), 2, cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        raise ValueError("No markers given")

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    tile_h = max(t.shape[0] for t in tiles)
    tile_w = max(t.shape[1] for t in tiles)
    pad = 20
    sheet = np.full(
        (rows * (tile_h + pad) + pad, cols * (tile_w + pad) + pad),
        255, dtype=np.uint8,
    )
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = pad + r * (tile_h + pad)
        x0 = pad + c * (tile_w + pad)
        sheet[y0:y0 + tile.shape[0], x0:x0 + tile.shape[1]] = tile

    ok, buf = cv2.imencode(".png", sheet)
    if not ok:
        raise RuntimeError("Failed to encode sheet PNG")
    return bytes(buf)


def to_data_uri(png_bytes: bytes) -> str:
    payload = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{payload}"


def default_hand_marker() -> PrintableMarker:
    return PrintableMarker(
        marker_id=HAND_MARKER_ID,
        size_m=HAND_MARKER_SIZE_M,
        role="hand",
        png_bytes=generate_marker_png(HAND_MARKER_ID, HAND_MARKER_SIZE_M),
    )


def default_object_markers() -> List[PrintableMarker]:
    return [
        PrintableMarker(
            marker_id=mid,
            size_m=OBJECT_MARKER_SIZE_M,
            role="object",
            png_bytes=generate_marker_png(mid, OBJECT_MARKER_SIZE_M),
        )
        for mid in OBJECT_MARKER_IDS
    ]


PLACEMENT_INSTRUCTIONS = f"""
**Print settings** — print at 100% / "actual size". Disable "fit to page" \
or "shrink to fit" in your printer dialog, then measure the printed black \
square with a ruler before cutting: it must read exactly the size shown \
under each marker (do not trust the print preview).

**Object tags** (dictionary DICT_4X4_50, ids {OBJECT_MARKER_IDS[0]}-\
{OBJECT_MARKER_IDS[-1]}, {OBJECT_MARKER_SIZE_M * _MM_PER_M:.0f} mm square) \
— cut one out per object you want pixel-perfect pose tracking on, and stick \
it flat on the most visible, least-curved face of the object, facing the \
camera when the object sits on the table in its normal resting orientation. \
Avoid wrapping it around a curved surface (bottles/cans) — pick the \
flattest face available, even if small.

**Hand tag** (id {HAND_MARKER_ID}, {HAND_MARKER_SIZE_M * _MM_PER_M:.0f} mm \
square) — stick this on the back of the Dex3 palm (the flat plate opposite \
the fingers), centered, tag facing straight out. This is optional: the arm's \
own joint encoders already give an exact hand pose via forward kinematics. \
The hand tag is only there so the recognition UI can show you, live, whether \
the camera-to-base calibration still lines up with what FK predicts — a \
quick visual check, not something the grasp pipeline depends on.
"""
