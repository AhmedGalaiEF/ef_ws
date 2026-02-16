#!/usr/bin/env python3
"""
Headless runner for navigation/obstacle_avoidance/live_slam_save.py.

It reuses LiveSLAMDemo map-saving behavior while monkey-patching the viewer to
avoid opening an Open3D window.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path


def _load_live_slam_save_module():
    script_dir = Path(__file__).resolve().parent
    nav_dir = script_dir.parent.parent / "navigation" / "obstacle_avoidance"
    if not nav_dir.exists():
        raise SystemExit(f"Missing navigation path: {nav_dir}")
    if str(nav_dir) not in sys.path:
        sys.path.insert(0, str(nav_dir))
    import live_slam_save as lss  # type: ignore

    return lss


class _HeadlessViewer:
    def __init__(self):
        self._latest_pts = None
        self._latest_pose = None

    def push(self, xyz, pose):
        self._latest_pts = xyz
        self._latest_pose = pose

    def tick(self) -> bool:
        return True

    def close(self):
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Headless SLAM with auto-saving point cloud map")
    ap.add_argument("--save-dir", default="./maps", help="Directory to save maps (use empty to disable)")
    ap.add_argument("--save-every", type=int, default=1, help="Save every N updates")
    ap.add_argument("--save-latest", action="store_true", help="Overwrite a single latest file")
    ap.add_argument("--save-prefix", default="live_slam", help="Filename prefix for saved maps")
    args = ap.parse_args()

    lss = _load_live_slam_save_module()
    lss._Viewer = _HeadlessViewer  # type: ignore[attr-defined]

    save_dir = Path(args.save_dir) if args.save_dir else None
    demo = lss.LiveSLAMDemo(save_dir, args.save_every, args.save_latest, args.save_prefix)

    stop = False

    def _sigint(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    try:
        while not stop and demo._viewer.tick():
            time.sleep(0.01)
    finally:
        demo.shutdown()


if __name__ == "__main__":
    main()
