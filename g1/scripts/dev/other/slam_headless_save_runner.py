#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SLAM_DIR = REPO_ROOT / "scripts" / "navigation" / "obstacle_avoidance"
if str(SLAM_DIR) not in sys.path:
    sys.path.insert(0, str(SLAM_DIR))

import live_slam_save as live_slam_mod  # noqa: E402


class _HeadlessViewer:
    def push(self, _xyz, _pose) -> None:
        pass

    def tick(self) -> bool:
        return True

    def close(self) -> None:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless compatibility wrapper for live_slam_save.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--save-dir", default="./maps")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--save-prefix", default="live_slam")
    parser.add_argument("--save-latest", action="store_true")
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", default="0")
    parser.add_argument("--overlay-plan-file", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    live_slam_mod._Viewer = _HeadlessViewer  # type: ignore[attr-defined]

    save_dir = Path(args.save_dir) if args.save_dir else None
    demo = live_slam_mod.LiveSLAMDemo(
        save_dir=save_dir,
        save_every=max(1, int(args.save_every)),
        save_latest=bool(args.save_latest),
        save_prefix=str(args.save_prefix),
    )

    stop = False

    def _sigint(*_args) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    try:
        while not stop:
            time.sleep(0.05)
    finally:
        demo.shutdown()


if __name__ == "__main__":
    main()
