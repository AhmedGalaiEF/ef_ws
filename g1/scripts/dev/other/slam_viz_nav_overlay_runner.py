#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SLAM_DIR = REPO_ROOT / "scripts" / "navigation" / "obstacle_avoidance"
if str(SLAM_DIR) not in sys.path:
    sys.path.insert(0, str(SLAM_DIR))

import live_slam_save as live_slam_mod  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualization compatibility wrapper for live_slam_save.py",
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
    sys.argv = [
        str(SLAM_DIR / "live_slam_save.py"),
        "--save-dir",
        str(args.save_dir),
        "--save-every",
        str(max(1, int(args.save_every))),
        "--save-prefix",
        str(args.save_prefix),
        *(["--save-latest"] if args.save_latest else []),
    ]
    live_slam_mod.main()


if __name__ == "__main__":
    main()
