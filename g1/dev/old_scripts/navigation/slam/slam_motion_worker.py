#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SDK_CLIENT_PATH = ROOT.parents[2] / "modules" / "sdk_client.py"


def _force_cyclonedds_no_shm() -> None:
    if os.environ.get("CYCLONEDDS_URI"):
        return
    os.environ["CYCLONEDDS_URI"] = (
        "<CycloneDDS>"
        "<Domain>"
        "<Tracing><Category>none</Category></Tracing>"
        "<SharedMemory><Enable>false</Enable></SharedMemory>"
        "</Domain>"
        "</CycloneDDS>"
    )


def _load_robot_type():
    spec = importlib.util.spec_from_file_location("slam_sdk_client", SDK_CLIENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sdk_client from {SDK_CLIENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["slam_sdk_client"] = module
    spec.loader.exec_module(module)
    return module.Robot


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", required=True)
    args = parser.parse_args()

    _force_cyclonedds_no_shm()

    robot = None

    Robot = _load_robot_type()

    def ensure_robot():
        nonlocal robot
        if robot is not None:
            return robot
        robot = Robot(iface=args.iface, safety_boot=False, auto_start_sensors=False)
        return robot

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        cmd = "?"
        try:
            msg = json.loads(raw)
            cmd = msg.get("cmd", "")
            if cmd == "quit":
                break
            if cmd == "boot":
                ensure_robot()
            elif cmd == "set_balance":
                ensure_robot().balanced_stand(int(msg.get("mode", 0)))
            elif cmd == "move":
                rob = ensure_robot()
                rob.move_for(
                    float(msg.get("duration", 2.0)),
                    vx=float(msg.get("vx", 0.0)),
                    vy=float(msg.get("vy", 0.0)),
                    vyaw=float(msg.get("omega", 0.0)),
                )
            elif cmd == "stop":
                rob = ensure_robot()
                rob.stop()
                try:
                    rob.balanced_stand(0)
                except Exception:
                    pass
            elif cmd == "free_walk":
                rob = ensure_robot()
                fn = getattr(getattr(rob, "_client", None), "FreeWalk", None)
                if callable(fn):
                    fn()
            else:
                continue
            print(json.dumps({"ok": True, "cmd": cmd}), flush=True)
        except Exception as exc:  # pylint: disable=broad-except
            print(json.dumps({"ok": False, "cmd": cmd, "error": str(exc)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
