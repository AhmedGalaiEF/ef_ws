#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import queue
import sys
import threading
import time
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


class _MoveHandle:
    """Per-move cancellation token so overlapping commands never interfere."""

    def __init__(self) -> None:
        self._evt = threading.Event()

    def cancel(self) -> None:
        self._evt.set()

    def is_cancelled(self) -> bool:
        return self._evt.is_set()

    def wait(self, timeout: float) -> bool:
        return self._evt.wait(timeout=timeout)


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

    cmd_queue: "queue.Queue[str | None]" = queue.Queue()
    _current_handle: list[_MoveHandle | None] = [None]
    _move_thread: list[threading.Thread | None] = [None]

    def _stdin_reader() -> None:
        try:
            for raw in sys.stdin:
                raw = raw.strip()
                if raw:
                    cmd_queue.put(raw)
        except Exception:
            pass
        cmd_queue.put(None)

    threading.Thread(target=_stdin_reader, daemon=True).start()

    def _do_move(rob, vx: float, vy: float, omega: float, duration: float, handle: _MoveHandle) -> None:
        t_end = time.time() + duration
        try:
            rob.loco_move(vx, vy, omega)
            while True:
                remaining = t_end - time.time()
                if remaining <= 0.0:
                    break
                if handle.wait(timeout=min(0.15, remaining)):
                    return  # cancelled — caller handles stop
                try:
                    rob.loco_move(vx, vy, omega)  # keep SDK watchdog satisfied
                except Exception:
                    break
        except Exception:
            pass
        if not handle.is_cancelled():
            try:
                rob.stop()
            except Exception:
                pass

    def _cancel_current() -> None:
        h = _current_handle[0]
        if h is not None:
            h.cancel()
        t = _move_thread[0]
        if t is not None and t.is_alive():
            t.join(timeout=0.3)

    while True:
        try:
            raw = cmd_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if raw is None:
            _cancel_current()
            break
        cmd = "?"
        try:
            msg = json.loads(raw)
            cmd = msg.get("cmd", "")
            if cmd == "quit":
                _cancel_current()
                break
            if cmd == "boot":
                ensure_robot()
            elif cmd == "set_balance":
                ensure_robot().balanced_stand(int(msg.get("mode", 0)))
            elif cmd == "move":
                _cancel_current()
                handle = _MoveHandle()
                _current_handle[0] = handle
                rob = ensure_robot()
                t = threading.Thread(
                    target=_do_move,
                    args=(
                        rob,
                        float(msg.get("vx", 0.0)),
                        float(msg.get("vy", 0.0)),
                        float(msg.get("omega", 0.0)),
                        float(msg.get("duration", 2.0)),
                        handle,
                    ),
                    daemon=True,
                )
                _move_thread[0] = t
                t.start()
            elif cmd == "stop":
                _cancel_current()
                _current_handle[0] = None
                try:
                    ensure_robot().stop()
                    ensure_robot().balanced_stand(0)
                except Exception:
                    pass
            elif cmd == "free_walk":
                _cancel_current()
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
