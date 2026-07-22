#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dds_env import default_dds_iface, ensure_channel_factory_initialized
from sdk_boot import create_loco_client, read_fsm_state

try:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed. Install it with:\n"
        "  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc


ERROR_HINTS = {
    0: "success",
    7001: "request parameter error",
    7002: "service busy; retry",
    7004: "unsupported mode name",
    7005: "internal command execute error",
    7006: "check command execute error",
    7007: "switch command execute error",
    7008: "release command execute error",
    7009: "custom config set error",
}


@dataclass(frozen=True)
class ModeSpec:
    key: str
    label: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class RobotState:
    fsm_id: int | None
    fsm_mode: int | None
    motion_name: str | None
    motion_raw: Any
    error: str | None = None


DEFAULT_MODES = (
    ModeSpec(
        "wake",
        "wake-up mode",
        ("wake_up", "wake-up", "wake_up_mode", "wake-up-mode"),
    ),
    ModeSpec(
        "key",
        "keymode",
        ("keymode", "key_mode", "key-mode"),
    ),
    ModeSpec(
        "close",
        "close-interaction-mode",
        (
            "close_interaction_mode",
            "close-interaction-mode",
            "close_interaction",
            "close-interaction",
        ),
    ),
)


def _mode_name(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    for key in ("name", "mode", "alias"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _result_code(result: Any) -> int:
    if isinstance(result, tuple):
        return int(result[0])
    if result is None:
        return 0
    return int(result)


def _error_hint(code: int) -> str:
    return ERROR_HINTS.get(int(code), "unknown SDK return code")


class WakeUpModeTester:
    def __init__(
        self,
        iface: str,
        domain_id: int,
        timeout: float,
        poll_interval: float,
        modes: Sequence[ModeSpec],
    ) -> None:
        self.iface = iface
        self.domain_id = int(domain_id)
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self.modes = tuple(modes)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None

        ensure_channel_factory_initialized(self.domain_id, self.iface)

        self.motion = MotionSwitcherClient()
        self.motion.SetTimeout(self.timeout)
        self.motion.Init()

        self.loco = create_loco_client(self.domain_id, self.iface, timeout=self.timeout)

    def state(self) -> RobotState:
        with self._lock:
            try:
                fsm_id, fsm_mode = read_fsm_state(self.loco, retries=2, retry_delay=0.05)
                motion_code, motion_raw = self.motion.CheckMode()
                motion_code = int(motion_code)
                motion_name = _mode_name(motion_raw) if motion_code == 0 else None

                if motion_code != 0:
                    return RobotState(
                        fsm_id,
                        fsm_mode,
                        None,
                        motion_raw,
                        error=f"CheckMode failed: code={motion_code} ({_error_hint(motion_code)})",
                    )
                return RobotState(fsm_id, fsm_mode, motion_name, motion_raw)
            except Exception as exc:
                return RobotState(None, None, None, None, error=str(exc))

    def print_state(self, prefix: str = "state") -> None:
        state = self.state()
        motion = state.motion_name or "<released/none>"
        line = (
            f"{prefix}: fsm_id={state.fsm_id} fsm_mode={state.fsm_mode} "
            f"motion={motion}"
        )
        if state.error:
            line += f" error={state.error}"
        print(line, flush=True)

    def start_monitor(self) -> None:
        if self._monitor_thread is not None:
            return

        def run() -> None:
            while not self._stop.wait(self.poll_interval):
                self.print_state("monitor")

        self._monitor_thread = threading.Thread(target=run, daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        self._stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)

    def select_mode(self, spec: ModeSpec) -> tuple[int, str | None]:
        last_code = -1
        with self._lock:
            for alias in spec.aliases:
                code = _result_code(self.motion.SelectMode(alias))
                last_code = code
                print(
                    f"SelectMode({alias!r}) -> code={code} ({_error_hint(code)})",
                    flush=True,
                )
                if code == 0:
                    return code, alias
                time.sleep(0.2)
        return last_code, None

    def cycle_modes(self, dwell_s: float) -> None:
        print("\nCycling through requested modes.")
        for spec in self.modes:
            print(f"\nSelecting {spec.label}...")
            code, alias = self.select_mode(spec)
            if code != 0:
                print(f"  Could not select {spec.label}; continuing to next mode.")
                continue
            print(f"  Active alias: {alias}")
            deadline = time.monotonic() + max(0.0, dwell_s)
            while time.monotonic() < deadline:
                self.print_state("after-select")
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            if dwell_s <= 0.0:
                self.print_state("after-select")
            input("Press Enter to continue to the next mode...")

    def prompt_final_selection(self) -> None:
        while True:
            print("\nSelect the mode to leave toggled on:")
            for index, spec in enumerate(self.modes, start=1):
                print(f"  {index}. {spec.label} ({', '.join(spec.aliases)})")
            choice = input("Mode number, name, or q to abort: ").strip().lower()
            if choice in {"q", "quit", "exit"}:
                print("Aborted; leaving the current robot mode unchanged.")
                return

            spec = self._resolve_mode_choice(choice)
            if spec is None:
                print("Unknown selection.")
                continue

            code, alias = self.select_mode(spec)
            self.print_state("final")
            if code == 0:
                print(f"{spec.label} is toggled on via SelectMode({alias!r}).")
            else:
                print(f"Failed to toggle {spec.label}; last code={code} ({_error_hint(code)}).")
            return

    def _resolve_mode_choice(self, choice: str) -> ModeSpec | None:
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(self.modes):
                return self.modes[index]
        for spec in self.modes:
            names = {spec.key, spec.label.lower(), *(alias.lower() for alias in spec.aliases)}
            if choice in names:
                return spec
        return None


def _split_aliases(value: str) -> tuple[str, ...]:
    aliases = tuple(part.strip() for part in value.split(",") if part.strip())
    if not aliases:
        raise argparse.ArgumentTypeError("at least one alias is required")
    return aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor G1 SDK state, cycle wake-up/key/close-interaction modes, "
            "then leave one selected through MotionSwitcherClient."
        )
    )
    parser.add_argument("--iface", default=default_dds_iface("eth0"), help="DDS network interface.")
    parser.add_argument("--domain", type=int, default=0, help="DDS domain id.")
    parser.add_argument("--timeout", type=float, default=5.0, help="SDK RPC timeout in seconds.")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between background state monitor prints.",
    )
    parser.add_argument(
        "--dwell-s",
        type=float,
        default=2.0,
        help="Seconds to observe each selected mode before prompting for the next one.",
    )
    parser.add_argument(
        "--wake-aliases",
        type=_split_aliases,
        default=DEFAULT_MODES[0].aliases,
        help="Comma-separated SelectMode aliases for wake-up mode.",
    )
    parser.add_argument(
        "--key-aliases",
        type=_split_aliases,
        default=DEFAULT_MODES[1].aliases,
        help="Comma-separated SelectMode aliases for keymode.",
    )
    parser.add_argument(
        "--close-aliases",
        type=_split_aliases,
        default=DEFAULT_MODES[2].aliases,
        help="Comma-separated SelectMode aliases for close-interaction-mode.",
    )
    parser.add_argument(
        "--no-monitor",
        action="store_true",
        help="Disable background monitor prints while waiting for input.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modes = (
        ModeSpec("wake", "wake-up mode", tuple(args.wake_aliases)),
        ModeSpec("key", "keymode", tuple(args.key_aliases)),
        ModeSpec("close", "close-interaction-mode", tuple(args.close_aliases)),
    )

    tester = WakeUpModeTester(
        iface=args.iface,
        domain_id=args.domain,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        modes=modes,
    )
    print(f"Connected on domain={args.domain} iface={args.iface!r}.")
    tester.print_state("initial")

    if not args.no_monitor:
        tester.start_monitor()
    try:
        input("\nPress Enter to start cycling modes, or Ctrl-C to abort...")
        tester.cycle_modes(dwell_s=args.dwell_s)
        tester.prompt_final_selection()
    except KeyboardInterrupt:
        print("\nInterrupted; leaving the current robot mode unchanged.")
        return 130
    finally:
        tester.stop_monitor()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
