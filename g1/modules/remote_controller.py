#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from dds_env import (
    default_dds_iface,
    ensure_channel_factory_initialized,
    ensure_cyclonedds_environment,
)
from sdk_client import Robot
from sdk_audio import parse_color, scale_color
from arm_sdk import ArmSdk

ensure_cyclonedds_environment()

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber
except ImportError as exc:
    raise SystemExit("unitree_sdk2py not installed.") from exc


TOPIC_LOWSTATE = "rt/lowstate"
MODE_NAME = "Endeffector mode"
MODE_TOGGLE_BUTTONS = ("R1", "B")
ARM_CONTROL_MODES = ("right", "left", "both")
DEFAULT_HEADLIGHT_COLOR = "red"
DEFAULT_HEADLIGHT_INTENSITY = 100
DEFAULT_HEADLIGHT_INTERVAL_S = 0.2
DEFAULT_START_SPEECH = "end effector mode started"
DEFAULT_END_SPEECH = "end effector mode ended"
DEFAULT_CONTROL_RATE_HZ = 8.0
DEFAULT_XY_SPEED_M_S = 0.035
DEFAULT_Z_SPEED_M_S = 0.025
DEFAULT_REMOTE_DEADBAND = 0.12
DEFAULT_MAX_DQ_RAD = 0.08


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_axis(value: float, deadband: float) -> float:
    if abs(value) <= deadband:
        return 0.0
    scaled = (abs(value) - deadband) / max(1e-6, 1.0 - deadband)
    return clamp(scaled, 0.0, 1.0) if value > 0.0 else -clamp(scaled, 0.0, 1.0)


def _resolve_lowstate_type() -> type[Any]:
    for path in (
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        "unitree_sdk2py.idl.unitree_go.msg.dds_",
    ):
        try:
            mod = __import__(path, fromlist=["LowState_"])
        except Exception:
            continue
        if hasattr(mod, "LowState_"):
            return getattr(mod, "LowState_")
    raise RuntimeError("LowState_ not found in unitree_sdk2py.")


def decode_buttons(data: bytes | bytearray | list[int] | tuple[int, ...]) -> dict[str, int]:
    raw = bytes(data)
    if len(raw) < 4:
        return {}
    data1 = int(raw[2])
    data2 = int(raw[3])
    return {
        "R1": (data1 >> 0) & 1,
        "L1": (data1 >> 1) & 1,
        "Start": (data1 >> 2) & 1,
        "Select": (data1 >> 3) & 1,
        "R2": (data1 >> 4) & 1,
        "L2": (data1 >> 5) & 1,
        "F1": (data1 >> 6) & 1,
        "F3": (data1 >> 7) & 1,
        "A": (data2 >> 0) & 1,
        "B": (data2 >> 1) & 1,
        "X": (data2 >> 2) & 1,
        "Y": (data2 >> 3) & 1,
        "Up": (data2 >> 4) & 1,
        "Right": (data2 >> 5) & 1,
        "Down": (data2 >> 6) & 1,
        "Left": (data2 >> 7) & 1,
    }


def _unpack_float(raw: bytes, start: int) -> float:
    if len(raw) < start + 4:
        return 0.0
    return float(struct.unpack("<f", raw[start:start + 4])[0])


def decode_remote(data: bytes | bytearray | list[int] | tuple[int, ...]) -> dict[str, Any]:
    raw = bytes(data)
    return {
        "lx": _unpack_float(raw, 4),
        "rx": _unpack_float(raw, 8),
        "ry": _unpack_float(raw, 12),
        "ly": _unpack_float(raw, 20),
        "buttons": decode_buttons(raw),
        "raw_hex": raw.hex(),
    }


@dataclass
class RemoteControllerSnapshot:
    mode_active: bool = False
    arm_control_mode: str = ARM_CONTROL_MODES[0]
    handoff_in_progress: bool = False
    last_button_event: str = ""
    last_button_time: float = 0.0
    last_lowstate_time: float = 0.0
    last_status: str = "waiting for lowstate"
    speech_code: Optional[int] = None
    headlight_code: Optional[int] = None
    remote_lx: float = 0.0
    remote_ly: float = 0.0
    remote_rx: float = 0.0
    remote_ry: float = 0.0
    last_ik_success: Optional[bool] = None
    last_ik_error: str = ""


class PersistentHeadlight(threading.Thread):
    def __init__(
        self,
        client: object,
        rgb: tuple[int, int, int],
        interval_s: float,
        stop_event: threading.Event,
        status_cb: Optional[Callable[[int], None]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self.client = client
        self.rgb = rgb
        self.interval_s = max(0.02, float(interval_s))
        self.stop_event = stop_event
        self.status_cb = status_cb
        self.last_code = 0

    def run(self) -> None:
        next_call = time.monotonic()
        while not self.stop_event.is_set():
            wait_s = next_call - time.monotonic()
            if wait_s > 0 and self.stop_event.wait(wait_s):
                break
            self.last_code = int(self.client.LedControl(*self.rgb))
            if self.status_cb is not None:
                self.status_cb(self.last_code)
            if self.last_code != 0:
                break
            next_call += self.interval_s


class G1RemoteController:
    """Remote mode gate and basic end-effector jog controller.

    R1+B toggles the custom mode. While active, Select cycles arm ownership,
    the right stick jogs x/y, and D-pad Up/Down jogs z.
    """

    def __init__(
        self,
        *,
        iface: str = default_dds_iface(),
        domain_id: int = 0,
        headlight_color: str = DEFAULT_HEADLIGHT_COLOR,
        headlight_intensity: int = DEFAULT_HEADLIGHT_INTENSITY,
        headlight_interval_s: float = DEFAULT_HEADLIGHT_INTERVAL_S,
        start_speech_text: str = DEFAULT_START_SPEECH,
        end_speech_text: str = DEFAULT_END_SPEECH,
        speech_volume: Optional[int] = None,
        speech_language: Optional[str] = None,
        control_rate_hz: float = DEFAULT_CONTROL_RATE_HZ,
        xy_speed_m_s: float = DEFAULT_XY_SPEED_M_S,
        z_speed_m_s: float = DEFAULT_Z_SPEED_M_S,
        remote_deadband: float = DEFAULT_REMOTE_DEADBAND,
        max_dq_rad: float = DEFAULT_MAX_DQ_RAD,
    ) -> None:
        self.iface = str(iface)
        self.domain_id = int(domain_id)
        self.headlight_rgb = scale_color(parse_color(headlight_color), int(headlight_intensity))
        self.headlight_interval_s = float(headlight_interval_s)
        self.start_speech_text = str(start_speech_text)
        self.end_speech_text = str(end_speech_text)
        self.speech_volume = speech_volume
        self.speech_language = speech_language
        self.control_rate_hz = max(1.0, float(control_rate_hz))
        self.xy_speed_m_s = max(0.0, float(xy_speed_m_s))
        self.z_speed_m_s = max(0.0, float(z_speed_m_s))
        self.remote_deadband = max(0.0, min(0.95, float(remote_deadband)))
        self.max_dq_rad = max(0.005, float(max_dq_rad))

        self._lock = threading.Lock()
        self._speech_lock = threading.Lock()
        self._arm_lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._prev_toggle_combo = False
        self._prev_buttons: dict[str, int] = {}
        self._buttons: dict[str, int] = {}
        self._snapshot = RemoteControllerSnapshot()
        self._headlight_stop = threading.Event()
        self._headlight_thread: PersistentHeadlight | None = None
        self._robot: Robot | None = None
        self._arm_sdk: ArmSdk | None = None
        self._control_thread: threading.Thread | None = None
        self._lowstate_sub: ChannelSubscriber | None = None

        ensure_channel_factory_initialized(self.domain_id, self.iface)
        lowstate_type = _resolve_lowstate_type()
        self._lowstate_sub = ChannelSubscriber(TOPIC_LOWSTATE, lowstate_type)
        self._lowstate_sub.Init(self._on_lowstate, 10)
        self._control_thread = threading.Thread(
            target=self._control_loop,
            name="g1_remote_ee_control",
            daemon=True,
        )
        self._control_thread.start()

    def snapshot(self) -> RemoteControllerSnapshot:
        with self._lock:
            return RemoteControllerSnapshot(**self._snapshot.__dict__)

    def stop(self) -> None:
        self._running.clear()
        self._deactivate_mode(turn_off_headlight=True)
        if self._control_thread is not None and self._control_thread.is_alive():
            self._control_thread.join(timeout=1.0)

    def run_forever(self) -> None:
        while self._running.is_set():
            time.sleep(0.2)

    def _get_robot(self) -> Robot:
        if self._robot is None:
            self._robot = Robot(
                iface=self.iface,
                domain_id=self.domain_id,
                safety_boot=False,
                recover_dev_mode_on_init=False,
                auto_start_sensors=False,
            )
        return self._robot

    def _get_arm_sdk(self) -> ArmSdk:
        if self._arm_sdk is None:
            self._arm_sdk = ArmSdk(iface=self.iface, domain_id=self.domain_id)
            self._arm_sdk.resync(timeout=3.0)
        return self._arm_sdk

    def _on_lowstate(self, msg: Any) -> None:
        try:
            remote = decode_remote(msg.wireless_remote)
        except Exception as exc:
            with self._lock:
                self._snapshot.last_status = f"remote decode failed: {exc}"
            return

        buttons = remote.get("buttons", {})
        combo = all(bool(buttons.get(name)) for name in MODE_TOGGLE_BUTTONS)
        pressed = [
            name for name, value in buttons.items()
            if value and not self._prev_buttons.get(name, 0)
        ]

        with self._lock:
            now = time.time()
            self._snapshot.last_lowstate_time = now
            self._snapshot.remote_lx = float(remote.get("lx", 0.0))
            self._snapshot.remote_ly = float(remote.get("ly", 0.0))
            self._snapshot.remote_rx = float(remote.get("rx", 0.0))
            self._snapshot.remote_ry = float(remote.get("ry", 0.0))
            self._buttons = dict(buttons)
            if pressed:
                self._snapshot.last_button_event = "+".join(pressed)
                self._snapshot.last_button_time = now

        with self._lock:
            mode_active = self._snapshot.mode_active

        if combo and not self._prev_toggle_combo:
            self._toggle_mode_async()
        elif mode_active and pressed:
            if "Select" in pressed:
                self._cycle_arm_control_mode()

        self._prev_toggle_combo = combo
        self._prev_buttons = dict(buttons)

    def _control_loop(self) -> None:
        dt = 1.0 / self.control_rate_hz
        next_tick = time.monotonic()
        while self._running.is_set():
            wait_s = next_tick - time.monotonic()
            if wait_s > 0:
                time.sleep(wait_s)
            if not self._running.is_set():
                break
            started = time.monotonic()
            self._apply_remote_ee_command(dt)
            next_tick = max(next_tick + dt, started + dt)

    def _apply_remote_ee_command(self, dt: float) -> None:
        with self._lock:
            if not self._snapshot.mode_active:
                return
            arm_mode = self._snapshot.arm_control_mode
            rx = self._snapshot.remote_rx
            ry = self._snapshot.remote_ry
            buttons = dict(self._buttons)

        dx = normalize_axis(ry, self.remote_deadband) * self.xy_speed_m_s * dt
        dy = normalize_axis(rx, self.remote_deadband) * self.xy_speed_m_s * dt
        z_dir = float(bool(buttons.get("Up"))) - float(bool(buttons.get("Down")))
        dz = z_dir * self.z_speed_m_s * dt
        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(dz) < 1e-6:
            return

        inc = [dx, dy, dz, 0.0, 0.0, 0.0]
        try:
            with self._arm_lock:
                info = self._get_arm_sdk().ik_move_EE(
                    inc,
                    arm=arm_mode,
                    position_only=True,
                    selected_axis=None,
                    max_dq=self.max_dq_rad,
                    timeout=0.5,
                )
            with self._lock:
                self._snapshot.last_ik_success = bool(info.get("success"))
                self._snapshot.last_ik_error = ""
                if not info.get("success"):
                    self._snapshot.last_status = (
                        f"IK failed {arm_mode}: pos={float(info.get('error_pos_m', 0.0)):.4f}m"
                    )
        except Exception as exc:
            with self._lock:
                self._snapshot.last_ik_success = False
                self._snapshot.last_ik_error = str(exc)
                self._snapshot.last_status = f"IK command failed: {exc}"

    def _toggle_mode_async(self) -> None:
        with self._lock:
            if self._snapshot.handoff_in_progress:
                return
            self._snapshot.handoff_in_progress = True
            activate = not self._snapshot.mode_active
        thread = threading.Thread(
            target=self._set_mode_worker,
            args=(activate,),
            name="g1_remote_mode_toggle",
            daemon=True,
        )
        thread.start()

    def _set_mode_worker(self, activate: bool) -> None:
        try:
            if activate:
                self._activate_mode()
            else:
                self._deactivate_mode(turn_off_headlight=True)
        finally:
            with self._lock:
                self._snapshot.handoff_in_progress = False

    def _activate_mode(self) -> None:
        with self._lock:
            self._snapshot.mode_active = True
            self._snapshot.last_status = f"{MODE_NAME} active"

        robot = self._get_robot()
        self._start_headlight(robot)
        try:
            self._get_arm_sdk().resync(timeout=3.0)
        except Exception as exc:
            with self._lock:
                self._snapshot.last_status = f"{MODE_NAME} active; arm resync failed: {exc}"
        self._speak_async(self.start_speech_text)

    def _deactivate_mode(self, *, turn_off_headlight: bool) -> None:
        self._stop_headlight(turn_off=turn_off_headlight)
        with self._lock:
            was_active = self._snapshot.mode_active
            self._snapshot.mode_active = False
            self._snapshot.last_status = f"{MODE_NAME} inactive"
        if was_active:
            self._speak_async(self.end_speech_text)

    def _cycle_arm_control_mode(self) -> None:
        with self._lock:
            current = self._snapshot.arm_control_mode
            try:
                idx = ARM_CONTROL_MODES.index(current)
            except ValueError:
                idx = 0
            next_mode = ARM_CONTROL_MODES[(idx + 1) % len(ARM_CONTROL_MODES)]
            self._snapshot.arm_control_mode = next_mode
            self._snapshot.last_status = f"{MODE_NAME} active; arm control {next_mode}"
        self._speak_async(f"{next_mode} arm control")

    def _speak_async(self, text: str) -> None:
        if not str(text).strip():
            return
        thread = threading.Thread(
            target=self._speak_once,
            args=(str(text),),
            name="g1_remote_mode_speech",
            daemon=True,
        )
        thread.start()

    def _speak_once(self, text: str) -> None:
        try:
            with self._speech_lock:
                code = int(self._get_robot().say(
                    text,
                    volume=self.speech_volume,
                    language=self.speech_language,
                ))
            with self._lock:
                self._snapshot.speech_code = code
                self._snapshot.last_status = f'speech "{text}" code {code}'
        except Exception as exc:
            with self._lock:
                self._snapshot.last_status = f'speech "{text}" failed: {exc}'

    def _start_headlight(self, robot: Robot) -> None:
        self._stop_headlight(turn_off=False)
        self._headlight_stop = threading.Event()
        client = robot._get_audio()._client
        self._headlight_thread = PersistentHeadlight(
            client,
            self.headlight_rgb,
            self.headlight_interval_s,
            self._headlight_stop,
            self._set_headlight_code,
        )
        self._headlight_thread.start()

    def _stop_headlight(self, *, turn_off: bool) -> None:
        self._headlight_stop.set()
        thread = self._headlight_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._headlight_thread = None
        if turn_off and self._robot is not None:
            try:
                code = int(self._robot._get_audio()._client.LedControl(0, 0, 0))
                self._set_headlight_code(code)
            except Exception as exc:
                with self._lock:
                    self._snapshot.last_status = f"headlight off failed: {exc}"

    def _set_headlight_code(self, code: int) -> None:
        with self._lock:
            self._snapshot.headlight_code = int(code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G1 remote-controller custom end-effector mode gate."
    )
    parser.add_argument("--iface", default=default_dds_iface(), help="DDS network interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain ID.")
    parser.add_argument("--headlight-color", default=DEFAULT_HEADLIGHT_COLOR)
    parser.add_argument("--headlight-intensity", type=int, default=DEFAULT_HEADLIGHT_INTENSITY)
    parser.add_argument("--headlight-interval-s", type=float, default=DEFAULT_HEADLIGHT_INTERVAL_S)
    parser.add_argument("--start-speech", default=DEFAULT_START_SPEECH)
    parser.add_argument("--end-speech", default=DEFAULT_END_SPEECH)
    parser.add_argument("--speech-volume", type=int, default=None)
    parser.add_argument("--speech-language", default=None)
    parser.add_argument("--control-rate-hz", type=float, default=DEFAULT_CONTROL_RATE_HZ)
    parser.add_argument("--xy-speed-m-s", type=float, default=DEFAULT_XY_SPEED_M_S)
    parser.add_argument("--z-speed-m-s", type=float, default=DEFAULT_Z_SPEED_M_S)
    parser.add_argument("--remote-deadband", type=float, default=DEFAULT_REMOTE_DEADBAND)
    parser.add_argument("--max-dq-rad", type=float, default=DEFAULT_MAX_DQ_RAD)
    parser.add_argument("--status-interval-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    controller = G1RemoteController(
        iface=str(args.iface),
        domain_id=int(args.domain_id),
        headlight_color=str(args.headlight_color),
        headlight_intensity=int(args.headlight_intensity),
        headlight_interval_s=float(args.headlight_interval_s),
        start_speech_text=str(args.start_speech),
        end_speech_text=str(args.end_speech),
        speech_volume=args.speech_volume,
        speech_language=args.speech_language,
        control_rate_hz=float(args.control_rate_hz),
        xy_speed_m_s=float(args.xy_speed_m_s),
        z_speed_m_s=float(args.z_speed_m_s),
        remote_deadband=float(args.remote_deadband),
        max_dq_rad=float(args.max_dq_rad),
    )

    stop_event = threading.Event()

    def _stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        print("G1 remote controller ready. Press R1+B to toggle Endeffector mode.", flush=True)
        while not stop_event.wait(max(0.1, float(args.status_interval_s))):
            snap = controller.snapshot()
            lowstate_age = (
                "--"
                if snap.last_lowstate_time <= 0.0
                else f"{time.time() - snap.last_lowstate_time:.1f}s"
            )
            print(
                f"mode={'on' if snap.mode_active else 'off'} "
                f"arms={snap.arm_control_mode} "
                f"handoff={'yes' if snap.handoff_in_progress else 'no'} "
                f"lowstate_age={lowstate_age} "
                f"rx/ry={snap.remote_rx:+.2f}/{snap.remote_ry:+.2f} "
                f"last_button={snap.last_button_event or '--'} "
                f"ik={snap.last_ik_success} "
                f"status={snap.last_status}",
                flush=True,
            )
    finally:
        controller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
