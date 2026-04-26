from __future__ import annotations

import os
import time
import threading
from typing import Dict

from unitree_sdk2py.core import channel as channel_module
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_


channel_module.ChannelConfigHasInterface = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>"""
channel_module.ChannelConfigAutoDetermine = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface autodetermine="true" priority="default" multicast="default" />
      </Interfaces>
    </General>
  </Domain>
</CycloneDDS>"""
os.environ.setdefault(
    "CYCLONEDDS_URI",
    "<CycloneDDS><Domain><Tracing><Category>none</Category></Tracing></Domain></CycloneDDS>",
)


TOPIC_HAND_BY_SIDE = {
    "left": "rt/dex3/left/cmd",
    "right": "rt/dex3/right/cmd",
}

HAND_MAX_LIMITS = {
    "left": [1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0],
    "right": [1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75],
}

HAND_MIN_LIMITS = {
    "left": [-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75],
    "right": [-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0],
}

HAND_THUMB_0_HOLD_TARGETS = {
    "left": -0.09927542507648468,
    "right": -0.03510913997888565,
}

# Backwards-compatible right-hand presets. New code should use
# hand_open_targets(), hand_closed_targets(), or hand_grip_targets().
HAND_OPEN = [
    HAND_THUMB_0_HOLD_TARGETS["right"],
    HAND_MAX_LIMITS["right"][1],
    HAND_MAX_LIMITS["right"][2],
    HAND_MIN_LIMITS["right"][3],
    HAND_MIN_LIMITS["right"][4],
    HAND_MIN_LIMITS["right"][5],
    HAND_MIN_LIMITS["right"][6],
]

HAND_CLOSED = [
    HAND_THUMB_0_HOLD_TARGETS["right"],
    HAND_MIN_LIMITS["right"][1],
    HAND_MIN_LIMITS["right"][2],
    HAND_MAX_LIMITS["right"][3],
    HAND_MAX_LIMITS["right"][4],
    HAND_MAX_LIMITS["right"][5],
    HAND_MAX_LIMITS["right"][6],
]

HAND_CLOSED_LIMITS = {
    "left": [
        HAND_THUMB_0_HOLD_TARGETS["left"],
        HAND_MAX_LIMITS["left"][1],
        HAND_MAX_LIMITS["left"][2],
        HAND_MIN_LIMITS["left"][3],
        HAND_MIN_LIMITS["left"][4],
        HAND_MIN_LIMITS["left"][5],
        HAND_MIN_LIMITS["left"][6],
    ],
    "right": [
        HAND_THUMB_0_HOLD_TARGETS["right"],
        HAND_MIN_LIMITS["right"][1],
        HAND_MIN_LIMITS["right"][2],
        HAND_MAX_LIMITS["right"][3],
        HAND_MAX_LIMITS["right"][4],
        HAND_MAX_LIMITS["right"][5],
        HAND_MAX_LIMITS["right"][6],
    ],
}

HAND_OPEN_LIMITS = {
    side: [
        closed[0],
        *[
            hi if abs(closed_value - lo) < abs(closed_value - hi) else lo
            for closed_value, lo, hi in zip(
                closed[1:],
                HAND_MIN_LIMITS[side][1:],
                HAND_MAX_LIMITS[side][1:],
            )
        ],
    ]
    for side, closed in HAND_CLOSED_LIMITS.items()
}

FINGER_TO_IDXS: Dict[str, list[int]] = {
    "thumb": [0, 1, 2],
    "middle": [3, 4],
    "index": [5, 6],
}


def hand_mid_targets(hand: str) -> list[float]:
    side = str(hand).strip().lower()
    return [
        (lo + hi) / 2.0
        for lo, hi in zip(HAND_MIN_LIMITS[side], HAND_MAX_LIMITS[side])
    ]


def _with_thumb_0(targets: list[float], thumb_0: float | None) -> list[float]:
    if thumb_0 is not None:
        targets[0] = float(thumb_0)
    return targets


def hand_open_targets(hand: str, thumb_0: float | None = None) -> list[float]:
    side = str(hand).strip().lower()
    return _with_thumb_0(list(HAND_OPEN_LIMITS[side]), thumb_0)


def hand_closed_targets(hand: str, thumb_0: float | None = None) -> list[float]:
    side = str(hand).strip().lower()
    return _with_thumb_0(list(HAND_CLOSED_LIMITS[side]), thumb_0)


def hand_grip_targets(hand: str, percent: float, thumb_0: float | None = None) -> list[float]:
    alpha = min(1.0, max(0.0, float(percent) / 100.0))
    open_targets = hand_open_targets(hand, thumb_0=thumb_0)
    closed_targets = hand_closed_targets(hand, thumb_0=thumb_0)
    return [
        start + (stop - start) * alpha
        for start, stop in zip(open_targets, closed_targets)
    ]


def pack_ris_mode(motor_id: int, status: int = 1, timeout: int = 0) -> int:
    return (
        (int(motor_id) & 0x0F)
        | ((int(status) & 0x07) << 4)
        | ((int(timeout) & 0x01) << 7)
    )


def build_hand_msg(
    targets: list[float],
    kp: float,
    kd: float,
    tau: float,
    *,
    timeout: int = 0,
) -> HandCmd_:
    if len(targets) != 7:
        raise ValueError("Hand targets must contain 7 joint values.")
    msg = unitree_hg_msg_dds__HandCmd_()
    for idx in range(7):
        cmd = msg.motor_cmd[idx]
        cmd.mode = pack_ris_mode(idx, timeout=timeout)
        cmd.tau = float(tau)
        cmd.q = float(targets[idx])
        cmd.dq = 0.0
        cmd.kp = float(kp)
        cmd.kd = float(kd)
    return msg


class Dex3HandController:
    def __init__(self, hand: str = "right", iface: str = "eth0", domain_id: int = 0) -> None:
        side = str(hand).strip().lower()
        if side not in TOPIC_HAND_BY_SIDE:
            raise ValueError(f"Invalid hand '{hand}'.")
        self.hand = side
        self.iface = str(iface)
        self.domain_id = int(domain_id)
        ChannelFactoryInitialize(self.domain_id, self.iface)
        self._pub = ChannelPublisher(TOPIC_HAND_BY_SIDE[self.hand], HandCmd_)
        self._pub.Init()
        self._last_targets: list[float] | None = None
        self._release_stop: threading.Event | None = None
        self._release_thread: threading.Thread | None = None

    @staticmethod
    def _interpolate_targets(
        start: list[float],
        stop: list[float],
        *,
        alpha: float,
    ) -> list[float]:
        blend = min(1.0, max(0.0, float(alpha)))
        return [s + (e - s) * blend for s, e in zip(start, stop)]

    def _publish_targets_for(
        self,
        targets: list[float],
        *,
        seconds: float,
        rate_hz: float,
        kp: float,
        kd: float,
        tau: float,
    ) -> None:
        self.publish_for(
            build_hand_msg(targets, kp=kp, kd=kd, tau=tau),
            seconds=seconds,
            rate_hz=rate_hz,
        )
        self._last_targets = list(targets)

    def write_targets_once(
        self,
        targets: list[float],
        *,
        kp: float = 0.5,
        kd: float = 0.1,
        tau: float = 0.0,
        timeout: int = 0,
        first_write_timeout_s: float | None = None,
    ) -> bool:
        if len(targets) != 7:
            raise ValueError("Hand targets must contain 7 joint values.")
        msg = build_hand_msg(targets, kp=kp, kd=kd, tau=tau, timeout=timeout)
        ok = self._pub.Write(msg, timeout=first_write_timeout_s)
        self._last_targets = [float(value) for value in targets]
        return ok is not False

    def publish_for(
        self,
        msg: HandCmd_,
        seconds: float,
        rate_hz: float = 50.0,
        *,
        first_write_timeout_s: float | None = None,
    ) -> bool:
        steps = max(1, int(max(0.01, float(seconds)) * max(1.0, float(rate_hz))))
        dt = 1.0 / max(1.0, float(rate_hz))
        matched = True
        for step_idx in range(steps):
            timeout = first_write_timeout_s if step_idx == 0 else None
            ok = self._pub.Write(msg, timeout=timeout)
            if step_idx == 0 and ok is False:
                matched = False
            time.sleep(dt)
        return matched

    def _stop_release_thread(self) -> None:
        if self._release_stop is not None:
            self._release_stop.set()
        if self._release_thread is not None and self._release_thread.is_alive():
            self._release_thread.join(timeout=1.0)
        self._release_stop = None
        self._release_thread = None

    def set_targets(
        self,
        targets: list[float],
        hold_s: float = 0.6,
        rate_hz: float = 50.0,
        kp: float = 1.2,
        kd: float = 0.05,
        tau: float = 0.05,
        ramp_s: float | None = None,
    ) -> None:
        self._stop_release_thread()
        if len(targets) != 7:
            raise ValueError("Hand targets must contain 7 joint values.")

        target_list = [float(value) for value in targets]
        rate = max(1.0, float(rate_hz))
        total_hold_s = max(0.0, float(hold_s))
        ramp_duration_s = min(
            total_hold_s,
            max(1.0 / rate, 0.25 if ramp_s is None else float(ramp_s)),
        )

        start_targets = hand_open_targets(self.hand) if self._last_targets is None else list(self._last_targets)
        if any(abs(dst - src) > 1e-6 for src, dst in zip(start_targets, target_list)) and ramp_duration_s > 0.0:
            ramp_steps = max(2, int(round(ramp_duration_s * rate)))
            step_dt = ramp_duration_s / float(ramp_steps)
            for step_idx in range(1, ramp_steps + 1):
                alpha = float(step_idx) / float(ramp_steps)
                interp_targets = self._interpolate_targets(start_targets, target_list, alpha=alpha)
                self._publish_targets_for(
                    interp_targets,
                    seconds=step_dt,
                    rate_hz=rate,
                    kp=kp,
                    kd=kd,
                    tau=tau,
                )

        remaining_hold_s = max(0.0, total_hold_s - ramp_duration_s)
        if remaining_hold_s > 0.0 or self._last_targets is None:
            self._publish_targets_for(
                target_list,
                seconds=remaining_hold_s if remaining_hold_s > 0.0 else (1.0 / rate),
                rate_hz=rate,
                kp=kp,
                kd=kd,
                tau=tau,
            )

    def _pose_targets(self, pose_targets: list[float]) -> list[float]:
        targets = list(pose_targets)
        if self._last_targets is not None:
            targets[0] = float(self._last_targets[0])
        return targets

    def open(self, hold_s: float = 0.6, rate_hz: float = 50.0, ramp_s: float | None = None) -> None:
        self.set_targets(
            hand_open_targets(self.hand),
            hold_s=hold_s,
            rate_hz=rate_hz,
            ramp_s=ramp_s,
        )

    def close(self, hold_s: float = 0.6, rate_hz: float = 50.0, ramp_s: float | None = None) -> None:
        self.set_targets(
            self._pose_targets(hand_closed_targets(self.hand)),
            hold_s=hold_s,
            rate_hz=rate_hz,
            tau=0.0,
            ramp_s=ramp_s,
        )

    def release_fingers(
        self,
        hold_s: float = 0.5,
        rate_hz: float = 50.0,
        *,
        persistent: bool = False,
    ) -> None:
        self._stop_release_thread()
        targets = list(self._last_targets) if self._last_targets is not None else hand_open_targets(self.hand)
        msg = build_hand_msg(targets, kp=0.0, kd=0.0, tau=0.0, timeout=1)
        if persistent:
            stop_event = threading.Event()

            def _loop() -> None:
                dt = 1.0 / max(1.0, float(rate_hz))
                while not stop_event.is_set():
                    self._pub.Write(msg)
                    time.sleep(dt)

            self._release_stop = stop_event
            self._release_thread = threading.Thread(
                target=_loop,
                name=f"dex3-{self.hand}-release",
                daemon=True,
            )
            self._release_thread.start()
        else:
            self.publish_for(
                msg,
                seconds=hold_s,
                rate_hz=rate_hz,
                first_write_timeout_s=1.0,
            )
        self._last_targets = None

    def stop_release_fingers(self) -> None:
        self._stop_release_thread()

    def move_finger(self, finger_name: str, hold_s: float = 1.0, settle_s: float = 0.6, rate_hz: float = 50.0) -> None:
        finger = str(finger_name).strip().lower()
        if finger not in FINGER_TO_IDXS:
            raise ValueError(f"Unknown finger '{finger_name}'.")
        targets = self._pose_targets(hand_open_targets(self.hand))
        closed_targets = self._pose_targets(hand_closed_targets(self.hand))
        for idx in FINGER_TO_IDXS[finger]:
            targets[idx] = closed_targets[idx]
        self.open(hold_s=settle_s, rate_hz=rate_hz)
        self.set_targets(targets, hold_s=hold_s, rate_hz=rate_hz, tau=0.12)
        self.open(hold_s=settle_s, rate_hz=rate_hz)


__all__ = [
    "Dex3HandController",
    "FINGER_TO_IDXS",
    "HAND_CLOSED",
    "HAND_CLOSED_LIMITS",
    "HAND_MAX_LIMITS",
    "HAND_MIN_LIMITS",
    "HAND_OPEN",
    "HAND_OPEN_LIMITS",
    "HAND_THUMB_0_HOLD_TARGETS",
    "build_hand_msg",
    "hand_closed_targets",
    "hand_grip_targets",
    "hand_mid_targets",
    "hand_open_targets",
    "pack_ris_mode",
]
