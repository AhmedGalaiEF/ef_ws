from __future__ import annotations

import os
import time
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

HAND_OPEN = [
    -0.15717165172100067,
    -0.41322529315948486,
    0.02846403606235981,
    0.17782948911190033,
    -0.025226416066288948,
    0.17983606457710266,
    -0.027690349146723747,
]

HAND_CLOSED = [
    0.07452802360057831,
    0.9478388428688049,
    1.766921877861023,
    -1.4442411661148071,
    -1.4384468793869019,
    -1.5298594236373901,
    -1.4153316020965576,
]

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


def hand_open_targets(hand: str) -> list[float]:
    side = str(hand).strip().lower()
    return list(HAND_MIN_LIMITS[side])


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

    def open(self, hold_s: float = 0.6, rate_hz: float = 50.0, ramp_s: float | None = None) -> None:
        self.set_targets(hand_open_targets(self.hand), hold_s=hold_s, rate_hz=rate_hz, ramp_s=ramp_s)

    def close(self, hold_s: float = 0.6, rate_hz: float = 50.0, ramp_s: float | None = None) -> None:
        self.set_targets(hand_mid_targets(self.hand), hold_s=hold_s, rate_hz=rate_hz, tau=0.0, ramp_s=ramp_s)

    def move_finger(self, finger_name: str, hold_s: float = 1.0, settle_s: float = 0.6, rate_hz: float = 50.0) -> None:
        finger = str(finger_name).strip().lower()
        if finger not in FINGER_TO_IDXS:
            raise ValueError(f"Unknown finger '{finger_name}'.")
        targets = hand_open_targets(self.hand)
        mid_targets = hand_mid_targets(self.hand)
        for idx in FINGER_TO_IDXS[finger]:
            targets[idx] = mid_targets[idx]
        self.open(hold_s=settle_s, rate_hz=rate_hz)
        self.set_targets(targets, hold_s=hold_s, rate_hz=rate_hz, tau=0.12)
        self.open(hold_s=settle_s, rate_hz=rate_hz)


__all__ = [
    "Dex3HandController",
    "FINGER_TO_IDXS",
    "HAND_CLOSED",
    "HAND_MAX_LIMITS",
    "HAND_MIN_LIMITS",
    "HAND_OPEN",
    "build_hand_msg",
    "hand_mid_targets",
    "hand_open_targets",
    "pack_ris_mode",
]
