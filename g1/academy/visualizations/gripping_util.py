"""Thin compatibility shim so recognition_app_v3.py can talk to
sdk_wrapper.G1 through the same method names/signatures it already calls on
sdk_client.Robot for grasping/arm control, without rewriting call sites
across a 3000+ line file.

No robot capability lives here -- every ControlAdapter method is a one-line
forward onto a G1 instance; the real DDS/RPC work (ramped Dex3 control,
release/engage arms, locomotion, ...) lives on sdk_wrapper.G1. If a method
here grows logic of its own beyond a rename/forward, it belongs on G1
instead.
"""
from __future__ import annotations

from typing import Any, Optional


class ControlAdapter:
    """Wraps a sdk_wrapper.G1 instance behind sdk_client.Robot's control
    method names, so it can be dropped in as recognition_app_v3.py's
    `_CONTROL_ROBOT` unchanged."""

    def __init__(self, g1: Any) -> None:
        self.g1 = g1

    # -- arm authority ------------------------------------------------------

    def release_arms(self, duration_s: float = 3.0) -> dict:
        return self.g1.release_arms(duration_s=duration_s)

    def unrelease_arms(self, duration_s: float = 1.0) -> dict:
        # sdk_client.Robot names this "unrelease"; G1 calls the same ramp
        # engage_arms().
        return self.g1.engage_arms(duration_s=duration_s)

    # -- locomotion / FSM -----------------------------------------------------

    def damp(self) -> None:
        self.g1.damp_mode()

    def prepare(self) -> None:
        self.g1.prepare_mode()

    def walk_mode(self) -> None:
        self.g1.walk_mode()

    def loco_move(self, vx: float, vy: float, vyaw: float) -> int:
        # sdk_client.Robot.loco_move() is fire-and-forget: one Move() RPC,
        # no pacing/auto-stop of its own (that's move_for()'s job). G1's own
        # loco_move() bundles in a blocking sleep+auto-stop by default, so
        # pass duration_s=None to get the same fire-and-forget behavior --
        # otherwise e.g. sending (0, 0, 0) to stop immediately would itself
        # block for ~2s first.
        return self.g1.loco_move(vx, vy, vyaw, duration_s=None)

    def move_for(self, duration: float, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> int:
        # This is the paced one: fire, sleep for `duration`, then stop --
        # exactly G1.loco_move()'s default (duration_s) behavior.
        return self.g1.loco_move(vx, vy, vyaw, duration_s=duration)

    def stop(self) -> None:
        self.g1.loco_stop()

    # -- upper-body joints ----------------------------------------------------

    def move_upper_body_joint(self, joint_index: int, target: float, *, max_speed_rad_s: float = 0.45,
                               command_rate_hz: float = 50.0, timeout: float = 3.0, **_ignored: Any) -> dict:
        # sdk_client.Robot's max_speed_rad_s is the same velocity ceiling as
        # G1's max_joint_speed (shared with interpolate_to_pose()).
        # command_rate_hz/timeout have no equivalent knob on G1's version
        # (it uses interpolate_to_pose()'s own step count and
        # _upper_body_pose()'s default read timeout) -- accepted and
        # ignored rather than raising, so an old call site doesn't break.
        return self.g1.move_upper_body_joint(joint_index, target, max_joint_speed=max_speed_rad_s)

    # -- Dex3 hand --------------------------------------------------------------

    def hand_open(self, hand: str = "right", hold_s: float = 0.6, rate_hz: float = 50.0, ramp_s: float | None = None) -> None:
        self.g1.open_dex3_hand(hand=hand, hold_s=hold_s, rate_hz=rate_hz, ramp_s=ramp_s)

    def hand_close(self, hand: str = "right", hold_s: float = 0.6, rate_hz: float = 50.0, ramp_s: float | None = None) -> None:
        self.g1.close_dex3_hand(hand=hand, hold_s=hold_s, rate_hz=rate_hz, ramp_s=ramp_s)

    def hand_pose(self, targets: list[float], hand: str = "right", **kwargs: Any) -> None:
        self.g1.hand_pose(targets, hand=hand, **kwargs)

    def get_hand_state_snapshot(self, hand: str = "right") -> Optional[dict]:
        snapshot = self.g1.get_dex3_hand_sensors(hand)
        # G1's own convention on a missing/errored hand is an {"ok": False,
        # "error": ...} dict (see G1._hand_error()), not None -- translate
        # that here so this matches sdk_client.Robot's Optional[dict] return.
        return snapshot if isinstance(snapshot, dict) and "positions" in snapshot else None

    def get_tactile_pressures(self, hand: str = "right"):
        return self.g1.get_dex3_tactile_pressures(hand)

    def _get_hand(self, hand: str = "right") -> Any:
        # Kept for parity with sdk_client.Robot's private accessor -- some
        # call sites reach into it directly as a fallback when
        # get_hand_state_snapshot() comes back empty.
        return self.g1._dex3_hand(hand)


def hand_open_targets(hand: str) -> list[float]:
    from sdk_wrapper import HAND_OPEN
    return list(HAND_OPEN[str(hand).strip().lower()])


def hand_closed_targets(hand: str) -> list[float]:
    from sdk_wrapper import HAND_CLOSED
    return list(HAND_CLOSED[str(hand).strip().lower()])


__all__ = ["ControlAdapter", "hand_open_targets", "hand_closed_targets"]
