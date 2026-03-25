#!/usr/bin/env python3
"""
test_solution.py
================
 
Demonstrates the correct pattern for combining arm control with normal walking
on the Unitree G1.
 
PROBLEM:
  Any use of the rt/arm_sdk channel (motor_cmd[29].q = 1) takes over arm
  joints from the locomotion controller.  If the arm_sdk is simply abandoned
  (stop publishing) without an explicit release, the arms remain frozen —
  they no longer swing naturally during walking.
 
ROOT CAUSE (confirmed from g1_arm7_sdk_dds_example.py):
  motor_cmd[NOT_USED_IDX=29].q acts as a blend weight between arm_sdk control
  and the locomotion controller's default arm motion.
    1 → arm_sdk owns the arm joints fully
    0 → locomotion controller owns the arm joints (natural swing)
 
  Existing scripts (e.g. pbd_demonstrate.py) set the flag to 1 in __init__
  and never ramp it back, so the locomotion controller never reclaims the arms.
 
SOLUTION:
  When done with arm_sdk control, publish a ramp that drives
  motor_cmd[29].q from 1.0 → 0.0 over ~1-2 seconds, then stop publishing.
  This smoothly hands arm control back to the locomotion controller.
 
DEMO SEQUENCE:
  1. Safe boot  → balanced stand (FSM-200) via hanger_boot_sequence
  2. Arm SDK    → arms go to zero-torque (compliant / limp)
  3. Walk       → robot walks forward while arms are compliant
  4. Stop walk  → StopMove()
  5. Release    → ramp motor_cmd[29].q 1→0, default arm swing resumes
"""
from __future__ import annotations
 
import argparse
import sys
import time
import threading
from pathlib import Path
from typing import List
 
# ---------------------------------------------------------------------------
# Path setup — works regardless of which directory the script is run from
# ---------------------------------------------------------------------------
_SCRIPTS_ROOT = Path(__file__).resolve().parent
_DEV_DIR      = _SCRIPTS_ROOT / "dev"
_SAFETY_DIR   = _DEV_DIR / "other" / "safety"
 
for _p in [str(_DEV_DIR), str(_SAFETY_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
 
# ---------------------------------------------------------------------------
# Unitree SDK imports
# ---------------------------------------------------------------------------
try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
except ImportError as exc:
    raise SystemExit(
        "unitree_sdk2py is not installed.\n"
        "Install it with:  pip install -e <path-to-unitree_sdk2_python>"
    ) from exc
 
try:
    from safety.hanger_boot_sequence import hanger_boot_sequence
except ImportError:
    from hanger_boot_sequence import hanger_boot_sequence  # type: ignore
 
# ---------------------------------------------------------------------------
# Joint index constants  (G1 29-DoF)
# ---------------------------------------------------------------------------
LEFT_ARM_IDX  = [15, 16, 17, 18, 19, 20, 21]   # shoulder_pitch … wrist_yaw
RIGHT_ARM_IDX = [22, 23, 24, 25, 26, 27, 28]
ALL_ARM_IDX: List[int] = LEFT_ARM_IDX + RIGHT_ARM_IDX
 
NOT_USED_IDX = 29  # motor_cmd[29].q: 1 = arm_sdk active, 0 = loco controller
 
 
# ---------------------------------------------------------------------------
# ArmSdkController
# ---------------------------------------------------------------------------
class ArmSdkController:
    """
    Manages arm joints via the rt/arm_sdk DDS channel.
 
    Lifecycle
    ---------
    1. Construct  → publisher + subscriber initialised
    2. wait_for_state()  → block until rt/lowstate received
    3. enable_zero_torque(ramp_s)  → ramp motor_cmd[29].q 0→1, kp=kd=tau=0
    4. [robot does things]
    5. release(ramp_s)  → ramp motor_cmd[29].q 1→0, stop publishing
       After step 5 the locomotion controller reclaims the arms and natural
       arm swing during walking is fully restored.
    """
 
    def __init__(self, cmd_hz: float = 50.0) -> None:
        self._dt      = 1.0 / max(10.0, float(cmd_hz))
        self._cmd_hz  = max(10.0, float(cmd_hz))
 
        self._lock    = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
 
        # Blend weight published to motor_cmd[NOT_USED_IDX].q
        self._sdk_weight = 0.0
 
        # Per-joint PD gains (default: zero torque)
        self._kp = [0.0] * 35
        self._kd = [0.0] * 35
 
        # Desired joint positions (used as hold reference)
        self._q_hold = [0.0] * 35
 
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
 
        # Current joint state from rt/lowstate
        self._q_cur: List[float] = [0.0] * 35
        self._state_ready = threading.Event()
 
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
 
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._lowstate_cb, 10)
 
    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------
 
    def _lowstate_cb(self, msg: LowState_) -> None:
        q: List[float] = []
        for i in range(35):
            try:
                q.append(float(msg.motor_state[i].q))
            except Exception:
                q.append(0.0)
        with self._lock:
            self._q_cur = q
        self._state_ready.set()
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def wait_for_state(self, timeout: float = 3.0) -> bool:
        """Block until rt/lowstate has been received at least once."""
        ok = self._state_ready.wait(timeout=timeout)
        if ok:
            with self._lock:
                self._q_hold = list(self._q_cur)
        return ok
 
    def enable_zero_torque(self, ramp_s: float = 1.0) -> None:
        """
        Enable arm_sdk and set all arm joints to zero torque.
 
        Arms become fully compliant (backdrivable).  The blend weight
        motor_cmd[29].q is ramped from 0 → 1 over ramp_s seconds so the
        transition is smooth and does not disturb balance.
        """
        with self._lock:
            self._q_hold = list(self._q_cur)
            for i in ALL_ARM_IDX:
                self._kp[i] = 0.0
                self._kd[i] = 0.0
            self._sdk_weight = 0.0
 
        self._start_loop()
        self._ramp_weight(0.0, 1.0, ramp_s)
        print("[ArmCtrl] Zero-torque active — arms are fully compliant.")
 
    def release(self, ramp_s: float = 1.5) -> None:
        """
        Gradually release arm_sdk control.
 
        Ramps motor_cmd[29].q from 1 → 0 over ramp_s seconds, then stops
        the publish loop.  After this call the locomotion controller fully
        reclaims the arm joints and the default walking arm-swing resumes.
 
        This is the CRITICAL step that existing scripts were missing.
        """
        print("[ArmCtrl] Releasing arm SDK (ramp %.1fs) …" % ramp_s)
        self._ramp_weight(1.0, 0.0, ramp_s)
 
        # Stop the publish loop — arm_sdk is now fully disabled
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        print("[ArmCtrl] Arm SDK released — default arm swing restored.")
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _ramp_weight(self, start: float, end: float, duration: float) -> None:
        steps = max(1, int(self._cmd_hz * max(0.05, float(duration))))
        for step in range(steps + 1):
            alpha = step / steps
            w = start + (end - start) * alpha
            with self._lock:
                self._sdk_weight = float(w)
            time.sleep(self._dt)
 
    def _start_loop(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()
 
    def _publish_loop(self) -> None:
        """Continuously publish arm_sdk commands at cmd_hz."""
        while self._running:
            with self._lock:
                weight  = self._sdk_weight
                q_hold  = list(self._q_hold)
                kp      = list(self._kp)
                kd      = list(self._kd)
 
            # Blend-weight sentinel
            self._cmd.motor_cmd[NOT_USED_IDX].q = float(weight)
 
            for i in ALL_ARM_IDX:
                mc      = self._cmd.motor_cmd[i]
                mc.q    = float(q_hold[i])   # position reference (irrelevant when kp=0)
                mc.dq   = 0.0
                mc.kp   = float(kp[i])
                mc.kd   = float(kd[i])
                mc.tau  = 0.0
 
            self._cmd.crc = self._crc.Crc(self._cmd)
            self._pub.Write(self._cmd)
            time.sleep(self._dt)
 
 
# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "G1 safe boot + zero-torque arms + walk demo.\n"
            "Demonstrates the correct arm_sdk release pattern that\n"
            "restores default arm swing after arm control."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--iface",          default="eth0",
                   help="Network interface for DDS (default: eth0)")
    p.add_argument("--domain-id",      type=int, default=0,
                   help="DDS domain ID (default: 0)")
    p.add_argument("--vx",             type=float, default=0.2,
                   help="Forward velocity in m/s (default: 0.2)")
    p.add_argument("--walk-duration",  type=float, default=5.0,
                   help="How long to walk forward in seconds (default: 5.0)")
    p.add_argument("--enable-arms",    action="store_true", default=True,
                   help="Enable zero-torque arm mode during walk (default: on)")
    p.add_argument("--no-arms",        dest="enable_arms", action="store_false",
                   help="Skip arm control (walk with default arm swing only)")
    p.add_argument("--ramp-in",        type=float, default=1.0,
                   help="Seconds to ramp arm_sdk weight 0→1 (default: 1.0)")
    p.add_argument("--ramp-out",       type=float, default=1.5,
                   help="Seconds to ramp arm_sdk weight 1→0 (default: 1.5)")
    return p.parse_args()
 
 
def main() -> None:
    args = parse_args()
 
    print("=" * 60)
    print("G1 Test: safe boot + zero-torque arms + walk")
    print("=" * 60)
    print(f"  Interface    : {args.iface}")
    print(f"  Domain ID    : {args.domain_id}")
    print(f"  Forward vel  : {args.vx:.2f} m/s")
    print(f"  Walk duration: {args.walk_duration:.1f} s