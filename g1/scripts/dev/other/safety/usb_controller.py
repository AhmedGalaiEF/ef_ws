#!/usr/bin/env python3
"""
usb_controller.py — USB gamepad teleop for Unitree G1.

Requires: pip install pygame

Controls:
    Left stick Y  : forward / backward (vx)
    Left stick X  : strafe left / right (vy)
    Right stick X : turn left / right (vyaw)

    Start  : BalancedStand(0) and re-enable controller output
    Select : StopMove and disable controller output

    L2 + DPad Up    : Damp
    L2 + DPad Right : ZeroTorque

    Hold R2         : run gait (SetGaitType(1))
    Release R2      : walk gait (SetGaitType(0))
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pygame

from hanger_boot_sequence import hanger_boot_sequence


# Common button indices (can vary by controller)
BTN_SELECT = 6
BTN_START = 7
BTN_L2_FALLBACK = 4
BTN_R2_FALLBACK = 5

# Axis indices
AXIS_LX = 0   # left stick X  -> vy (strafe)
AXIS_LY = 1   # left stick Y  -> vx (forward/back, inverted)
AXIS_RX = 3   # right stick X -> vyaw (turn)
AXIS_L2 = 2
AXIS_R2 = 5

MAX_VX = 0.5    # m/s
MAX_VY = 0.3    # m/s
MAX_VYAW = 0.8  # rad/s
DEADZONE = 0.1
SEND_HZ = 10


def apply_deadzone(value: float, dz: float) -> float:
    if abs(value) < dz:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - dz) / (1.0 - dz)


def _axis_pressed(v: float) -> bool:
    # Supports both [-1..1] and [0..1] trigger ranges.
    val = float(v)
    if val < -1.0:
        val = -1.0
    if val > 1.0:
        val = 1.0
    norm = 0.5 * (val + 1.0)
    return norm >= 0.60


def _set_red_headlight(iface: str) -> None:
    script = Path(__file__).resolve().parents[2] / "basic" / "headlight_client" / "headlight.py"
    if not script.exists():
        return
    cmd = [sys.executable, str(script), "--iface", iface, "--color", "red", "--intensity", "100"]
    try:
        subprocess.run(cmd, check=False)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="USB gamepad teleop for G1.")
    parser.add_argument("--iface", default="eth0", help="network interface")
    parser.add_argument("--joy", type=int, default=0, help="joystick index")
    args = parser.parse_args()

    bot = hanger_boot_sequence(iface=args.iface)
    try:
        bot.BalanceStand(0)
    except Exception:
        pass

    _set_red_headlight(args.iface)

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        raise SystemExit("No joystick detected. Connect a USB gamepad and retry.")

    joy = pygame.joystick.Joystick(args.joy)
    joy.init()
    print(f"Using: {joy.get_name()}  (axes={joy.get_numaxes()}, buttons={joy.get_numbuttons()})")
    print("Start=BalancedStand(re-enable)  Select=Stop+Disable  L2+Up=Damp  L2+Right=ZeroTorque")

    active = True  # False after Damp/ZeroTorque (don't send Move commands)
    dt = 1.0 / SEND_HZ
    run_mode = False
    last_start = False
    last_select = False
    last_damp_combo = False
    last_zt_combo = False
    last_mode: int | None = None

    try:
        while True:
            pygame.event.pump()

            # --- gating buttons ---
            start_pressed = joy.get_numbuttons() > BTN_START and bool(joy.get_button(BTN_START))
            select_pressed = joy.get_numbuttons() > BTN_SELECT and bool(joy.get_button(BTN_SELECT))

            if start_pressed and not last_start:
                print("[Start] BalanceStand + enable controller")
                bot.StopMove()
                bot.BalanceStand(0)
                active = True
            if select_pressed and not last_select:
                print("[Select] Stop + disable controller")
                bot.StopMove()
                active = False

            last_start = start_pressed
            last_select = select_pressed

            # --- trigger + dpad combos for FSM ---
            l2_pressed = False
            if joy.get_numaxes() > AXIS_L2:
                l2_pressed = _axis_pressed(joy.get_axis(AXIS_L2))
            if not l2_pressed and joy.get_numbuttons() > BTN_L2_FALLBACK:
                l2_pressed = bool(joy.get_button(BTN_L2_FALLBACK))
            hat_x, hat_y = (0, 0)
            if joy.get_numhats() > 0:
                hat_x, hat_y = joy.get_hat(0)

            damp_combo = l2_pressed and hat_y > 0
            zt_combo = l2_pressed and hat_x > 0

            if damp_combo and not last_damp_combo:
                print("[L2 + DPad Up] Damp")
                bot.Damp()
                active = False
            if zt_combo and not last_zt_combo:
                print("[L2 + DPad Right] ZeroTorque — EMERGENCY")
                bot.ZeroTorque()
                active = False

            last_damp_combo = damp_combo
            last_zt_combo = zt_combo

            # --- run/walk gait mode from R2 hold ---
            r2_pressed = False
            if joy.get_numaxes() > AXIS_R2:
                r2_pressed = _axis_pressed(joy.get_axis(AXIS_R2))
            if not r2_pressed and joy.get_numbuttons() > BTN_R2_FALLBACK:
                r2_pressed = bool(joy.get_button(BTN_R2_FALLBACK))
            if r2_pressed != run_mode:
                run_mode = r2_pressed
                mode = 1 if run_mode else 0
                if mode != last_mode:
                    try:
                        bot.SetGaitType(mode)
                        last_mode = mode
                        print("[R2] run gait" if run_mode else "[R2 released] walk gait")
                    except Exception:
                        pass

            # --- sticks ---
            if active:
                lx = apply_deadzone(joy.get_axis(AXIS_LX), DEADZONE)
                ly = apply_deadzone(joy.get_axis(AXIS_LY), DEADZONE)
                rx = apply_deadzone(joy.get_axis(AXIS_RX), DEADZONE)

                vx = -ly * MAX_VX     # stick up = negative axis = forward
                vy = -lx * MAX_VY     # stick left = negative axis = strafe left (positive vy)
                vyaw = -rx * MAX_VYAW  # stick left = negative axis = turn left (positive vyaw)

                bot.Move(vx, vy, vyaw)

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        bot.StopMove()
        pygame.quit()


if __name__ == "__main__":
    main()
