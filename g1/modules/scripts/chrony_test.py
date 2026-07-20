#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import socket
import statistics
import struct
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

NTP_EPOCH_DELTA = 2_208_988_800
DEFAULT_PC1_IP = "192.168.123.161"


def run_cmd(args: list[str], *, timeout_s: float = 5.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(0.1, float(timeout_s)),
        )
        return int(proc.returncode), proc.stdout.strip()
    except FileNotFoundError:
        return 127, f"{args[0]}: command not found"
    except subprocess.TimeoutExpired:
        return 124, f"{' '.join(args)}: timed out after {timeout_s:.1f}s"


def unix_to_ntp(value: float) -> tuple[int, int]:
    ntp_value = float(value) + NTP_EPOCH_DELTA
    seconds = int(ntp_value)
    fraction = int((ntp_value - seconds) * 2**32)
    return seconds, fraction


def ntp_to_unix(seconds: int, fraction: int) -> float:
    return int(seconds) + int(fraction) / 2**32 - NTP_EPOCH_DELTA


def ntp_offset_once_ms(host: str, *, timeout_s: float = 3.0) -> tuple[float, float]:
    packet = bytearray(b"\x1b" + 47 * b"\0")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(max(0.1, float(timeout_s)))
    try:
        t0 = time.time()
        tx_sec, tx_frac = unix_to_ntp(t0)
        struct.pack_into("!II", packet, 40, tx_sec, tx_frac)
        sock.sendto(bytes(packet), (host, 123))
        data, _addr = sock.recvfrom(512)
        t3 = time.time()
    finally:
        sock.close()

    if len(data) < 48:
        raise RuntimeError(f"short NTP response: {len(data)} bytes")
    values = struct.unpack("!12I", data[:48])
    origin = ntp_to_unix(values[6], values[7]) or t0
    server_recv = ntp_to_unix(values[8], values[9])
    server_tx = ntp_to_unix(values[10], values[11])
    offset = ((server_recv - origin) + (server_tx - t3)) / 2.0
    delay = (t3 - origin) - (server_tx - server_recv)
    return offset * 1000.0, delay * 1000.0


def ntp_offset_ms(host: str, *, timeout_s: float = 3.0, samples: int = 5) -> tuple[float, float, list[tuple[float, float]]]:
    measurements: list[tuple[float, float]] = []
    for _idx in range(max(1, int(samples))):
        measurements.append(ntp_offset_once_ms(host, timeout_s=timeout_s))
        time.sleep(0.05)
    offsets = [item[0] for item in measurements]
    delays = [item[1] for item in measurements]
    return statistics.median(offsets), statistics.median(delays), measurements


def chrony_ok(pc1_ip: str) -> tuple[bool, str]:
    src_code, sources = run_cmd(["chronyc", "sources", "-v"])
    trk_code, tracking = run_cmd(["chronyc", "tracking"])
    ok = (
        src_code == 0
        and trk_code == 0
        and pc1_ip in sources
        and "Leap status" in tracking
        and "Normal" in tracking
    )
    return ok, f"$ chronyc sources -v\n{sources}\n\n$ chronyc tracking\n{tracking}"


def pose_delta(start: tuple[float, float, float] | None, pose: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if start is None or pose is None:
        return None
    x0, y0, yaw0 = start
    x, y, yaw = pose
    dx = float(x) - float(x0)
    dy = float(y) - float(y0)
    dyaw = math.atan2(math.sin(float(yaw) - float(yaw0)), math.cos(float(yaw) - float(yaw0)))
    return dx, dy, dyaw


def fmt_pose(pose: tuple[float, float, float] | None) -> str:
    if pose is None:
        return "n/a"
    return f"({pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f})"


def prepare_locomotion(robot: Any) -> str:
    client = getattr(robot, "_client", None)
    if client is not None:
        for method_name, args in (
            ("BalanceStand", (0,)),
            ("Start", ()),
            ("SetFsmId", (501,)),
        ):
            method = getattr(client, method_name, None)
            if callable(method):
                method(*args)
                time.sleep(0.3)
        return "_client.BalanceStand/Start/SetFsmId"
    for method_name in ("balanced_stand", "walk_mode", "fsm_4_prepare"):
        method = getattr(robot, method_name, None)
        if callable(method):
            method()
            return method_name
    return "none"


def send_velocity(robot: Any, vx: float, vy: float, vyaw: float, duration_s: float) -> Any:
    client = getattr(robot, "_client", None)
    if client is not None and callable(getattr(client, "SetVelocity", None)):
        return client.SetVelocity(float(vx), float(vy), float(vyaw), float(duration_s))
    return robot.loco_move(float(vx), float(vy), float(vyaw))


def parse_turn_degrees(raw: str) -> list[float]:
    values: list[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if not text:
            continue
        values.append(float(text))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Test G1 PC1 time synchronization while running a short locoClient.Move motion. "
            "The motion part requires --yes."
        )
    )
    parser.add_argument("--pc1-ip", default=DEFAULT_PC1_IP, help="G1 PC1 NTP address.")
    parser.add_argument("--iface", default="eth0", help="Robot SDK DDS interface.")
    parser.add_argument("--domain-id", type=int, default=0, help="Robot SDK DDS domain ID.")
    parser.add_argument("--vx", type=float, default=0.20, help="Forward velocity for the test motion in m/s.")
    parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity for the test motion in m/s.")
    parser.add_argument("--vyaw", type=float, default=0.0, help="Yaw velocity for the test motion in rad/s.")
    parser.add_argument("--duration-s", type=float, default=2.0, help="Motion command duration.")
    parser.add_argument(
        "--turn-deg",
        default="90,180",
        help="Comma-separated yaw turn angle(s) after the forward motion. Default: 90,180.",
    )
    parser.add_argument("--turn-rate", type=float, default=0.35, help="Absolute yaw velocity for the turn in rad/s.")
    parser.add_argument("--skip-turn", action="store_true", help="Only run the linear motion segment.")
    parser.add_argument("--sample-hz", type=float, default=20.0, help="Pose logging sample rate.")
    parser.add_argument("--max-offset-ms", type=float, default=5.0, help="Maximum acceptable PC1/local offset.")
    parser.add_argument("--ntp-samples", type=int, default=5, help="Number of PC1 NTP samples to median-filter.")
    parser.add_argument("--log-path", default=None, help="CSV output path.")
    parser.add_argument("--skip-stand", action="store_true", help="Do not call a stand/walk-mode helper before moving.")
    parser.add_argument("--yes", action="store_true", help="Actually move the robot.")
    args = parser.parse_args()

    pc1_ip = str(args.pc1_ip)
    print(f"Testing time sync against PC1 {pc1_ip}")
    print("This script uses time.monotonic() for motion timing so NTP wall-clock changes cannot stretch the command.")

    try:
        offset_before, delay_before, ntp_before = ntp_offset_ms(
            pc1_ip,
            samples=int(args.ntp_samples),
        )
    except Exception as exc:
        print(f"FAIL: PC1 NTP query failed before motion: {exc}")
        return 1
    print(
        f"PC1/local offset before motion: {offset_before:+.3f} ms, "
        f"NTP delay {delay_before:.3f} ms, samples={len(ntp_before)}"
    )

    ok, chrony_report = chrony_ok(pc1_ip)
    print()
    print(chrony_report)
    if not ok:
        print("FAIL: Chrony is not synced to PC1 with normal leap status.")
        return 1
    if abs(offset_before) > float(args.max_offset_ms):
        print(f"FAIL: PC1/local offset exceeds --max-offset-ms ({args.max_offset_ms:.3f} ms).")
        return 1

    if not args.yes:
        print()
        print("Time sync looks OK. Re-run with --yes to perform the locoClient.Move motion test.")
        return 2

    duration_s = max(0.0, float(args.duration_s))
    sample_hz = max(1.0, float(args.sample_hz))
    sample_dt = 1.0 / sample_hz
    log_path = args.log_path or f"/tmp/g1_loco_time_sync_{int(time.time())}.csv"
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    from sdk_client import Robot

    robot = Robot(iface=args.iface, domain_id=args.domain_id, auto_start_sensors=True)
    if not args.skip_stand:
        method_name = prepare_locomotion(robot)
        if method_name == "none":
            print("No stand/walk-mode helper exists on Robot; continuing without one.")
        else:
            print(f"Called {method_name}() before motion.")
        time.sleep(1.0)

    start_pose = robot.get_odom_pose() or robot.get_position()
    print(f"Start pose: {fmt_pose(start_pose)}")
    try:
        print(f"FSM before motion: {robot.get_fsm()}")
        print(f"Mode/gait before motion: mode={robot.get_mode()} gait={robot.get_gait()}")
    except Exception as exc:
        print(f"Could not read FSM before motion: {exc}")
    print(f"Sending SetVelocity vx={args.vx:+.3f}, vy={args.vy:+.3f}, vyaw={args.vyaw:+.3f}, duration={duration_s:.2f}s")

    rows: list[dict[str, Any]] = []
    status_code: int | None = None

    def log_sample(phase: str, phase_start_mono: float) -> None:
        wall = time.time()
        now_mono = time.monotonic()
        odom_pose = robot.get_odom_pose()
        sport_pose = robot.get_position()
        velocity = robot.get_velocity()
        rows.append(
            {
                "phase": phase,
                "wall_time_utc": dt.datetime.utcfromtimestamp(wall).isoformat() + "Z",
                "wall_time_s": f"{wall:.9f}",
                "monotonic_elapsed_s": f"{now_mono - phase_start_mono:.9f}",
                "odom_x": "" if odom_pose is None else f"{odom_pose[0]:.9f}",
                "odom_y": "" if odom_pose is None else f"{odom_pose[1]:.9f}",
                "odom_yaw": "" if odom_pose is None else f"{odom_pose[2]:.9f}",
                "sport_x": "" if sport_pose is None else f"{sport_pose[0]:.9f}",
                "sport_y": "" if sport_pose is None else f"{sport_pose[1]:.9f}",
                "sport_yaw": "" if sport_pose is None else f"{sport_pose[2]:.9f}",
                "vel_x": "" if velocity is None else f"{velocity[0]:.9f}",
                "vel_y": "" if velocity is None else f"{velocity[1]:.9f}",
                "vel_yaw": "" if velocity is None else f"{velocity[2]:.9f}",
            }
        )

    def run_segment(phase: str, vx: float, vy: float, vyaw: float, segment_duration_s: float) -> int:
        nonlocal status_code
        segment_duration_s = max(0.0, float(segment_duration_s))
        phase_start = time.monotonic()
        t_end = phase_start + segment_duration_s
        next_sample = phase_start
        print(
            f"[{phase}] SetVelocity vx={vx:+.3f}, vy={vy:+.3f}, "
            f"vyaw={vyaw:+.3f}, duration={segment_duration_s:.2f}s"
        )
        status_raw = send_velocity(robot, vx, vy, vyaw, segment_duration_s)
        status_code = 0 if status_raw is None else int(status_raw)
        while time.monotonic() < t_end:
            now_mono = time.monotonic()
            if now_mono < next_sample:
                time.sleep(min(0.01, next_sample - now_mono))
                continue
            next_sample += sample_dt
            log_sample(phase, phase_start)
        return status_code

    try:
        status_code = run_segment("forward", float(args.vx), float(args.vy), float(args.vyaw), duration_s)
        if not args.skip_turn:
            turn_rate = max(0.05, abs(float(args.turn_rate)))
            for turn_deg in parse_turn_degrees(str(args.turn_deg)):
                if abs(turn_deg) <= 1e-6:
                    continue
                turn_angle_rad = math.radians(float(turn_deg))
                turn_vyaw = math.copysign(turn_rate, turn_angle_rad)
                turn_duration_s = abs(turn_angle_rad) / turn_rate
                time.sleep(0.2)
                status_code = run_segment(f"turn_{turn_deg:g}deg", 0.0, 0.0, turn_vyaw, turn_duration_s)
    finally:
        robot.stop()
        time.sleep(0.2)

    end_pose = robot.get_odom_pose() or robot.get_position()
    delta = pose_delta(start_pose, end_pose)
    offset_after, delay_after, ntp_after = ntp_offset_ms(
        pc1_ip,
        samples=int(args.ntp_samples),
    )

    with open(log_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "phase",
            "wall_time_utc",
            "wall_time_s",
            "monotonic_elapsed_s",
            "odom_x",
            "odom_y",
            "odom_yaw",
            "sport_x",
            "sport_y",
            "sport_yaw",
            "vel_x",
            "vel_y",
            "vel_yaw",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Stop command sent.")
    print(f"SetVelocity status: {status_code}")
    try:
        print(f"FSM after motion: {robot.get_fsm()}")
        print(f"Mode/gait after motion: mode={robot.get_mode()} gait={robot.get_gait()}")
    except Exception as exc:
        print(f"Could not read FSM after motion: {exc}")
    print(f"End pose: {fmt_pose(end_pose)}")
    if delta is not None:
        print(f"Pose delta: dx={delta[0]:+.3f}m dy={delta[1]:+.3f}m dyaw={delta[2]:+.3f}rad")
    print(
        f"PC1/local offset after motion: {offset_after:+.3f} ms, "
        f"NTP delay {delay_after:.3f} ms, samples={len(ntp_after)}"
    )
    print(f"Offset drift during test: {offset_after - offset_before:+.3f} ms")
    print(f"Samples logged: {len(rows)}")
    print(f"CSV log: {os.path.abspath(log_path)}")

    if abs(offset_after) > float(args.max_offset_ms):
        print(f"FAIL: PC1/local offset after motion exceeds --max-offset-ms ({args.max_offset_ms:.3f} ms).")
        return 1
    print("PASS: Chrony stayed synced to PC1 during the locoClient.Move motion test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
