#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import time


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


def ntp_query(host: str, *, timeout_s: float = 3.0) -> dict[str, float | str]:
    msg = b"\x1b" + 47 * b"\0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(max(0.1, float(timeout_s)))
    try:
        t0 = time.time()
        sock.sendto(msg, (host, 123))
        data, _addr = sock.recvfrom(512)
        t3 = time.time()
    finally:
        sock.close()

    if len(data) < 48:
        raise RuntimeError(f"short NTP response: {len(data)} bytes")
    values = struct.unpack("!12I", data[:48])
    tx = values[10] + values[11] / 2**32 - NTP_EPOCH_DELTA
    midpoint = (t0 + t3) / 2.0
    return {
        "pc1_time_utc": _dt.datetime.utcfromtimestamp(tx).isoformat() + "Z",
        "local_midpoint_utc": _dt.datetime.utcfromtimestamp(midpoint).isoformat() + "Z",
        "roundtrip_ms": (t3 - t0) * 1000.0,
        "offset_ms": (tx - midpoint) * 1000.0,
    }


def parse_chrony_source(output: str, pc1_ip: str) -> bool:
    return pc1_ip in output


def parse_timesync_server(output: str) -> str | None:
    match = re.search(r"^\s*Server:\s+(.+?)\s*$", output, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def print_section(title: str) -> None:
    print()
    print(f"== {title} ==")


def print_cmd(label: str, args: list[str], *, timeout_s: float = 5.0) -> tuple[int, str]:
    code, output = run_cmd(args, timeout_s=timeout_s)
    print(f"$ {' '.join(args)}")
    print(output if output else "(no output)")
    print(f"[exit {code}]")
    return code, output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only diagnostic for Unitree G1 PC1 time synchronization."
    )
    parser.add_argument("--pc1-ip", default=DEFAULT_PC1_IP, help="G1 PC1 NTP address.")
    parser.add_argument("--ntp-timeout-s", type=float, default=3.0, help="NTP query timeout.")
    parser.add_argument(
        "--warn-offset-ms",
        type=float,
        default=100.0,
        help="Warn when PC1 and local system time differ by more than this many ms.",
    )
    args = parser.parse_args()
    pc1_ip = str(args.pc1_ip)

    findings: list[str] = []
    warnings: list[str] = []
    chrony_uses_pc1 = False
    chrony_leap_normal = False

    print(f"G1 PC1 time-sync diagnostic for {pc1_ip}")
    print("This script is read-only: it does not install packages, edit config, restart services, or set time.")

    print_section("Reachability")
    _ping_code, _ping_output = print_cmd("ping", ["ping", "-c", "3", pc1_ip], timeout_s=6.0)

    print_section("PC1 NTP Probe")
    try:
        ntp = ntp_query(pc1_ip, timeout_s=float(args.ntp_timeout_s))
        offset_ms = float(ntp["offset_ms"])
        print(f"pc1_ntp_tx_utc:    {ntp['pc1_time_utc']}")
        print(f"local_midpoint_utc: {ntp['local_midpoint_utc']}")
        print(f"roundtrip_ms:       {float(ntp['roundtrip_ms']):.3f}")
        print(f"estimated_offset_ms:{offset_ms:+.3f}")
        findings.append("PC1 NTP source responded on UDP/123.")
        if abs(offset_ms) > float(args.warn_offset_ms):
            warnings.append(
                f"PC1 differs from local clock by {offset_ms:+.1f} ms; syncing may step the clock."
            )
    except PermissionError as exc:
        warnings.append(f"NTP UDP probe was blocked by permissions: {exc}")
        print(f"ERROR: {exc}")
        print("Try running this script outside a sandbox or with sufficient permissions.")
    except Exception as exc:
        warnings.append(f"PC1 NTP probe failed: {exc}")
        print(f"ERROR: {exc}")

    print_section("Installed Tools")
    chronyc = shutil.which("chronyc")
    chronyd = shutil.which("chronyd")
    timedatectl = shutil.which("timedatectl")
    print(f"chronyc:     {chronyc or 'not found'}")
    print(f"chronyd:     {chronyd or 'not found'}")
    print(f"timedatectl: {timedatectl or 'not found'}")

    print_section("Chrony")
    if chronyc:
        _sources_code, sources = print_cmd("chronyc sources", ["chronyc", "sources", "-v"])
        _tracking_code, tracking = print_cmd("chronyc tracking", ["chronyc", "tracking"])
        if parse_chrony_source(sources, pc1_ip):
            chrony_uses_pc1 = True
            findings.append("Chrony sources include G1 PC1.")
        else:
            warnings.append("Chrony is installed but its sources do not show G1 PC1.")
        if "Leap status" in tracking and "Normal" in tracking:
            chrony_leap_normal = True
            findings.append("Chrony reports normal leap status.")
    else:
        warnings.append("Chrony is not installed or chronyc is not in PATH.")
        print("chronyc not available; skipping Chrony source/tracking checks.")

    print_section("systemd-timesyncd")
    if timedatectl:
        _td_code, td_status = print_cmd("timedatectl status", ["timedatectl", "status"])
        _ts_code, ts_status = print_cmd("timedatectl timesync-status", ["timedatectl", "timesync-status"])
        server = parse_timesync_server(ts_status)
        if server:
            print(f"Parsed timesyncd server: {server}")
            if pc1_ip in server:
                findings.append("systemd-timesyncd is using G1 PC1.")
            else:
                warnings.append(f"systemd-timesyncd is using {server}, not G1 PC1.")
        if "System clock synchronized: yes" in td_status:
            findings.append("System clock is synchronized according to timedatectl.")
    else:
        print("timedatectl not available; skipping systemd-timesyncd checks.")

    print_section("Config Files")
    for path in ("/etc/chrony/chrony.conf", "/etc/default/chrony", "/etc/systemd/timesyncd.conf"):
        if os.path.exists(path):
            print(f"{path}: exists")
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
                if pc1_ip in text:
                    findings.append(f"{path} references G1 PC1.")
                elif path.endswith("chrony.conf"):
                    warnings.append(f"{path} does not reference G1 PC1.")
                elif path.endswith("timesyncd.conf") and not (chrony_uses_pc1 and chrony_leap_normal):
                    warnings.append(f"{path} does not reference G1 PC1.")
            except OSError as exc:
                warnings.append(f"Could not read {path}: {exc}")
        else:
            print(f"{path}: missing")

    print_section("Summary")
    if findings:
        print("Findings:")
        for item in findings:
            print(f"  - {item}")
    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")
    if not warnings:
        print("No warnings detected.")

    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
