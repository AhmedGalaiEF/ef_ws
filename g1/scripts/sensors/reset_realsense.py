#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import Optional

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError as exc:
    raise SystemExit(
        "pyrealsense2 is not installed. Install it in the active Python environment first."
    ) from exc


def _device_info(device: rs.device) -> str:
    name = device.get_info(rs.camera_info.name)
    serial = device.get_info(rs.camera_info.serial_number)
    firmware = device.get_info(rs.camera_info.firmware_version)
    parts = [f"{name} serial={serial}", f"firmware={firmware}"]
    if device.supports(rs.camera_info.usb_type_descriptor):
        parts.append(f"usb={device.get_info(rs.camera_info.usb_type_descriptor)}")
    return " ".join(parts)


def _find_device(serial: Optional[str]) -> Optional[rs.device]:
    context = rs.context()
    for device in context.query_devices():
        if serial is None or device.get_info(rs.camera_info.serial_number) == serial:
            return device
    return None


def _run_systemctl(action: str, service: str) -> None:
    cmd = ["systemctl", action, service]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _wait_for_device(serial: Optional[str], timeout: float) -> rs.device:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        device = _find_device(serial)
        if device is not None:
            return device
        time.sleep(0.5)
    label = serial if serial is not None else "any RealSense device"
    raise TimeoutError(f"Timed out waiting for {label} to reconnect after {timeout:.1f}s.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hardware-reset an Intel RealSense camera and wait for reconnect.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--serial", help="Reset only the camera with this serial number")
    parser.add_argument("--wait-seconds", type=float, default=12.0, help="Reconnect wait timeout")
    parser.add_argument(
        "--service",
        help="Systemd service to stop before reset and start after reset, e.g. real-sense.service",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="When --service is set, stop it before reset but do not start it afterwards",
    )
    args = parser.parse_args()

    service_was_stopped = False
    try:
        if args.service:
            _run_systemctl("stop", args.service)
            service_was_stopped = True
            time.sleep(1.0)

        device = _find_device(args.serial)
        if device is None:
            label = args.serial if args.serial else "any RealSense device"
            print(f"No {label} found.", file=sys.stderr)
            return 1

        serial = device.get_info(rs.camera_info.serial_number)
        print(f"Found: {_device_info(device)}")
        print("Issuing hardware reset...")
        device.hardware_reset()

        print("Waiting for reconnect...")
        reconnected = _wait_for_device(serial, args.wait_seconds)
        print(f"Reconnected: {_device_info(reconnected)}")
        return 0
    except subprocess.CalledProcessError as exc:
        print(f"systemctl failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    except RuntimeError as exc:
        print(f"RealSense reset failed: {exc}", file=sys.stderr)
        return 1
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if args.service and service_was_stopped and not args.no_start:
            try:
                _run_systemctl("start", args.service)
            except subprocess.CalledProcessError as exc:
                print(f"Failed to start {args.service}: exit code {exc.returncode}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
