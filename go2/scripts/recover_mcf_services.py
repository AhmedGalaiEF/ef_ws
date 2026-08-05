from __future__ import annotations

import argparse
import math
import time
from typing import Any


LOWLEVEL_STOP_WAIT = 3.0
SERVICE_RESTART_WAIT = 2.0
SERVICES_TO_ENABLE = ("mcf", "sport_mode")
MODE_ALIASES = ("normal", "mcf")


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _nonnegative_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite value >= 0")
    return parsed


def try_stop_lowlevel(duration_sec: float) -> None:
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    if not math.isfinite(duration_sec) or duration_sec < 0.0:
        raise ValueError("low-level stop duration must be a finite value >= 0")

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x00
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    crc = CRC()
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        time.sleep(0.02)


def dump_services(robot_state: Any) -> dict[str, Any]:
    code, services = robot_state.ServiceList()
    print("ServiceList:", code)
    by_name: dict[str, Any] = {}
    if code != 0 or services is None:
        return by_name

    for service in services:
        by_name[service.name] = service
        print(
            f"  - {service.name}: status={service.status}, protect={service.protect}"
        )
    return by_name


def release_mode(motion_switcher: Any) -> None:
    code, data = motion_switcher.CheckMode()
    print("CheckMode before:", code, data)
    if code != 0:
        raise RuntimeError(f"CheckMode failed before release: {code}")

    code, _ = motion_switcher.ReleaseMode()
    print("ReleaseMode:", code)
    if code != 0:
        raise RuntimeError(f"ReleaseMode failed: {code}")
    time.sleep(1.0)

    code, data = motion_switcher.CheckMode()
    print("CheckMode after:", code, data)
    if code != 0 or (data and data.get("name")):
        raise RuntimeError(f"Motion mode was not released: code={code} data={data!r}")


def reacquire_motion_mode(motion_switcher: Any) -> None:
    code, data = motion_switcher.CheckMode()
    print("CheckMode before select:", code, data)
    if code == 0 and data is not None and data.get("name"):
        print("Motion mode already owned by:", data.get("name"))
        return

    for alias in MODE_ALIASES:
        code, _ = motion_switcher.SelectMode(alias)
        print(f"SelectMode({alias}):", code)
        if code != 0:
            continue
        time.sleep(1.0)
        check_code, check_data = motion_switcher.CheckMode()
        print(f"CheckMode after {alias}:", check_code, check_data)
        if check_code == 0 and check_data is not None and check_data.get("name"):
            return

    raise RuntimeError(f"Unable to reacquire a motion mode using {MODE_ALIASES!r}")


def set_service(robot_state: Any, service_name: str, enabled: bool) -> int:
    code = robot_state.ServiceSwitch(service_name, enabled)
    state = "on" if enabled else "off"
    print(f"ServiceSwitch({service_name}, {state}):", code)
    if code != 0:
        raise RuntimeError(f"ServiceSwitch({service_name}, {state}) failed: {code}")
    return code


def ensure_services_enabled(robot_state: Any) -> None:
    services = dump_services(robot_state)

    for service_name in SERVICES_TO_ENABLE:
        service = services.get(service_name)
        if service is not None and service.status == 1:
            print(f"{service_name} already enabled.")
            continue

        set_service(robot_state, service_name, False)
        time.sleep(0.5)
        set_service(robot_state, service_name, True)
        time.sleep(SERVICE_RESTART_WAIT)

    services = dump_services(robot_state)
    print("Recovery summary:")
    for service_name in SERVICES_TO_ENABLE:
        service = services.get(service_name)
        if service is None:
            raise RuntimeError(f"Required service {service_name!r} is not present in ServiceList")
        else:
            print(
                f"  - {service_name}: status={service.status}, protect={service.protect}"
            )
            if service.status != 1:
                raise RuntimeError(f"Required service {service_name!r} is not enabled")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recover mcf and sport_mode services after low-level control."
    )
    parser.add_argument("iface", nargs="?", default="enp1s0")
    parser.add_argument("--domain-id", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--lowlevel-seconds",
        type=_nonnegative_finite_float,
        default=LOWLEVEL_STOP_WAIT,
    )
    args = parser.parse_args(argv)

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient

    ChannelFactoryInitialize(args.domain_id, args.iface)
    print(f"Interface: {args.iface}, domain_id: {args.domain_id}")
    print(f"Neutralizing low-level control for {args.lowlevel_seconds:.1f}s...")
    try_stop_lowlevel(args.lowlevel_seconds)

    motion_switcher = MotionSwitcherClient()
    motion_switcher.SetTimeout(5.0)
    motion_switcher.Init()
    release_mode(motion_switcher)

    robot_state = RobotStateClient()
    robot_state.SetTimeout(5.0)
    robot_state.Init()
    ensure_services_enabled(robot_state)
    reacquire_motion_mode(motion_switcher)
    print("Recovery complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Recovery failed: {exc}")
        raise SystemExit(1)
