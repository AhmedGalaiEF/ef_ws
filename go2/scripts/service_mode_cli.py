import argparse
import sys
import time

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient


DEFAULT_TOGGLE_DELAY = 0.5
DEFAULT_RESTART_WAIT = 2.0


def print_jsonish(label: str, value):
    print(f"{label}: {value}")


def make_clients():
    motion_switcher = MotionSwitcherClient()
    motion_switcher.SetTimeout(5.0)
    motion_switcher.Init()

    robot_state = RobotStateClient()
    robot_state.SetTimeout(5.0)
    robot_state.Init()
    return motion_switcher, robot_state


def service_list(robot_state: RobotStateClient):
    code, services = robot_state.ServiceList()
    print(f"ServiceList: {code}")
    if code != 0 or services is None:
        return code, {}

    by_name = {}
    for service in services:
        by_name[service.name] = service
    for name in sorted(by_name):
        service = by_name[name]
        print(f"  - {service.name}: status={service.status}, protect={service.protect}")
    return code, by_name


def service_switch(robot_state: RobotStateClient, service_name: str, enabled: bool):
    code = robot_state.ServiceSwitch(service_name, enabled)
    state = "on" if enabled else "off"
    print(f"ServiceSwitch({service_name}, {state}): {code}")
    return code


def service_toggle(robot_state: RobotStateClient, service_name: str, delay_sec: float):
    code, services = service_list(robot_state)
    if code != 0:
        return code
    service = services.get(service_name)
    if service is None:
        print(f"Service not found: {service_name}")
        return 1

    target = service.status != 1
    return service_switch(robot_state, service_name, target)


def service_restart(robot_state: RobotStateClient, service_name: str, toggle_delay: float, restart_wait: float):
    off_code = service_switch(robot_state, service_name, False)
    time.sleep(toggle_delay)
    on_code = service_switch(robot_state, service_name, True)
    time.sleep(restart_wait)
    return off_code if off_code != 0 else on_code


def mode_check(motion_switcher: MotionSwitcherClient):
    code, data = motion_switcher.CheckMode()
    print_jsonish("CheckMode", {"code": code, "data": data})
    return code, data


def mode_release(motion_switcher: MotionSwitcherClient):
    code, data = motion_switcher.ReleaseMode()
    print_jsonish("ReleaseMode", {"code": code, "data": data})
    return code


def mode_select(motion_switcher: MotionSwitcherClient, name: str, settle_sec: float):
    code, data = motion_switcher.SelectMode(name)
    print_jsonish(f"SelectMode({name})", {"code": code, "data": data})
    if settle_sec > 0.0:
        time.sleep(settle_sec)
        mode_check(motion_switcher)
    return code


def cmd_list(args):
    motion_switcher, robot_state = make_clients()
    mode_check(motion_switcher)
    service_list(robot_state)
    return 0


def cmd_service(args):
    _, robot_state = make_clients()

    if args.action == "list":
        code, _ = service_list(robot_state)
        return 0 if code == 0 else code
    if args.action == "on":
        return service_switch(robot_state, args.name, True)
    if args.action == "off":
        return service_switch(robot_state, args.name, False)
    if args.action == "toggle":
        return service_toggle(robot_state, args.name, args.toggle_delay)
    if args.action == "restart":
        return service_restart(robot_state, args.name, args.toggle_delay, args.restart_wait)

    print(f"Unknown service action: {args.action}")
    return 2


def cmd_mode(args):
    motion_switcher, _ = make_clients()

    if args.action == "check":
        code, _ = mode_check(motion_switcher)
        return 0 if code == 0 else code
    if args.action == "release":
        return mode_release(motion_switcher)
    if args.action == "select":
        return mode_select(motion_switcher, args.name, args.settle)

    print(f"Unknown mode action: {args.action}")
    return 2


def cmd_watch(args):
    motion_switcher, robot_state = make_clients()
    try:
        while True:
            print("")
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            mode_check(motion_switcher)
            service_list(robot_state)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inspect and control Unitree services and motion modes."
    )
    parser.add_argument("iface", nargs="?", default="enp1s0", help="Robot network interface")
    parser.add_argument("--domain-id", type=int, default=0)

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser("list", help="Show current mode and all services")
    p_list.set_defaults(func=cmd_list)

    p_service = subparsers.add_parser("service", help="Control a service")
    p_service.add_argument("action", choices=["list", "on", "off", "toggle", "restart"])
    p_service.add_argument("name", nargs="?", help="Service name")
    p_service.add_argument("--toggle-delay", type=float, default=DEFAULT_TOGGLE_DELAY)
    p_service.add_argument("--restart-wait", type=float, default=DEFAULT_RESTART_WAIT)
    p_service.set_defaults(func=cmd_service)

    p_mode = subparsers.add_parser("mode", help="Control motion mode ownership")
    p_mode.add_argument("action", choices=["check", "release", "select"])
    p_mode.add_argument("name", nargs="?", help="Mode name for select")
    p_mode.add_argument("--settle", type=float, default=1.0, help="Seconds to wait before re-checking mode")
    p_mode.set_defaults(func=cmd_mode)

    p_watch = subparsers.add_parser("watch", help="Continuously print mode and service status")
    p_watch.add_argument("--interval", type=float, default=1.0)
    p_watch.set_defaults(func=cmd_watch)

    return parser


def validate_args(args, parser):
    if args.command == "service" and args.action != "list" and not args.name:
        parser.error("service name is required for this action")
    if args.command == "mode" and args.action == "select" and not args.name:
        parser.error("mode name is required for select")


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    ChannelFactoryInitialize(args.domain_id, args.iface)
    rc = args.func(args)
    raise SystemExit(0 if rc is None else rc)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
