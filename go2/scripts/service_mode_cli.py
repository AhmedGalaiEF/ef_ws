import argparse
import math
import sys
import time

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_


DEFAULT_TOGGLE_DELAY = 0.5
DEFAULT_RESTART_WAIT = 2.0
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_ODOM = "rt/odom"


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite value >= 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = _nonnegative_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def print_jsonish(label: str, value):
    print(f"{label}: {value}")


def age_text(ts: float) -> str:
    if ts <= 0.0:
        return "--"
    return f"{max(0.0, time.time() - ts):4.1f}s"


class StateMonitor:
    def __init__(self):
        self.low_state = None
        self.low_state_time = 0.0
        self.odom = None
        self.odom_time = 0.0

    def init(self):
        self.lowstate_sub = ChannelSubscriber(TOPIC_LOWSTATE, LowState_)
        self.lowstate_sub.Init(self._low_state_handler, 10)
        self.odom_sub = ChannelSubscriber(TOPIC_ODOM, Odometry_)
        self.odom_sub.Init(self._odom_handler, 10)

    def _low_state_handler(self, msg: LowState_):
        self.low_state = msg
        self.low_state_time = time.time()

    def _odom_handler(self, msg: Odometry_):
        self.odom = msg
        self.odom_time = time.time()

    def print_state(self):
        print(f"LowState age: {age_text(self.low_state_time)}")
        if self.low_state is not None:
            imu = self.low_state.imu_state
            print(
                "  power_v/power_a:",
                f"{float(self.low_state.power_v):.2f}",
                f"{float(self.low_state.power_a):.2f}",
            )
            print(
                "  imu rpy:",
                f"{float(imu.rpy[0]):+.3f}",
                f"{float(imu.rpy[1]):+.3f}",
                f"{float(imu.rpy[2]):+.3f}",
            )
            print("  foot_force:", " ".join(str(int(v)) for v in self.low_state.foot_force))
        print(f"Odom age: {age_text(self.odom_time)}")
        if self.odom is not None:
            pos = self.odom.pose.pose.position
            print("  pos xyz:", f"{pos.x:+.3f}", f"{pos.y:+.3f}", f"{pos.z:+.3f}")


def make_clients():
    motion_switcher = MotionSwitcherClient()
    motion_switcher.SetTimeout(5.0)
    motion_switcher.Init()

    robot_state = RobotStateClient()
    robot_state.SetTimeout(5.0)
    robot_state.Init()

    obstacles_avoid = ObstaclesAvoidClient()
    obstacles_avoid.SetTimeout(5.0)
    obstacles_avoid.Init()
    return motion_switcher, robot_state, obstacles_avoid


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


def obstacles_status(obstacles_avoid: ObstaclesAvoidClient):
    code, enabled = obstacles_avoid.SwitchGet()
    print_jsonish("ObstaclesAvoid.SwitchGet", {"code": code, "enabled": enabled})
    return code


def obstacles_switch(obstacles_avoid: ObstaclesAvoidClient, enabled: bool):
    code = obstacles_avoid.SwitchSet(enabled)
    state = "on" if enabled else "off"
    print_jsonish(f"ObstaclesAvoid.SwitchSet({state})", {"code": code})
    return code


def obstacles_remote(obstacles_avoid: ObstaclesAvoidClient, enabled: bool):
    code = obstacles_avoid.UseRemoteCommandFromApi(enabled)
    state = "api" if enabled else "remote"
    print_jsonish(f"ObstaclesAvoid.UseRemoteCommandFromApi({state})", {"code": code})
    return code


def cmd_list(args):
    motion_switcher, robot_state, obstacles_avoid = make_clients()
    monitor = StateMonitor()
    monitor.init()
    time.sleep(args.settle)
    mode_check(motion_switcher)
    service_list(robot_state)
    obstacles_status(obstacles_avoid)
    monitor.print_state()
    return 0


def cmd_service(args):
    _, robot_state, _ = make_clients()

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
    motion_switcher, _, _ = make_clients()

    if args.action == "check":
        code, _ = mode_check(motion_switcher)
        return 0 if code == 0 else code
    if args.action == "release":
        return mode_release(motion_switcher)
    if args.action == "select":
        return mode_select(motion_switcher, args.name, args.settle)

    print(f"Unknown mode action: {args.action}")
    return 2


def cmd_avoid(args):
    _, _, obstacles_avoid = make_clients()

    if args.action == "status":
        return obstacles_status(obstacles_avoid)
    if args.action == "on":
        return obstacles_switch(obstacles_avoid, True)
    if args.action == "off":
        return obstacles_switch(obstacles_avoid, False)
    if args.action == "api":
        return obstacles_remote(obstacles_avoid, True)
    if args.action == "remote":
        return obstacles_remote(obstacles_avoid, False)

    print(f"Unknown avoid action: {args.action}")
    return 2


def cmd_watch(args):
    motion_switcher, robot_state, obstacles_avoid = make_clients()
    monitor = StateMonitor()
    monitor.init()
    try:
        while True:
            print("")
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            mode_check(motion_switcher)
            service_list(robot_state)
            obstacles_status(obstacles_avoid)
            monitor.print_state()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Inspect and control Unitree services and motion modes."
    )
    parser.add_argument("iface", nargs="?", default="enp1s0", help="Robot network interface")
    parser.add_argument("--domain-id", type=_nonnegative_int, default=0)
    parser.add_argument("--settle", type=_nonnegative_float, default=0.5, help="Seconds to wait for DDS state before printing")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser("list", help="Show current mode and all services")
    p_list.set_defaults(func=cmd_list)

    p_service = subparsers.add_parser("service", help="Control a service")
    p_service.add_argument("action", choices=["list", "on", "off", "toggle", "restart"])
    p_service.add_argument("name", nargs="?", help="Service name")
    p_service.add_argument("--toggle-delay", type=_nonnegative_float, default=DEFAULT_TOGGLE_DELAY)
    p_service.add_argument("--restart-wait", type=_nonnegative_float, default=DEFAULT_RESTART_WAIT)
    p_service.set_defaults(func=cmd_service)

    p_mode = subparsers.add_parser("mode", help="Control motion mode ownership")
    p_mode.add_argument("action", choices=["check", "release", "select"])
    p_mode.add_argument("name", nargs="?", help="Mode name for select")
    p_mode.add_argument("--settle", type=_nonnegative_float, default=1.0, help="Seconds to wait before re-checking mode")
    p_mode.set_defaults(func=cmd_mode)

    p_avoid = subparsers.add_parser("avoid", help="Control obstacle avoidance")
    p_avoid.add_argument("action", choices=["status", "on", "off", "api", "remote"])
    p_avoid.set_defaults(func=cmd_avoid)

    p_watch = subparsers.add_parser("watch", help="Continuously print mode and service status")
    p_watch.add_argument("--interval", type=_positive_float, default=1.0)
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
