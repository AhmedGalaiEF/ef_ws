#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

SLAM_VIEWER_SCRIPT = Path(SCRIPT_DIR).resolve() / "slam_points_viewer.py"

from sdk_client import Robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive SLAM mapping and navigation demo for the G1 robot."
    )
    parser.add_argument("--iface", default="eth0", help="Network interface for the robot SDK.")
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--no-safety-boot",
        action="store_true",
        help="Skip the robot safety boot sequence during initialization.",
    )
    parser.add_argument(
        "--slam-type",
        default="indoor",
        help="SLAM type passed to start_mapping().",
    )
    parser.add_argument(
        "--default-move-seconds",
        type=float,
        default=1.5,
        help="Default duration for manual timed motion commands.",
    )
    return parser.parse_args()


def print_section(title: str, payload: object) -> None:
    print(f"\n=== {title} ===")
    if isinstance(payload, (dict, list, tuple)):
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(payload)


def print_current_pose(robot: Robot, title: str = "Current SLAM Pose") -> None:
    status = robot.get_slam_pose_status(timeout_s=0.4)
    print_section(
        title,
        {
            "pose": status.get("pose"),
            "usable": status.get("usable"),
            "reason": status.get("reason"),
            "sport_pose": status.get("sport_pose"),
            "sport_vs_slam_xy_gap_m": status.get("sport_vs_slam_xy_gap_m"),
        },
    )


def _viewer_terminal_command(iface: str, domain_id: int) -> list[str] | None:
    script_dir = str(SLAM_VIEWER_SCRIPT.parent)
    python_exe = shlex.quote(sys.executable)
    script_path = shlex.quote(str(SLAM_VIEWER_SCRIPT))
    shell_cmd = (
        f"cd {shlex.quote(script_dir)} && exec {python_exe} {script_path} "
        f"--iface {shlex.quote(str(iface))} "
        f"--domain-id {int(domain_id)}"
    )

    terminal_candidates: list[list[str]] = [
        ["x-terminal-emulator", "-e", "bash", "-lc", shell_cmd],
        ["gnome-terminal", "--", "bash", "-lc", shell_cmd],
        ["konsole", "-e", "bash", "-lc", shell_cmd],
    ]
    for cmd in terminal_candidates:
        if shutil.which(cmd[0]):
            return cmd
    return None


def launch_live_slam_viewer(*, iface: str, domain_id: int) -> None:
    if not SLAM_VIEWER_SCRIPT.is_file():
        print(f"SLAM viewer script not found: {SLAM_VIEWER_SCRIPT}")
        return
    cmd = _viewer_terminal_command(iface=iface, domain_id=domain_id)
    if cmd is None:
        print("No supported terminal emulator found for launching the live SLAM viewer.")
        return
    print(f"Launching SLAM DDS viewer in a new terminal: {SLAM_VIEWER_SCRIPT}")
    try:
        subprocess.Popen(
            cmd,
            cwd=str(SLAM_VIEWER_SCRIPT.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"Failed to launch live SLAM viewer: {exc}")


def prompt_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else (default or "")


def prompt_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    default_text = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{default_text}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


def show_status(robot: Robot, slam_type: str, named_points: dict[str, tuple[float, float, float]]) -> None:
    slam_status = robot.get_slam_pose_status(timeout_s=0.6)
    print_section(
        "Status",
        {
            "slam_type": slam_type,
            "slam_is_running": robot.slam_is_running,
            "slam_pose": slam_status.get("pose"),
            "slam_pose_usable": slam_status.get("usable"),
            "slam_pose_reason": slam_status.get("reason"),
            "sport_pose": slam_status.get("sport_pose"),
            "sport_vs_slam_xy_gap_m": slam_status.get("sport_vs_slam_xy_gap_m"),
            "queued_path_points": robot.get_path_points(),
            "named_map_points": named_points,
        },
    )


def run_timed_move(
    robot: Robot,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    vyaw: float = 0.0,
    duration_s: float = 1.0,
) -> None:
    duration_s = max(0.0, float(duration_s))
    print_section(
        "Manual Motion",
        {
            "vx_mps": vx,
            "vy_mps": vy,
            "vyaw_radps": vyaw,
            "duration_s": duration_s,
        },
    )
    robot.walk(vx=vx, vy=vy, vyaw=vyaw)
    time.sleep(duration_s)
    robot.stop()
    time.sleep(0.5)


def manual_motion_menu(robot: Robot, default_move_seconds: float) -> None:
    while True:
        print_current_pose(robot)
        print(
            "\nManual motion options:\n"
            "  1. Forward\n"
            "  2. Backward\n"
            "  3. Strafe left\n"
            "  4. Strafe right\n"
            "  5. Turn left\n"
            "  6. Turn right\n"
            "  7. Custom velocity\n"
            "  8. Stop\n"
            "  9. Back"
        )
        choice = input("Select motion command: ").strip()

        if choice == "1":
            run_timed_move(robot, vx=0.20, duration_s=prompt_float("Move time in seconds", default_move_seconds))
        elif choice == "2":
            run_timed_move(robot, vx=-0.20, duration_s=prompt_float("Move time in seconds", default_move_seconds))
        elif choice == "3":
            run_timed_move(robot, vy=0.12, duration_s=prompt_float("Move time in seconds", default_move_seconds))
        elif choice == "4":
            run_timed_move(robot, vy=-0.12, duration_s=prompt_float("Move time in seconds", default_move_seconds))
        elif choice == "5":
            run_timed_move(robot, vyaw=0.50, duration_s=prompt_float("Turn time in seconds", default_move_seconds))
        elif choice == "6":
            run_timed_move(robot, vyaw=-0.50, duration_s=prompt_float("Turn time in seconds", default_move_seconds))
        elif choice == "7":
            vx = prompt_float("vx (m/s)", 0.0)
            vy = prompt_float("vy (m/s)", 0.0)
            vyaw = prompt_float("vyaw (rad/s)", 0.0)
            duration_s = prompt_float("Duration (s)", default_move_seconds)
            run_timed_move(robot, vx=vx, vy=vy, vyaw=vyaw, duration_s=duration_s)
        elif choice == "8":
            robot.stop()
            print("Stop command sent.")
        elif choice == "9":
            return
        else:
            print("Unknown option.")


def capture_named_point(
    robot: Robot,
    named_points: dict[str, tuple[float, float, float]],
) -> None:
    slam_status = robot.get_slam_pose_status(timeout_s=1.0)
    pose = slam_status.get("pose")
    if not bool(slam_status.get("usable")) or pose is None:
        print(
            "No usable SLAM pose available. "
            f"reason={slam_status.get('reason')} pose={pose} sport_pose={slam_status.get('sport_pose')}"
        )
        return

    name = prompt_text("Point name")
    if not name:
        print("Point name cannot be empty.")
        return

    named_points[name] = pose
    print_section("Saved Map Point", {"name": name, "pose": pose})


def queue_manual_path_point(robot: Robot) -> None:
    x = prompt_float("Target x", 0.0)
    y = prompt_float("Target y", 0.0)
    yaw = prompt_float("Target yaw (rad)", 0.0)
    robot.set_path_point(x, y, yaw)
    print_section("Queued Path Point", {"x": x, "y": y, "yaw": yaw})


def queue_named_point(
    robot: Robot,
    named_points: dict[str, tuple[float, float, float]],
) -> None:
    if not named_points:
        print("No named map points saved yet.")
        return

    print_section("Named Map Points", named_points)
    name = prompt_text("Name to queue")
    pose = named_points.get(name)
    if pose is None:
        print(f"No map point named '{name}'.")
        return

    robot.set_path_point(*pose)
    print_section("Queued Named Point", {"name": name, "pose": pose})


def navigate_to_named_point(
    robot: Robot,
    named_points: dict[str, tuple[float, float, float]],
) -> None:
    if not named_points:
        print("No named map points saved yet.")
        return

    print_section("Named Map Points", named_points)
    name = prompt_text("Name to navigate to")
    pose = named_points.get(name)
    if pose is None:
        print(f"No map point named '{name}'.")
        return

    robot.clear_path_points()
    robot.set_path_point(*pose)
    ok = robot.navigate_path(clear_on_finish=True)
    print_section("Navigate To Named Point", {"name": name, "pose": pose, "completed": ok})


def navigate_queued_path(robot: Robot) -> None:
    points = robot.get_path_points()
    if not points:
        print("No queued path points. Add one or more points first.")
        return

    print_section("Queued Path", points)
    ok = robot.navigate_path(clear_on_finish=True)
    print_section("Navigate Path", {"completed": ok})


def print_menu() -> None:
    print(
        "\nInteractive SLAM menu:\n"
        "  1. Show status\n"
        "  2. Start SLAM mapping\n"
        "  3. Stop SLAM mapping\n"
        "  4. Show current SLAM pose\n"
        "  5. Move robot manually\n"
        "  6. Save current pose as named map point\n"
        "  7. List named map points\n"
        "  8. Queue manual path point\n"
        "  9. Queue named map point\n"
        " 10. Show queued path points\n"
        " 11. Clear queued path points\n"
        " 12. Navigate queued path\n"
        " 13. Navigate directly to a named map point\n"
        " 14. Stop robot motion\n"
        " 15. Launch live SLAM viewer\n"
        " 16. Exit"
    )


def main() -> int:
    args = parse_args()

    try:
        robot = Robot(
            iface=args.iface,
            domain_id=args.domain_id,
            safety_boot=not args.no_safety_boot,
            auto_start_sensors=True,
        )
    except Exception as exc:
        print(f"Failed to connect to robot: {exc}")
        return 1

    named_points: dict[str, tuple[float, float, float]] = {}
    slam_type = args.slam_type

    print_section(
        "SLAM Control",
        "Use the menu to start mapping, drive the robot, save poses, queue path points, and navigate.",
    )

    try:
        while True:
            print_current_pose(robot)
            print_menu()
            choice = input("Select option: ").strip()

            if choice == "1":
                show_status(robot, slam_type, named_points)
            elif choice == "2":
                slam_type = prompt_text("SLAM type", slam_type) or slam_type
                code = robot.start_slam(slam_type=slam_type)
                print_section("Start SLAM", {"slam_type": slam_type, "code": code})
            elif choice == "3":
                save_path = prompt_text("Optional map save path", "")
                code = robot.stop_slam(save_path=save_path or None)
                print_section("Stop SLAM", {"save_path": save_path or None, "code": code})
            elif choice == "4":
                print_section("Current SLAM Pose", robot.get_slam_pose(timeout_s=1.0))
            elif choice == "5":
                manual_motion_menu(robot, default_move_seconds=args.default_move_seconds)
            elif choice == "6":
                capture_named_point(robot, named_points)
            elif choice == "7":
                print_section("Named Map Points", named_points or "No named map points saved yet.")
            elif choice == "8":
                queue_manual_path_point(robot)
            elif choice == "9":
                queue_named_point(robot, named_points)
            elif choice == "10":
                print_section("Queued Path Points", robot.get_path_points())
            elif choice == "11":
                if prompt_yes_no("Clear all queued path points?", default=True):
                    robot.clear_path_points()
                    print("Queued path points cleared.")
            elif choice == "12":
                navigate_queued_path(robot)
            elif choice == "13":
                navigate_to_named_point(robot, named_points)
            elif choice == "14":
                robot.stop()
                print("Stop command sent.")
            elif choice == "15":
                launch_live_slam_viewer(iface=args.iface, domain_id=args.domain_id)
            elif choice == "16":
                if robot.slam_is_running and prompt_yes_no("SLAM is still running. Stop it before exit?", default=True):
                    save_path = prompt_text("Optional map save path", "")
                    code = robot.stop_slam(save_path=save_path or None)
                    print_section("Stop SLAM", {"save_path": save_path or None, "code": code})
                robot.stop()
                break
            else:
                print("Unknown option.")
    except KeyboardInterrupt:
        robot.stop()
        print("\nInterrupted. Stop command sent.")
        return 1
    except Exception as exc:
        robot.stop()
        print(f"Interactive SLAM session failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
