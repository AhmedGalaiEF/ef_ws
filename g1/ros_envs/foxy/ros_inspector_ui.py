#!/usr/bin/env python3
"""Interactive ROS 2 topic inspector - Textual TUI.

Sidebar lists topics grouped by category; selecting one streams its live
`ros2 topic echo` output into the main panel. Requires the `textual`
package (`pip install textual`) and a sourced ROS 2 environment on PATH.
If `ros2` isn't reachable, the sidebar falls back to the last known
topic list captured in summary.md so the layout can still be browsed.
"""

import asyncio
import re
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Header, Footer, Tree, Static, RichLog
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This tool requires the 'textual' package. Install it with:\n"
        "    pip install textual\n"
        f"(import failed: {exc})"
    )

# -- Fallback topic set, used only when `ros2 topic list` is unreachable -------

FALLBACK_TOPICS = {
    "Robot State / Cmd": [
        "/lowcmd", "/lowstate", "/user_lowcmd",
        "/multiplestate", "/sportmodestate", "/odommodestate",
        "/lowstate_doubleimu", "/wirelesscontroller", "/servicestate",
        "/servicestateactivate", "/selftest", "/config_change_status",
        "/public_network_status", "/SymState",
        "/lf/lowstate", "/lf/sportmodestate", "/lf/odommodestate",
        "/lf/agvalarmstate", "/lf/agvbmsstate", "/lf/battery_alarm",
        "/lf/bmsstate", "/lf/emergency_stop", "/lf/mainboardstate",
    ],
    "API Request/Response": [
        "/api/action_store/request", "/api/action_store/response",
        "/api/sport/request",      "/api/sport/response",
        "/api/arm/request",        "/api/arm/response",
        "/api/robot_state/request","/api/robot_state/response",
        "/api/motion_switcher/request", "/api/motion_switcher/response",
        "/api/slam_operate/request",    "/api/slam_operate/response",
        "/api/audiohub/request", "/api/audiohub/response",
        "/api/bashrunner/request", "/api/bashrunner/response",
        "/api/basic_clearoip/request", "/api/basic_clearoip/response",
        "/api/basic_clearoip_lease/request", "/api/basic_clearoip_lease/response",
        "/api/basic_demarcate/request", "/api/basic_demarcate/response",
        "/api/basic_demarcate_lease/request", "/api/basic_demarcate_lease/response",
        "/api/basic_softlimit/request", "/api/basic_softlimit/response",
        "/api/basic_softlimit_lease/request", "/api/basic_softlimit_lease/response",
        "/api/basic_taumax/request", "/api/basic_taumax/response",
        "/api/basic_taumax_lease/request", "/api/basic_taumax_lease/response",
        "/api/config/request", "/api/config/response",
        "/api/dex3_msg_controller/request", "/api/dex3_msg_controller/response",
        "/api/gesture/request", "/api/gpt/request", "/api/gpt/response",
        "/api/rm_con/request", "/api/robot_type_service/request",
        "/api/robot_type_service/response", "/api/videohub/request",
        "/api/videohub/response", "/api/voice/request", "/api/voice/response",
        "/api/vui/request", "/api/vui/response",
    ],
    "SLAM / Mapping": [
        "/unitree/slam_mapping/odom",    "/unitree/slam_mapping/points",
        "/unitree/slam_relocation/odom", "/unitree/slam_relocation/points",
        "/unitree/slam_relocation/global_map", "/unitree/slam_relocation/web_points",
        "/unitree_slam/waypoints",       "/slam_info",
        "/slam_key_info",                "/global_map",
        "/planner_map",                  "/gridmap",
        "/collision_clouds", "/ele_clouds", "/grid_clouds",
        "/no_warning_clouds", "/pre_collision_clouds", "/pre_safe_clouds",
        "/safe_clouds", "/warning_clouds",
    ],
    "LiDAR / IMU": [
        "/utlidar/cloud_livox_mid360",
        "/utlidar/imu_livox_mid360",
        "/utlidar/map_state",
        "/utlidar/range_info",
        "/dog_imu_raw",
        "/dog_odom",
        "/secondary_imu",
        "/lf/secondary_imu",
    ],
    "Audio / Voice / AI": [
        "/audio_msg", "/audio_msg/filter", "/audiosender",
        "/gpt_cmd", "/gpt_state", "/gptflowfeedback",
        "/gesture/result",
    ],
    "Camera / WebRTC": [
        "/frontvideostream",
        "/videohub/inner",
        "/webrtcreq",
        "/webrtcres",
        "/xfk_webrtcreq",
        "/xfk_webrtcres",
        "/rtc/state",
        "/rtc_status",
    ],
    "Arm / Dexterous Hand": [
        "/arm/action/state", "/arm_sdk", "/armsdk", "/loco_sdk",
        "/dex3/left/cmd",   "/dex3/right/cmd",
        "/dex3/left/state", "/dex3/right/state",
        "/lf/dex3/left/state", "/lf/dex3/right/state",
    ],
    "ROS System": [
        "/parameter_events", "/rosout", "/event/action_store",
        "/log_system_inbound", "/log_system_outbound",
    ],
    "Other": [],
}

CATEGORY_COLORS = {
    "Robot State / Cmd":     "cyan",
    "API Request/Response":  "yellow",
    "SLAM / Mapping":        "magenta",
    "LiDAR / IMU":           "blue",
    "Audio / Voice / AI":    "green",
    "Camera / WebRTC":       "green",
    "Arm / Dexterous Hand":  "red",
    "ROS System":            "bright_black",
    "Other":                 "white",
}

CATEGORY_RULES: List[Tuple[str, Tuple[str, ...]]] = [
    ("API Request/Response", ("/api/",)),
    ("SLAM / Mapping", ("/unitree/slam_", "/unitree_slam/", "/slam_", "/global_map",
                        "/planner_map", "/gridmap", "/collision_clouds", "/ele_clouds",
                        "/grid_clouds", "/safe_clouds", "/warning_clouds",
                        "/pre_collision_clouds", "/pre_safe_clouds", "/no_warning_clouds")),
    ("LiDAR / IMU", ("/utlidar/", "/dog_imu", "/dog_odom", "/secondary_imu", "/lf/secondary_imu")),
    ("Audio / Voice / AI", ("/audio", "/audiosender", "/api/audio", "/api/voice",
                            "/api/vui", "/api/gpt", "/gpt", "/gesture")),
    ("Camera / WebRTC", ("/frontvideo", "/videohub", "/api/videohub", "/webrtc",
                         "/xfk_webrtc", "/rtc")),
    ("Arm / Dexterous Hand", ("/dex3/", "/lf/dex3/", "/arm", "/armsdk", "/loco_sdk")),
    ("ROS System", ("/parameter_events", "/rosout", "/event/", "/log_system_")),
    ("Robot State / Cmd", ("/low", "/user_lowcmd", "/multiplestate", "/sportmodestate",
                           "/odommodestate", "/wirelesscontroller", "/servicestate",
                           "/selftest", "/config_change_status", "/public_network_status",
                           "/SymState", "/lf/")),
]


def short_type(msg_type: str) -> str:
    """Keep sidebar type labels readable."""
    if not msg_type:
        return ""
    return msg_type.rsplit("/", 1)[-1]


def categorize_topic(topic: str) -> str:
    for category, prefixes in CATEGORY_RULES:
        if any(topic.startswith(prefix) for prefix in prefixes):
            return category
    return "Other"


def ordered_categories(categorized: Dict[str, List[str]]) -> Iterable[Tuple[str, List[str]]]:
    for category in FALLBACK_TOPICS:
        topics = categorized.get(category, [])
        if topics:
            yield category, sorted(topics)
    for category in sorted(set(categorized) - set(FALLBACK_TOPICS)):
        topics = categorized[category]
        if topics:
            yield category, sorted(topics)


def parse_topic_list_with_types(output: str) -> Tuple[List[str], Dict[str, str]]:
    topics: List[str] = []
    types: Dict[str, str] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(?P<topic>/\S+)(?:\s+\[(?P<type>[^\]]+)\])?$", line)
        if not match:
            continue
        topic = match.group("topic")
        topics.append(topic)
        msg_type = match.group("type")
        if msg_type:
            types[topic] = msg_type
    return topics, types


def discover_live_topics() -> Optional[Tuple[Dict[str, List[str]], Dict[str, str]]]:
    """Return categorized live topics plus known message types, or None if unreachable."""
    try:
        result = subprocess.run(
            ["ros2", "topic", "list", "-t"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        try:
            result = subprocess.run(
                ["ros2", "topic", "list"],
                capture_output=True, text=True, timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0 or not result.stdout.strip():
            return None

    live_topics, topic_types = parse_topic_list_with_types(result.stdout)
    categorized: Dict[str, List[str]] = {cat: [] for cat in FALLBACK_TOPICS}
    for topic in live_topics:
        categorized.setdefault(categorize_topic(topic), []).append(topic)
    return {cat: topics for cat, topics in categorized.items() if topics}, topic_types


class ROSInspectorApp(App):
    """Sidebar topic browser + live `ros2 topic echo` viewer."""

    CSS = """
    #sidebar {
        width: 52;
        border: solid $accent;
    }
    #main {
        width: 1fr;
    }
    #overview {
        height: 5;
        border: solid $accent;
        padding: 0 1;
    }
    #topic_info {
        height: 4;
        border: solid $accent;
        padding: 0 1;
        content-align: left middle;
    }
    #echo_log {
        border: solid $accent;
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_topics", "Refresh topic list"),
        ("c", "clear_echo", "Clear echo output"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.current_topic: Optional[str] = None
        self.topic_types: Dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield Tree("Topics", id="sidebar")
            with Vertical(id="main"):
                yield Static("", id="overview")
                yield Static("Select a topic from the sidebar to inspect it.", id="topic_info")
                yield RichLog(id="echo_log", wrap=True, highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "ROS 2 Topic Inspector"
        self.populate_tree()

    def populate_tree(self) -> None:
        tree = self.query_one("#sidebar", Tree)
        tree.clear()
        tree.root.expand()

        discovered = discover_live_topics()
        live = discovered is not None
        if discovered is None:
            categorized = FALLBACK_TOPICS
            self.topic_types = {}
        else:
            categorized, self.topic_types = discovered

        total_topics = sum(len(topics) for topics in categorized.values())
        source = "live" if live else "offline snapshot - ros2 unreachable"
        tree.root.label = f"Topics ({source}, {total_topics})"

        overview_lines = [
            f"[bold]ROS 2 Topic Inspector[/bold]  [dim]{source}[/dim]",
            f"topics: [bold]{total_topics}[/bold]    groups: [bold]{len([t for t in categorized.values() if t])}[/bold]    types shown: [bold]{len(self.topic_types)}[/bold]",
            "[dim]Select a leaf topic to stream it. Press r to refresh, c to clear echo, q to quit.[/dim]",
        ]
        self.query_one("#overview", Static).update("\n".join(overview_lines))

        for category, topics in ordered_categories(categorized):
            color = CATEGORY_COLORS.get(category, "white")
            cat_node = tree.root.add(f"[{color}]{category}[/] ({len(topics)})", expand=True)
            for topic in topics:
                msg_type = self.topic_types.get(topic, "")
                type_label = f" [dim]{short_type(msg_type)}[/dim]" if msg_type else ""
                cat_node.add_leaf(f"{topic}{type_label}", data=topic)

    def action_refresh_topics(self) -> None:
        self.populate_tree()
        self.query_one("#topic_info", Static).update(
            "Topic list refreshed. Select a topic from the sidebar."
        )

    def action_clear_echo(self) -> None:
        self.query_one("#echo_log", RichLog).clear()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        topic = event.node.data
        if not topic:
            return  # a category header was selected, not a leaf topic
        self.select_topic(topic)

    def select_topic(self, topic: str) -> None:
        self.current_topic = topic
        self.query_one("#echo_log", RichLog).clear()
        cached_type = self.topic_types.get(topic)
        if cached_type:
            self.query_one("#topic_info", Static).update(
                f"[bold]{topic}[/bold]\n[dim]type:[/dim] {cached_type}\n[dim]stream:[/dim] ros2 topic echo"
            )
        else:
            self.query_one("#topic_info", Static).update(
                f"[bold]{topic}[/bold]\n[dim]type:[/dim] resolving...\n[dim]stream:[/dim] ros2 topic echo"
            )
        self.run_worker(self._stream_echo(topic), exclusive=True, group="echo")
        if not cached_type:
            self.run_worker(self._fetch_topic_type(topic), exclusive=True, group="type")

    async def _fetch_topic_type(self, topic: str) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ros2", "topic", "type", topic,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            msg_type = stdout.decode().strip() or "unknown"
        except (FileNotFoundError, asyncio.TimeoutError):
            msg_type = "unavailable"

        if self.current_topic == topic:
            self.topic_types[topic] = msg_type
            self.query_one("#topic_info", Static).update(
                f"[bold]{topic}[/bold]\n[dim]type:[/dim] {msg_type}\n[dim]stream:[/dim] ros2 topic echo"
            )

    async def _stream_echo(self, topic: str) -> None:
        log = self.query_one("#echo_log", RichLog)
        try:
            proc = await asyncio.create_subprocess_exec(
                "ros2", "topic", "echo", topic,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
        except FileNotFoundError:
            log.write("[bold red]ros2 CLI not found - source your ROS 2 environment first.[/]")
            return

        assert proc.stdout is not None
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                log.write(line.decode(errors="replace").rstrip())
        except asyncio.CancelledError:
            raise
        finally:
            if proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2)
                except asyncio.TimeoutError:
                    proc.kill()


def main() -> None:
    ROSInspectorApp().run()


if __name__ == "__main__":
    main()
