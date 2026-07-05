#!/usr/bin/env python3
"""Interactive ROS 2 topic inspector - Textual TUI.

Sidebar lists topics grouped by category; selecting one streams its live
`ros2 topic echo` output into the main panel. Requires the `textual`
package (`pip install textual`) and a sourced ROS 2 environment on PATH.
If `ros2` isn't reachable, the sidebar falls back to the last known
topic list captured in summary.md so the layout can still be browsed.
"""

import asyncio
import subprocess

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

# ── Fallback topic set, used only when `ros2 topic list` is unreachable ─────

FALLBACK_TOPICS = {
    "Robot State / Cmd": [
        "/lowcmd", "/lowstate", "/user_lowcmd",
        "/multiplestate", "/sportmodestate", "/odommodestate",
    ],
    "API Request/Response": [
        "/api/sport/request",      "/api/sport/response",
        "/api/arm/request",        "/api/arm/response",
        "/api/robot_state/request","/api/robot_state/response",
        "/api/motion_switcher/request", "/api/motion_switcher/response",
        "/api/slam_operate/request",    "/api/slam_operate/response",
    ],
    "SLAM / Mapping": [
        "/unitree/slam_mapping/odom",    "/unitree/slam_mapping/points",
        "/unitree/slam_relocation/odom", "/unitree/slam_relocation/points",
        "/unitree_slam/waypoints",       "/slam_info",
        "/slam_key_info",                "/global_map",
        "/planner_map",                  "/gridmap",
    ],
    "LiDAR / IMU": [
        "/utlidar/cloud_livox_mid360",
        "/utlidar/imu_livox_mid360",
        "/dog_imu_raw",
        "/secondary_imu",
    ],
    "Camera / WebRTC": [
        "/frontvideostream",
        "/videohub/inner",
        "/webrtcreq",
        "/webrtcres",
    ],
    "Arm / Dexterous Hand": [
        "/dex3/left/cmd",   "/dex3/right/cmd",
        "/dex3/left/state", "/dex3/right/state",
    ],
}

CATEGORY_COLORS = {
    "Robot State / Cmd":     "cyan",
    "API Request/Response":  "yellow",
    "SLAM / Mapping":        "magenta",
    "LiDAR / IMU":           "blue",
    "Camera / WebRTC":       "green",
    "Arm / Dexterous Hand":  "red",
    "Other":                 "white",
}


def discover_live_topics() -> dict[str, list[str]] | None:
    """Return {category: [topics]} from a live `ros2 topic list`, or None if unreachable."""
    try:
        result = subprocess.run(
            ["ros2", "topic", "list"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None

    live_topics = result.stdout.strip().splitlines()
    categorized: dict[str, list[str]] = {cat: [] for cat in FALLBACK_TOPICS}
    categorized["Other"] = []
    for topic in live_topics:
        for cat, known in FALLBACK_TOPICS.items():
            if topic in known:
                categorized[cat].append(topic)
                break
        else:
            categorized["Other"].append(topic)
    return {cat: topics for cat, topics in categorized.items() if topics}


class ROSInspectorApp(App):
    """Sidebar topic browser + live `ros2 topic echo` viewer."""

    CSS = """
    #sidebar {
        width: 40;
        border: solid $accent;
    }
    #main {
        width: 1fr;
    }
    #topic_info {
        height: 3;
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
    ]

    def __init__(self) -> None:
        super().__init__()
        self.current_topic: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield Tree("Topics", id="sidebar")
            with Vertical(id="main"):
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

        categorized = discover_live_topics()
        live = categorized is not None
        if categorized is None:
            categorized = FALLBACK_TOPICS
        tree.root.label = "Topics (live)" if live else "Topics (offline snapshot - ros2 unreachable)"

        for category, topics in categorized.items():
            color = CATEGORY_COLORS.get(category, "white")
            cat_node = tree.root.add(f"[{color}]{category}[/] ({len(topics)})", expand=True)
            for topic in topics:
                cat_node.add_leaf(topic, data=topic)

    def action_refresh_topics(self) -> None:
        self.populate_tree()
        self.query_one("#topic_info", Static).update(
            "Topic list refreshed. Select a topic from the sidebar."
        )

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        topic = event.node.data
        if not topic:
            return  # a category header was selected, not a leaf topic
        self.select_topic(topic)

    def select_topic(self, topic: str) -> None:
        self.current_topic = topic
        self.query_one("#echo_log", RichLog).clear()
        self.query_one("#topic_info", Static).update(f"[bold]{topic}[/bold]  [dim](resolving type...)[/dim]")
        self.run_worker(self._stream_echo(topic), exclusive=True, group="echo")
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
            self.query_one("#topic_info", Static).update(
                f"[bold]{topic}[/bold]  [dim]type:[/dim] {msg_type}"
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


def main() -> None:
    ROSInspectorApp().run()


if __name__ == "__main__":
    main()
