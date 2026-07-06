#!/usr/bin/env python3
"""Interactive ROS 2 topic inspector - Textual TUI.

Sidebar lists topics grouped by category; selecting one streams its live
`ros2 topic echo` output into the main panel. Requires the `textual`
package (`pip install textual`) and a sourced ROS 2 environment on PATH.
If `ros2` isn't reachable, the sidebar falls back to the last known
topic list captured in summary.md so the layout can still be browsed.
"""

import asyncio
import contextlib
import re
import subprocess
from functools import lru_cache
from pathlib import Path

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

# ── Fallback topic set, used when `ros2 topic list` is unreachable ───────────

SUMMARY_PATH = Path(__file__).with_name("summary.md")

DEFAULT_TOPICS = {
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

SUMMARY_CATEGORY_ALIASES = {
    "low-level robot state and command topics": "Robot State / Cmd",
    "api request/response topics": "API Request/Response",
    "slam / mapping topics": "SLAM / Mapping",
    "lidar / imu topics": "LiDAR / IMU",
    "camera / media / webrtc topics": "Camera / WebRTC",
    "arm / dexterous hand topics": "Arm / Dexterous Hand",
}


def infer_topic_category(topic: str) -> str:
    """Infer a display category for topics not already in the known catalog."""
    if topic.startswith("/api/"):
        return "API Request/Response"
    if topic.startswith("/dex3/"):
        return "Arm / Dexterous Hand"
    if topic.startswith("/utlidar/") or "imu" in topic or "lidar" in topic:
        return "LiDAR / IMU"
    if any(part in topic for part in ("webrtc", "video", "camera", "image")):
        return "Camera / WebRTC"
    if any(part in topic for part in ("slam", "map", "waypoint")):
        return "SLAM / Mapping"
    if topic in DEFAULT_TOPICS["Robot State / Cmd"]:
        return "Robot State / Cmd"
    return "Other"


def categorize_topics(topics: list[str], known_topics: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return ordered topic groups, deduplicated and enriched with inferred categories."""
    topic_to_category = {
        topic: category
        for category, category_topics in known_topics.items()
        for topic in category_topics
    }

    categorized: dict[str, list[str]] = {category: [] for category in known_topics}
    categorized["Other"] = []
    for topic in sorted(set(topics)):
        category = topic_to_category.get(topic, infer_topic_category(topic))
        categorized.setdefault(category, []).append(topic)
    return {category: category_topics for category, category_topics in categorized.items() if category_topics}


@lru_cache(maxsize=1)
def load_fallback_topics() -> dict[str, list[str]]:
    """Load the offline topic catalog from summary.md when available."""
    if not SUMMARY_PATH.exists():
        return DEFAULT_TOPICS

    extracted: dict[str, list[str]] = {}
    current_category: str | None = None
    in_topic_section = False
    for line in SUMMARY_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "### Topics":
            in_topic_section = True
            continue
        if in_topic_section and stripped.startswith("## "):
            break
        if not in_topic_section:
            continue

        category_match = re.match(r"^- (.+ topics):$", stripped, flags=re.IGNORECASE)
        if category_match:
            label = category_match.group(1).strip().lower()
            current_category = SUMMARY_CATEGORY_ALIASES.get(label)
            if current_category:
                extracted.setdefault(current_category, [])
            continue

        if current_category is None:
            continue

        topic_matches = re.findall(r"/[A-Za-z0-9_./-]+", stripped)
        extracted[current_category].extend(topic_matches)

    if not extracted:
        return DEFAULT_TOPICS

    merged = {category: list(topics) for category, topics in DEFAULT_TOPICS.items()}
    for category, topics in extracted.items():
        merged[category] = sorted(set(topics))
    return merged


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
    return categorize_topics(live_topics, load_fallback_topics())


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
            categorized = load_fallback_topics()
        topic_count = sum(len(topics) for topics in categorized.values())
        tree.root.label = (
            f"Topics (live, {topic_count})"
            if live
            else f"Topics (offline snapshot, {topic_count})"
        )

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
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            raise
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=1)


def main() -> None:
    ROSInspectorApp().run()


if __name__ == "__main__":
    main()
