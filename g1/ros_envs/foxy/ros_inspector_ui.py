#!/usr/bin/env python3
"""Jetson Board Inspection Summary - htop-style Rich TUI viewer."""

import time
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.progress import BarColumn, Progress, TextColumn
from rich import box
from rich.live import Live
from rich.align import Align
from rich.rule import Rule

# ── Data extracted from summary.md ──────────────────────────────────────────

SYSTEM = {
    "Hostname": "ubuntu",
    "OS": "Ubuntu 20.04.6 LTS (focal)",
    "Kernel": "5.10.104-tegra",
    "Arch": "aarch64",
    "ROS Distro": "foxy (ROS 2)",
    "RMW": "rmw_cyclonedds_cpp",
    "Uptime": "~6 min at capture",
}

RESOURCES = {
    "RAM Used": (2.38, 15.4),   # (used, total) GB
    "Swap Used": (0.0, 7.5),
    "CPU": "light-moderate",
    "GPU": "0%",
    "Temps": "39–42 °C",
    "Disk": (21, 1900),         # GB used / total
}

SERVICES = [
    "ssh.service", "docker.service", "containerd.service",
    "master_service.service", "unitree-upgrade.service",
    "nvargus-daemon.service", "NetworkManager.service", "gdm.service",
]

UNITREE_PROCS = [
    "/unitree/module/master_service/master_service",
    "/unitree/ota/pipe/ota_pipe_service",
    "/unitree/module/video_hub_pc4/videohub_pc4 /dev/video4",
    "/unitree/module/video_hub_pc4/videohub_pc4_chest /dev/video10",
]

PORTS = [
    ("22", "ssh"),
    ("80", "http"),
    ("4000", "TCP+UDP board svc"),
    ("7400/7401", "DDS / CycloneDDS"),
    ("7001", "localhost / IPv6"),
    ("111", "rpcbind"),
]

TOPICS = {
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

GRAPH_STATUS = {
    "Topics": ("ACTIVE", "green"),
    "Nodes":   ("EMPTY",  "red"),
    "Services":("EMPTY",  "red"),
    "Actions": ("EMPTY",  "red"),
}

WORKSPACES = [
    ("cyclonedds_ws", "CycloneDDS + RMW build", "ROS 2"),
    ("Odometer_service", "SVO/VIO visual odometry", "ROS 1 / catkin"),
]

TAKEAWAYS = [
    "ROS 2 Foxy installed and sourced",
    "CycloneDDS is the active middleware",
    "Unitree services, video & DDS traffic are active",
    "Rich topic graph: motion, SLAM, LiDAR, audio, video, arm",
    "Node/service/action discovery returned empty (DDS namespace / discovery quirk)",
    "No explicit ROS 2 application workspace or launch files found",
]

CATEGORY_COLORS = {
    "Robot State / Cmd":    "cyan",
    "API Request/Response": "yellow",
    "SLAM / Mapping":       "magenta",
    "LiDAR / IMU":          "blue",
    "Camera / WebRTC":      "green",
    "Arm / Dexterous Hand": "red",
}

# ── Builder helpers ──────────────────────────────────────────────────────────

def header_bar() -> Panel:
    txt = Text()
    txt.append("  JETSON BOARD INSPECTION SUMMARY  ", style="bold white on dark_blue")
    txt.append("  ROS 2 Foxy / CycloneDDS / aarch64  ", style="bold white on dark_blue")
    txt.append("  [CAPTURED SNAPSHOT]  ", style="bold yellow on dark_blue")
    return Panel(Align.center(txt), style="bold blue", box=box.HEAVY, padding=(0, 1))


def resource_bar(label: str, used: float, total: float, color: str) -> Text:
    pct = used / total
    filled = int(pct * 30)
    bar = "[" + "█" * filled + " " * (30 - filled) + "]"
    t = Text()
    t.append(f"{label:<5}", style="bold white")
    t.append(bar, style=color)
    t.append(f"  {used:.1f}/{total:.1f} GB  ({pct*100:.0f}%)", style="white")
    return t


def system_panel() -> Panel:
    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column(style="bold cyan", no_wrap=True)
    tbl.add_column(style="white")
    for k, v in SYSTEM.items():
        tbl.add_row(k, v)
    return Panel(tbl, title="[bold cyan]System[/]", border_style="cyan", box=box.ROUNDED)


def resource_panel() -> Panel:
    lines = [
        resource_bar("RAM", *RESOURCES["RAM Used"], "green"),
        resource_bar("Swap", *RESOURCES["Swap Used"], "yellow"),
        resource_bar("Disk", *RESOURCES["Disk"], "blue"),
    ]
    extra = Text()
    extra.append(f"\n  CPU: ", style="bold white")
    extra.append(RESOURCES["CPU"], style="white")
    extra.append("   GPU: ", style="bold white")
    extra.append(RESOURCES["GPU"], style="white")
    extra.append("   Temp: ", style="bold white")
    extra.append(RESOURCES["Temps"], style="red")
    from rich.console import Group
    content = Group(*lines, extra)
    return Panel(content, title="[bold green]Resources[/]", border_style="green", box=box.ROUNDED)


def graph_status_panel() -> Panel:
    tbl = Table(box=None, show_header=False, padding=(0, 2))
    tbl.add_column(style="bold white", no_wrap=True)
    tbl.add_column()
    for name, (status, color) in GRAPH_STATUS.items():
        tbl.add_row(name, f"[bold {color}]{status}[/]")
    return Panel(tbl, title="[bold yellow]ROS Graph[/]", border_style="yellow", box=box.ROUNDED)


def services_panel() -> Panel:
    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column(style="green")
    for s in SERVICES:
        tbl.add_row(f"● {s}")
    return Panel(tbl, title="[bold green]Services[/]", border_style="green", box=box.ROUNDED)


def ports_panel() -> Panel:
    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column(style="bold yellow", no_wrap=True, min_width=12)
    tbl.add_column(style="white")
    for port, desc in PORTS:
        tbl.add_row(port, desc)
    return Panel(tbl, title="[bold yellow]Open Ports[/]", border_style="yellow", box=box.ROUNDED)


def workspaces_panel() -> Panel:
    tbl = Table(box=None, show_header=True, padding=(0, 1))
    tbl.add_column("Workspace", style="bold magenta", no_wrap=True)
    tbl.add_column("Description", style="white")
    tbl.add_column("Type", style="cyan", no_wrap=True)
    for ws, desc, kind in WORKSPACES:
        tbl.add_row(ws, desc, kind)
    return Panel(tbl, title="[bold magenta]Workspaces[/]", border_style="magenta", box=box.ROUNDED)


def topic_panel(category: str, topics: list) -> Panel:
    color = CATEGORY_COLORS[category]
    tbl = Table(box=None, show_header=False, padding=(0, 0))
    tbl.add_column(style=color)
    for t in topics:
        tbl.add_row(t)
    count = f"[bold {color}]({len(topics)} topics)[/]"
    return Panel(
        tbl,
        title=f"[bold {color}]{category}[/] {count}",
        border_style=color,
        box=box.ROUNDED,
    )


def topics_section() -> Columns:
    panels = [topic_panel(cat, topics) for cat, topics in TOPICS.items()]
    return Columns(panels, equal=False, expand=True)


def procs_panel() -> Panel:
    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column(style="dim white")
    for p in UNITREE_PROCS:
        tbl.add_row(p)
    return Panel(tbl, title="[bold red]Unitree Processes[/]", border_style="red", box=box.ROUNDED)


def takeaways_panel() -> Panel:
    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column()
    for i, t in enumerate(TAKEAWAYS):
        icon = "[bold green]✔[/]" if i < 4 else "[bold yellow]⚠[/]"
        tbl.add_row(f"{icon}  {t}")
    return Panel(
        tbl,
        title="[bold white]Key Takeaways[/]",
        border_style="white",
        box=box.HEAVY,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def build_page() -> Layout:
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=10),
        Layout(name="mid_info", size=8),
        Layout(name="topics_label", size=1),
        Layout(name="topics", size=None),
        Layout(name="takeaways", size=12),
    )

    layout["top"].split_row(
        Layout(name="sys", ratio=3),
        Layout(name="res", ratio=4),
        Layout(name="graph", ratio=2),
    )

    layout["mid_info"].split_row(
        Layout(name="svcs", ratio=3),
        Layout(name="ports", ratio=2),
        Layout(name="procs", ratio=4),
        Layout(name="ws", ratio=3),
    )

    layout["header"].update(header_bar())
    layout["top"]["sys"].update(system_panel())
    layout["top"]["res"].update(resource_panel())
    layout["top"]["graph"].update(graph_status_panel())

    layout["mid_info"]["svcs"].update(services_panel())
    layout["mid_info"]["ports"].update(ports_panel())
    layout["mid_info"]["procs"].update(procs_panel())
    layout["mid_info"]["ws"].update(workspaces_panel())

    layout["topics_label"].update(
        Rule("[bold white]━  ACTIVE ROS 2 TOPICS  ━[/]", style="bold blue")
    )
    layout["topics"].update(topics_section())
    layout["takeaways"].update(takeaways_panel())

    return layout


def main():
    console = Console()
    console.clear()
    page = build_page()
    console.print(page)
    console.print()


if __name__ == "__main__":
    main()
