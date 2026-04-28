#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sdk_client import Robot


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


@dataclass
class TelemetrySnapshot:
    stamp: float = 0.0
    pose: tuple[float, float, float] | None = None
    pose_source: str = "none"
    slam_pose: tuple[float, float, float] | None = None
    odom_pose: tuple[float, float, float] | None = None
    sport_position: tuple[float, float, float] | None = None
    velocity: tuple[float, float, float] | None = None
    imu_rpy: tuple[float, float, float] | None = None
    slam_running: bool = False
    nav_active: bool = False
    mapping_enabled: bool = True
    goal: tuple[float, float] | None = None
    status: str = "idle"
    robot_connected: bool = False
    sensor_stale: dict[str, bool] = field(default_factory=dict)
    ros_bridge_ready: bool = False
    ros_bridge_status: str = "disabled"


@dataclass
class MapSnapshot:
    rgb: np.ndarray | None = None
    occupancy: np.ndarray | None = None
    origin_x: float = 0.0
    origin_y: float = 0.0
    resolution: float = 0.05
    updated_at: float = 0.0
    robot_pose: tuple[float, float, float] | None = None
    goal_pose: tuple[float, float] | None = None
    path_world: list[tuple[float, float]] = field(default_factory=list)

    @property
    def width(self) -> int:
        return 0 if self.occupancy is None else int(self.occupancy.shape[1])

    @property
    def height(self) -> int:
        return 0 if self.occupancy is None else int(self.occupancy.shape[0])


class OccupancyGrid:
    def __init__(
        self,
        *,
        resolution: float = 0.05,
        size_m: float = 24.0,
        decay_s: float = 12.0,
        occ_logit: float = 0.85,
        free_logit: float = 0.22,
        l_min: float = -2.5,
        l_max: float = 3.5,
    ) -> None:
        self.resolution = float(resolution)
        self.size_m = float(size_m)
        self.decay_s = float(decay_s)
        self.occ_logit = float(occ_logit)
        self.free_logit = float(free_logit)
        self.l_min = float(l_min)
        self.l_max = float(l_max)
        self.width = max(64, int(round(self.size_m / self.resolution)))
        self.height = max(64, int(round(self.size_m / self.resolution)))
        self.origin_x = -0.5 * self.width * self.resolution
        self.origin_y = -0.5 * self.height * self.resolution
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.age = np.zeros((self.height, self.width), dtype=np.float32)
        self.last_update = 0.0

    def reset(self) -> None:
        self.log_odds.fill(0.0)
        self.age.fill(0.0)
        self.last_update = 0.0

    def world_to_grid(self, x: float, y: float) -> tuple[int, int] | None:
        gx = int((float(x) - self.origin_x) / self.resolution)
        gy = int((float(y) - self.origin_y) / self.resolution)
        if gx < 0 or gy < 0 or gx >= self.width or gy >= self.height:
            return None
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        x = self.origin_x + (float(gx) + 0.5) * self.resolution
        y = self.origin_y + (float(gy) + 0.5) * self.resolution
        return x, y

    def _raytrace(self, x0: int, y0: int, x1: int, y1: int, stamp: float) -> None:
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if x == x1 and y == y1:
                break
            self.log_odds[y, x] = np.clip(self.log_odds[y, x] - self.free_logit, self.l_min, self.l_max)
            self.age[y, x] = stamp
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def insert(
        self,
        pose: tuple[float, float, float],
        points_xyz: np.ndarray,
        *,
        min_range: float = 0.50,
        max_range: float = 8.0,
        z_min: float = -0.20,
        z_max: float = 0.90,
    ) -> None:
        if points_xyz.size == 0:
            return
        robot = self.world_to_grid(pose[0], pose[1])
        if robot is None:
            return
        stamp = time.time()
        px, py = robot
        yaw = float(pose[2])
        c = math.cos(yaw)
        s = math.sin(yaw)

        for x_l, y_l, z_l in points_xyz:
            rng = math.hypot(float(x_l), float(y_l))
            if rng < min_range or rng > max_range:
                continue
            if float(z_l) < z_min or float(z_l) > z_max:
                continue
            x_w = c * float(x_l) - s * float(y_l) + float(pose[0])
            y_w = s * float(x_l) + c * float(y_l) + float(pose[1])
            cell = self.world_to_grid(x_w, y_w)
            if cell is None:
                continue
            gx, gy = cell
            self._raytrace(px, py, gx, gy, stamp)
            self.log_odds[gy, gx] = np.clip(self.log_odds[gy, gx] + self.occ_logit, self.l_min, self.l_max)
            self.age[gy, gx] = stamp
        self.last_update = stamp

    def occupancy_mask(self) -> np.ndarray:
        if self.decay_s > 0.0:
            age = np.maximum(0.0, time.time() - self.age)
            active = (self.age > 0.0) & (age <= self.decay_s)
        else:
            active = self.age > 0.0
        return active & (self.log_odds > 0.0)

    def render(
        self,
        *,
        robot_pose: tuple[float, float, float] | None,
        goal_pose: tuple[float, float] | None,
        path_world: list[tuple[float, float]] | None,
    ) -> MapSnapshot:
        occ = self.occupancy_mask()
        rgb = np.full((self.height, self.width, 3), 26, dtype=np.uint8)
        rgb[occ] = (235, 235, 235)

        if path_world:
            path_px = []
            for x_w, y_w in path_world:
                cell = self.world_to_grid(x_w, y_w)
                if cell is not None:
                    path_px.append(cell)
            for idx in range(1, len(path_px)):
                self._draw_line(rgb, path_px[idx - 1], path_px[idx], (41, 98, 255), thickness=1)

        if goal_pose is not None:
            goal = self.world_to_grid(goal_pose[0], goal_pose[1])
            if goal is not None:
                self._draw_marker(rgb, goal, (255, 106, 0))

        if robot_pose is not None:
            rob = self.world_to_grid(robot_pose[0], robot_pose[1])
            if rob is not None:
                self._draw_robot(rgb, rob, robot_pose[2])

        rgb = np.flipud(rgb)
        occ_out = np.flipud(occ.copy())
        return MapSnapshot(
            rgb=rgb,
            occupancy=occ_out,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            resolution=self.resolution,
            updated_at=self.last_update,
            robot_pose=robot_pose,
            goal_pose=goal_pose,
            path_world=list(path_world or []),
        )

    @staticmethod
    def _draw_line(
        img: np.ndarray,
        p0: tuple[int, int],
        p1: tuple[int, int],
        color: tuple[int, int, int],
        *,
        thickness: int = 1,
    ) -> None:
        x0, y0 = p0
        x1, y1 = p1
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            for oy in range(-thickness, thickness + 1):
                for ox in range(-thickness, thickness + 1):
                    xx = x0 + ox
                    yy = y0 + oy
                    if 0 <= yy < img.shape[0] and 0 <= xx < img.shape[1]:
                        img[yy, xx] = color
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    @staticmethod
    def _draw_marker(img: np.ndarray, p: tuple[int, int], color: tuple[int, int, int]) -> None:
        x, y = p
        for d in range(-4, 5):
            if 0 <= y < img.shape[0] and 0 <= x + d < img.shape[1]:
                img[y, x + d] = color
            if 0 <= y + d < img.shape[0] and 0 <= x < img.shape[1]:
                img[y + d, x] = color

    @staticmethod
    def _draw_robot(img: np.ndarray, p: tuple[int, int], yaw: float) -> None:
        x, y = p
        for oy in range(-3, 4):
            for ox in range(-3, 4):
                xx = x + ox
                yy = y + oy
                if 0 <= yy < img.shape[0] and 0 <= xx < img.shape[1]:
                    img[yy, xx] = (255, 220, 0)
        tip = (
            int(round(x + 10.0 * math.cos(yaw))),
            int(round(y + 10.0 * math.sin(yaw))),
        )
        OccupancyGrid._draw_line(img, p, tip, (255, 220, 0), thickness=1)


class RosSensorBridge:
    def __init__(
        self,
        *,
        enabled: bool = True,
        point_topic: str = "/livox/points",
        rgb_topic: str = "/rgbd/color/image_raw",
        depth_topic: str = "/rgbd/depth/image_raw",
    ) -> None:
        self.enabled = bool(enabled)
        self.point_topic = str(point_topic)
        self.rgb_topic = str(rgb_topic)
        self.depth_topic = str(depth_topic)
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._init_error: str | None = None
        self._ready = False
        self._status_message = "disabled" if not self.enabled else "starting"
        self._latest_points: np.ndarray | None = None
        self._latest_points_ts = 0.0
        self._latest_rgb: np.ndarray | None = None
        self._latest_rgb_ts = 0.0
        self._latest_depth: np.ndarray | None = None
        self._latest_depth_ts = 0.0
        if self.enabled:
            self._thread = threading.Thread(target=self._reader_loop, name="g1-nav-ros2-bridge", daemon=True)
            self._thread.start()

    def shutdown(self) -> None:
        self._stop_evt.set()
        self._terminate_worker()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def status(self) -> dict[str, bool]:
        with self._lock:
            return {
                "ros_bridge_error": self._init_error is not None,
                "ros_bridge_ready": self._ready,
                "ros_lidar_stale": self._is_stale(self._latest_points_ts, max_age=1.5),
                "ros_rgb_stale": self._is_stale(self._latest_rgb_ts, max_age=1.5),
                "ros_depth_stale": self._is_stale(self._latest_depth_ts, max_age=1.5),
            }

    def get_points(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return (
                None if self._latest_points is None else np.array(self._latest_points, copy=True),
                self._latest_points_ts,
            )

    def get_rgb(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return (
                None if self._latest_rgb is None else np.array(self._latest_rgb, copy=True),
                self._latest_rgb_ts,
            )

    def get_depth(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            return (
                None if self._latest_depth is None else np.array(self._latest_depth, copy=True),
                self._latest_depth_ts,
            )

    @property
    def init_error(self) -> str | None:
        with self._lock:
            return self._init_error

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def status_message(self) -> str:
        with self._lock:
            return self._status_message

    @staticmethod
    def _is_stale(stamp: float, max_age: float) -> bool:
        if stamp <= 0.0:
            return True
        return (time.time() - stamp) > float(max_age)

    def _reader_loop(self) -> None:
        try:
            worker = Path(__file__).resolve().parent / "g1_nav_ros_bridge_worker.py"
            env = dict(os.environ)
            env.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    str(worker),
                    "--lidar-topic",
                    self.point_topic,
                    "--rgb-topic",
                    self.rgb_topic,
                    "--depth-topic",
                    self.depth_topic,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                env=env,
                bufsize=0,
            )
            if self._proc.stdout is None:
                raise RuntimeError("ROS bridge worker stdout pipe is unavailable.")
            with self._lock:
                self._status_message = f"worker pid={self._proc.pid}"
            while not self._stop_evt.is_set():
                header = self._proc.stdout.read(4)
                if not header:
                    break
                size = struct.unpack("<I", header)[0]
                payload = self._proc.stdout.read(size)
                if len(payload) != size:
                    break
                self._handle_message(pickle.loads(payload))
            if self._proc.poll() not in (None, 0):
                stderr_text = b""
                if self._proc.stderr is not None:
                    try:
                        stderr_text = self._proc.stderr.read() or b""
                    except Exception:
                        stderr_text = b""
                raise RuntimeError(
                    f"ROS bridge worker exited rc={self._proc.returncode}: {stderr_text.decode(errors='replace').strip()}"
                )
        except Exception as exc:
            with self._lock:
                self._init_error = str(exc)
                self._status_message = str(exc)
        finally:
            self._terminate_worker()

    def _terminate_worker(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _handle_message(self, msg: dict[str, Any]) -> None:
        kind = str(msg.get("kind", ""))
        stamp = float(msg.get("stamp", time.time()))
        if kind == "status":
            if bool(msg.get("ok", False)):
                with self._lock:
                    self._ready = True
                    self._status_message = str(msg.get("message", "ready"))
            else:
                with self._lock:
                    self._init_error = str(msg.get("message", "unknown ROS bridge worker error"))
                    self._status_message = self._init_error
            return
        if kind == "points":
            with self._lock:
                self._latest_points = np.array(msg.get("data"), copy=True)
                self._latest_points_ts = stamp
            return
        if kind == "rgb":
            with self._lock:
                self._latest_rgb = np.array(msg.get("data"), copy=True)
                self._latest_rgb_ts = stamp
            return
        if kind == "depth":
            with self._lock:
                self._latest_depth = np.array(msg.get("data"), copy=True)
                self._latest_depth_ts = stamp


class NavigationController:
    def __init__(
        self,
        *,
        iface: str = "eth0",
        map_resolution: float = 0.05,
        map_size_m: float = 24.0,
        ros_topics_enabled: bool = True,
        ros_lidar_topic: str = "/livox/points",
        ros_rgb_topic: str = "/rgbd/color/image_raw",
        ros_depth_topic: str = "/rgbd/depth/image_raw",
    ) -> None:
        self.iface = str(iface)
        self.map = OccupancyGrid(resolution=map_resolution, size_m=map_size_m)
        self.robot: Robot | None = None
        self._lock = threading.Lock()
        self._telemetry = TelemetrySnapshot(status="disconnected")
        self._map_snapshot = self.map.render(robot_pose=None, goal_pose=None, path_world=None)
        self._camera_rgb: np.ndarray | None = None
        self._camera_enabled = True
        self._logs: list[str] = []
        self._goal_world: tuple[float, float] | None = None
        self._path_world: list[tuple[float, float]] = []
        self._nav_active = False
        self._mapping_enabled = True
        self._avoid_enabled = True
        self._ros_bridge = RosSensorBridge(
            enabled=ros_topics_enabled,
            point_topic=ros_lidar_topic,
            rgb_topic=ros_rgb_topic,
            depth_topic=ros_depth_topic,
        )
        self._ros_bridge_error_logged = False
        self._ros_bridge_ready_logged = False
        self._stop_evt = threading.Event()
        self._worker = threading.Thread(target=self._loop, name="g1-nav-backend", daemon=True)
        self._worker.start()

    def shutdown(self) -> None:
        self._stop_evt.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._ros_bridge.shutdown()
        if self.robot is not None:
            try:
                self.robot.stop()
            except Exception:
                pass

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with self._lock:
            self._logs.append(line)
            self._logs = self._logs[-200:]
            self._telemetry.status = msg

    def get_ui_state(self) -> dict[str, Any]:
        with self._lock:
            telemetry = TelemetrySnapshot(**self._telemetry.__dict__)
            map_snapshot = MapSnapshot(
                rgb=None if self._map_snapshot.rgb is None else np.array(self._map_snapshot.rgb, copy=True),
                occupancy=None if self._map_snapshot.occupancy is None else np.array(self._map_snapshot.occupancy, copy=True),
                origin_x=self._map_snapshot.origin_x,
                origin_y=self._map_snapshot.origin_y,
                resolution=self._map_snapshot.resolution,
                updated_at=self._map_snapshot.updated_at,
                robot_pose=self._map_snapshot.robot_pose,
                goal_pose=self._map_snapshot.goal_pose,
                path_world=list(self._map_snapshot.path_world),
            )
            camera = None if self._camera_rgb is None else np.array(self._camera_rgb, copy=True)
            logs = list(self._logs)
            return {
                "telemetry": telemetry,
                "map": map_snapshot,
                "camera_rgb": camera,
                "logs": logs,
                "mapping_enabled": self._mapping_enabled,
                "avoid_enabled": self._avoid_enabled,
                "nav_active": self._nav_active,
            }

    def connect_robot(self) -> None:
        if self.robot is not None:
            self.log("Robot is already connected.")
            return

        def _job():
            try:
                from sdk_client import Robot

                robot = Robot(iface=self.iface, safety_boot=False, auto_start_sensors=True)
                with self._lock:
                    self.robot = robot
                    self._telemetry.robot_connected = True
                    self._telemetry.status = "robot connected"
                self.log(f"Connected to robot on iface={self.iface}.")
            except Exception as exc:
                self.log(f"Robot connect failed: {exc}")

        threading.Thread(target=_job, name="g1-nav-connect", daemon=True).start()

    def start_slam(self, slam_type: str = "indoor") -> None:
        def _job():
            robot = self.robot
            if robot is None:
                self.log("Cannot start SLAM: robot not connected.")
                return
            try:
                rc = int(robot.start_slam(slam_type=slam_type))
                self.log(f"Start SLAM rc={rc} ({slam_type}).")
            except Exception as exc:
                self.log(f"Start SLAM failed: {exc}")

        threading.Thread(target=_job, name="g1-nav-start-slam", daemon=True).start()

    def stop_slam(self, save_path: str | None = None) -> None:
        def _job():
            robot = self.robot
            if robot is None:
                self.log("Cannot stop SLAM: robot not connected.")
                return
            try:
                rc = int(robot.stop_slam(save_path=save_path))
                self.log(f"Stop SLAM rc={rc}.")
            except Exception as exc:
                self.log(f"Stop SLAM failed: {exc}")

        threading.Thread(target=_job, name="g1-nav-stop-slam", daemon=True).start()

    def pause_nav(self) -> None:
        def _job():
            robot = self.robot
            if robot is None:
                self.log("Cannot pause nav: robot not connected.")
                return
            try:
                rc = int(robot._get_slam_client().pause_nav().code)  # noqa: SLF001
                self.log(f"Pause nav rc={rc}.")
            except Exception as exc:
                self.log(f"Pause nav failed: {exc}")

        threading.Thread(target=_job, name="g1-nav-pause", daemon=True).start()

    def resume_nav(self) -> None:
        def _job():
            robot = self.robot
            if robot is None:
                self.log("Cannot resume nav: robot not connected.")
                return
            try:
                rc = int(robot._get_slam_client().resume_nav().code)  # noqa: SLF001
                self.log(f"Resume nav rc={rc}.")
            except Exception as exc:
                self.log(f"Resume nav failed: {exc}")

        threading.Thread(target=_job, name="g1-nav-resume", daemon=True).start()

    def set_mapping_enabled(self, enabled: bool) -> None:
        self._mapping_enabled = bool(enabled)
        self.log("Local occupancy mapping enabled." if enabled else "Local occupancy mapping frozen.")

    def toggle_avoidance(self) -> None:
        self._avoid_enabled = not self._avoid_enabled
        self.log(f"Obstacle avoidance {'enabled' if self._avoid_enabled else 'disabled'}.")

    def clear_goal(self) -> None:
        self._goal_world = None
        self._path_world = []
        self._nav_active = False
        self.log("Goal cleared.")

    def reset_map(self) -> None:
        self._path_world = []
        self._goal_world = None
        self._nav_active = False
        self.map.reset()
        self.log("Local map reset.")

    def set_goal_world(self, x: float, y: float) -> None:
        self._goal_world = (float(x), float(y))
        self._nav_active = False
        if self._replan() is None:
            self.log(f"Selected goal ({x:+.2f}, {y:+.2f}), but no path was found.")
        else:
            self.log(f"Selected goal ({x:+.2f}, {y:+.2f}).")

    def start_navigation(self) -> None:
        if self._goal_world is None:
            self.log("Select a goal before starting navigation.")
            return
        if not self._path_world and self._replan() is None:
            self.log("Navigation cannot start: planner found no path.")
            return
        self._nav_active = True
        self.log("Autonomous navigation started.")

    def stop_navigation(self) -> None:
        self._nav_active = False
        robot = self.robot
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        self.log("Navigation stopped.")

    def step_move(self, vx: float, vy: float, vyaw: float, duration: float = 0.35) -> None:
        def _job():
            robot = self.robot
            if robot is None:
                self.log("Cannot move: robot not connected.")
                return
            try:
                robot.move_for(duration=float(duration), vx=float(vx), vy=float(vy), vyaw=float(vyaw))
                self.log(f"Step move vx={vx:+.2f} vy={vy:+.2f} yaw={vyaw:+.2f}.")
            except Exception as exc:
                self.log(f"Step move failed: {exc}")

        threading.Thread(target=_job, name="g1-nav-step", daemon=True).start()

    def free_walk(self) -> None:
        def _job():
            robot = self.robot
            if robot is None:
                self.log("Cannot enter free walk: robot not connected.")
                return
            try:
                fn = getattr(robot._client, "FreeWalk", None)  # noqa: SLF001
                if not callable(fn):
                    raise RuntimeError("SDK locomotion client has no FreeWalk().")
                fn()
                self.log("Free walk requested.")
            except Exception as exc:
                self.log(f"Free walk failed: {exc}")

        threading.Thread(target=_job, name="g1-nav-freewalk", daemon=True).start()

    def save_snapshot(self, name: str | None = None) -> Path | None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = (name or f"g1_nav_snapshot_{stamp}").strip()
        out_dir = Path(__file__).resolve().parent / "maps"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{base}.npz"
        state = self.get_ui_state()
        map_snapshot: MapSnapshot = state["map"]
        telemetry: TelemetrySnapshot = state["telemetry"]
        np.savez_compressed(
            out_path,
            occupancy=map_snapshot.occupancy,
            rgb=map_snapshot.rgb,
            origin_x=np.array([map_snapshot.origin_x], dtype=np.float32),
            origin_y=np.array([map_snapshot.origin_y], dtype=np.float32),
            resolution=np.array([map_snapshot.resolution], dtype=np.float32),
            robot_pose=np.array(telemetry.pose if telemetry.pose is not None else [], dtype=np.float32),
            goal=np.array(self._goal_world if self._goal_world is not None else [], dtype=np.float32),
            path=np.array(self._path_world, dtype=np.float32),
        )
        self.log(f"Saved snapshot to {out_path.name}.")
        return out_path

    def _loop(self) -> None:
        last_rgb = 0.0
        while not self._stop_evt.is_set():
            robot = self.robot
            if robot is None:
                time.sleep(0.10)
                continue
            try:
                state = robot.get_robot_state()
                slam_pose = state.get("slam_pose")
                odom_pose = state.get("odom_pose")
                sport_position = state.get("position")
                pose_source, pose = self._choose_pose(slam_pose, odom_pose, sport_position)
                imu = state.get("imu")
                velocity = state.get("velocity")
                telemetry = TelemetrySnapshot(
                    stamp=time.time(),
                    pose=pose,
                    pose_source=pose_source,
                    slam_pose=slam_pose,
                    odom_pose=odom_pose,
                    sport_position=sport_position,
                    velocity=velocity,
                    imu_rpy=None if imu is None else tuple(float(v) for v in imu.rpy),
                    slam_running=bool(state.get("slam_is_running")),
                    nav_active=bool(self._nav_active),
                    mapping_enabled=bool(self._mapping_enabled),
                    goal=self._goal_world,
                    status=self._telemetry.status,
                    robot_connected=True,
                    sensor_stale={**dict(state.get("sensor_stale", {})), **self._ros_bridge.status()},
                    ros_bridge_ready=self._ros_bridge.ready,
                    ros_bridge_status=self._ros_bridge.status_message,
                )
                with self._lock:
                    self._telemetry = telemetry
                if self._ros_bridge.ready and not self._ros_bridge_ready_logged:
                    self._ros_bridge_ready_logged = True
                    self.log(f"ROS 2 bridge ready on topics lidar={self._ros_bridge.point_topic} rgb={self._ros_bridge.rgb_topic} depth={self._ros_bridge.depth_topic}.")
                ros_error = self._ros_bridge.init_error
                if ros_error and not self._ros_bridge_error_logged:
                    self._ros_bridge_error_logged = True
                    self.log(f"ROS 2 CycloneDDS bridge unavailable: {ros_error}")
                if pose is not None and self._mapping_enabled:
                    cloud, cloud_ts = self._ros_bridge.get_points()
                    if cloud is None or (time.time() - cloud_ts) > 1.5:
                        points = robot.get_lidar_points(max_points=2500)
                        cloud = self._points_to_numpy(points)
                    if cloud.size > 0:
                        self.map.insert(pose, cloud)
                if pose is not None and self._goal_world is not None and not self._path_world:
                    self._replan()
                self._apply_navigation(robot, pose)
                self._refresh_map_snapshot(pose)
                now = time.time()
                if self._camera_enabled and now - last_rgb >= 0.8:
                    last_rgb = now
                    self._refresh_camera(robot)
            except Exception as exc:
                self.log(f"Backend loop error: {exc}")
                time.sleep(0.4)
            time.sleep(0.10)

    def _refresh_camera(self, robot: "Robot") -> None:
        try:
            frame, stamp = self._ros_bridge.get_rgb()
            if frame is None or (time.time() - stamp) > 1.5:
                frame = robot.get_camera_frame_rgb()
            if frame is not None:
                with self._lock:
                    self._camera_rgb = np.array(frame, copy=True)
        except Exception:
            pass

    def _refresh_map_snapshot(self, pose: tuple[float, float, float] | None) -> None:
        snapshot = self.map.render(robot_pose=pose, goal_pose=self._goal_world, path_world=self._path_world)
        with self._lock:
            self._map_snapshot = snapshot

    @staticmethod
    def _points_to_numpy(points: list[dict[str, float]]) -> np.ndarray:
        if not points:
            return np.empty((0, 3), dtype=np.float32)
        xyz = np.array([[p["x"], p["y"], p["z"]] for p in points], dtype=np.float32)
        if xyz.shape[0] > 2000:
            step = max(1, xyz.shape[0] // 2000)
            xyz = xyz[::step]
        return xyz

    @staticmethod
    def _choose_pose(
        slam_pose: tuple[float, float, float] | None,
        odom_pose: tuple[float, float, float] | None,
        sport_position: tuple[float, float, float] | None,
    ) -> tuple[str, tuple[float, float, float] | None]:
        if slam_pose is not None:
            return "slam", tuple(float(v) for v in slam_pose)
        if odom_pose is not None:
            return "odom", tuple(float(v) for v in odom_pose)
        if sport_position is not None:
            return "sport", tuple(float(v) for v in sport_position)
        return "none", None

    def _replan(self) -> list[tuple[float, float]] | None:
        pose = self._telemetry.pose
        goal = self._goal_world
        if pose is None or goal is None:
            return None
        occ = self._inflate_occupancy(
            self.map.occupancy_mask(),
            inflation_m=0.35 if self._avoid_enabled else 0.05,
            resolution=self.map.resolution,
        )
        start = self.map.world_to_grid(pose[0], pose[1])
        goal_cell = self.map.world_to_grid(goal[0], goal[1])
        if start is None or goal_cell is None:
            return None
        path_px = self._astar(occ, start, goal_cell)
        if not path_px:
            return None
        world = [self.map.grid_to_world(px, py) for px, py in path_px]
        self._path_world = self._compress_waypoints(world)
        return self._path_world

    @staticmethod
    def _inflate_occupancy(occ: np.ndarray, inflation_m: float, resolution: float = 0.05) -> np.ndarray:
        radius = max(1, int(round(float(inflation_m) / float(resolution))))
        pad = radius
        padded = np.pad(occ.astype(np.uint8), pad, mode="constant")
        out = np.zeros_like(occ, dtype=bool)
        ys, xs = np.where(padded > 0)
        for y, x in zip(ys, xs):
            y0 = max(0, y - radius - pad)
            y1 = min(out.shape[0], y + radius + 1 - pad)
            x0 = max(0, x - radius - pad)
            x1 = min(out.shape[1], x + radius + 1 - pad)
            out[y0:y1, x0:x1] = True
        return out

    @staticmethod
    def _nearest_free(cell: tuple[int, int], occ: np.ndarray, max_radius: int = 20) -> tuple[int, int] | None:
        x, y = cell
        h, w = occ.shape
        if not (0 <= x < w and 0 <= y < h):
            return None
        if not occ[y, x]:
            return (x, y)
        for radius in range(1, max_radius + 1):
            x0 = max(0, x - radius)
            x1 = min(w - 1, x + radius)
            y0 = max(0, y - radius)
            y1 = min(h - 1, y + radius)
            for xx in range(x0, x1 + 1):
                for yy in (y0, y1):
                    if not occ[yy, xx]:
                        return (xx, yy)
            for yy in range(y0 + 1, y1):
                for xx in (x0, x1):
                    if not occ[yy, xx]:
                        return (xx, yy)
        return None

    def _astar(
        self,
        occ: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        import heapq

        start = self._nearest_free(start, occ) or start
        goal = self._nearest_free(goal, occ) or goal
        if occ[start[1], start[0]] or occ[goal[1], goal[0]]:
            return None

        open_set: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {start: 0.0}

        def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
            return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))

        while open_set:
            _f, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            cx, cy = current
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx = cx + dx
                    ny = cy + dy
                    if nx < 0 or ny < 0 or ny >= occ.shape[0] or nx >= occ.shape[1]:
                        continue
                    if occ[ny, nx]:
                        continue
                    if dx != 0 and dy != 0 and (occ[cy, nx] or occ[ny, cx]):
                        continue
                    step = math.hypot(float(dx), float(dy))
                    tentative = g_score[current] + step
                    if tentative < g_score.get((nx, ny), float("inf")):
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative
                        heapq.heappush(open_set, (tentative + heuristic((nx, ny), goal), (nx, ny)))
        return None

    @staticmethod
    def _compress_waypoints(path: list[tuple[float, float]], stride: int = 10) -> list[tuple[float, float]]:
        if not path:
            return []
        out = [path[0]]
        for idx in range(stride, max(1, len(path) - 1), stride):
            out.append(path[idx])
        if out[-1] != path[-1]:
            out.append(path[-1])
        return out

    def _apply_navigation(self, robot: "Robot", pose: tuple[float, float, float] | None) -> None:
        if not self._nav_active or pose is None:
            return
        if not self._path_world and self._goal_world is not None:
            if self._replan() is None:
                self._nav_active = False
                self.log("Navigation stopped: no path available.")
                return

        while self._path_world:
            tx, ty = self._path_world[0]
            if math.hypot(tx - pose[0], ty - pose[1]) > 0.22:
                break
            self._path_world.pop(0)
        if not self._path_world:
            self._nav_active = False
            try:
                robot.stop()
            except Exception:
                pass
            self.log("Goal reached.")
            return

        tx, ty = self._path_world[0]
        dx = tx - pose[0]
        dy = ty - pose[1]
        dist = math.hypot(dx, dy)
        target_yaw = math.atan2(dy, dx)
        err = _wrap_angle(target_yaw - pose[2])
        if abs(err) > 0.80:
            vx = 0.0
            omega = max(-0.32, min(0.32, -0.95 * err))
        else:
            vx = min(0.22, 0.55 * dist)
            if dist < 0.40:
                vx = min(vx, 0.12)
            omega = max(-0.22, min(0.22, -0.70 * err))
            if abs(err) > 0.30:
                vx *= 0.60
        try:
            robot.loco_move(vx=float(vx), vy=0.0, vyaw=float(omega))
        except Exception as exc:
            self._nav_active = False
            self.log(f"Navigation command failed: {exc}")
