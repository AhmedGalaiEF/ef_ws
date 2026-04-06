import argparse
import math
import tkinter as tk
from pathlib import Path
from tkinter import ttk
import xml.etree.ElementTree as ET

import numpy as np


DEFAULT_URDF = (
    Path(__file__).resolve().parent
    / "../G1_rviz_simulation-main/G1_rviz_simulation-main/urdf/g1_29dof_with_hand_rev_1_0_pkg.urdf"
).resolve()

NOMINAL_STAND_POSE = {}


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        return np.eye(3)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def make_transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0), rot=None):
    tf = np.eye(4, dtype=float)
    tf[:3, :3] = rpy_matrix(rpy) if rot is None else rot
    tf[:3, 3] = np.asarray(xyz, dtype=float)
    return tf


class Joint:
    def __init__(self, name, joint_type, parent, child, origin_xyz, origin_rpy, axis, limit):
        self.name = name
        self.type = joint_type
        self.parent = parent
        self.child = child
        self.origin_xyz = origin_xyz
        self.origin_rpy = origin_rpy
        self.axis = axis
        self.limit = limit


class G1UrdfModel:
    def __init__(self, urdf_path):
        self.urdf_path = Path(urdf_path).resolve()
        root = ET.parse(self.urdf_path).getroot()
        self.links = {link.attrib["name"] for link in root.findall("link")}
        self.joints = []
        self.children_by_parent = {}
        child_links = set()

        for joint_node in root.findall("joint"):
            parent = joint_node.find("parent").attrib["link"]
            child = joint_node.find("child").attrib["link"]
            origin_node = joint_node.find("origin")
            axis_node = joint_node.find("axis")
            limit_node = joint_node.find("limit")

            origin_xyz = self._parse_vec(origin_node.attrib.get("xyz", "0 0 0")) if origin_node is not None else np.zeros(3)
            origin_rpy = self._parse_vec(origin_node.attrib.get("rpy", "0 0 0")) if origin_node is not None else np.zeros(3)
            axis = self._parse_vec(axis_node.attrib.get("xyz", "0 0 1")) if axis_node is not None else np.array([0.0, 0.0, 1.0])
            limit = None
            if limit_node is not None and "lower" in limit_node.attrib and "upper" in limit_node.attrib:
                limit = (float(limit_node.attrib["lower"]), float(limit_node.attrib["upper"]))

            joint = Joint(
                name=joint_node.attrib["name"],
                joint_type=joint_node.attrib["type"],
                parent=parent,
                child=child,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
                limit=limit,
            )
            self.joints.append(joint)
            self.children_by_parent.setdefault(parent, []).append(joint)
            child_links.add(child)

        root_links = sorted(self.links - child_links)
        if not root_links:
            raise ValueError(f"No root link found in {self.urdf_path}")
        self.root_link = root_links[0]
        self.actuated_joints = [joint for joint in self.joints if joint.type != "fixed"]

    @staticmethod
    def _parse_vec(text):
        return np.array([float(value) for value in text.split()], dtype=float)

    def forward_kinematics(self, joint_positions):
        link_transforms = {self.root_link: np.eye(4, dtype=float)}
        joint_frames = {}
        edges = []

        def walk(parent_link):
            parent_tf = link_transforms[parent_link]
            for joint in self.children_by_parent.get(parent_link, []):
                joint_origin_tf = parent_tf @ make_transform(joint.origin_xyz, joint.origin_rpy)
                if joint.type in ("revolute", "continuous"):
                    angle = joint_positions.get(joint.name, 0.0)
                    child_tf = joint_origin_tf @ make_transform(rot=axis_angle_matrix(joint.axis, angle))
                elif joint.type == "prismatic":
                    distance = joint_positions.get(joint.name, 0.0)
                    child_tf = joint_origin_tf @ make_transform(xyz=joint.axis * distance)
                else:
                    child_tf = joint_origin_tf

                joint_frames[joint.name] = child_tf
                link_transforms[joint.child] = child_tf
                edges.append((parent_link, joint.child))
                walk(joint.child)

        walk(self.root_link)
        return link_transforms, joint_frames, edges


class G1Viewer:
    def __init__(self, model, show_fixed):
        self.model = model
        self.show_fixed = show_fixed
        self.joint_values = {joint.name: 0.0 for joint in self.model.actuated_joints}
        self.camera_yaw = math.radians(55.0)
        self.camera_pitch = math.radians(-16.0)
        self.camera_distance = 2.8
        self.camera_target = np.array([0.0, 0.0, 0.7], dtype=float)
        self.drag_start = None

        self.root = tk.Tk()
        self.root.title("G1 URDF Viewer")
        self.root.geometry("1500x940")
        self.show_labels = tk.BooleanVar(master=self.root, value=False)
        self.show_root_axes = tk.BooleanVar(master=self.root, value=True)
        self._build_ui()
        self.set_pose(NOMINAL_STAND_POSE)
        self.redraw()

    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=10)
        controls.grid(row=0, column=0, sticky="ns")
        controls.columnconfigure(0, weight=1)

        canvas_frame = ttk.Frame(self.root, padding=(0, 0, 10, 10))
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#0e1116", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonPress-3>", self._on_right_press)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

        ttk.Label(controls, text=f"URDF: {self.model.urdf_path.name}").grid(row=0, column=0, sticky="w")
        ttk.Label(
            controls,
            text="Left-drag orbit, right-drag pan, wheel zoom",
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))

        toggles = ttk.Frame(controls)
        toggles.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ttk.Checkbutton(toggles, text="Joint labels", variable=self.show_labels, command=self.redraw).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(toggles, text="Base frame", variable=self.show_root_axes, command=self.redraw).grid(row=1, column=0, sticky="w")

        pose_buttons = ttk.Frame(controls)
        pose_buttons.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(pose_buttons, text="Nominal pose", command=lambda: self.set_pose(NOMINAL_STAND_POSE)).grid(row=0, column=0, sticky="ew")
        ttk.Button(pose_buttons, text="Zero pose", command=self.zero_pose).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        slider_frame = ttk.Frame(controls)
        slider_frame.grid(row=4, column=0, sticky="nsew")
        slider_frame.columnconfigure(0, weight=1)

        self.slider_vars = {}
        row = 0
        for joint in self.model.actuated_joints:
            limit = joint.limit if joint.limit is not None else (-math.pi, math.pi)
            ttk.Label(slider_frame, text=joint.name).grid(row=row, column=0, sticky="w")
            row += 1

            slider_var = tk.DoubleVar(value=0.0)
            slider = tk.Scale(
                slider_frame,
                from_=math.degrees(limit[0]),
                to=math.degrees(limit[1]),
                orient="horizontal",
                resolution=0.1,
                variable=slider_var,
                command=lambda _value, name=joint.name, var=slider_var: self._on_slider(name, var),
                length=320,
            )
            slider.grid(row=row, column=0, sticky="ew", pady=(0, 8))
            self.slider_vars[joint.name] = slider_var
            row += 1

    def set_pose(self, pose):
        for joint in self.model.actuated_joints:
            value = pose.get(joint.name, 0.0)
            self.joint_values[joint.name] = value
            self.slider_vars[joint.name].set(math.degrees(value))
        self.redraw()

    def zero_pose(self):
        self.set_pose({})

    def _on_slider(self, joint_name, var):
        self.joint_values[joint_name] = math.radians(var.get())
        self.redraw()

    def _on_left_press(self, event):
        self.drag_start = ("orbit", event.x, event.y)

    def _on_left_drag(self, event):
        if self.drag_start is None:
            return
        _, last_x, last_y = self.drag_start
        self.camera_yaw += (event.x - last_x) * 0.008
        self.camera_pitch = max(-1.45, min(1.45, self.camera_pitch + (event.y - last_y) * 0.008))
        self.drag_start = ("orbit", event.x, event.y)
        self.redraw()

    def _on_right_press(self, event):
        self.drag_start = ("pan", event.x, event.y)

    def _on_right_drag(self, event):
        if self.drag_start is None:
            return
        _, last_x, last_y = self.drag_start
        dx = event.x - last_x
        dy = event.y - last_y
        right, up, _ = self._camera_basis()
        self.camera_target -= right * dx * 0.002
        self.camera_target += up * dy * 0.002
        self.drag_start = ("pan", event.x, event.y)
        self.redraw()

    def _on_mousewheel(self, event):
        self.camera_distance = max(0.6, min(10.0, self.camera_distance * (0.92 if event.delta > 0 else 1.08)))
        self.redraw()

    def _camera_basis(self):
        cam_pos = np.array(
            [
                self.camera_distance * math.cos(self.camera_pitch) * math.cos(self.camera_yaw),
                self.camera_distance * math.cos(self.camera_pitch) * math.sin(self.camera_yaw),
                self.camera_distance * math.sin(self.camera_pitch),
            ],
            dtype=float,
        ) + self.camera_target
        forward = self.camera_target - cam_pos
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-8:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        return right, up, forward

    def _view_matrix(self):
        right, up, forward = self._camera_basis()
        cam_pos = self.camera_target - forward * self.camera_distance
        view = np.eye(4, dtype=float)
        view[:3, :3] = np.vstack([right, up, -forward])
        view[:3, 3] = -view[:3, :3] @ cam_pos
        return view

    def _project(self, point_world):
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        point_cam = self._view_matrix() @ np.append(point_world, 1.0)
        z = point_cam[2]
        if z >= -0.05:
            return None
        focal = 0.9 * min(width, height)
        x = (point_cam[0] / -z) * focal + width * 0.5
        y = (-point_cam[1] / -z) * focal + height * 0.5
        return x, y

    def _draw_line_3d(self, p0, p1, color, width=2):
        s0 = self._project(np.asarray(p0, dtype=float))
        s1 = self._project(np.asarray(p1, dtype=float))
        if s0 is None or s1 is None:
            return
        self.canvas.create_line(s0[0], s0[1], s1[0], s1[1], fill=color, width=width)

    def _draw_frame(self, transform, scale, label=None):
        origin = transform[:3, 3]
        rotation = transform[:3, :3]
        for color, axis in (
            ("#ff5c5c", rotation[:, 0]),
            ("#4bd47a", rotation[:, 1]),
            ("#57a6ff", rotation[:, 2]),
        ):
            self._draw_line_3d(origin, origin + axis * scale, color=color, width=2)
        if label and self.show_labels.get():
            screen = self._project(origin)
            if screen is not None:
                self.canvas.create_text(
                    screen[0] + 8,
                    screen[1] - 8,
                    text=label,
                    fill="#d7dde8",
                    anchor="sw",
                    font=("TkDefaultFont", 9),
                )

    def _draw_ground_hint(self):
        span = 0.8
        steps = 6
        for i in range(-steps, steps + 1):
            offset = i * span / steps
            self._draw_line_3d(np.array([-span, offset, 0.0]), np.array([span, offset, 0.0]), "#28303b", width=1)
            self._draw_line_3d(np.array([offset, -span, 0.0]), np.array([offset, span, 0.0]), "#28303b", width=1)

    def redraw(self):
        self.canvas.delete("all")
        self._draw_ground_hint()
        link_transforms, joint_frames, edges = self.model.forward_kinematics(self.joint_values)

        if self.show_root_axes.get():
            self._draw_frame(link_transforms[self.model.root_link], scale=0.08, label=self.model.root_link)

        for parent, child in edges:
            self._draw_line_3d(link_transforms[parent][:3, 3], link_transforms[child][:3, 3], color="#d2d7de", width=4)

        for joint in self.model.joints:
            if joint.type == "fixed" and not self.show_fixed:
                continue
            self._draw_frame(joint_frames[joint.name], scale=0.035, label=joint.name)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Standalone G1 URDF visualizer without MuJoCo.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help=f"URDF file to load (default: {DEFAULT_URDF})")
    parser.add_argument("--show-fixed-joints", action="store_true", help="Draw coordinate frames for fixed joints in addition to actuated joints.")
    args = parser.parse_args()

    model = G1UrdfModel(args.urdf)
    viewer = G1Viewer(model=model, show_fixed=args.show_fixed_joints)
    viewer.run()


if __name__ == "__main__":
    main()
