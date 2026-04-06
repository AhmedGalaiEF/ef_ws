import argparse
import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

import numpy as np

from g1_urdf_viewer import DEFAULT_URDF, G1UrdfModel, NOMINAL_STAND_POSE


def skew(vec):
    x, y, z = vec
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def exp_so3(omega_dt):
    angle = np.linalg.norm(omega_dt)
    if angle < 1e-9:
        return np.eye(3) + skew(omega_dt)
    axis = omega_dt / angle
    k = skew(axis)
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


@dataclass
class JointCommand:
    q: float
    dq: float
    tau: float
    kp: float
    kd: float


class G1PhysicsModel:
    def __init__(self, urdf_path):
        self.urdf = G1UrdfModel(urdf_path)
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.contact_k = 18000.0
        self.contact_c = 260.0
        self.tangent_damping = 120.0
        self.friction_coeff = 0.85
        self.dt = 0.002
        self.passive_joint_damping = 0.35

        self.link_inertials = self._parse_inertials()
        self.contact_offsets_by_link = self._parse_contact_offsets()
        self.total_mass = sum(inertial["mass"] for inertial in self.link_inertials.values())
        if self.total_mass <= 0.0:
            raise ValueError("URDF mass properties are missing or invalid")

        self.joint_lookup = {joint.name: joint for joint in self.urdf.actuated_joints}
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_commands = {}
        self.last_joint_torques = {}

        self.link_com_body = {}
        self.foot_points_body = []
        self.com_body = np.zeros(3, dtype=float)
        self.inertia_body = np.eye(3, dtype=float)
        self.current_link_transforms_body = {}
        self.current_joint_frames_body = {}
        self.current_edges = []
        self._initialize_joint_state()
        self.reset()

    def _parse_inertials(self):
        import xml.etree.ElementTree as ET

        root = ET.parse(self.urdf.urdf_path).getroot()
        inertials = {}
        for link_node in root.findall("link"):
            inertial = link_node.find("inertial")
            if inertial is None:
                continue
            mass_node = inertial.find("mass")
            if mass_node is None:
                continue
            origin_node = inertial.find("origin")
            xyz = np.zeros(3, dtype=float)
            if origin_node is not None and "xyz" in origin_node.attrib:
                xyz = np.array([float(value) for value in origin_node.attrib["xyz"].split()], dtype=float)
            inertials[link_node.attrib["name"]] = {"mass": float(mass_node.attrib["value"]), "xyz": xyz}
        return inertials

    def _parse_contact_offsets(self):
        import xml.etree.ElementTree as ET

        root = ET.parse(self.urdf.urdf_path).getroot()
        offsets = {}
        for link_node in root.findall("link"):
            link_name = link_node.attrib["name"]
            if link_name not in ("left_ankle_roll_link", "right_ankle_roll_link"):
                continue
            for index, collision_node in enumerate(link_node.findall("collision")):
                origin_node = collision_node.find("origin")
                xyz = np.zeros(3, dtype=float)
                if origin_node is not None and "xyz" in origin_node.attrib:
                    xyz = np.array([float(value) for value in origin_node.attrib["xyz"].split()], dtype=float)
                offsets.setdefault(link_name, []).append((f"{link_name}_{index}", xyz))
        return offsets

    def _default_hold_gains(self, joint_name):
        if "hip" in joint_name or "knee" in joint_name:
            return 80.0, 6.0
        if "ankle" in joint_name or "waist" in joint_name:
            return 50.0, 4.0
        if "shoulder" in joint_name or "elbow" in joint_name:
            return 35.0, 3.0
        if "wrist" in joint_name:
            return 18.0, 1.5
        return 8.0, 0.8

    def _initialize_joint_state(self):
        for joint in self.urdf.actuated_joints:
            q0 = NOMINAL_STAND_POSE.get(joint.name, 0.0)
            kp, kd = self._default_hold_gains(joint.name)
            self.joint_positions[joint.name] = q0
            self.joint_velocities[joint.name] = 0.0
            self.joint_commands[joint.name] = JointCommand(q=q0, dq=0.0, tau=0.0, kp=kp, kd=kd)
            self.last_joint_torques[joint.name] = 0.0

    def _joint_inertia(self, joint_name):
        if "hip" in joint_name or "knee" in joint_name:
            return 0.06
        if "ankle" in joint_name or "waist" in joint_name:
            return 0.035
        if "shoulder" in joint_name or "elbow" in joint_name:
            return 0.02
        if "wrist" in joint_name:
            return 0.01
        return 0.004

    def _update_body_properties(self):
        link_transforms, joint_frames, edges = self.urdf.forward_kinematics(self.joint_positions)
        self.current_link_transforms_body = link_transforms
        self.current_joint_frames_body = joint_frames
        self.current_edges = edges

        com_accum = np.zeros(3, dtype=float)
        self.link_com_body = {}
        for link_name, link_tf in link_transforms.items():
            if link_name not in self.link_inertials:
                continue
            mass = self.link_inertials[link_name]["mass"]
            local_com = self.link_inertials[link_name]["xyz"]
            com_pos = link_tf[:3, :3] @ local_com + link_tf[:3, 3]
            self.link_com_body[link_name] = com_pos
            com_accum += mass * com_pos

        self.com_body = com_accum / self.total_mass

        inertia = np.zeros((3, 3), dtype=float)
        for link_name, com_pos in self.link_com_body.items():
            mass = self.link_inertials[link_name]["mass"]
            offset = com_pos - self.com_body
            inertia += mass * ((offset @ offset) * np.eye(3) - np.outer(offset, offset))
        inertia += np.diag([0.3, 0.28, 0.15])
        self.inertia_body = inertia

        self.foot_points_body = []
        for link_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            if link_name not in link_transforms:
                continue
            link_tf = link_transforms[link_name]
            contact_offsets = self.contact_offsets_by_link.get(link_name, [(link_name, np.zeros(3, dtype=float))])
            for contact_name, offset in contact_offsets:
                point = link_tf[:3, :3] @ offset + link_tf[:3, 3] - self.com_body
                self.foot_points_body.append((contact_name, point))

    def reset(self):
        for joint in self.urdf.actuated_joints:
            q0 = NOMINAL_STAND_POSE.get(joint.name, 0.0)
            kp, kd = self._default_hold_gains(joint.name)
            self.joint_positions[joint.name] = q0
            self.joint_velocities[joint.name] = 0.0
            self.joint_commands[joint.name] = JointCommand(q=q0, dq=0.0, tau=0.0, kp=kp, kd=kd)
            self.last_joint_torques[joint.name] = 0.0

        self._update_body_properties()
        foot_z = [point_body[2] for _, point_body in self.foot_points_body]
        base_height = -min(foot_z)

        self.com_position = np.array([0.0, 0.0, base_height], dtype=float)
        self.com_velocity = np.zeros(3, dtype=float)
        self.rotation = np.eye(3, dtype=float)
        self.omega_body = np.zeros(3, dtype=float)
        self.sim_time = 0.0
        self.last_contacts = []
        self.last_force_world = np.zeros(3, dtype=float)

    def issue_joint_command(self, joint_name, q, dq, tau, kp, kd):
        if joint_name not in self.joint_lookup:
            raise KeyError(f"Unknown joint {joint_name}")
        self.joint_commands[joint_name] = JointCommand(q=q, dq=dq, tau=tau, kp=kp, kd=kd)

    def hold_joint(self, joint_name):
        kp, kd = self._default_hold_gains(joint_name)
        self.issue_joint_command(
            joint_name=joint_name,
            q=self.joint_positions[joint_name],
            dq=0.0,
            tau=0.0,
            kp=kp,
            kd=kd,
        )

    def _step_joint_dynamics(self):
        for joint in self.urdf.actuated_joints:
            name = joint.name
            q = self.joint_positions[name]
            dq = self.joint_velocities[name]
            cmd = self.joint_commands[name]

            applied_tau = cmd.tau + cmd.kp * (cmd.q - q) + cmd.kd * (cmd.dq - dq) - self.passive_joint_damping * dq
            qdd = applied_tau / self._joint_inertia(name)
            dq += qdd * self.dt
            q += dq * self.dt

            if joint.limit is not None:
                lower, upper = joint.limit
                if q < lower:
                    q = lower
                    dq = 0.0
                elif q > upper:
                    q = upper
                    dq = 0.0

            self.joint_positions[name] = q
            self.joint_velocities[name] = dq
            self.last_joint_torques[name] = applied_tau

    def step(self):
        self._step_joint_dynamics()
        self._update_body_properties()

        force_world = self.total_mass * self.gravity
        torque_world = np.zeros(3, dtype=float)
        contacts = []
        omega_world = self.rotation @ self.omega_body

        for foot_name, point_body in self.foot_points_body:
            point_world = self.com_position + self.rotation @ point_body
            point_vel = self.com_velocity + np.cross(omega_world, self.rotation @ point_body)
            penetration = max(0.0, -point_world[2])
            if penetration <= 0.0 and point_vel[2] >= 0.0:
                continue

            normal_force = max(0.0, self.contact_k * penetration - self.contact_c * point_vel[2])
            tangent_vel = point_vel.copy()
            tangent_vel[2] = 0.0
            tangent_force = -self.tangent_damping * tangent_vel
            tangent_limit = self.friction_coeff * normal_force
            tangent_norm = np.linalg.norm(tangent_force[:2])
            if tangent_norm > tangent_limit > 0.0:
                tangent_force[:2] *= tangent_limit / tangent_norm

            contact_force = tangent_force + np.array([0.0, 0.0, normal_force], dtype=float)
            force_world += contact_force
            torque_world += np.cross(self.rotation @ point_body, contact_force)
            contacts.append((foot_name, point_world, penetration))

        self.com_velocity += (force_world / self.total_mass) * self.dt
        self.com_position += self.com_velocity * self.dt

        torque_body = self.rotation.T @ torque_world
        gyro = np.cross(self.omega_body, self.inertia_body @ self.omega_body)
        self.omega_body += np.linalg.solve(self.inertia_body, torque_body - gyro) * self.dt
        self.rotation = self.rotation @ exp_so3(self.omega_body * self.dt)
        u, _, vh = np.linalg.svd(self.rotation)
        self.rotation = u @ vh

        self.sim_time += self.dt
        self.last_contacts = contacts
        self.last_force_world = force_world

    def world_link_transforms(self):
        def world_tf(local_tf):
            tf = np.eye(4, dtype=float)
            tf[:3, :3] = self.rotation @ local_tf[:3, :3]
            tf[:3, 3] = self.com_position + self.rotation @ (local_tf[:3, 3] - self.com_body)
            return tf

        link_world = {name: world_tf(tf) for name, tf in self.current_link_transforms_body.items()}
        joint_world = {name: world_tf(tf) for name, tf in self.current_joint_frames_body.items()}
        return link_world, joint_world, self.current_edges


class G1PhysicsViewer:
    def __init__(self, physics, show_fixed):
        self.physics = physics
        self.show_fixed = show_fixed
        self.camera_yaw = math.radians(55.0)
        self.camera_pitch = math.radians(-16.0)
        self.camera_distance = 3.0
        self.camera_target = np.array([0.0, 0.0, 0.75], dtype=float)
        self.drag_start = None
        self.running = True
        self.sim_substeps = 5

        self.root = tk.Tk()
        self.root.title("G1 URDF Physics Viewer")
        self.root.geometry("1520x960")
        self.show_labels = tk.BooleanVar(master=self.root, value=False)
        self.show_joint_frames = tk.BooleanVar(master=self.root, value=True)
        self.show_contact_points = tk.BooleanVar(master=self.root, value=True)
        self.selected_joint = tk.StringVar(master=self.root, value=self.physics.urdf.actuated_joints[0].name)
        self.cmd_q_var = tk.StringVar(master=self.root, value="")
        self.cmd_dq_var = tk.StringVar(master=self.root, value="")
        self.cmd_tau_var = tk.StringVar(master=self.root, value="")
        self.cmd_kp_var = tk.StringVar(master=self.root, value="")
        self.cmd_kd_var = tk.StringVar(master=self.root, value="")
        self.command_status = tk.StringVar(master=self.root, value="")
        self._build_ui()
        self._populate_command_fields()
        self.redraw()
        self._tick()

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

        ttk.Label(controls, text=f"URDF: {self.physics.urdf.urdf_path.name}").grid(row=0, column=0, sticky="w")
        ttk.Label(controls, text="Gravity + flat-floor contacts on the ankle roll links", wraplength=320).grid(row=1, column=0, sticky="w", pady=(0, 8))
        ttk.Label(controls, text="Left-drag orbit, right-drag pan, wheel zoom").grid(row=2, column=0, sticky="w", pady=(0, 8))

        buttons = ttk.Frame(controls)
        buttons.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(buttons, text="Pause / Resume", command=self.toggle_running).grid(row=0, column=0, sticky="ew")
        ttk.Button(buttons, text="Reset", command=self.reset).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        toggles = ttk.Frame(controls)
        toggles.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        ttk.Checkbutton(toggles, text="Joint labels", variable=self.show_labels, command=self.redraw).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(toggles, text="Joint frames", variable=self.show_joint_frames, command=self.redraw).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(toggles, text="Contact points", variable=self.show_contact_points, command=self.redraw).grid(row=2, column=0, sticky="w")

        stats_frame = ttk.LabelFrame(controls, text="State", padding=8)
        stats_frame.grid(row=5, column=0, sticky="ew")
        self.stats_var = tk.StringVar(master=self.root, value="")
        ttk.Label(stats_frame, textvariable=self.stats_var, justify="left").grid(row=0, column=0, sticky="w")

        command_frame = ttk.LabelFrame(controls, text="LowCmd-style Joint Command", padding=8)
        command_frame.grid(row=6, column=0, sticky="ew", pady=(10, 0))
        command_frame.columnconfigure(1, weight=1)

        ttk.Label(command_frame, text="joint").grid(row=0, column=0, sticky="w")
        joint_names = [joint.name for joint in self.physics.urdf.actuated_joints]
        joint_box = ttk.Combobox(command_frame, textvariable=self.selected_joint, values=joint_names, state="readonly")
        joint_box.grid(row=0, column=1, sticky="ew", pady=(0, 4))
        joint_box.bind("<<ComboboxSelected>>", lambda _event: self._populate_command_fields())

        for row, (label, var) in enumerate(
            [("q", self.cmd_q_var), ("dq", self.cmd_dq_var), ("tau", self.cmd_tau_var), ("kp", self.cmd_kp_var), ("kd", self.cmd_kd_var)],
            start=1,
        ):
            ttk.Label(command_frame, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(command_frame, textvariable=var).grid(row=row, column=1, sticky="ew", pady=(0, 4))

        ttk.Button(command_frame, text="Send Command", command=self._apply_joint_command).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(command_frame, text="Hold Current Joint", command=self._hold_selected_joint).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(command_frame, textvariable=self.command_status, justify="left", wraplength=320).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def toggle_running(self):
        self.running = not self.running

    def reset(self):
        self.physics.reset()
        self._populate_command_fields()
        self.redraw()

    def _populate_command_fields(self):
        joint_name = self.selected_joint.get()
        cmd = self.physics.joint_commands[joint_name]
        self.cmd_q_var.set(f"{cmd.q:.6f}")
        self.cmd_dq_var.set(f"{cmd.dq:.6f}")
        self.cmd_tau_var.set(f"{cmd.tau:.6f}")
        self.cmd_kp_var.set(f"{cmd.kp:.6f}")
        self.cmd_kd_var.set(f"{cmd.kd:.6f}")
        self.command_status.set("")

    def _apply_joint_command(self):
        joint_name = self.selected_joint.get()
        try:
            q = float(self.cmd_q_var.get())
            dq = float(self.cmd_dq_var.get())
            tau = float(self.cmd_tau_var.get())
            kp = float(self.cmd_kp_var.get())
            kd = float(self.cmd_kd_var.get())
        except ValueError:
            self.command_status.set("Command rejected: all fields must be valid floats.")
            return
        self.physics.issue_joint_command(joint_name, q=q, dq=dq, tau=tau, kp=kp, kd=kd)
        self.command_status.set(f"Sent {joint_name}: q={q:.3f}, dq={dq:.3f}, tau={tau:.3f}, kp={kp:.3f}, kd={kd:.3f}")

    def _hold_selected_joint(self):
        joint_name = self.selected_joint.get()
        self.physics.hold_joint(joint_name)
        self._populate_command_fields()
        self.command_status.set(f"Holding {joint_name} at current q with default gains.")

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
        self.camera_distance = max(0.7, min(10.0, self.camera_distance * (0.92 if event.delta > 0 else 1.08)))
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
        s0 = self._project(p0)
        s1 = self._project(p1)
        if s0 is None or s1 is None:
            return
        self.canvas.create_line(s0[0], s0[1], s1[0], s1[1], fill=color, width=width)

    def _draw_frame(self, tf, scale, label=None):
        origin = tf[:3, 3]
        rot = tf[:3, :3]
        for color, axis in (("#ff5c5c", rot[:, 0]), ("#4bd47a", rot[:, 1]), ("#57a6ff", rot[:, 2])):
            self._draw_line_3d(origin, origin + axis * scale, color, width=2)
        if label and self.show_labels.get():
            screen = self._project(origin)
            if screen is not None:
                self.canvas.create_text(screen[0] + 8, screen[1] - 8, text=label, fill="#d7dde8", anchor="sw", font=("TkDefaultFont", 9))

    def _draw_floor(self):
        span = 1.0
        steps = 10
        for i in range(-steps, steps + 1):
            offset = i * span / steps
            self._draw_line_3d(np.array([-span, offset, 0.0]), np.array([span, offset, 0.0]), "#28303b", width=1)
            self._draw_line_3d(np.array([offset, -span, 0.0]), np.array([offset, span, 0.0]), "#28303b", width=1)

    def _update_stats(self):
        pos = self.physics.com_position
        vel = self.physics.com_velocity
        omega = self.physics.omega_body
        contacts = len(self.physics.last_contacts)
        joint_name = self.selected_joint.get()
        q = self.physics.joint_positions[joint_name]
        dq = self.physics.joint_velocities[joint_name]
        tau = self.physics.last_joint_torques[joint_name]
        self.stats_var.set(
            f"time: {self.physics.sim_time:6.3f} s\n"
            f"com:  [{pos[0]: .3f}, {pos[1]: .3f}, {pos[2]: .3f}] m\n"
            f"vel:  [{vel[0]: .3f}, {vel[1]: .3f}, {vel[2]: .3f}] m/s\n"
            f"omega:[{omega[0]: .3f}, {omega[1]: .3f}, {omega[2]: .3f}] rad/s\n"
            f"contacts: {contacts}\n"
            f"{joint_name}: q={q: .3f}, dq={dq: .3f}, tau={tau: .3f}\n"
            f"mode: {'running' if self.running else 'paused'}"
        )

    def redraw(self):
        self.canvas.delete("all")
        self._draw_floor()
        link_world, joint_world, edges = self.physics.world_link_transforms()

        for parent, child in edges:
            self._draw_line_3d(link_world[parent][:3, 3], link_world[child][:3, 3], color="#d2d7de", width=3)

        self._draw_frame(link_world[self.physics.urdf.root_link], scale=0.09, label=self.physics.urdf.root_link)

        if self.show_joint_frames.get():
            for joint in self.physics.urdf.joints:
                if joint.type == "fixed" and not self.show_fixed:
                    continue
                self._draw_frame(joint_world[joint.name], scale=0.032, label=joint.name)

        if self.show_contact_points.get():
            for foot_name, point_world, penetration in self.physics.last_contacts:
                screen = self._project(point_world)
                if screen is None:
                    continue
                r = 4
                self.canvas.create_oval(screen[0] - r, screen[1] - r, screen[0] + r, screen[1] + r, fill="#ffcf5c", outline="")
                if self.show_labels.get():
                    self.canvas.create_text(screen[0] + 8, screen[1] - 8, text=f"{foot_name} {penetration*1000:.1f} mm", fill="#ffcf5c", anchor="sw", font=("TkDefaultFont", 9))

        self._update_stats()

    def _tick(self):
        if self.running:
            for _ in range(self.sim_substeps):
                self.physics.step()
        self.redraw()
        self.root.after(int(self.physics.dt * self.sim_substeps * 1000), self._tick)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Standalone G1 viewer with simple gravity/contact physics.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help=f"URDF file to load (default: {DEFAULT_URDF})")
    parser.add_argument("--show-fixed-joints", action="store_true", help="Draw coordinate frames for fixed joints as well.")
    args = parser.parse_args()

    physics = G1PhysicsModel(args.urdf)
    viewer = G1PhysicsViewer(physics=physics, show_fixed=args.show_fixed_joints)
    viewer.run()


if __name__ == "__main__":
    main()
