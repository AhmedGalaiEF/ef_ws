from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import struct
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go


LEFT_ARM_7DOF_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

LEFT_HAND_7DOF_JOINTS = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]

LEFT_ARM_HAND_14DOF_JOINTS = LEFT_ARM_7DOF_JOINTS + LEFT_HAND_7DOF_JOINTS

LEFT_LEG_6DOF_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]

LEFT_ARM_VISIBLE_LINKS = [
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_hand_palm_link",
    "left_hand_thumb_0_link",
    "left_hand_thumb_1_link",
    "left_hand_thumb_2_link",
    "left_hand_middle_0_link",
    "left_hand_middle_1_link",
    "left_hand_index_0_link",
    "left_hand_index_1_link",
]

LEFT_LEG_VISIBLE_LINKS = [
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
]

# Right leg: 3 hip (pitch, roll, yaw) + 1 knee + 2 ankle (pitch, roll) = 6 DOF
RIGHT_LEG_6DOF_JOINTS = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

RIGHT_LEG_VISIBLE_LINKS = [
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
]

RIGHT_ARM_7DOF_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

RIGHT_HAND_7DOF_JOINTS = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

RIGHT_ARM_HAND_14DOF_JOINTS = RIGHT_ARM_7DOF_JOINTS + RIGHT_HAND_7DOF_JOINTS

RIGHT_ARM_VISIBLE_LINKS = [
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_hand_palm_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
]

WAIST_3DOF_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

# Full body: 2×(3 hip + 1 knee + 2 ankle) + 3 waist + 2×7 arm = 29 DOF
FULL_BODY_29DOF_JOINTS = (
    LEFT_LEG_6DOF_JOINTS + RIGHT_LEG_6DOF_JOINTS
    + WAIST_3DOF_JOINTS
    + LEFT_ARM_7DOF_JOINTS + RIGHT_ARM_7DOF_JOINTS
)

FULL_BODY_VISIBLE_LINKS = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_yaw_link", "waist_roll_link", "torso_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link",
    "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link",
    "left_wrist_yaw_link", "left_hand_palm_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link",
    "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link",
    "right_wrist_yaw_link", "right_hand_palm_link",
]

# Approximate bounding-sphere radii per link (metres) for collision checking
_LINK_SPHERE_RADII: dict[str, float] = {
    "pelvis": 0.12, "torso_link": 0.14,
    "waist_yaw_link": 0.09, "waist_roll_link": 0.09,
    "left_hip_pitch_link": 0.07, "left_hip_roll_link": 0.07, "left_hip_yaw_link": 0.07,
    "left_knee_link": 0.065, "left_ankle_pitch_link": 0.05, "left_ankle_roll_link": 0.05,
    "right_hip_pitch_link": 0.07, "right_hip_roll_link": 0.07, "right_hip_yaw_link": 0.07,
    "right_knee_link": 0.065, "right_ankle_pitch_link": 0.05, "right_ankle_roll_link": 0.05,
    "left_shoulder_pitch_link": 0.07, "left_shoulder_roll_link": 0.06,
    "left_shoulder_yaw_link": 0.06, "left_elbow_link": 0.055,
    "left_wrist_roll_link": 0.045, "left_wrist_pitch_link": 0.04,
    "left_wrist_yaw_link": 0.04, "left_hand_palm_link": 0.05,
    "right_shoulder_pitch_link": 0.07, "right_shoulder_roll_link": 0.06,
    "right_shoulder_yaw_link": 0.06, "right_elbow_link": 0.055,
    "right_wrist_roll_link": 0.045, "right_wrist_pitch_link": 0.04,
    "right_wrist_yaw_link": 0.04, "right_hand_palm_link": 0.05,
}


def find_g1_urdf(search_root: Path | None = None) -> Path:
    search_root = (search_root or Path.cwd()).resolve()
    explicit_candidates = [
        search_root / "../G1_rviz_simulation-main/G1_rviz_simulation-main/install/g1_description/share/g1_description/urdf/g1_29dof_with_hand_rev_1_0_pkg.urdf",
        search_root / "G1_rviz_simulation-main/G1_rviz_simulation-main/install/g1_description/share/g1_description/urdf/g1_29dof_with_hand_rev_1_0_pkg.urdf",
    ]
    for candidate in explicit_candidates:
        if candidate.exists():
            return candidate.resolve()
    for root in [search_root, search_root.parent]:
        matches = sorted(root.rglob("g1_29dof_with_hand_rev_1_0_pkg.urdf"))
        if matches:
            return matches[0].resolve()
    raise FileNotFoundError("Could not find g1_29dof_with_hand_rev_1_0_pkg.urdf in this workspace.")


URDF_PATH = find_g1_urdf()
PACKAGE_DIR = URDF_PATH.parent.parent


def _float3(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=float)
    return np.asarray([float(x) for x in text.split()], dtype=float)


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    return rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(3)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=float,
    )


def make_transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=float)
    if rotation is not None:
        T[:3, :3] = rotation
    if translation is not None:
        T[:3, 3] = translation
    return T


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    flat = pts.reshape(-1, 3)
    out = flat @ transform[:3, :3].T + transform[:3, 3]
    return out.reshape(pts.shape)


def rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=float)


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    R_err = target @ current.T
    return 0.5 * np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ],
        dtype=float,
    )


@dataclass
class Visual:
    mesh_path: Path
    origin: np.ndarray
    color: tuple[float, float, float, float]
    scale: np.ndarray


@dataclass
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower_limit: float
    upper_limit: float


@dataclass
class RobotModel:
    name: str
    links: list[str]
    joints: list[Joint]
    visuals: dict[str, list[Visual]]
    parent_joint_by_link: dict[str, Joint]
    child_joints_by_link: dict[str, list[Joint]]
    root_link: str


def _parse_color(material_elem: ET.Element | None) -> tuple[float, float, float, float]:
    if material_elem is None:
        return (0.7, 0.7, 0.7, 1.0)
    color_elem = material_elem.find("color")
    if color_elem is None:
        return (0.7, 0.7, 0.7, 1.0)
    rgba = [float(x) for x in color_elem.attrib.get("rgba", "0.7 0.7 0.7 1.0").split()]
    return tuple(rgba[:4])


def _visual_origin(visual_elem: ET.Element) -> np.ndarray:
    origin = visual_elem.find("origin")
    xyz = _float3(origin.attrib.get("xyz") if origin is not None else None)
    rpy = _float3(origin.attrib.get("rpy") if origin is not None else None)
    return make_transform(rpy_matrix(rpy), xyz)


def _mesh_scale(mesh_elem: ET.Element) -> np.ndarray:
    return _float3(mesh_elem.attrib.get("scale"), default=(1.0, 1.0, 1.0))


def load_robot_model(urdf_path: Path = URDF_PATH) -> RobotModel:
    root = ET.parse(urdf_path).getroot()
    links = [link.attrib["name"] for link in root.findall("link")]
    material_library = {mat.attrib["name"]: _parse_color(mat) for mat in root.findall("material")}

    visuals: dict[str, list[Visual]] = {name: [] for name in links}
    for link_elem in root.findall("link"):
        link_name = link_elem.attrib["name"]
        for visual_elem in link_elem.findall("visual"):
            geometry = visual_elem.find("geometry")
            if geometry is None:
                continue
            mesh_elem = geometry.find("mesh")
            if mesh_elem is None:
                continue
            filename = mesh_elem.attrib["filename"].replace("package://g1_description/", f"{PACKAGE_DIR.as_posix()}/")
            material_elem = visual_elem.find("material")
            if material_elem is not None and "name" in material_elem.attrib and material_elem.find("color") is None:
                color = material_library.get(material_elem.attrib["name"], (0.7, 0.7, 0.7, 1.0))
            else:
                color = _parse_color(material_elem)
            visuals[link_name].append(
                Visual(
                    mesh_path=Path(filename),
                    origin=_visual_origin(visual_elem),
                    color=color,
                    scale=_mesh_scale(mesh_elem),
                )
            )

    joints: list[Joint] = []
    parent_joint_by_link: dict[str, Joint] = {}
    child_joints_by_link: dict[str, list[Joint]] = {name: [] for name in links}
    for joint_elem in root.findall("joint"):
        origin_elem = joint_elem.find("origin")
        xyz = _float3(origin_elem.attrib.get("xyz") if origin_elem is not None else None)
        rpy = _float3(origin_elem.attrib.get("rpy") if origin_elem is not None else None)
        axis_elem = joint_elem.find("axis")
        axis = _float3(axis_elem.attrib.get("xyz") if axis_elem is not None else None, default=(1.0, 0.0, 0.0))
        limit_elem = joint_elem.find("limit")
        lower = float(limit_elem.attrib.get("lower", 0.0)) if limit_elem is not None else 0.0
        upper = float(limit_elem.attrib.get("upper", 0.0)) if limit_elem is not None else 0.0
        joint = Joint(
            name=joint_elem.attrib["name"],
            joint_type=joint_elem.attrib["type"],
            parent=joint_elem.find("parent").attrib["link"],
            child=joint_elem.find("child").attrib["link"],
            origin=make_transform(rpy_matrix(rpy), xyz),
            axis=axis,
            lower_limit=lower,
            upper_limit=upper,
        )
        joints.append(joint)
        parent_joint_by_link[joint.child] = joint
        child_joints_by_link.setdefault(joint.parent, []).append(joint)

    child_links = {joint.child for joint in joints}
    root_link = next(link for link in links if link not in child_links)
    return RobotModel(
        name=root.attrib.get("name", urdf_path.stem),
        links=links,
        joints=joints,
        visuals=visuals,
        parent_joint_by_link=parent_joint_by_link,
        child_joints_by_link=child_joints_by_link,
        root_link=root_link,
    )


ROBOT_MODEL = load_robot_model()
JOINT_INDEX_BY_NAME = {joint.name: idx for idx, joint in enumerate(ROBOT_MODEL.joints)}
ACTUATED_JOINTS = [joint for joint in ROBOT_MODEL.joints if joint.joint_type == "revolute"]


def actuated_joint_table() -> pd.DataFrame:
    rows = []
    for idx, joint in enumerate(ACTUATED_JOINTS):
        rows.append(
            {
                "joint_index": idx,
                "joint_name": joint.name,
                "parent_link": joint.parent,
                "child_link": joint.child,
                "lower_limit": joint.lower_limit,
                "upper_limit": joint.upper_limit,
            }
        )
    return pd.DataFrame(rows)


def forward_kinematics(joint_positions: dict[str, float] | None = None):
    joint_positions = joint_positions or {}
    link_transforms = {ROBOT_MODEL.root_link: np.eye(4, dtype=float)}
    joint_frames = {}
    joint_axes_world = {}

    def visit(parent_link: str):
        parent_T = link_transforms[parent_link]
        for joint in ROBOT_MODEL.child_joints_by_link.get(parent_link, []):
            joint_origin_T = parent_T @ joint.origin
            if joint.joint_type == "revolute":
                q = float(joint_positions.get(joint.name, 0.0))
                motion_R = axis_angle_matrix(joint.axis, q)
                child_T = joint_origin_T @ make_transform(motion_R, np.zeros(3))
                axis_world = joint_origin_T[:3, :3] @ (joint.axis / max(np.linalg.norm(joint.axis), 1e-12))
            else:
                child_T = joint_origin_T
                axis_world = joint_origin_T[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=float)
            joint_frames[joint.name] = child_T
            joint_axes_world[joint.name] = axis_world
            link_transforms[joint.child] = child_T
            visit(joint.child)

    visit(ROBOT_MODEL.root_link)
    return link_transforms, joint_frames, joint_axes_world


def end_effector_pose(link_name: str, joint_positions: dict[str, float] | None = None):
    link_transforms, _, _ = forward_kinematics(joint_positions)
    T = link_transforms[link_name]
    return T[:3, 3].copy(), T[:3, :3].copy()


def ordered_joint_subset(joint_names):
    joint_names = list(joint_names)
    table = actuated_joint_table()
    subset = table[table["joint_name"].isin(set(joint_names))].copy()
    subset["joint_name"] = pd.Categorical(subset["joint_name"], categories=joint_names, ordered=True)
    return subset.sort_values("joint_name").reset_index(drop=True)


def frame_data(joint_names, joint_positions: dict[str, float] | None = None):
    _, joint_frames, _ = forward_kinematics(joint_positions)
    rows = ordered_joint_subset(joint_names)
    data = []
    for row in rows.itertuples(index=False):
        T = joint_frames[row.joint_name]
        data.append({"joint_name": row.joint_name, "origin": T[:3, 3].copy(), "rotation": T[:3, :3].copy()})
    return data


@lru_cache(maxsize=256)
def load_stl_triangles(mesh_path: str):
    path = Path(mesh_path)
    data = path.read_bytes()
    if len(data) >= 84:
        tri_count = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + tri_count * 50
        if expected == len(data):
            triangles = np.empty((tri_count, 3, 3), dtype=np.float32)
            offset = 84
            for idx in range(tri_count):
                offset += 12
                vertices = np.frombuffer(data, dtype="<f4", count=9, offset=offset).reshape(3, 3)
                triangles[idx] = vertices
                offset += 36
                offset += 2
            return triangles

    vertices = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("vertex "):
            vertices.append([float(v) for v in line.split()[1:4]])
    arr = np.asarray(vertices, dtype=np.float32)
    if len(arr) == 0 or len(arr) % 3 != 0:
        raise ValueError(f"Could not parse STL mesh: {path}")
    return arr.reshape(-1, 3, 3)


def visual_mesh_traces(visible_link_names, *, joint_positions=None, opacity: float = 1.0):
    link_transforms, _, _ = forward_kinematics(joint_positions)
    traces = []
    for link_name in visible_link_names:
        link_T = link_transforms[link_name]
        for visual in ROBOT_MODEL.visuals.get(link_name, []):
            triangles = load_stl_triangles(str(visual.mesh_path)).astype(float)
            triangles *= visual.scale.reshape(1, 1, 3)
            world = transform_points(transform_points(triangles, visual.origin), link_T)
            verts = world.reshape(-1, 3)
            n_tri = world.shape[0]
            i = np.arange(0, n_tri * 3, 3)
            j = i + 1
            k = i + 2
            rgba = visual.color
            color = f"rgb({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)})"
            traces.append(
                go.Mesh3d(
                    x=verts[:, 0],
                    y=verts[:, 1],
                    z=verts[:, 2],
                    i=i,
                    j=j,
                    k=k,
                    color=color,
                    opacity=min(opacity, rgba[3]),
                    flatshading=True,
                    hoverinfo="skip",
                    name=visual.mesh_path.stem,
                    showscale=False,
                )
            )
    return traces


def make_urdf_plotly_figure(
    joint_names,
    *,
    visible_link_names,
    joint_positions=None,
    tip_link_name=None,
    frame_scale: float = 0.04,
    show_labels: bool = True,
    title: str = "",
    mesh_opacity: float = 1.0,
):
    frames = frame_data(joint_names, joint_positions)
    fig = go.Figure()
    for trace in visual_mesh_traces(visible_link_names, joint_positions=joint_positions, opacity=mesh_opacity):
        fig.add_trace(trace)

    axis_specs = [(0, "x", "red"), (1, "y", "green"), (2, "z", "blue")]
    for axis_index, axis_name, color in axis_specs:
        xs, ys, zs = [], [], []
        for item in frames:
            origin = item["origin"]
            endpoint = origin + item["rotation"][:, axis_index] * frame_scale
            xs.extend([origin[0], endpoint[0], None])
            ys.extend([origin[1], endpoint[1], None])
            zs.extend([origin[2], endpoint[2], None])
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line={"color": color, "width": 6},
                name=axis_name,
                hoverinfo="skip",
            )
        )

    if tip_link_name is not None:
        tip_pos, _ = end_effector_pose(tip_link_name, joint_positions)
        fig.add_trace(
            go.Scatter3d(
                x=[tip_pos[0]],
                y=[tip_pos[1]],
                z=[tip_pos[2]],
                mode="markers",
                marker={"size": 4, "color": "black"},
                name="tip",
                hoverinfo="skip",
            )
        )

    if show_labels:
        labels = [item["joint_name"] for item in frames]
        origins = np.vstack([item["origin"] for item in frames])
        fig.add_trace(
            go.Scatter3d(
                x=origins[:, 0],
                y=origins[:, 1],
                z=origins[:, 2],
                mode="text",
                text=labels,
                textposition="top center",
                name="labels",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=title,
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "dragmode": "orbit",
        },
        uirevision="keep",
    )
    return fig


def solve_left_arm_hand_pose(x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
    target_pos = np.array([x, y, z], dtype=float)
    target_rot = rpy_matrix(np.array([roll, pitch, yaw], dtype=float))
    limits = actuated_joint_table().set_index("joint_name")[["lower_limit", "upper_limit"]]
    lower = np.array([float(limits.loc[name, "lower_limit"]) for name in LEFT_ARM_7DOF_JOINTS], dtype=float)
    upper = np.array([float(limits.loc[name, "upper_limit"]) for name in LEFT_ARM_7DOF_JOINTS], dtype=float)
    q = np.zeros(len(LEFT_ARM_7DOF_JOINTS), dtype=float)

    for _ in range(120):
        joint_positions = {name: float(value) for name, value in zip(LEFT_ARM_7DOF_JOINTS, q)}
        link_transforms, joint_frames, joint_axes = forward_kinematics(joint_positions)
        end_T = link_transforms["left_hand_palm_link"]
        cur_pos = end_T[:3, 3]
        cur_rot = end_T[:3, :3]
        pos_error = target_pos - cur_pos
        rot_err = rotation_error(target_rot, cur_rot)
        error = np.concatenate([pos_error, rot_err])
        if np.linalg.norm(pos_error) < 1e-4 and np.linalg.norm(rot_err) < 1e-3:
            break

        J = np.zeros((6, len(LEFT_ARM_7DOF_JOINTS)), dtype=float)
        for idx, joint_name in enumerate(LEFT_ARM_7DOF_JOINTS):
            joint_T = joint_frames[joint_name]
            joint_origin = joint_T[:3, 3]
            axis_world = joint_axes[joint_name]
            J[:3, idx] = np.cross(axis_world, cur_pos - joint_origin)
            J[3:, idx] = axis_world

        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), error)
        q = np.clip(q + 0.6 * dq, lower, upper)

    return {name: float(value) for name, value in zip(LEFT_ARM_7DOF_JOINTS, q)}


def rrt_arm_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    joint_names: list[str],
    obstacles: list[tuple],
    body_links: list[str],
    arm_links: list[str],
    n_iter: int = 3000,
    step_size: float = 0.12,
    goal_bias: float = 0.15,
    seed: int = 42,
) -> "list[np.ndarray] | None":
    """RRT planner in joint space with bounding-sphere self-collision and obstacle checks.

    Each link is approximated as a sphere (see _LINK_SPHERE_RADII).  Non-adjacent
    arm links and arm-vs-body pairs are checked for penetration at every new node.
    External obstacles are sphere primitives supplied by the caller.

    Returns a list of joint-angle arrays from q_start to q_goal, or None on failure.
    """
    rng = np.random.default_rng(seed)
    limits = actuated_joint_table().set_index("joint_name")[["lower_limit", "upper_limit"]]
    lower = np.array([float(limits.loc[n, "lower_limit"]) for n in joint_names])
    upper = np.array([float(limits.loc[n, "upper_limit"]) for n in joint_names])

    obs_centers = [np.asarray(oc, dtype=float) for oc, _ in obstacles]
    obs_radii   = [float(r)                      for _, r  in obstacles]

    def _valid(q: np.ndarray) -> bool:
        jp = {n: float(v) for n, v in zip(joint_names, q)}
        link_T, _, _ = forward_kinematics(jp)

        arm_cr = [
            (link_T[lk][:3, 3], _LINK_SPHERE_RADII.get(lk, 0.06))
            for lk in arm_links
            if lk in link_T
        ]

        # Arm-vs-body self-collision
        for bl in body_links:
            if bl not in link_T:
                continue
            bc = link_T[bl][:3, 3]
            br = _LINK_SPHERE_RADII.get(bl, 0.06)
            for ac, ar in arm_cr:
                if float(np.linalg.norm(ac - bc)) < ar + br:
                    return False

        # Arm-vs-obstacle collision
        for oc, or_ in zip(obs_centers, obs_radii):
            for ac, ar in arm_cr:
                if float(np.linalg.norm(ac - oc)) < ar + or_:
                    return False

        # Non-adjacent arm-link self-collision (skip 2 neighbours each side)
        n = len(arm_cr)
        for i in range(n):
            for j in range(i + 3, n):
                ac1, ar1 = arm_cr[i]
                ac2, ar2 = arm_cr[j]
                if float(np.linalg.norm(ac1 - ac2)) < ar1 + ar2:
                    return False

        return True

    if not _valid(q_start):
        print("rrt_arm_trajectory: start configuration is in collision")
        return None
    if not _valid(q_goal):
        print("rrt_arm_trajectory: goal configuration is in collision")
        return None

    tree: list[np.ndarray] = [q_start.copy()]
    parent: dict[int, int]  = {0: -1}

    for _iter in range(n_iter):
        q_rand = q_goal.copy() if rng.random() < goal_bias else rng.uniform(lower, upper)

        dists   = np.fromiter((float(np.linalg.norm(q_rand - nd)) for nd in tree), float, len(tree))
        near_i  = int(dists.argmin())
        q_near  = tree[near_i]

        diff = q_rand - q_near
        d    = float(np.linalg.norm(diff))
        if d < 1e-9:
            continue
        q_new = np.clip(q_near + step_size * diff / d, lower, upper)

        if _valid(q_new):
            new_i        = len(tree)
            tree.append(q_new)
            parent[new_i] = near_i

            if float(np.linalg.norm(q_new - q_goal)) < step_size:
                path: list[np.ndarray] = [q_goal.copy()]
                idx = new_i
                while idx != -1:
                    path.append(tree[idx].copy())
                    idx = parent[idx]
                return list(reversed(path))

    return None
