import argparse
import atexit
import math
import tempfile
import time
from pathlib import Path

import pybullet as p
import pybullet_data


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_URDF = (SCRIPT_DIR / "../unitree_description-master/model/go2/go2.urdf").resolve()
DEFAULT_PACKAGE_ROOT = (SCRIPT_DIR / "../unitree_description-master").resolve()
DEFAULT_PLANE_URDF = Path(pybullet_data.getDataPath()) / "plane.urdf"
LEG_NAMES = ("FL", "FR", "RL", "RR")
LEG_SIGNS = {
    "FL": {"left": 1.0, "front": 1.0},
    "FR": {"left": -1.0, "front": 1.0},
    "RL": {"left": 1.0, "front": -1.0},
    "RR": {"left": -1.0, "front": -1.0},
}
STAND_POSE = {
    "FL_hip_joint": 0.06,
    "FR_hip_joint": -0.06,
    "RL_hip_joint": 0.08,
    "RR_hip_joint": -0.08,
    "FL_thigh_joint": 0.67,
    "FR_thigh_joint": 0.67,
    "RL_thigh_joint": 0.78,
    "RR_thigh_joint": 0.78,
    "FL_calf_joint": -1.32,
    "FR_calf_joint": -1.32,
    "RL_calf_joint": -1.38,
    "RR_calf_joint": -1.38,
}
FOOT_LINK_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
SUPPORT_POLYGON_COLORS = {
    "polygon": (0.95, 0.35, 0.9),
    "com": (0.25, 0.95, 0.95),
    "center": (0.95, 0.25, 0.25),
}
TEMP_URDFS = []


def clamp(value, limit):
    return max(-limit, min(limit, value))


def resolve_package_root(urdf_path: Path) -> Path:
    urdf_path = urdf_path.resolve()
    for parent in urdf_path.parents:
        if (parent / "model" / "go2" / "go2.urdf").is_file():
            return parent
    return DEFAULT_PACKAGE_ROOT


def build_resolved_urdf(urdf_path: Path, package_root: Path) -> Path:
    package_prefix = package_root.resolve().as_posix().rstrip("/") + "/"
    urdf_text = urdf_path.read_text(encoding="utf-8")
    urdf_text = urdf_text.replace("package://unitree_description/", package_prefix)

    handle = tempfile.NamedTemporaryFile(prefix="go2_balance_", suffix=".urdf", delete=False, mode="w", encoding="utf-8")
    handle.write(urdf_text)
    handle.flush()
    handle.close()

    resolved_path = Path(handle.name)
    TEMP_URDFS.append(resolved_path)
    return resolved_path


def cleanup_temp_urdfs():
    for path in TEMP_URDFS:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def reset_all_joints(body_id):
    for joint_index in range(p.getNumJoints(body_id)):
        p.resetJointState(body_id, joint_index, targetValue=0.0, targetVelocity=0.0)


def reset_joints_to_pose(body_id, actuators, targets):
    for joint_name, target_position in targets.items():
        actuator = actuators.get(joint_name)
        if actuator is None:
            continue
        p.resetJointState(body_id, actuator["index"], targetValue=target_position, targetVelocity=0.0)


def build_joint_actuator_map(body_id):
    actuators = {}
    for joint_index in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, joint_index)
        if joint_info[2] == p.JOINT_FIXED:
            continue

        joint_name = joint_info[1].decode("utf-8")
        max_force = joint_info[10] if joint_info[10] > 0.0 else 45.0
        actuators[joint_name] = {"index": joint_index, "force": max_force}
    return actuators


def build_link_name_map(body_id):
    link_name_map = {}
    for joint_index in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, joint_index)
        link_name_map[joint_info[12].decode("utf-8")] = joint_index
    return link_name_map


def apply_joint_targets(body_id, actuators, targets, position_gain, velocity_gain, force_scale):
    for joint_name, target_position in targets.items():
        actuator = actuators.get(joint_name)
        if actuator is None:
            continue

        p.setJointMotorControl2(
            body_id,
            actuator["index"],
            p.POSITION_CONTROL,
            targetPosition=target_position,
            positionGain=position_gain,
            velocityGain=velocity_gain,
            force=force_scale * actuator["force"],
        )


def read_dummy_imu(body_id):
    base_position, base_orientation = p.getBasePositionAndOrientation(body_id)
    base_linear_velocity, base_angular_velocity = p.getBaseVelocity(body_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(base_orientation)
    return {
        "position": base_position,
        "orientation": base_orientation,
        "linear_velocity": base_linear_velocity,
        "angular_velocity": base_angular_velocity,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "roll_rate": base_angular_velocity[0],
        "pitch_rate": base_angular_velocity[1],
    }


def build_mass_properties(body_id):
    masses = [(-1, p.getDynamicsInfo(body_id, -1)[0])]
    for joint_index in range(p.getNumJoints(body_id)):
        masses.append((joint_index, p.getDynamicsInfo(body_id, joint_index)[0]))
    return [(link_index, mass) for link_index, mass in masses if mass > 0.0]


def compute_total_mass(body_id):
    return sum(mass for _, mass in build_mass_properties(body_id))


def compute_center_of_mass(body_id, mass_properties):
    total_mass = sum(mass for _, mass in mass_properties)
    weighted_position = [0.0, 0.0, 0.0]
    weighted_velocity = [0.0, 0.0, 0.0]
    base_position, _ = p.getBasePositionAndOrientation(body_id)
    base_velocity, _ = p.getBaseVelocity(body_id)

    for link_index, mass in mass_properties:
        if link_index == -1:
            position = base_position
            linear_velocity = base_velocity
        else:
            link_state = p.getLinkState(body_id, link_index, computeLinkVelocity=True, computeForwardKinematics=True)
            position = link_state[0]
            linear_velocity = link_state[6]

        for axis in range(3):
            weighted_position[axis] += position[axis] * mass
            weighted_velocity[axis] += linear_velocity[axis] * mass

    com_position = tuple(component / total_mass for component in weighted_position)
    com_velocity = tuple(component / total_mass for component in weighted_velocity)
    return com_position, com_velocity


def project_world_vector_to_body_frame(vector_xy, yaw):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        cos_yaw * vector_xy[0] + sin_yaw * vector_xy[1],
        -sin_yaw * vector_xy[0] + cos_yaw * vector_xy[1],
    )


def build_balance_targets(imu_state, com_position, com_velocity, support_center, args):
    base_pitch_feedback = -(args.pitch_kp * imu_state["pitch"] + args.pitch_kd * imu_state["pitch_rate"])
    base_roll_feedback = -(args.roll_kp * imu_state["roll"] + args.roll_kd * imu_state["roll_rate"])
    com_error_body = (0.0, 0.0)
    com_velocity_body = (0.0, 0.0)
    if support_center is not None:
        com_error_body = project_world_vector_to_body_frame(
            (com_position[0] - support_center[0], com_position[1] - support_center[1]),
            imu_state["yaw"],
        )
        com_velocity_body = project_world_vector_to_body_frame((com_velocity[0], com_velocity[1]), imu_state["yaw"])

    com_pitch_feedback = -(args.com_balance_kp * com_error_body[0] + args.com_balance_kd * com_velocity_body[0])
    com_roll_feedback = -(args.com_balance_roll_kp * com_error_body[1] + args.com_balance_roll_kd * com_velocity_body[1])

    pitch_correction = clamp(base_pitch_feedback + com_pitch_feedback, args.pitch_limit)
    roll_correction = clamp(base_roll_feedback + com_roll_feedback, args.roll_limit)
    height_error = args.target_base_height - imu_state["position"][2]
    height_correction = clamp(
        args.height_kp * height_error - args.height_kd * imu_state["linear_velocity"][2],
        args.height_limit,
    )

    targets = dict(STAND_POSE)
    for leg_name in LEG_NAMES:
        leg_sign = LEG_SIGNS[leg_name]
        left_sign = leg_sign["left"]
        front_sign = leg_sign["front"]

        hip_name = f"{leg_name}_hip_joint"
        thigh_name = f"{leg_name}_thigh_joint"
        calf_name = f"{leg_name}_calf_joint"

        targets[hip_name] += left_sign * roll_correction * args.hip_roll_gain
        targets[thigh_name] += height_correction + front_sign * pitch_correction * args.thigh_pitch_gain
        targets[calf_name] += -2.0 * height_correction - front_sign * pitch_correction * args.calf_pitch_gain

    return targets, pitch_correction, roll_correction, height_correction, com_error_body


def build_status_text(imu_state, com_position, polygon_margin, pitch_correction, roll_correction, height_correction, feet_in_contact):
    base_position = imu_state["position"]
    return "\n".join(
        (
            f"base z: {base_position[2]:.3f} m",
            f"com xy: ({com_position[0]:+.3f}, {com_position[1]:+.3f}) m",
            f"support margin: {polygon_margin:+.3f} m",
            f"roll/pitch: {imu_state['roll']:+.3f}, {imu_state['pitch']:+.3f} rad",
            f"feedback pitch: {pitch_correction:+.3f}",
            f"feedback roll: {roll_correction:+.3f}",
            f"feedback height: {height_correction:+.3f}",
            f"feet in contact: {feet_in_contact}/4",
        )
    )


def count_feet_in_contact(body_id, plane_id, link_name_map):
    contact_count = 0
    for link_name in FOOT_LINK_NAMES:
        link_index = link_name_map.get(link_name)
        if link_index is None:
            continue
        contacts = p.getContactPoints(bodyA=body_id, bodyB=plane_id, linkIndexA=link_index)
        if any(contact[9] > 1e-3 for contact in contacts):
            contact_count += 1
    return contact_count


def get_default_foot_link_indices(link_name_map):
    return {link_name: link_name_map[link_name] for link_name in FOOT_LINK_NAMES if link_name in link_name_map}


def build_contact_link_map(link_name_map, contact_links_arg):
    foot_indices = get_default_foot_link_indices(link_name_map)
    if contact_links_arg == "feet":
        return foot_indices, True
    if contact_links_arg == "all":
        return {link_name: joint_index for link_name, joint_index in link_name_map.items()}, False

    selected_links = {}
    for link_name in (part.strip() for part in contact_links_arg.split(",")):
        if link_name and link_name in link_name_map:
            selected_links[link_name] = link_name_map[link_name]
    return selected_links, True


def get_link_contact_points(body_id, plane_id, link_index):
    contacts = p.getContactPoints(bodyA=body_id, bodyB=plane_id, linkIndexA=link_index)
    return [contact for contact in contacts if contact[9] > 1e-3]


def get_active_foot_supports(body_id, plane_id, foot_link_indices):
    supports = {}
    for link_name, link_index in foot_link_indices.items():
        contacts = get_link_contact_points(body_id, plane_id, link_index)
        if contacts:
            supports[link_name] = contacts
    return supports


def get_link_support_corners(body_id, link_index, ground_height=0.0):
    aabb_min, aabb_max = p.getAABB(body_id, linkIndex=link_index)
    return [
        (aabb_min[0], aabb_min[1], ground_height),
        (aabb_min[0], aabb_max[1], ground_height),
        (aabb_max[0], aabb_max[1], ground_height),
        (aabb_max[0], aabb_min[1], ground_height),
    ]


def cross_2d(origin, point_a, point_b):
    return (point_a[0] - origin[0]) * (point_b[1] - origin[1]) - (point_a[1] - origin[1]) * (point_b[0] - origin[0])


def convex_hull_2d(points):
    unique_points = sorted({(round(point[0], 5), round(point[1], 5), point[2]) for point in points})
    if len(unique_points) <= 1:
        return [(point[0], point[1], point[2]) for point in unique_points]

    lower = []
    for point in unique_points:
        while len(lower) >= 2 and cross_2d(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross_2d(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    return [(point[0], point[1], point[2]) for point in hull]


def polygon_centroid_2d(points):
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    if len(points) == 2:
        return (
            0.5 * (points[0][0] + points[1][0]),
            0.5 * (points[0][1] + points[1][1]),
            0.5 * (points[0][2] + points[1][2]),
        )

    area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        cross = point[0] * next_point[1] - next_point[0] * point[1]
        area += cross
        centroid_x += (point[0] + next_point[0]) * cross
        centroid_y += (point[1] + next_point[1]) * cross

    if abs(area) < 1e-8:
        return (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
            sum(point[2] for point in points) / len(points),
        )

    area *= 0.5
    return (centroid_x / (6.0 * area), centroid_y / (6.0 * area), sum(point[2] for point in points) / len(points))


def point_to_segment_distance_2d(point, segment_start, segment_end):
    seg_x = segment_end[0] - segment_start[0]
    seg_y = segment_end[1] - segment_start[1]
    seg_len_sq = seg_x * seg_x + seg_y * seg_y
    if seg_len_sq < 1e-10:
        return math.hypot(point[0] - segment_start[0], point[1] - segment_start[1])

    projection = ((point[0] - segment_start[0]) * seg_x + (point[1] - segment_start[1]) * seg_y) / seg_len_sq
    projection = max(0.0, min(1.0, projection))
    closest_x = segment_start[0] + projection * seg_x
    closest_y = segment_start[1] + projection * seg_y
    return math.hypot(point[0] - closest_x, point[1] - closest_y)


def point_in_polygon_2d(point, polygon_points):
    if len(polygon_points) < 3:
        return False
    inside = False
    x, y = point
    for index, point_a in enumerate(polygon_points):
        point_b = polygon_points[(index + 1) % len(polygon_points)]
        if ((point_a[1] > y) != (point_b[1] > y)) and (
            x < (point_b[0] - point_a[0]) * (y - point_a[1]) / max(point_b[1] - point_a[1], 1e-10) + point_a[0]
        ):
            inside = not inside
    return inside


def support_polygon_margin(point, polygon_points):
    if not polygon_points:
        return -1.0
    if len(polygon_points) == 1:
        return -math.hypot(point[0] - polygon_points[0][0], point[1] - polygon_points[0][1])
    if len(polygon_points) == 2:
        return -point_to_segment_distance_2d(point, polygon_points[0], polygon_points[1])

    distances = [
        point_to_segment_distance_2d(point, polygon_points[index], polygon_points[(index + 1) % len(polygon_points)])
        for index in range(len(polygon_points))
    ]
    margin = min(distances)
    return margin if point_in_polygon_2d(point, polygon_points) else -margin


def build_support_polygon(body_id, foot_link_indices, active_supports, polygon_height):
    polygon_points = []
    for link_name in active_supports:
        polygon_points.extend(get_link_support_corners(body_id, foot_link_indices[link_name], polygon_height))
    return convex_hull_2d(polygon_points)


def summarize_contact_forces(body_id, plane_id, contact_link_indices, include_inactive=True):
    summaries = {}
    if include_inactive:
        summaries = {
            label: {
                "normal_force": 0.0,
                "force_vector": [0.0, 0.0, 0.0],
                "weighted_position": [0.0, 0.0, 0.0],
                "position": p.getLinkState(body_id, link_index, computeForwardKinematics=True)[4],
            }
            for label, link_index in contact_link_indices.items()
        }

    contacts = p.getContactPoints(bodyA=body_id, bodyB=plane_id)
    link_to_label = {link_index: label for label, link_index in contact_link_indices.items()}
    for contact in contacts:
        link_index = contact[3]
        if link_index not in link_to_label:
            continue

        label = link_to_label[link_index]
        position = contact[6]
        normal = contact[7]
        normal_force = contact[9]
        lateral_force_1 = contact[10]
        lateral_dir_1 = contact[11]
        lateral_force_2 = contact[12]
        lateral_dir_2 = contact[13]
        force_vector = (
            normal[0] * normal_force + lateral_dir_1[0] * lateral_force_1 + lateral_dir_2[0] * lateral_force_2,
            normal[1] * normal_force + lateral_dir_1[1] * lateral_force_1 + lateral_dir_2[1] * lateral_force_2,
            normal[2] * normal_force + lateral_dir_1[2] * lateral_force_1 + lateral_dir_2[2] * lateral_force_2,
        )

        summary = summaries.setdefault(
            label,
            {
                "normal_force": 0.0,
                "force_vector": [0.0, 0.0, 0.0],
                "weighted_position": [0.0, 0.0, 0.0],
                "position": position,
            },
        )
        summary["normal_force"] += normal_force
        for index in range(3):
            summary["force_vector"][index] += force_vector[index]
            summary["weighted_position"][index] += position[index] * normal_force

    for summary in summaries.values():
        if summary["normal_force"] > 1e-6:
            summary["position"] = tuple(component / summary["normal_force"] for component in summary["weighted_position"])

    return summaries


def update_contact_visuals(contact_summaries, debug_items, total_weight, force_scale, show_text=True):
    total_normal_force = sum(summary["normal_force"] for summary in contact_summaries.values())
    for label, summary in contact_summaries.items():
        item = debug_items.setdefault(label, {})
        position = summary["position"]
        if position is None:
            anchor = item.get("last_position", (0.0, 0.0, 0.15))
            zero_end = (anchor[0], anchor[1], anchor[2] + 0.001)
            item["arrow"] = p.addUserDebugLine(
                anchor,
                zero_end,
                (0.5, 0.5, 0.5),
                lineWidth=2.0,
                replaceItemUniqueId=item.get("arrow", -1),
            )
            if show_text:
                item["text"] = p.addUserDebugText(
                    f"{label}: 0 N (0.0%)",
                    (anchor[0], anchor[1], anchor[2] + 0.04),
                    textColorRGB=(0.8, 0.8, 0.8),
                    textSize=1.2,
                    replaceItemUniqueId=item.get("text", -1),
                )
            continue

        item["last_position"] = position
        force_vector = summary["force_vector"]
        arrow_end = (
            position[0] + force_vector[0] * force_scale,
            position[1] + force_vector[1] * force_scale,
            position[2] + force_vector[2] * force_scale,
        )
        load_fraction = (summary["normal_force"] / total_normal_force) if total_normal_force > 1e-6 else 0.0
        weight_fraction = (summary["normal_force"] / total_weight) if total_weight > 1e-6 else 0.0
        item["arrow"] = p.addUserDebugLine(
            position,
            arrow_end,
            (1.0, 0.75, 0.2),
            lineWidth=4.0,
            replaceItemUniqueId=item.get("arrow", -1),
        )
        if show_text:
            item["text"] = p.addUserDebugText(
                f"{label}: {summary['normal_force']:.1f} N | load {load_fraction * 100.0:.1f}% | body wt {weight_fraction * 100.0:.1f}%",
                (position[0], position[1], position[2] + 0.05),
                textColorRGB=(1.0, 0.92, 0.55),
                textSize=1.2,
                replaceItemUniqueId=item.get("text", -1),
            )


def clear_contact_visuals(debug_items, active_labels):
    active_labels = set(active_labels)
    for label in list(debug_items):
        if label in active_labels:
            continue
        item = debug_items.pop(label)
        for key in ("arrow", "text"):
            item_id = item.get(key)
            if item_id is not None and item_id >= 0:
                p.removeUserDebugItem(item_id)


def clear_debug_item_group(item_group):
    for item_id in item_group.values():
        if isinstance(item_id, list):
            for nested_item_id in item_id:
                if nested_item_id is not None and nested_item_id >= 0:
                    p.removeUserDebugItem(nested_item_id)
            continue
        if item_id is not None and item_id >= 0:
            p.removeUserDebugItem(item_id)
    item_group.clear()


def update_support_polygon_visuals(polygon_points, com_position, polygon_center, debug_items, show_text=True):
    line_ids = debug_items.setdefault("lines", [])
    required_line_count = len(polygon_points) + 2 if polygon_points else 0
    while len(line_ids) < required_line_count:
        line_ids.append(-1)

    active_line_count = 0
    if polygon_points:
        for index, point in enumerate(polygon_points):
            next_point = polygon_points[(index + 1) % len(polygon_points)]
            line_ids[index] = p.addUserDebugLine(
                point,
                next_point,
                SUPPORT_POLYGON_COLORS["polygon"],
                lineWidth=3.0,
                replaceItemUniqueId=line_ids[index],
            )
            active_line_count += 1

        com_projection = (com_position[0], com_position[1], polygon_points[0][2])
        line_ids[active_line_count] = p.addUserDebugLine(
            com_projection,
            (com_projection[0], com_projection[1], com_projection[2] + 0.08),
            SUPPORT_POLYGON_COLORS["com"],
            lineWidth=4.0,
            replaceItemUniqueId=line_ids[active_line_count],
        )
        active_line_count += 1
        line_ids[active_line_count] = p.addUserDebugLine(
            polygon_center,
            (polygon_center[0], polygon_center[1], polygon_center[2] + 0.06),
            SUPPORT_POLYGON_COLORS["center"],
            lineWidth=4.0,
            replaceItemUniqueId=line_ids[active_line_count],
        )
        active_line_count += 1

    for index in range(active_line_count, len(line_ids)):
        if line_ids[index] >= 0:
            p.removeUserDebugItem(line_ids[index])
            line_ids[index] = -1

    if not polygon_points:
        if "text" in debug_items and debug_items["text"] >= 0:
            p.removeUserDebugItem(debug_items["text"])
            debug_items["text"] = -1
        return

    if show_text:
        debug_items["text"] = p.addUserDebugText(
            "support polygon",
            (polygon_center[0], polygon_center[1], polygon_center[2] + 0.07),
            textColorRGB=SUPPORT_POLYGON_COLORS["polygon"],
            textSize=1.1,
            replaceItemUniqueId=debug_items.get("text", -1),
        )
    elif "text" in debug_items and debug_items["text"] >= 0:
        p.removeUserDebugItem(debug_items["text"])
        debug_items["text"] = -1


def place_base_so_feet_touch_ground(body_id, foot_link_indices, clearance):
    min_foot_z = None
    for link_index in foot_link_indices.values():
        aabb_min, _ = p.getAABB(body_id, linkIndex=link_index)
        if min_foot_z is None or aabb_min[2] < min_foot_z:
            min_foot_z = aabb_min[2]
    if min_foot_z is None:
        return

    base_position, base_orientation = p.getBasePositionAndOrientation(body_id)
    adjusted_position = (base_position[0], base_position[1], base_position[2] + clearance - min_foot_z)
    p.resetBasePositionAndOrientation(body_id, adjusted_position, base_orientation)
    p.resetBaseVelocity(body_id, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def configure_camera(body_id):
    base_position, _ = p.getBasePositionAndOrientation(body_id)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.4,
        cameraYaw=42.0,
        cameraPitch=-22.0,
        cameraTargetPosition=(base_position[0], base_position[1], base_position[2] + 0.1),
    )


def compatibility_notes(args):
    notes = []
    if args.motion != "stand":
        notes.append(f"--motion={args.motion} is accepted for CLI compatibility but this script only runs the stand controller.")
    if args.contact_links != "feet":
        notes.append(
            f"--contact-links={args.contact_links} is accepted for CLI compatibility; contact display still reports feet-ground contacts."
        )

    recovery_options = {
        "recovery_margin": args.recovery_margin,
        "recovery_step_length": args.recovery_step_length,
        "recovery_step_placement_kp": args.recovery_step_placement_kp,
        "recovery_step_placement_kd": args.recovery_step_placement_kd,
        "recovery_lateral_step_kp": args.recovery_lateral_step_kp,
        "recovery_lateral_step_kd": args.recovery_lateral_step_kd,
        "recovery_arm_pitch_kp": args.recovery_arm_pitch_kp,
        "recovery_arm_roll_kp": args.recovery_arm_roll_kp,
    }
    if any(value is not None for value in recovery_options.values()):
        notes.append("Recovery-step tuning flags are accepted for compatibility but are not used by the standing controller.")

    return notes


def parse_args():
    parser = argparse.ArgumentParser(description="Load the Unitree Go2 in PyBullet and hold a standing balance pose.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help=f"URDF to load (default: {DEFAULT_URDF})")
    parser.add_argument(
        "--package-root",
        type=Path,
        default=None,
        help="Root directory used to resolve package://unitree_description mesh paths.",
    )
    parser.add_argument("--headless", action="store_true", help="Run in DIRECT mode instead of opening the PyBullet GUI.")
    parser.add_argument("--fixed-base", action="store_true", help="Keep the base fixed for quick URDF inspection.")
    parser.add_argument("--start-height", type=float, default=0.42, help="Initial base height in meters.")
    parser.add_argument("--target-base-height", type=float, default=0.343, help="Nominal floating-base height during standing.")
    parser.add_argument("--time-step", type=float, default=1.0 / 240.0, help="Physics step size in seconds.")
    parser.add_argument("--max-fps", type=float, default=240.0, help="GUI pacing limit. Set to 0 to run uncapped.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional simulation-step limit for testing. Set to 0 to run until closed.")
    parser.add_argument(
        "--balance-control",
        action="store_true",
        help="Accepted for CLI compatibility. This script always runs the balance-standing controller.",
    )
    parser.add_argument(
        "--contact-links",
        choices=("feet", "all"),
        default="feet",
        help="Accepted for CLI compatibility. Contact status currently tracks only foot-ground contacts.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("full", "minimal", "off"),
        default="full",
        help="Select the amount of PyBullet GUI chrome when not running headless.",
    )
    parser.add_argument(
        "--motion",
        choices=("stand", "manual"),
        default="stand",
        help="Accepted for CLI compatibility. Only standing balance is implemented in this script.",
    )
    parser.add_argument("--pitch-kp", type=float, default=0.25, help="Pitch proportional gain.")
    parser.add_argument("--pitch-kd", type=float, default=0.04, help="Pitch derivative gain.")
    parser.add_argument("--roll-kp", type=float, default=0.20, help="Roll proportional gain.")
    parser.add_argument("--roll-kd", type=float, default=0.04, help="Roll derivative gain.")
    parser.add_argument("--height-kp", type=float, default=0.0, help="Base-height proportional gain.")
    parser.add_argument("--height-kd", type=float, default=0.0, help="Base-height derivative gain.")
    parser.add_argument("--pitch-limit", type=float, default=0.05, help="Clamp for pitch feedback.")
    parser.add_argument("--roll-limit", type=float, default=0.05, help="Clamp for roll feedback.")
    parser.add_argument("--height-limit", type=float, default=0.0, help="Clamp for height feedback.")
    parser.add_argument("--hip-roll-gain", type=float, default=0.10, help="How strongly roll feedback drives the hip joints.")
    parser.add_argument("--thigh-pitch-gain", type=float, default=0.08, help="How strongly pitch feedback drives the thigh joints.")
    parser.add_argument("--calf-pitch-gain", type=float, default=0.12, help="How strongly pitch feedback drives the calf joints.")
    parser.add_argument("--position-gain", type=float, default=0.15, help="PyBullet position gain for the joint motors.")
    parser.add_argument("--velocity-gain", type=float, default=0.5, help="PyBullet velocity gain for the joint motors.")
    parser.add_argument("--force-scale", type=float, default=1.5, help="Extra scale applied on top of the URDF joint effort limits.")
    parser.add_argument("--com-balance-kp", type=float, default=0.0, help="COM-to-support-center proportional gain in the sagittal plane.")
    parser.add_argument("--com-balance-kd", type=float, default=0.0, help="COM velocity gain in the sagittal plane.")
    parser.add_argument("--com-balance-roll-kp", type=float, default=0.0, help="COM-to-support-center proportional gain in the lateral plane.")
    parser.add_argument("--com-balance-roll-kd", type=float, default=0.0, help="COM velocity gain in the lateral plane.")
    parser.add_argument("--contact-force-scale", type=float, default=0.0025, help="Meters per Newton for contact-force arrows.")
    parser.add_argument("--support-polygon-height", type=float, default=0.005, help="Height offset for the support polygon debug overlay.")
    parser.add_argument("--spawn-foot-clearance", type=float, default=0.02, help="Initial vertical foot clearance above the plane.")
    parser.add_argument("--settle-steps", type=int, default=120, help="Physics steps used to settle into the standing pose before normal control.")
    parser.add_argument("--debug-update-every", type=int, default=4, help="Only refresh debug overlays every N physics steps.")
    parser.add_argument("--recovery-margin", type=float, default=None, help="Accepted for CLI compatibility; currently unused.")
    parser.add_argument("--recovery-step-length", type=float, default=None, help="Accepted for CLI compatibility; currently unused.")
    parser.add_argument(
        "--recovery-step-placement-kp",
        type=float,
        default=None,
        help="Accepted for CLI compatibility; currently unused.",
    )
    parser.add_argument(
        "--recovery-step-placement-kd",
        type=float,
        default=None,
        help="Accepted for CLI compatibility; currently unused.",
    )
    parser.add_argument(
        "--recovery-lateral-step-kp",
        type=float,
        default=None,
        help="Accepted for CLI compatibility; currently unused.",
    )
    parser.add_argument(
        "--recovery-lateral-step-kd",
        type=float,
        default=None,
        help="Accepted for CLI compatibility; currently unused.",
    )
    parser.add_argument(
        "--recovery-arm-pitch-kp",
        type=float,
        default=None,
        help="Accepted for CLI compatibility; currently unused.",
    )
    parser.add_argument(
        "--recovery-arm-roll-kp",
        type=float,
        default=None,
        help="Accepted for CLI compatibility; currently unused.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for note in compatibility_notes(args):
        print(f"[go2_balance_stand] {note}")

    urdf_path = args.urdf.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    if not DEFAULT_PLANE_URDF.is_file():
        raise FileNotFoundError(f"PyBullet plane URDF not found: {DEFAULT_PLANE_URDF}")

    package_root = args.package_root.resolve() if args.package_root is not None else resolve_package_root(urdf_path)
    if not package_root.is_dir():
        raise FileNotFoundError(f"Package root not found: {package_root}")

    resolved_urdf_path = build_resolved_urdf(urdf_path, package_root)
    connection_mode = p.DIRECT if args.headless else p.GUI
    client_id = p.connect(connection_mode)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(args.time_step)
    p.setGravity(0.0, 0.0, -9.81)
    show_gui = 1 if (not args.headless and args.render_mode == "full") else 0
    show_shadows = 1 if (not args.headless and args.render_mode == "full") else 0
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, show_gui)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, show_shadows)

    plane_id = p.loadURDF(str(DEFAULT_PLANE_URDF))
    p.changeDynamics(plane_id, -1, lateralFriction=1.2, spinningFriction=0.04, rollingFriction=0.02)

    body_id = p.loadURDF(
        str(resolved_urdf_path),
        basePosition=(0.0, 0.0, args.start_height),
        baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, 0.0)),
        useFixedBase=args.fixed_base,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )
    actuators = build_joint_actuator_map(body_id)
    link_name_map = build_link_name_map(body_id)
    foot_link_indices = get_default_foot_link_indices(link_name_map)
    reset_joints_to_pose(body_id, actuators, STAND_POSE)
    place_base_so_feet_touch_ground(body_id, foot_link_indices, args.spawn_foot_clearance)

    for joint_index in range(-1, p.getNumJoints(body_id)):
        p.changeDynamics(body_id, joint_index, lateralFriction=1.0, spinningFriction=0.04, rollingFriction=0.02)

    mass_properties = build_mass_properties(body_id)
    total_weight = compute_total_mass(body_id) * 9.81
    contact_link_indices, include_inactive_contact_links = build_contact_link_map(link_name_map, args.contact_links)
    contact_debug_items = {}
    support_polygon_items = {}
    status_item = -1
    configure_camera(body_id)

    for _ in range(max(0, args.settle_steps)):
        apply_joint_targets(body_id, actuators, STAND_POSE, args.position_gain, args.velocity_gain, args.force_scale)
        p.stepSimulation()

    step_index = 0
    last_wall_time = time.perf_counter()
    try:
        while p.isConnected():
            imu_state = read_dummy_imu(body_id)
            com_position, com_velocity = compute_center_of_mass(body_id, mass_properties)
            active_foot_supports = get_active_foot_supports(body_id, plane_id, foot_link_indices)
            support_polygon = build_support_polygon(body_id, foot_link_indices, active_foot_supports, args.support_polygon_height)
            support_center = polygon_centroid_2d(support_polygon)
            polygon_margin = support_polygon_margin((com_position[0], com_position[1]), support_polygon)
            if support_center is None:
                support_center = (com_position[0], com_position[1], args.support_polygon_height)

            targets, pitch_correction, roll_correction, height_correction, _ = build_balance_targets(
                imu_state,
                com_position,
                com_velocity,
                support_center,
                args,
            )
            apply_joint_targets(body_id, actuators, targets, args.position_gain, args.velocity_gain, args.force_scale)
            p.stepSimulation()

            if not args.headless and step_index % max(1, args.debug_update_every) == 0:
                feet_in_contact = count_feet_in_contact(body_id, plane_id, link_name_map)
                status_item = p.addUserDebugText(
                    build_status_text(
                        imu_state,
                        com_position,
                        polygon_margin,
                        pitch_correction,
                        roll_correction,
                        height_correction,
                        feet_in_contact,
                    ),
                    (-0.45, -0.28, 0.55),
                    textColorRGB=(0.95, 0.95, 0.95),
                    textSize=1.1,
                    replaceItemUniqueId=status_item,
                )
                if args.render_mode in ("full", "minimal") and support_polygon:
                    update_support_polygon_visuals(
                        support_polygon,
                        com_position,
                        support_center,
                        support_polygon_items,
                        show_text=args.render_mode == "full",
                    )
                elif support_polygon_items:
                    clear_debug_item_group(support_polygon_items)

                if args.render_mode in ("full", "minimal") and contact_link_indices:
                    contact_summaries = summarize_contact_forces(
                        body_id,
                        plane_id,
                        contact_link_indices,
                        include_inactive=include_inactive_contact_links,
                    )
                    if not include_inactive_contact_links:
                        clear_contact_visuals(contact_debug_items, contact_summaries.keys())
                    update_contact_visuals(
                        contact_summaries,
                        contact_debug_items,
                        total_weight=total_weight,
                        force_scale=args.contact_force_scale,
                        show_text=args.render_mode == "full",
                    )
                configure_camera(body_id)

            if not args.headless and args.max_fps > 0.0:
                target_dt = 1.0 / args.max_fps
                now = time.perf_counter()
                sleep_time = target_dt - (now - last_wall_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
                    now = time.perf_counter()
                last_wall_time = now

            step_index += 1
            if args.max_steps > 0 and step_index >= args.max_steps:
                break
    finally:
        clear_debug_item_group(contact_debug_items)
        clear_debug_item_group(support_polygon_items)
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    atexit.register(cleanup_temp_urdfs)
    main()
