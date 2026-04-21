import argparse
import math
import time
from pathlib import Path

import pybullet as p
import pybullet_data


DEFAULT_URDF = (
    Path(__file__).resolve().parent
    / "../G1_rviz_simulation-main/G1_rviz_simulation-main/install/g1_description/share/g1_description/urdf/g1_29dof_with_hand_rev_1_0_pkg.urdf"
).resolve()
DEFAULT_PACKAGE_SHARE = DEFAULT_URDF.parents[2]
DEFAULT_PLANE_URDF = Path(pybullet_data.getDataPath()) / "plane.urdf"
DEFAULT_CONTACT_LINK_NAMES = {
    "Left": "left_ankle_roll_link",
    "Right": "right_ankle_roll_link",
}
SUPPORT_POLYGON_COLORS = {
    "polygon": (0.95, 0.35, 0.9),
    "com": (0.25, 0.95, 0.95),
    "center": (0.95, 0.25, 0.25),
}


def update_axes(position, orientation, axis_length, axis_width, item_ids=None):
    rotation = p.getMatrixFromQuaternion(orientation)
    x_axis = (rotation[0], rotation[3], rotation[6])
    y_axis = (rotation[1], rotation[4], rotation[7])
    z_axis = (rotation[2], rotation[5], rotation[8])

    axes = (
        (x_axis, (1.0, 0.2, 0.2)),
        (y_axis, (0.2, 1.0, 0.2)),
        (z_axis, (0.2, 0.4, 1.0)),
    )
    line_ids = []
    for axis, color in axes:
        end = (
            position[0] + axis_length * axis[0],
            position[1] + axis_length * axis[1],
            position[2] + axis_length * axis[2],
        )
        replace_id = -1 if item_ids is None else item_ids[len(line_ids)]
        line_id = p.addUserDebugLine(position, end, color, lineWidth=axis_width, replaceItemUniqueId=replace_id)
        line_ids.append(line_id)
    return line_ids


def is_fixed_joint(joint_type):
    return joint_type == p.JOINT_FIXED


def reset_all_joints(body_id):
    for joint_index in range(p.getNumJoints(body_id)):
        p.resetJointState(body_id, joint_index, targetValue=0.0, targetVelocity=0.0)


def enumerate_visualized_joints(body_id, show_fixed_joints):
    visualized_joints = []
    for joint_index in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, joint_index)
        joint_type = joint_info[2]
        if is_fixed_joint(joint_type) and not show_fixed_joints:
            continue
        visualized_joints.append((joint_index, joint_info))
    return visualized_joints


def update_joint_frames(body_id, visualized_joints, axis_length, axis_width, debug_items, show_labels=True):
    for joint_index, joint_info in visualized_joints:
        joint_name = joint_info[1].decode("utf-8")
        link_state = p.getLinkState(body_id, joint_index, computeForwardKinematics=True)
        position = link_state[4]
        orientation = link_state[5]

        item = debug_items.setdefault(joint_index, {})
        item["axes"] = update_axes(
            position,
            orientation,
            axis_length=axis_length,
            axis_width=axis_width,
            item_ids=item.get("axes"),
        )
        if show_labels:
            label_position = (position[0], position[1], position[2] + axis_length * 0.35)
            item["text"] = p.addUserDebugText(
                joint_name,
                label_position,
                textColorRGB=(0.95, 0.95, 0.95),
                textSize=1.0,
                replaceItemUniqueId=item.get("text", -1),
            )


def configure_camera(body_id):
    base_position, _ = p.getBasePositionAndOrientation(body_id)
    p.resetDebugVisualizerCamera(
        cameraDistance=2.2,
        cameraYaw=52.0,
        cameraPitch=-18.0,
        cameraTargetPosition=(base_position[0], base_position[1], base_position[2] + 0.65),
    )


def resolve_default_package_share(urdf_path):
    urdf_path = urdf_path.resolve()
    for parent in urdf_path.parents:
        if parent.name == "g1_description":
            return parent.parent
    return DEFAULT_PACKAGE_SHARE


def build_joint_controllers(body_id):
    controllers = []
    for joint_index in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, joint_index)
        joint_type = joint_info[2]
        if is_fixed_joint(joint_type):
            continue

        joint_name = joint_info[1].decode("utf-8")
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        max_force = joint_info[10] if joint_info[10] > 0.0 else 80.0
        initial_position = p.getJointState(body_id, joint_index)[0]
        slider_id = p.addUserDebugParameter(joint_name, lower_limit, upper_limit, initial_position)
        controllers.append((joint_index, slider_id, max_force))
    return controllers


def apply_joint_controllers(body_id, controllers):
    for joint_index, slider_id, max_force in controllers:
        target_position = p.readUserDebugParameter(slider_id)
        p.setJointMotorControl2(
            body_id,
            joint_index,
            p.POSITION_CONTROL,
            targetPosition=target_position,
            force=max_force,
        )


def build_joint_actuator_map(body_id):
    actuators = {}
    for joint_index in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, joint_index)
        if is_fixed_joint(joint_info[2]):
            continue

        joint_name = joint_info[1].decode("utf-8")
        max_force = joint_info[10] if joint_info[10] > 0.0 else 80.0
        actuators[joint_name] = (joint_index, max_force)
    return actuators


def build_march_targets(t):
    targets = {}
    cycle_hz = 0.75
    left_phase = (t * cycle_hz) % 1.0
    right_phase = (left_phase + 0.5) % 1.0

    stance_pose = {"hip": -0.22, "knee": 0.42, "ankle": -0.22}
    liftoff_pose = {"hip": -0.08, "knee": 0.92, "ankle": -0.52}
    swing_pose = {"hip": 0.24, "knee": 0.86, "ankle": -0.34}
    placement_pose = {"hip": 0.06, "knee": 0.54, "ankle": -0.18}

    def step_pose(phase):
        if phase < 0.12:
            return interpolate_pose(stance_pose, liftoff_pose, phase / 0.12)
        if phase < 0.28:
            return interpolate_pose(liftoff_pose, swing_pose, (phase - 0.12) / 0.16)
        if phase < 0.42:
            return interpolate_pose(swing_pose, placement_pose, (phase - 0.28) / 0.14)
        if phase < 0.5:
            return interpolate_pose(placement_pose, stance_pose, (phase - 0.42) / 0.08)
        return dict(stance_pose)

    for side, phase in (("left", left_phase), ("right", right_phase)):
        pose = step_pose(phase)
        targets[f"{side}_hip_pitch_joint"] = pose["hip"]
        targets[f"{side}_knee_joint"] = pose["knee"]
        targets[f"{side}_ankle_pitch_joint"] = pose["ankle"]

    arm_swing = math.sin(2.0 * math.pi * cycle_hz * t)
    targets["left_shoulder_pitch_joint"] = -0.45 * arm_swing
    targets["right_shoulder_pitch_joint"] = 0.45 * arm_swing
    return targets


def read_dummy_imu(body_id):
    _, base_orientation = p.getBasePositionAndOrientation(body_id)
    _, angular_velocity = p.getBaseVelocity(body_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(base_orientation)
    return {
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "roll_rate": angular_velocity[0],
        "pitch_rate": angular_velocity[1],
        "yaw_rate": angular_velocity[2],
    }


def clamp(value, limit):
    return max(-limit, min(limit, value))


def blend(a, b, alpha):
    return a + (b - a) * alpha


def interpolate_pose(pose_a, pose_b, alpha):
    return {name: blend(pose_a[name], pose_b[name], alpha) for name in pose_a}


def apply_balance_feedback(targets, imu_state, args):
    pitch_correction = -(
        args.balance_pitch_kp * imu_state["pitch"]
        + args.balance_pitch_kd * imu_state["pitch_rate"]
    )
    roll_correction = -(
        args.balance_roll_kp * imu_state["roll"]
        + args.balance_roll_kd * imu_state["roll_rate"]
    )

    pitch_correction = clamp(pitch_correction, args.balance_pitch_limit)
    roll_correction = clamp(roll_correction, args.balance_roll_limit)

    for side in ("left", "right"):
        hip_pitch_name = f"{side}_hip_pitch_joint"
        ankle_pitch_name = f"{side}_ankle_pitch_joint"
        if hip_pitch_name in targets:
            targets[hip_pitch_name] += 0.35 * pitch_correction
        if ankle_pitch_name in targets:
            targets[ankle_pitch_name] += pitch_correction

    if "left_hip_roll_joint" in targets:
        targets["left_hip_roll_joint"] += roll_correction
    else:
        targets["left_hip_roll_joint"] = roll_correction
    if "right_hip_roll_joint" in targets:
        targets["right_hip_roll_joint"] += roll_correction
    else:
        targets["right_hip_roll_joint"] = roll_correction

    if "left_ankle_roll_joint" in targets:
        targets["left_ankle_roll_joint"] += 0.7 * roll_correction
    else:
        targets["left_ankle_roll_joint"] = 0.7 * roll_correction
    if "right_ankle_roll_joint" in targets:
        targets["right_ankle_roll_joint"] += 0.7 * roll_correction
    else:
        targets["right_ankle_roll_joint"] = 0.7 * roll_correction

    return targets


def build_mass_properties(body_id):
    masses = [(-1, p.getDynamicsInfo(body_id, -1)[0])]
    for joint_index in range(p.getNumJoints(body_id)):
        masses.append((joint_index, p.getDynamicsInfo(body_id, joint_index)[0]))
    return [(link_index, mass) for link_index, mass in masses if mass > 0.0]


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


def build_dynamic_balance_targets(body_id, t, controller_state, com_position, com_velocity, support_center, imu_state, args):
    controller_state["phase_time"] += args.controller_dt
    phase = controller_state["phase"]
    phase_duration = args.double_support_duration if phase == "double_support" else args.swing_duration
    if controller_state["phase_time"] >= phase_duration:
        controller_state["phase_time"] = 0.0
        if phase == "double_support":
            controller_state["phase"] = "swing"
        else:
            controller_state["phase"] = "double_support"
            controller_state["swing_side"] = "Right" if controller_state["swing_side"] == "Left" else "Left"
        phase = controller_state["phase"]

    _, base_orientation = p.getBasePositionAndOrientation(body_id)
    _, _, yaw = p.getEulerFromQuaternion(base_orientation)
    if support_center is None:
        support_center = (com_position[0], com_position[1], 0.0)

    com_error_world = (com_position[0] - support_center[0], com_position[1] - support_center[1])
    com_error_body = project_world_vector_to_body_frame(com_error_world, yaw)
    com_velocity_body = project_world_vector_to_body_frame((com_velocity[0], com_velocity[1]), yaw)

    pitch_feedback = clamp(
        -(args.com_balance_kp * com_error_body[0] + args.com_balance_kd * com_velocity_body[0]),
        args.balance_pitch_limit,
    )
    roll_feedback = clamp(
        -(args.com_balance_roll_kp * com_error_body[1] + args.com_balance_roll_kd * com_velocity_body[1]),
        args.balance_roll_limit,
    )

    targets = apply_balance_feedback({}, imu_state, args)
    pitch_feedback += 0.35 * targets.get("left_ankle_pitch_joint", 0.0)
    roll_feedback += 0.35 * targets.get("left_ankle_roll_joint", 0.0)

    stance_leg = {"hip": -0.28, "knee": 0.52, "ankle": -0.24}
    preload_leg = {"hip": -0.18, "knee": 0.62, "ankle": -0.30}
    liftoff_leg = {"hip": -0.02, "knee": 1.00, "ankle": -0.58}
    mid_swing_leg = {"hip": 0.18, "knee": 0.88, "ankle": -0.26}
    touchdown_leg = {"hip": 0.00, "knee": 0.56, "ankle": -0.18}

    swing_side = controller_state["swing_side"]
    support_sign = 1.0 if swing_side == "Left" else -1.0
    dynamic_targets = {}

    if phase == "double_support":
        preload_alpha = min(1.0, controller_state["phase_time"] / max(args.double_support_duration, 1e-6))
        swing_pose = interpolate_pose(stance_leg, preload_leg, preload_alpha)
        for side in ("Left", "Right"):
            pose = swing_pose if side == swing_side else dict(stance_leg)
            dynamic_targets[f"{side.lower()}_hip_pitch_joint"] = pose["hip"]
            dynamic_targets[f"{side.lower()}_knee_joint"] = pose["knee"]
            dynamic_targets[f"{side.lower()}_ankle_pitch_joint"] = pose["ankle"]
    else:
        swing_progress = min(1.0, controller_state["phase_time"] / max(args.swing_duration, 1e-6))
        step_forward = clamp(
            args.nominal_step_length
            + args.step_placement_kp * com_error_body[0]
            + args.step_placement_kd * com_velocity_body[0],
            args.max_step_length,
        )
        if swing_progress < 0.22:
            swing_pose = interpolate_pose(preload_leg, liftoff_leg, swing_progress / 0.22)
        elif swing_progress < 0.68:
            target_pose = dict(mid_swing_leg)
            target_pose["hip"] += step_forward
            swing_pose = interpolate_pose(liftoff_leg, target_pose, (swing_progress - 0.22) / 0.46)
        else:
            landing_pose = dict(touchdown_leg)
            landing_pose["hip"] += 0.35 * step_forward
            swing_pose = interpolate_pose(mid_swing_leg, landing_pose, (swing_progress - 0.68) / 0.32)

        for side in ("Left", "Right"):
            if side == swing_side:
                pose = swing_pose
            else:
                pose = dict(stance_leg)
                pose["hip"] += 0.2 * pitch_feedback
                pose["ankle"] += pitch_feedback
            dynamic_targets[f"{side.lower()}_hip_pitch_joint"] = pose["hip"]
            dynamic_targets[f"{side.lower()}_knee_joint"] = pose["knee"]
            dynamic_targets[f"{side.lower()}_ankle_pitch_joint"] = pose["ankle"]
        roll_feedback += support_sign * args.support_shift_roll

    dynamic_targets["left_hip_roll_joint"] = roll_feedback
    dynamic_targets["right_hip_roll_joint"] = roll_feedback
    dynamic_targets["left_ankle_roll_joint"] = 0.8 * roll_feedback
    dynamic_targets["right_ankle_roll_joint"] = 0.8 * roll_feedback

    gait_frequency = 1.0 / max(args.double_support_duration + args.swing_duration, 1e-6)
    arm_swing = math.sin(2.0 * math.pi * gait_frequency * t)
    dynamic_targets["left_shoulder_pitch_joint"] = -0.3 * arm_swing
    dynamic_targets["right_shoulder_pitch_joint"] = 0.3 * arm_swing
    return dynamic_targets


def select_recovery_swing_side(com_error_body, controller_state):
    lateral_error = com_error_body[1]
    if lateral_error > 0.0:
        return "Right"
    if lateral_error < 0.0:
        return "Left"
    return controller_state["swing_side"]


def maybe_activate_recovery_mode(controller_state, polygon_margin, com_error_body, com_velocity_body, args):
    outward_motion = (
        abs(com_velocity_body[0]) > args.recovery_velocity_threshold
        or abs(com_velocity_body[1]) > args.recovery_velocity_threshold
    )
    if polygon_margin < args.recovery_margin or (polygon_margin < args.recovery_margin * 1.6 and outward_motion):
        if controller_state["mode"] != "recovery":
            controller_state["mode"] = "recovery"
            controller_state["phase"] = "double_support"
            controller_state["phase_time"] = 0.0
            controller_state["recovery_hold"] = 0.0
            controller_state["swing_side"] = select_recovery_swing_side(com_error_body, controller_state)


def maybe_release_recovery_mode(controller_state, polygon_margin, com_velocity_body, args):
    if controller_state["mode"] != "recovery":
        return
    safe_margin = polygon_margin > args.recovery_release_margin
    low_velocity = (
        abs(com_velocity_body[0]) < args.recovery_release_velocity
        and abs(com_velocity_body[1]) < args.recovery_release_velocity
    )
    if safe_margin and low_velocity:
        controller_state["recovery_hold"] += args.controller_dt
        if controller_state["recovery_hold"] >= args.recovery_hold_time:
            controller_state["mode"] = "nominal"
            controller_state["phase"] = "double_support"
            controller_state["phase_time"] = 0.0
            controller_state["recovery_hold"] = 0.0
    else:
        controller_state["recovery_hold"] = 0.0


def build_recovery_targets(body_id, controller_state, com_position, com_velocity, support_center, imu_state, args):
    controller_state["phase_time"] += args.controller_dt
    phase = controller_state["phase"]
    phase_duration = args.recovery_double_support_duration if phase == "double_support" else args.recovery_swing_duration
    if controller_state["phase_time"] >= phase_duration:
        controller_state["phase_time"] = 0.0
        if phase == "double_support":
            controller_state["phase"] = "swing"
        else:
            controller_state["phase"] = "double_support"
            controller_state["swing_side"] = "Right" if controller_state["swing_side"] == "Left" else "Left"
        phase = controller_state["phase"]

    _, base_orientation = p.getBasePositionAndOrientation(body_id)
    _, _, yaw = p.getEulerFromQuaternion(base_orientation)
    com_error_world = (com_position[0] - support_center[0], com_position[1] - support_center[1])
    com_error_body = project_world_vector_to_body_frame(com_error_world, yaw)
    com_velocity_body = project_world_vector_to_body_frame((com_velocity[0], com_velocity[1]), yaw)

    pitch_feedback = clamp(
        -(args.recovery_com_balance_kp * com_error_body[0] + args.recovery_com_balance_kd * com_velocity_body[0]),
        args.recovery_pitch_limit,
    )
    roll_feedback = clamp(
        -(args.recovery_com_balance_roll_kp * com_error_body[1] + args.recovery_com_balance_roll_kd * com_velocity_body[1]),
        args.recovery_roll_limit,
    )
    imu_targets = apply_balance_feedback({}, imu_state, args)
    pitch_feedback += 0.45 * imu_targets.get("left_ankle_pitch_joint", 0.0)
    roll_feedback += 0.45 * imu_targets.get("left_ankle_roll_joint", 0.0)

    stance_leg = {"hip": -0.24, "knee": 0.54, "ankle": -0.20, "hip_roll": 0.0, "ankle_roll": 0.0}
    preload_leg = {"hip": -0.16, "knee": 0.68, "ankle": -0.30, "hip_roll": 0.0, "ankle_roll": 0.0}
    lift_leg = {"hip": 0.02, "knee": 1.05, "ankle": -0.56, "hip_roll": 0.0, "ankle_roll": 0.0}
    swing_leg = {"hip": 0.10, "knee": 0.92, "ankle": -0.22, "hip_roll": 0.0, "ankle_roll": 0.0}
    place_leg = {"hip": -0.02, "knee": 0.58, "ankle": -0.16, "hip_roll": 0.0, "ankle_roll": 0.0}

    step_forward = clamp(
        args.recovery_step_length
        + args.recovery_step_placement_kp * com_error_body[0]
        + args.recovery_step_placement_kd * com_velocity_body[0],
        args.recovery_max_step_length,
    )
    step_lateral = clamp(
        args.recovery_lateral_step_kp * com_error_body[1]
        + args.recovery_lateral_step_kd * com_velocity_body[1],
        args.recovery_max_lateral_step,
    )

    swing_side = controller_state["swing_side"]
    support_sign = 1.0 if swing_side == "Left" else -1.0
    targets = {}

    if phase == "double_support":
        alpha = min(1.0, controller_state["phase_time"] / max(args.recovery_double_support_duration, 1e-6))
        moving_pose = interpolate_pose(stance_leg, preload_leg, alpha)
        moving_pose["hip_roll"] += -support_sign * args.recovery_support_shift_roll
        moving_pose["ankle_roll"] += -support_sign * 0.8 * args.recovery_support_shift_roll
        for side in ("Left", "Right"):
            pose = moving_pose if side == swing_side else dict(stance_leg)
            targets[f"{side.lower()}_hip_pitch_joint"] = pose["hip"]
            targets[f"{side.lower()}_knee_joint"] = pose["knee"]
            targets[f"{side.lower()}_ankle_pitch_joint"] = pose["ankle"]
            targets[f"{side.lower()}_hip_roll_joint"] = pose["hip_roll"] + roll_feedback
            targets[f"{side.lower()}_ankle_roll_joint"] = pose["ankle_roll"] + 0.8 * roll_feedback
    else:
        alpha = min(1.0, controller_state["phase_time"] / max(args.recovery_swing_duration, 1e-6))
        if alpha < 0.25:
            moving_pose = interpolate_pose(preload_leg, lift_leg, alpha / 0.25)
        elif alpha < 0.7:
            swing_target = dict(swing_leg)
            swing_target["hip"] += step_forward
            swing_target["hip_roll"] += -step_lateral
            moving_pose = interpolate_pose(lift_leg, swing_target, (alpha - 0.25) / 0.45)
        else:
            landing_target = dict(place_leg)
            landing_target["hip"] += 0.35 * step_forward
            landing_target["hip_roll"] += -0.6 * step_lateral
            moving_pose = interpolate_pose(swing_leg, landing_target, (alpha - 0.7) / 0.3)

        stance_pose = dict(stance_leg)
        stance_pose["hip"] += 0.25 * pitch_feedback
        stance_pose["ankle"] += pitch_feedback
        stance_pose["hip_roll"] += support_sign * args.recovery_support_shift_roll + roll_feedback
        stance_pose["ankle_roll"] += support_sign * 0.8 * args.recovery_support_shift_roll + 0.8 * roll_feedback
        moving_pose["hip_roll"] += -step_lateral + 0.35 * roll_feedback
        moving_pose["ankle_roll"] += -0.5 * step_lateral + 0.25 * roll_feedback

        for side in ("Left", "Right"):
            pose = moving_pose if side == swing_side else stance_pose
            targets[f"{side.lower()}_hip_pitch_joint"] = pose["hip"]
            targets[f"{side.lower()}_knee_joint"] = pose["knee"]
            targets[f"{side.lower()}_ankle_pitch_joint"] = pose["ankle"]
            targets[f"{side.lower()}_hip_roll_joint"] = pose["hip_roll"]
            targets[f"{side.lower()}_ankle_roll_joint"] = pose["ankle_roll"]

    arm_pitch = clamp(-(args.recovery_arm_pitch_kp * com_error_body[0] + args.recovery_arm_pitch_kd * com_velocity_body[0]), 0.85)
    arm_roll = clamp(-(args.recovery_arm_roll_kp * com_error_body[1] + args.recovery_arm_roll_kd * com_velocity_body[1]), 0.55)
    targets["left_shoulder_pitch_joint"] = arm_pitch
    targets["right_shoulder_pitch_joint"] = arm_pitch
    targets["left_shoulder_roll_joint"] = arm_roll
    targets["right_shoulder_roll_joint"] = -arm_roll
    return targets, com_error_body, com_velocity_body


def apply_named_targets(body_id, actuators, targets):
    for joint_name, target_position in targets.items():
        actuator = actuators.get(joint_name)
        if actuator is None:
            continue

        joint_index, max_force = actuator
        p.setJointMotorControl2(
            body_id,
            joint_index,
            p.POSITION_CONTROL,
            targetPosition=target_position,
            force=max_force,
        )


def build_link_name_map(body_id):
    link_name_map = {}
    for joint_index in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, joint_index)
        link_name_map[joint_info[12].decode("utf-8")] = joint_index
    return link_name_map


def build_contact_link_map(link_name_map, contact_links_arg):
    if contact_links_arg == "feet":
        return {
            label: link_name_map[link_name]
            for label, link_name in DEFAULT_CONTACT_LINK_NAMES.items()
            if link_name in link_name_map
        }, True

    if contact_links_arg == "all":
        return {link_name: joint_index for link_name, joint_index in link_name_map.items()}, False

    selected_links = {}
    for link_name in (part.strip() for part in contact_links_arg.split(",")):
        if not link_name:
            continue
        if link_name in link_name_map:
            selected_links[link_name] = link_name_map[link_name]
    return selected_links, True


def get_default_foot_link_indices(link_name_map):
    return {
        side: link_name_map[link_name]
        for side, link_name in DEFAULT_CONTACT_LINK_NAMES.items()
        if link_name in link_name_map
    }


def get_link_contact_points(body_id, plane_id, link_index):
    contacts = p.getContactPoints(bodyA=body_id, bodyB=plane_id, linkIndexA=link_index)
    return [contact for contact in contacts if contact[9] > 1e-3]


def get_active_foot_supports(body_id, plane_id, foot_link_indices):
    supports = {}
    for side, link_index in foot_link_indices.items():
        contacts = get_link_contact_points(body_id, plane_id, link_index)
        if contacts:
            supports[side] = contacts
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

    projection = (
        ((point[0] - segment_start[0]) * seg_x + (point[1] - segment_start[1]) * seg_y) / seg_len_sq
    )
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
    for side in active_supports:
        polygon_points.extend(get_link_support_corners(body_id, foot_link_indices[side], polygon_height))
    return convex_hull_2d(polygon_points)


def compute_total_mass(body_id):
    total_mass = p.getDynamicsInfo(body_id, -1)[0]
    for joint_index in range(p.getNumJoints(body_id)):
        total_mass += p.getDynamicsInfo(body_id, joint_index)[0]
    return total_mass


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


def configure_visualizer(render_mode):
    gui_enabled = 1 if render_mode == "full" else 0
    shadows_enabled = 1 if render_mode == "full" else 0
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, gui_enabled)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, shadows_enabled)


def main():
    parser = argparse.ArgumentParser(description="Render the Unitree G1 URDF in PyBullet and show joint coordinate frames.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help=f"URDF file to load (default: {DEFAULT_URDF})")
    parser.add_argument(
        "--package-share",
        type=Path,
        default=None,
        help="Directory that contains the g1_description package, used to resolve package:// mesh paths.",
    )
    parser.add_argument("--axis-length", type=float, default=0.06, help="Length of each debug axis in meters.")
    parser.add_argument("--axis-width", type=float, default=2.0, help="Debug line width for the joint axes.")
    parser.add_argument("--show-fixed-joints", action="store_true", help="Show coordinate systems for fixed joints too.")
    parser.add_argument("--fixed-base", action="store_true", help="Keep the robot base fixed instead of simulating full-body gravity/contact.")
    parser.add_argument("--start-height", type=float, default=0.82, help="Initial pelvis height in meters.")
    parser.add_argument("--contact-force-scale", type=float, default=0.0025, help="Meters per Newton for contact-force arrows.")
    parser.add_argument(
        "--contact-links",
        type=str,
        default="feet",
        help="Contact force visualization target: 'feet', 'all', or a comma-separated list of link names.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("full", "minimal", "off"),
        default="full",
        help="Amount of nonessential rendering: full debug overlays, minimal overlays, or none.",
    )
    parser.add_argument(
        "--debug-update-every",
        type=int,
        default=1,
        help="Only refresh debug overlays every N physics steps to reduce rendering overhead.",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=240.0,
        help="GUI pacing limit. Set to 0 to run uncapped.",
    )
    parser.add_argument(
        "--motion",
        choices=("march", "manual"),
        default="march",
        help="Joint command source: marching trajectory or manual debug sliders.",
    )
    parser.add_argument(
        "--balance-control",
        action="store_true",
        help="Use base orientation and angular velocity as a dummy IMU and apply stabilizing joint offsets.",
    )
    parser.add_argument("--balance-pitch-kp", type=float, default=1.4, help="Pitch proportional gain for the balance layer.")
    parser.add_argument("--balance-pitch-kd", type=float, default=0.18, help="Pitch derivative gain for the balance layer.")
    parser.add_argument("--balance-roll-kp", type=float, default=0.9, help="Roll proportional gain for the balance layer.")
    parser.add_argument("--balance-roll-kd", type=float, default=0.12, help="Roll derivative gain for the balance layer.")
    parser.add_argument("--balance-pitch-limit", type=float, default=0.35, help="Maximum absolute pitch correction in radians.")
    parser.add_argument("--balance-roll-limit", type=float, default=0.25, help="Maximum absolute roll correction in radians.")
    parser.add_argument("--controller-dt", type=float, default=1.0 / 240.0, help="Internal balance-controller step in seconds.")
    parser.add_argument("--double-support-duration", type=float, default=0.16, help="Time spent with both feet planted before each swing.")
    parser.add_argument("--swing-duration", type=float, default=0.34, help="Time spent in single-support swing.")
    parser.add_argument("--nominal-step-length", type=float, default=0.08, help="Baseline swing-foot forward placement in joint-space radians.")
    parser.add_argument("--max-step-length", type=float, default=0.22, help="Maximum extra swing-foot forward placement in joint-space radians.")
    parser.add_argument("--step-placement-kp", type=float, default=0.55, help="COM position gain for adaptive foot placement.")
    parser.add_argument("--step-placement-kd", type=float, default=0.18, help="COM velocity gain for adaptive foot placement.")
    parser.add_argument("--com-balance-kp", type=float, default=3.0, help="COM-to-support-center proportional gain in the sagittal plane.")
    parser.add_argument("--com-balance-kd", type=float, default=1.1, help="COM velocity gain in the sagittal plane.")
    parser.add_argument("--com-balance-roll-kp", type=float, default=2.6, help="COM-to-support-center proportional gain in the lateral plane.")
    parser.add_argument("--com-balance-roll-kd", type=float, default=0.9, help="COM velocity gain in the lateral plane.")
    parser.add_argument("--support-shift-roll", type=float, default=0.08, help="Extra roll bias toward the stance foot during single support.")
    parser.add_argument("--support-polygon-height", type=float, default=0.005, help="Height offset for the support polygon debug overlay.")
    parser.add_argument("--recovery-margin", type=float, default=0.03, help="Support-polygon margin threshold that triggers recovery stepping.")
    parser.add_argument("--recovery-release-margin", type=float, default=0.055, help="Margin required before recovery mode can release.")
    parser.add_argument("--recovery-velocity-threshold", type=float, default=0.04, help="COM planar velocity threshold for early recovery triggering.")
    parser.add_argument("--recovery-release-velocity", type=float, default=0.02, help="COM planar velocity required to exit recovery mode.")
    parser.add_argument("--recovery-hold-time", type=float, default=0.18, help="Stable dwell time before leaving recovery mode.")
    parser.add_argument("--recovery-double-support-duration", type=float, default=0.10, help="Double-support time during recovery stepping.")
    parser.add_argument("--recovery-swing-duration", type=float, default=0.20, help="Swing time during recovery stepping.")
    parser.add_argument("--recovery-step-length", type=float, default=0.14, help="Baseline forward joint-space step used in recovery mode.")
    parser.add_argument("--recovery-max-step-length", type=float, default=0.32, help="Max forward joint-space recovery step.")
    parser.add_argument("--recovery-max-lateral-step", type=float, default=0.22, help="Max lateral joint-space recovery step.")
    parser.add_argument("--recovery-step-placement-kp", type=float, default=1.0, help="COM position gain for forward recovery stepping.")
    parser.add_argument("--recovery-step-placement-kd", type=float, default=0.38, help="COM velocity gain for forward recovery stepping.")
    parser.add_argument("--recovery-lateral-step-kp", type=float, default=1.4, help="COM position gain for lateral recovery stepping.")
    parser.add_argument("--recovery-lateral-step-kd", type=float, default=0.5, help="COM velocity gain for lateral recovery stepping.")
    parser.add_argument("--recovery-com-balance-kp", type=float, default=4.2, help="Sagittal COM gain during recovery mode.")
    parser.add_argument("--recovery-com-balance-kd", type=float, default=1.8, help="Sagittal COM velocity gain during recovery mode.")
    parser.add_argument("--recovery-com-balance-roll-kp", type=float, default=4.0, help="Lateral COM gain during recovery mode.")
    parser.add_argument("--recovery-com-balance-roll-kd", type=float, default=1.5, help="Lateral COM velocity gain during recovery mode.")
    parser.add_argument("--recovery-support-shift-roll", type=float, default=0.16, help="Extra stance-foot roll bias during recovery stepping.")
    parser.add_argument("--recovery-pitch-limit", type=float, default=0.5, help="Pitch correction clamp in recovery mode.")
    parser.add_argument("--recovery-roll-limit", type=float, default=0.42, help="Roll correction clamp in recovery mode.")
    parser.add_argument("--recovery-arm-pitch-kp", type=float, default=3.2, help="Arm pitch gain for COM compensation in recovery mode.")
    parser.add_argument("--recovery-arm-pitch-kd", type=float, default=1.2, help="Arm pitch velocity gain in recovery mode.")
    parser.add_argument("--recovery-arm-roll-kp", type=float, default=2.4, help="Arm roll gain for lateral COM compensation in recovery mode.")
    parser.add_argument("--recovery-arm-roll-kd", type=float, default=0.9, help="Arm roll velocity gain in recovery mode.")
    args = parser.parse_args()

    urdf_path = args.urdf.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    package_share = args.package_share.resolve() if args.package_share is not None else resolve_default_package_share(urdf_path)
    if not package_share.is_dir():
        raise FileNotFoundError(f"Package share directory not found: {package_share}")
    if not DEFAULT_PLANE_URDF.is_file():
        raise FileNotFoundError(f"PyBullet plane URDF not found: {DEFAULT_PLANE_URDF}")

    client_id = p.connect(p.GUI)
    if client_id < 0:
        raise RuntimeError("Failed to connect to the PyBullet GUI")

    p.setAdditionalSearchPath(str(package_share))
    p.setGravity(0.0, 0.0, -9.81)
    configure_visualizer(args.render_mode)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    plane_id = p.loadURDF(str(DEFAULT_PLANE_URDF))
    p.changeDynamics(plane_id, -1, lateralFriction=1.0, spinningFriction=0.02, rollingFriction=0.02)

    start_height = args.start_height
    start_orientation = p.getQuaternionFromEuler((0.0, 0.0, math.pi / 2.0))
    body_id = p.loadURDF(
        str(urdf_path),
        basePosition=(0.0, 0.0, start_height),
        baseOrientation=start_orientation,
        useFixedBase=args.fixed_base,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    reset_all_joints(body_id)
    for joint_index in range(p.getNumJoints(body_id)):
        p.changeDynamics(body_id, joint_index, lateralFriction=1.0, spinningFriction=0.02, rollingFriction=0.02)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    visualized_joints = enumerate_visualized_joints(body_id, args.show_fixed_joints)
    joint_frame_items = {}
    base_axes_items = None
    joint_controllers = build_joint_controllers(body_id) if args.motion == "manual" else []
    joint_actuators = build_joint_actuator_map(body_id)
    link_name_map = build_link_name_map(body_id)
    foot_link_indices = get_default_foot_link_indices(link_name_map)
    contact_link_indices, include_inactive_contact_links = build_contact_link_map(link_name_map, args.contact_links)
    contact_debug_items = {}
    support_polygon_items = {}
    mass_properties = build_mass_properties(body_id)
    dynamic_balance_state = {
        "mode": "nominal",
        "phase": "double_support",
        "phase_time": 0.0,
        "recovery_hold": 0.0,
        "swing_side": "Left",
    }
    total_weight = compute_total_mass(body_id) * 9.81

    configure_camera(body_id)

    try:
        motion_start_time = time.perf_counter()
        simulation_step = 0
        debug_update_every = max(1, args.debug_update_every)
        show_base_axes = args.render_mode == "full"
        show_joint_frames = args.render_mode in ("full", "minimal")
        show_joint_labels = args.render_mode == "full"
        show_contact_visuals = args.render_mode in ("full", "minimal")
        show_contact_text = args.render_mode == "full"
        while p.isConnected():
            com_position, com_velocity = compute_center_of_mass(body_id, mass_properties)
            active_foot_supports = get_active_foot_supports(body_id, plane_id, foot_link_indices)
            support_polygon = build_support_polygon(
                body_id,
                foot_link_indices,
                active_foot_supports,
                args.support_polygon_height,
            )
            support_center = polygon_centroid_2d(support_polygon)
            elapsed = time.perf_counter() - motion_start_time
            polygon_margin = support_polygon_margin((com_position[0], com_position[1]), support_polygon)
            imu_state = read_dummy_imu(body_id)
            _, _, yaw = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(body_id)[1])
            if support_center is None:
                support_center = (com_position[0], com_position[1], args.support_polygon_height)
            com_error_body = project_world_vector_to_body_frame(
                (com_position[0] - support_center[0], com_position[1] - support_center[1]),
                yaw,
            )
            com_velocity_body = project_world_vector_to_body_frame((com_velocity[0], com_velocity[1]), yaw)

            recovery_targets = None
            if args.balance_control and not args.fixed_base:
                maybe_activate_recovery_mode(dynamic_balance_state, polygon_margin, com_error_body, com_velocity_body, args)
                if dynamic_balance_state["mode"] == "recovery":
                    recovery_targets, com_error_body, com_velocity_body = build_recovery_targets(
                        body_id,
                        dynamic_balance_state,
                        com_position,
                        com_velocity,
                        support_center,
                        imu_state,
                        args,
                    )
                    maybe_release_recovery_mode(dynamic_balance_state, polygon_margin, com_velocity_body, args)

            if recovery_targets is not None:
                apply_named_targets(body_id, joint_actuators, recovery_targets)
            elif args.motion == "manual":
                apply_joint_controllers(body_id, joint_controllers)
            else:
                targets = build_march_targets(elapsed)
                if args.balance_control and not args.fixed_base:
                    dynamic_balance_state["mode"] = "nominal"
                    targets = build_dynamic_balance_targets(
                        body_id,
                        elapsed,
                        dynamic_balance_state,
                        com_position,
                        com_velocity,
                        support_center,
                        imu_state,
                        args,
                    )
                apply_named_targets(body_id, joint_actuators, targets)
            p.stepSimulation()
            simulation_step += 1

            if simulation_step % debug_update_every == 0:
                if show_base_axes:
                    base_position, base_orientation = p.getBasePositionAndOrientation(body_id)
                    base_axes_items = update_axes(
                        base_position,
                        base_orientation,
                        axis_length=args.axis_length * 1.4,
                        axis_width=args.axis_width,
                        item_ids=base_axes_items,
                    )
                if show_joint_frames:
                    update_joint_frames(
                        body_id,
                        visualized_joints,
                        axis_length=args.axis_length,
                        axis_width=args.axis_width,
                        debug_items=joint_frame_items,
                        show_labels=show_joint_labels,
                    )
                if args.balance_control and support_polygon:
                    update_support_polygon_visuals(
                        support_polygon,
                        com_position,
                        support_center,
                        support_polygon_items,
                        show_text=show_joint_labels,
                    )
                elif support_polygon_items:
                    clear_debug_item_group(support_polygon_items)
                if show_contact_visuals and contact_link_indices:
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
                        show_text=show_contact_text,
                    )

            if args.max_fps > 0.0:
                time.sleep(1.0 / args.max_fps)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
