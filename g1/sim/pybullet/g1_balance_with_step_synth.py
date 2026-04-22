import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import pybullet as p
import pybullet_data

from g1_test import (
    DEFAULT_PLANE_URDF,
    DEFAULT_URDF,
    build_link_name_map,
    build_mass_properties,
    build_support_polygon,
    compute_center_of_mass,
    get_active_foot_supports,
    get_default_foot_link_indices,
    polygon_centroid_2d,
    project_world_vector_to_body_frame,
    read_dummy_imu,
    resolve_default_package_share,
    support_polygon_margin,
    update_support_polygon_visuals,
)


STAND_POSE = {
    "left_hip_pitch_joint": -0.24,
    "left_hip_roll_joint": 0.02,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.52,
    "left_ankle_pitch_joint": -0.28,
    "left_ankle_roll_joint": -0.02,
    "right_hip_pitch_joint": -0.24,
    "right_hip_roll_joint": -0.02,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.52,
    "right_ankle_pitch_joint": -0.28,
    "right_ankle_roll_joint": 0.02,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.05,
    "left_shoulder_pitch_joint": 0.20,
    "left_shoulder_roll_joint": 0.08,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.45,
    "right_shoulder_pitch_joint": 0.20,
    "right_shoulder_roll_joint": -0.08,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.45,
}


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def lerp(a, b, alpha):
    return a + (b - a) * alpha


def interpolate_pose(pose_a, pose_b, alpha):
    return {name: lerp(pose_a[name], pose_b[name], alpha) for name in pose_a}


@dataclass
class BalanceSnapshot:
    com_position: tuple[float, float, float]
    com_velocity: tuple[float, float, float]
    support_center: tuple[float, float, float]
    support_margin: float
    support_polygon: list[tuple[float, float, float]]
    imu_state: dict
    com_error_body: tuple[float, float]
    com_velocity_body: tuple[float, float]
    capture_point_body: tuple[float, float]
    omega: float
    pitch_feedback: float
    roll_feedback: float
    assist_scale: float


@dataclass
class StepState:
    mode: str = "stand"
    swing_side: str = "Left"
    phase: str = "double_support"
    phase_time: float = 0.0
    cooldown: float = 0.0
    desired_forward_step: float = 0.0
    desired_lateral_step: float = 0.0


def connect_pybullet(headless):
    client_id = p.connect(p.DIRECT if headless else p.GUI)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet.")
    return client_id


def build_joint_catalog(body_id):
    catalog = {}
    for joint_index in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, joint_index)
        if joint_info[2] == p.JOINT_FIXED:
            continue
        joint_name = joint_info[1].decode("utf-8")
        max_force = joint_info[10] if joint_info[10] > 0.0 else 80.0
        catalog[joint_name] = {"index": joint_index, "max_force": max_force}
    return catalog


def reset_robot_to_stand_pose(body_id, joint_catalog):
    for joint_name, joint_data in joint_catalog.items():
        target = STAND_POSE.get(joint_name, 0.0)
        p.resetJointState(body_id, joint_data["index"], targetValue=target, targetVelocity=0.0)


def configure_scene(body_id, plane_id, dt):
    p.setGravity(0.0, 0.0, -9.81)
    p.setTimeStep(dt)
    p.changeDynamics(plane_id, -1, lateralFriction=1.2, spinningFriction=0.02, rollingFriction=0.02)
    for joint_index in range(p.getNumJoints(body_id)):
        p.changeDynamics(body_id, joint_index, lateralFriction=1.2, spinningFriction=0.02, rollingFriction=0.02)


def gain_profile_for_joint(joint_name):
    if "hip_pitch" in joint_name:
        return 0.42, 0.88
    if "hip_roll" in joint_name:
        return 0.40, 0.86
    if "hip_yaw" in joint_name:
        return 0.28, 0.75
    if "knee" in joint_name:
        return 0.50, 0.92
    if "ankle_pitch" in joint_name:
        return 0.34, 0.82
    if "ankle_roll" in joint_name:
        return 0.32, 0.80
    if "waist" in joint_name:
        return 0.24, 0.68
    if "shoulder" in joint_name:
        return 0.18, 0.55
    if "elbow" in joint_name:
        return 0.14, 0.45
    return 0.10, 0.35


def choose_recovery_swing_side(snapshot):
    return "Right" if snapshot.com_error_body[1] > 0.0 else "Left"


def compute_balance_snapshot(body_id, plane_id, mass_properties, foot_link_indices, args):
    com_position, com_velocity = compute_center_of_mass(body_id, mass_properties)
    active_supports = get_active_foot_supports(body_id, plane_id, foot_link_indices)
    support_polygon = build_support_polygon(body_id, foot_link_indices, active_supports, args.support_polygon_height)
    support_center = polygon_centroid_2d(support_polygon)
    if support_center is None:
        support_center = (com_position[0], com_position[1], args.support_polygon_height)

    imu_state = read_dummy_imu(body_id)
    _, base_orientation = p.getBasePositionAndOrientation(body_id)
    _, _, yaw = p.getEulerFromQuaternion(base_orientation)
    com_error_body = project_world_vector_to_body_frame(
        (com_position[0] - support_center[0], com_position[1] - support_center[1]),
        yaw,
    )
    com_velocity_body = project_world_vector_to_body_frame((com_velocity[0], com_velocity[1]), yaw)
    nominal_com_height = max(args.nominal_com_height, com_position[2] - args.support_polygon_height)
    omega = math.sqrt(9.81 / max(nominal_com_height, 1e-3))
    capture_point_body = (
        com_error_body[0] + com_velocity_body[0] / omega,
        com_error_body[1] + com_velocity_body[1] / omega,
    )

    pitch_feedback = -(
        args.com_pitch_kp * com_error_body[0]
        + args.com_pitch_kd * com_velocity_body[0]
        + args.imu_pitch_kp * imu_state["pitch"]
        + args.imu_pitch_kd * imu_state["pitch_rate"]
    )
    roll_feedback = -(
        args.com_roll_kp * com_error_body[1]
        + args.com_roll_kd * com_velocity_body[1]
        + args.imu_roll_kp * imu_state["roll"]
        + args.imu_roll_kd * imu_state["roll_rate"]
    )
    pitch_feedback = clamp(pitch_feedback, -args.pitch_limit, args.pitch_limit)
    roll_feedback = clamp(roll_feedback, -args.roll_limit, args.roll_limit)

    support_margin = support_polygon_margin((com_position[0], com_position[1]), support_polygon)
    assist_scale = 1.0
    if support_margin < args.assist_margin:
        assist_scale += clamp((args.assist_margin - support_margin) * args.assist_gain, 0.0, args.max_assist_scale - 1.0)

    return BalanceSnapshot(
        com_position=com_position,
        com_velocity=com_velocity,
        support_center=support_center,
        support_margin=support_margin,
        support_polygon=support_polygon,
        imu_state=imu_state,
        com_error_body=com_error_body,
        com_velocity_body=com_velocity_body,
        capture_point_body=capture_point_body,
        omega=omega,
        pitch_feedback=pitch_feedback,
        roll_feedback=roll_feedback,
        assist_scale=assist_scale,
    )


def update_step_state(step_state, snapshot, args):
    step_state.cooldown = max(0.0, step_state.cooldown - args.dt)
    if step_state.mode == "stand":
        capture_point_outside = (
            abs(snapshot.capture_point_body[0]) > args.capture_trigger_x
            or abs(snapshot.capture_point_body[1]) > args.capture_trigger_y
        )
        large_velocity = (
            abs(snapshot.com_velocity_body[0]) > args.step_trigger_velocity
            or abs(snapshot.com_velocity_body[1]) > args.step_trigger_velocity
        )
        if snapshot.support_margin < args.step_trigger_margin or (capture_point_outside and large_velocity and step_state.cooldown <= 0.0):
            step_state.mode = "step"
            step_state.phase = "double_support"
            step_state.phase_time = 0.0
            step_state.swing_side = choose_recovery_swing_side(snapshot)
            step_state.desired_forward_step = clamp(
                args.nominal_step_length
                + args.capture_step_gain_x * snapshot.capture_point_body[0],
                -args.max_step_length,
                args.max_step_length,
            )
            step_state.desired_lateral_step = clamp(
                args.capture_step_gain_y * snapshot.capture_point_body[1],
                -args.max_lateral_step,
                args.max_lateral_step,
            )
    else:
        step_state.phase_time += args.dt
        phase_duration = args.double_support_duration if step_state.phase == "double_support" else args.swing_duration
        if step_state.phase_time >= phase_duration:
            step_state.phase_time = 0.0
            if step_state.phase == "double_support":
                step_state.phase = "swing"
            else:
                step_state.mode = "stand"
                step_state.phase = "double_support"
                step_state.cooldown = args.step_cooldown


def build_joint_targets(snapshot, step_state, args):
    pitch = snapshot.pitch_feedback
    roll = snapshot.roll_feedback
    assist_scale = snapshot.assist_scale

    targets = dict(STAND_POSE)
    targets["left_hip_pitch_joint"] += 0.22 * pitch * assist_scale
    targets["right_hip_pitch_joint"] += 0.22 * pitch * assist_scale
    targets["left_knee_joint"] += -0.12 * pitch * assist_scale
    targets["right_knee_joint"] += -0.12 * pitch * assist_scale
    targets["left_ankle_pitch_joint"] += 0.90 * pitch * assist_scale
    targets["right_ankle_pitch_joint"] += 0.90 * pitch * assist_scale
    targets["left_hip_roll_joint"] += roll * assist_scale
    targets["right_hip_roll_joint"] += roll * assist_scale
    targets["left_ankle_roll_joint"] += 0.85 * roll * assist_scale
    targets["right_ankle_roll_joint"] += 0.85 * roll * assist_scale
    targets["waist_pitch_joint"] += -0.18 * pitch
    targets["waist_roll_joint"] += -0.24 * roll
    targets["left_shoulder_pitch_joint"] += -0.12 * pitch
    targets["right_shoulder_pitch_joint"] += -0.12 * pitch
    targets["left_shoulder_roll_joint"] += -0.18 * roll
    targets["right_shoulder_roll_joint"] += -0.18 * roll

    if step_state.mode != "step":
        return targets

    swing_side = step_state.swing_side
    stance_side = "Right" if swing_side == "Left" else "Left"
    support_sign = 1.0 if stance_side == "Left" else -1.0
    double_support_progress = min(1.0, step_state.phase_time / max(args.double_support_duration, 1e-6))
    swing_progress = min(1.0, step_state.phase_time / max(args.swing_duration, 1e-6))

    stance_leg = {"hip": -0.24, "knee": 0.52, "ankle": -0.26, "hip_roll": 0.0, "ankle_roll": 0.0}
    preload_leg = {"hip": -0.18, "knee": 0.66, "ankle": -0.34, "hip_roll": 0.0, "ankle_roll": 0.0}
    liftoff_leg = {"hip": -0.04, "knee": 1.00, "ankle": -0.58, "hip_roll": 0.0, "ankle_roll": 0.0}
    midswing_leg = {"hip": 0.10, "knee": 0.86, "ankle": -0.20, "hip_roll": 0.0, "ankle_roll": 0.0}
    touchdown_leg = {"hip": -0.02, "knee": 0.58, "ankle": -0.16, "hip_roll": 0.0, "ankle_roll": 0.0}

    if step_state.phase == "double_support":
        moving_pose = interpolate_pose(stance_leg, preload_leg, double_support_progress)
        moving_pose["hip_roll"] += -support_sign * args.support_shift_roll
        moving_pose["ankle_roll"] += -support_sign * 0.85 * args.support_shift_roll

        for side in ("Left", "Right"):
            pose = moving_pose if side == swing_side else dict(stance_leg)
            pose["hip"] += 0.12 * pitch
            pose["ankle"] += 0.25 * pitch
            targets[f"{side.lower()}_hip_pitch_joint"] = pose["hip"]
            targets[f"{side.lower()}_knee_joint"] = pose["knee"]
            targets[f"{side.lower()}_ankle_pitch_joint"] = pose["ankle"]
            targets[f"{side.lower()}_hip_roll_joint"] = pose["hip_roll"] + (roll if side == stance_side else 0.4 * roll)
            targets[f"{side.lower()}_ankle_roll_joint"] = pose["ankle_roll"] + 0.8 * roll
        return targets

    step_forward = step_state.desired_forward_step
    step_lateral = step_state.desired_lateral_step
    lateral_sign = 1.0 if swing_side == "Left" else -1.0

    if swing_progress < 0.25:
        swing_pose = interpolate_pose(preload_leg, liftoff_leg, swing_progress / 0.25)
    elif swing_progress < 0.70:
        swing_pose = interpolate_pose(liftoff_leg, midswing_leg, (swing_progress - 0.25) / 0.45)
    else:
        swing_pose = interpolate_pose(midswing_leg, touchdown_leg, (swing_progress - 0.70) / 0.30)

    swing_pose["hip"] += step_forward
    swing_pose["hip_roll"] += lateral_sign * step_lateral
    swing_pose["ankle_roll"] += lateral_sign * 0.75 * step_lateral

    stance_pose = dict(stance_leg)
    stance_pose["hip"] += 0.18 * pitch
    stance_pose["ankle"] += 0.45 * pitch
    stance_pose["hip_roll"] += support_sign * args.support_shift_roll + 0.8 * roll
    stance_pose["ankle_roll"] += support_sign * 0.9 * args.support_shift_roll + 0.9 * roll

    for side in ("Left", "Right"):
        pose = swing_pose if side == swing_side else stance_pose
        targets[f"{side.lower()}_hip_pitch_joint"] = pose["hip"]
        targets[f"{side.lower()}_knee_joint"] = pose["knee"]
        targets[f"{side.lower()}_ankle_pitch_joint"] = pose["ankle"]
        targets[f"{side.lower()}_hip_roll_joint"] = pose["hip_roll"]
        targets[f"{side.lower()}_ankle_roll_joint"] = pose["ankle_roll"]

    arm_phase = math.sin(math.pi * swing_progress)
    targets["left_shoulder_pitch_joint"] += -0.24 * arm_phase * (1.0 if swing_side == "Right" else -1.0)
    targets["right_shoulder_pitch_joint"] += 0.24 * arm_phase * (1.0 if swing_side == "Right" else -1.0)
    return targets


def apply_joint_targets(body_id, joint_catalog, joint_targets):
    for joint_name, target_position in joint_targets.items():
        joint_data = joint_catalog.get(joint_name)
        if joint_data is None:
            continue
        kp, kd = gain_profile_for_joint(joint_name)
        p.setJointMotorControl2(
            body_id,
            joint_data["index"],
            p.POSITION_CONTROL,
            targetPosition=target_position,
            targetVelocity=0.0,
            force=joint_data["max_force"],
            positionGain=kp,
            velocityGain=kd,
        )


def apply_virtual_harness(body_id, snapshot, args):
    base_position, base_orientation = p.getBasePositionAndOrientation(body_id)
    base_linear_velocity, base_angular_velocity = p.getBaseVelocity(body_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(base_orientation)

    force = (
        -args.harness_xy_kp * base_position[0] - args.harness_xy_kd * base_linear_velocity[0],
        -args.harness_xy_kp * base_position[1] - args.harness_xy_kd * base_linear_velocity[1],
        args.harness_z_kp * (args.base_target_height - base_position[2]) - args.harness_z_kd * base_linear_velocity[2],
    )
    torque = (
        -args.harness_roll_kp * roll - args.harness_roll_kd * base_angular_velocity[0],
        -args.harness_pitch_kp * pitch - args.harness_pitch_kd * base_angular_velocity[1],
        -args.harness_yaw_kp * (yaw - math.pi / 2.0) - args.harness_yaw_kd * base_angular_velocity[2],
    )
    scale = snapshot.assist_scale
    p.applyExternalForce(body_id, -1, tuple(scale * component for component in force), (0.0, 0.0, 0.0), p.WORLD_FRAME)
    p.applyExternalTorque(body_id, -1, tuple(scale * component for component in torque), p.WORLD_FRAME)


def update_status_text(debug_item_id, snapshot, step_state, harness_enabled):
    mode = f"{step_state.mode}:{step_state.phase}"
    text = (
        f"mode: {mode}\n"
        f"swing: {step_state.swing_side}\n"
        f"com err body: ({snapshot.com_error_body[0]:+.3f}, {snapshot.com_error_body[1]:+.3f})\n"
        f"cp body: ({snapshot.capture_point_body[0]:+.3f}, {snapshot.capture_point_body[1]:+.3f})\n"
        f"margin: {snapshot.support_margin:+.3f}\n"
        f"pitch fb: {snapshot.pitch_feedback:+.3f}\n"
        f"roll fb: {snapshot.roll_feedback:+.3f}\n"
        f"assist: {snapshot.assist_scale:.2f}\n"
        f"harness: {'on' if harness_enabled else 'off'}"
    )
    return p.addUserDebugText(
        text,
        (0.40, -0.35, 1.15),
        textColorRGB=(0.95, 0.95, 0.95),
        textSize=1.10,
        replaceItemUniqueId=debug_item_id,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="G1 basic balance stack with heuristic step synthesis and MPC-like reduced-order feedback.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help=f"URDF file to load (default: {DEFAULT_URDF})")
    parser.add_argument("--package-share", type=Path, default=None, help="Directory that contains the g1_description package.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the PyBullet GUI.")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds. Set to 0 to run until interrupted.")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Physics and controller time step.")
    parser.add_argument("--start-height", type=float, default=0.82, help="Initial pelvis height in meters.")
    parser.add_argument("--support-polygon-height", type=float, default=0.005, help="Support polygon overlay Z offset.")
    parser.add_argument("--disable-harness", action="store_true", help="Disable the virtual pelvis support layer.")
    parser.add_argument("--assist-margin", type=float, default=0.035, help="Support margin threshold that scales stabilization effort.")
    parser.add_argument("--assist-gain", type=float, default=14.0, help="How aggressively assistance ramps near the support edge.")
    parser.add_argument("--max-assist-scale", type=float, default=2.0, help="Maximum multiplier for corrective actions.")
    parser.add_argument("--com-pitch-kp", type=float, default=4.8, help="COM sagittal position gain.")
    parser.add_argument("--com-pitch-kd", type=float, default=1.6, help="COM sagittal velocity gain.")
    parser.add_argument("--com-roll-kp", type=float, default=4.4, help="COM lateral position gain.")
    parser.add_argument("--com-roll-kd", type=float, default=1.4, help="COM lateral velocity gain.")
    parser.add_argument("--nominal-com-height", type=float, default=0.78, help="Reduced-order COM height used for capture-point planning.")
    parser.add_argument("--imu-pitch-kp", type=float, default=1.7, help="IMU pitch angle gain.")
    parser.add_argument("--imu-pitch-kd", type=float, default=0.22, help="IMU pitch rate gain.")
    parser.add_argument("--imu-roll-kp", type=float, default=1.4, help="IMU roll angle gain.")
    parser.add_argument("--imu-roll-kd", type=float, default=0.18, help="IMU roll rate gain.")
    parser.add_argument("--pitch-limit", type=float, default=0.16, help="Pitch correction clamp in radians.")
    parser.add_argument("--roll-limit", type=float, default=0.12, help="Roll correction clamp in radians.")
    parser.add_argument("--step-trigger-margin", type=float, default=0.02, help="Support margin threshold that triggers a recovery step.")
    parser.add_argument("--step-trigger-velocity", type=float, default=0.08, help="Body-frame COM velocity threshold for step triggering.")
    parser.add_argument("--capture-trigger-x", type=float, default=0.035, help="Forward capture-point threshold for step triggering.")
    parser.add_argument("--capture-trigger-y", type=float, default=0.030, help="Lateral capture-point threshold for step triggering.")
    parser.add_argument("--double-support-duration", type=float, default=0.12, help="Pre-step weight-shift duration.")
    parser.add_argument("--swing-duration", type=float, default=0.26, help="Swing duration for the synthesized step.")
    parser.add_argument("--step-cooldown", type=float, default=0.20, help="Minimum time between completed steps.")
    parser.add_argument("--nominal-step-length", type=float, default=0.06, help="Baseline swing hip pitch offset.")
    parser.add_argument("--max-step-length", type=float, default=0.18, help="Maximum forward swing offset.")
    parser.add_argument("--max-lateral-step", type=float, default=0.12, help="Maximum lateral swing offset.")
    parser.add_argument("--capture-step-gain-x", type=float, default=1.35, help="Forward step placement gain from the capture point.")
    parser.add_argument("--capture-step-gain-y", type=float, default=1.55, help="Lateral step placement gain from the capture point.")
    parser.add_argument("--support-shift-roll", type=float, default=0.09, help="Roll bias toward the stance foot during stepping.")
    parser.add_argument("--base-target-height", type=float, default=0.82, help="Virtual harness target pelvis height.")
    parser.add_argument("--harness-xy-kp", type=float, default=40.0, help="Virtual harness planar position gain.")
    parser.add_argument("--harness-xy-kd", type=float, default=18.0, help="Virtual harness planar damping.")
    parser.add_argument("--harness-z-kp", type=float, default=180.0, help="Virtual harness vertical gain.")
    parser.add_argument("--harness-z-kd", type=float, default=35.0, help="Virtual harness vertical damping.")
    parser.add_argument("--harness-roll-kp", type=float, default=160.0, help="Virtual harness roll gain.")
    parser.add_argument("--harness-roll-kd", type=float, default=18.0, help="Virtual harness roll damping.")
    parser.add_argument("--harness-pitch-kp", type=float, default=180.0, help="Virtual harness pitch gain.")
    parser.add_argument("--harness-pitch-kd", type=float, default=20.0, help="Virtual harness pitch damping.")
    parser.add_argument("--harness-yaw-kp", type=float, default=35.0, help="Virtual harness yaw gain.")
    parser.add_argument("--harness-yaw-kd", type=float, default=8.0, help="Virtual harness yaw damping.")
    return parser.parse_args()


def main():
    args = parse_args()
    urdf_path = args.urdf.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    package_share = args.package_share.resolve() if args.package_share is not None else resolve_default_package_share(urdf_path)
    if not package_share.is_dir():
        raise FileNotFoundError(f"Package share directory not found: {package_share}")
    if not DEFAULT_PLANE_URDF.is_file():
        raise FileNotFoundError(f"PyBullet plane URDF not found: {DEFAULT_PLANE_URDF}")

    connect_pybullet(args.headless)
    p.setAdditionalSearchPath(str(package_share))
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    plane_id = p.loadURDF(str(DEFAULT_PLANE_URDF))
    body_id = p.loadURDF(
        str(urdf_path),
        basePosition=(0.0, 0.0, args.start_height),
        baseOrientation=p.getQuaternionFromEuler((0.0, 0.0, math.pi / 2.0)),
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )
    configure_scene(body_id, plane_id, args.dt)

    joint_catalog = build_joint_catalog(body_id)
    reset_robot_to_stand_pose(body_id, joint_catalog)
    mass_properties = build_mass_properties(body_id)
    link_name_map = build_link_name_map(body_id)
    foot_link_indices = get_default_foot_link_indices(link_name_map)

    step_state = StepState()
    debug_status_text = -1
    support_polygon_items = {}
    harness_enabled = not args.disable_harness
    start_time = time.perf_counter()

    try:
        while p.isConnected():
            snapshot = compute_balance_snapshot(body_id, plane_id, mass_properties, foot_link_indices, args)
            update_step_state(step_state, snapshot, args)
            joint_targets = build_joint_targets(snapshot, step_state, args)
            apply_joint_targets(body_id, joint_catalog, joint_targets)
            if harness_enabled:
                apply_virtual_harness(body_id, snapshot, args)

            p.stepSimulation()

            if not args.headless:
                update_support_polygon_visuals(
                    snapshot.support_polygon,
                    snapshot.com_position,
                    snapshot.support_center,
                    support_polygon_items,
                    show_text=True,
                )
                debug_status_text = update_status_text(debug_status_text, snapshot, step_state, harness_enabled)
                time.sleep(args.dt)

            if args.duration > 0.0 and time.perf_counter() - start_time >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
