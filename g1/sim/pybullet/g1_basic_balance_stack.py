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
    compute_center_of_mass,
    get_active_foot_supports,
    get_default_foot_link_indices,
    polygon_centroid_2d,
    project_world_vector_to_body_frame,
    read_dummy_imu,
    resolve_default_package_share,
    support_polygon_margin,
    update_support_polygon_visuals,
    build_support_polygon,
)


STAND_POSE = {
    "left_hip_pitch_joint": -0.25,
    "left_hip_roll_joint": 0.02,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.55,
    "left_ankle_pitch_joint": -0.30,
    "left_ankle_roll_joint": -0.02,
    "right_hip_pitch_joint": -0.25,
    "right_hip_roll_joint": -0.02,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.55,
    "right_ankle_pitch_joint": -0.30,
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


@dataclass
class LowLevelCommand:
    q: float
    dq: float
    kp: float
    kd: float
    tau: float = 0.0


@dataclass
class BalanceSnapshot:
    com_position: tuple[float, float, float]
    com_velocity: tuple[float, float, float]
    support_center: tuple[float, float, float]
    support_margin: float
    imu_state: dict
    pitch_feedback: float
    roll_feedback: float
    assist_scale: float


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def connect_pybullet(headless):
    mode = p.DIRECT if headless else p.GUI
    client_id = p.connect(mode)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet.")
    return client_id


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


def build_joint_catalog(body_id):
    catalog = {}
    for joint_index in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, joint_index)
        if joint_info[2] == p.JOINT_FIXED:
            continue
        joint_name = joint_info[1].decode("utf-8")
        max_force = joint_info[10] if joint_info[10] > 0.0 else 80.0
        catalog[joint_name] = {
            "index": joint_index,
            "max_force": max_force,
        }
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
        margin_error = args.assist_margin - support_margin
        assist_scale += clamp(margin_error * args.assist_gain, 0.0, args.max_assist_scale - 1.0)

    snapshot = BalanceSnapshot(
        com_position=com_position,
        com_velocity=com_velocity,
        support_center=support_center,
        support_margin=support_margin,
        imu_state=imu_state,
        pitch_feedback=pitch_feedback,
        roll_feedback=roll_feedback,
        assist_scale=assist_scale,
    )
    return snapshot, support_polygon


def build_low_level_commands(snapshot, joint_catalog, args):
    pitch = snapshot.pitch_feedback
    roll = snapshot.roll_feedback
    assist_scale = snapshot.assist_scale

    commands = {}
    for joint_name in joint_catalog:
        q_target = STAND_POSE.get(joint_name, 0.0)

        if joint_name in ("left_hip_pitch_joint", "right_hip_pitch_joint"):
            q_target += 0.24 * pitch * assist_scale
        elif joint_name in ("left_knee_joint", "right_knee_joint"):
            q_target += -0.14 * pitch * assist_scale
        elif joint_name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
            q_target += 0.95 * pitch * assist_scale
        elif joint_name in ("left_hip_roll_joint", "right_hip_roll_joint"):
            q_target += roll * assist_scale
        elif joint_name in ("left_ankle_roll_joint", "right_ankle_roll_joint"):
            q_target += 0.85 * roll * assist_scale
        elif joint_name == "waist_pitch_joint":
            q_target += -0.18 * pitch
        elif joint_name == "waist_roll_joint":
            q_target += -0.25 * roll
        elif joint_name == "left_shoulder_pitch_joint":
            q_target += -0.12 * pitch
        elif joint_name == "right_shoulder_pitch_joint":
            q_target += -0.12 * pitch
        elif joint_name == "left_shoulder_roll_joint":
            q_target += -0.20 * roll
        elif joint_name == "right_shoulder_roll_joint":
            q_target += -0.20 * roll

        kp, kd = gain_profile_for_joint(joint_name)
        commands[joint_name] = LowLevelCommand(q=q_target, dq=0.0, kp=kp, kd=kd, tau=0.0)
    return commands


def apply_low_level_commands(body_id, joint_catalog, commands):
    for joint_name, command in commands.items():
        joint_data = joint_catalog.get(joint_name)
        if joint_data is None:
            continue

        # PyBullet POSITION_CONTROL does not expose a true feedforward torque term,
        # so `tau` is kept as part of the command structure but not injected here.
        p.setJointMotorControl2(
            body_id,
            joint_data["index"],
            p.POSITION_CONTROL,
            targetPosition=command.q,
            targetVelocity=command.dq,
            force=joint_data["max_force"],
            positionGain=command.kp,
            velocityGain=command.kd,
        )


def apply_virtual_harness(body_id, snapshot, args):
    base_position, base_orientation = p.getBasePositionAndOrientation(body_id)
    base_linear_velocity, base_angular_velocity = p.getBaseVelocity(body_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(base_orientation)

    height_error = args.base_target_height - base_position[2]
    force = (
        -args.harness_xy_kp * base_position[0] - args.harness_xy_kd * base_linear_velocity[0],
        -args.harness_xy_kp * base_position[1] - args.harness_xy_kd * base_linear_velocity[1],
        args.harness_z_kp * height_error - args.harness_z_kd * base_linear_velocity[2],
    )
    torque = (
        -args.harness_roll_kp * roll - args.harness_roll_kd * base_angular_velocity[0],
        -args.harness_pitch_kp * pitch - args.harness_pitch_kd * base_angular_velocity[1],
        -args.harness_yaw_kp * (yaw - math.pi / 2.0) - args.harness_yaw_kd * base_angular_velocity[2],
    )

    scale = snapshot.assist_scale
    p.applyExternalForce(body_id, -1, tuple(scale * component for component in force), (0.0, 0.0, 0.0), p.WORLD_FRAME)
    p.applyExternalTorque(body_id, -1, tuple(scale * component for component in torque), p.WORLD_FRAME)


def update_status_text(debug_item_id, snapshot, harness_enabled):
    mode = "harness" if harness_enabled else "unsupported"
    text = (
        f"mode: {mode}\n"
        f"com xy: ({snapshot.com_position[0]:+.3f}, {snapshot.com_position[1]:+.3f})\n"
        f"margin: {snapshot.support_margin:+.3f}\n"
        f"pitch fb: {snapshot.pitch_feedback:+.3f}\n"
        f"roll fb: {snapshot.roll_feedback:+.3f}\n"
        f"assist: {snapshot.assist_scale:.2f}"
    )
    return p.addUserDebugText(
        text,
        (0.35, -0.35, 1.15),
        textColorRGB=(0.95, 0.95, 0.95),
        textSize=1.15,
        replaceItemUniqueId=debug_item_id,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Basic standing balance stack for the Unitree G1 in PyBullet.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help=f"URDF file to load (default: {DEFAULT_URDF})")
    parser.add_argument("--package-share", type=Path, default=None, help="Directory that contains the g1_description package.")
    parser.add_argument("--headless", action="store_true", help="Run without opening the PyBullet GUI.")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds. Set to 0 to run until interrupted.")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Physics and controller time step.")
    parser.add_argument("--start-height", type=float, default=0.82, help="Initial pelvis height in meters.")
    parser.add_argument("--support-polygon-height", type=float, default=0.005, help="Support polygon overlay Z offset.")
    parser.add_argument("--disable-harness", action="store_true", help="Disable the virtual pelvis support layer.")
    parser.add_argument("--assist-margin", type=float, default=0.035, help="Support margin threshold that scales balance assistance.")
    parser.add_argument("--assist-gain", type=float, default=14.0, help="How aggressively assistance ramps up near the edge of support.")
    parser.add_argument("--max-assist-scale", type=float, default=2.0, help="Maximum multiplier for corrective actions.")
    parser.add_argument("--com-pitch-kp", type=float, default=5.0, help="COM sagittal position gain.")
    parser.add_argument("--com-pitch-kd", type=float, default=1.6, help="COM sagittal velocity gain.")
    parser.add_argument("--com-roll-kp", type=float, default=4.6, help="COM lateral position gain.")
    parser.add_argument("--com-roll-kd", type=float, default=1.4, help="COM lateral velocity gain.")
    parser.add_argument("--imu-pitch-kp", type=float, default=1.8, help="IMU pitch angle gain.")
    parser.add_argument("--imu-pitch-kd", type=float, default=0.24, help="IMU pitch rate gain.")
    parser.add_argument("--imu-roll-kp", type=float, default=1.5, help="IMU roll angle gain.")
    parser.add_argument("--imu-roll-kd", type=float, default=0.20, help="IMU roll rate gain.")
    parser.add_argument("--pitch-limit", type=float, default=0.16, help="Pitch correction clamp in radians.")
    parser.add_argument("--roll-limit", type=float, default=0.12, help="Roll correction clamp in radians.")
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

    debug_status_text = -1
    support_polygon_items = {}
    harness_enabled = not args.disable_harness
    start_time = time.perf_counter()

    try:
        while p.isConnected():
            snapshot, support_polygon = compute_balance_snapshot(
                body_id,
                plane_id,
                mass_properties,
                foot_link_indices,
                args,
            )
            commands = build_low_level_commands(snapshot, joint_catalog, args)
            apply_low_level_commands(body_id, joint_catalog, commands)
            if harness_enabled:
                apply_virtual_harness(body_id, snapshot, args)

            p.stepSimulation()

            if not args.headless:
                update_support_polygon_visuals(
                    support_polygon,
                    snapshot.com_position,
                    snapshot.support_center,
                    support_polygon_items,
                    show_text=True,
                )
                debug_status_text = update_status_text(debug_status_text, snapshot, harness_enabled)
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
