import argparse
import atexit
import math
import time
from pathlib import Path

import pybullet as p
import pybullet_data

import go2_balance_stand as stand


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_URDF = stand.DEFAULT_URDF
DEFAULT_PACKAGE_ROOT = stand.DEFAULT_PACKAGE_ROOT
DEFAULT_PLANE_URDF = stand.DEFAULT_PLANE_URDF
LEG_ORDER = ("FR", "FL", "RR", "RL")
LEG_JOINTS = {
    "FR": ("FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"),
    "FL": ("FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"),
    "RR": ("RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"),
    "RL": ("RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"),
}
GAITS = {
    "Walk": {
        "description": "Four-beat lateral walk: RL -> FL -> RR -> FR",
        "phase_offsets": {"RL": 0.00, "FL": 0.25, "RR": 0.50, "FR": 0.75},
        "duty": 0.75,
        "cycle_sec": 2.0,
        "step_height": 0.16,
        "step_length": 0.16,
    },
    "Amble": {
        "description": "Lateral amble: same order as walk, quicker overlap",
        "phase_offsets": {"RL": 0.00, "FL": 0.20, "RR": 0.50, "FR": 0.70},
        "duty": 0.68,
        "cycle_sec": 1.6,
        "step_height": 0.18,
        "step_length": 0.18,
    },
    "Diagonal Amble": {
        "description": "Diagonal amble: RL -> FR -> RR -> FL",
        "phase_offsets": {"RL": 0.00, "FR": 0.25, "RR": 0.50, "FL": 0.75},
        "duty": 0.68,
        "cycle_sec": 1.55,
        "step_height": 0.18,
        "step_length": 0.18,
    },
    "Trot": {
        "description": "Two-beat diagonal gait",
        "phase_offsets": {"FR": 0.00, "RL": 0.00, "FL": 0.50, "RR": 0.50},
        "duty": 0.52,
        "cycle_sec": 1.2,
        "step_height": 0.20,
        "step_length": 0.22,
    },
    "Pace": {
        "description": "Two-beat lateral gait",
        "phase_offsets": {"FR": 0.00, "RR": 0.00, "FL": 0.50, "RL": 0.50},
        "duty": 0.52,
        "cycle_sec": 1.15,
        "step_height": 0.20,
        "step_length": 0.22,
    },
    "Canter": {
        "description": "Three-beat asymmetric gait",
        "phase_offsets": {"RL": 0.00, "RR": 0.18, "FR": 0.18, "FL": 0.55},
        "duty": 0.42,
        "cycle_sec": 0.95,
        "step_height": 0.24,
        "step_length": 0.28,
    },
}


def ramp_value(current, target, rate, dt):
    step = rate * dt
    if abs(target - current) <= step:
        return target
    return current + step if target > current else current - step


def stance_swing_value(leg_phase, duty):
    if leg_phase < duty:
        stance_phase = leg_phase / duty
        return 1.0 - 2.0 * stance_phase, 0.0
    swing_phase = (leg_phase - duty) / max(1.0 - duty, 1e-6)
    sweep = -1.0 + 2.0 * swing_phase
    lift = math.sin(math.pi * swing_phase)
    return sweep, lift


def build_gait_targets(controller_state, args):
    gait = GAITS[args.gait]
    controller_state["move_x"] = ramp_value(controller_state["move_x"], args.move_x, args.command_gradient, args.time_step)
    controller_state["move_yaw"] = ramp_value(
        controller_state["move_yaw"], args.move_yaw, args.command_gradient * 1.6, args.time_step
    )

    targets = dict(stand.STAND_POSE)
    move_mag = max(abs(controller_state["move_x"]), abs(controller_state["move_yaw"]))
    if move_mag < 0.02:
        controller_state["phase"] = 0.0
        return targets

    controller_state["phase"] = (controller_state["phase"] + args.time_step / gait["cycle_sec"]) % 1.0
    step_length = gait["step_length"] * abs(controller_state["move_x"]) * args.step_scale
    step_height = gait["step_height"] * abs(controller_state["move_x"]) * args.step_scale
    turn_amount = args.turn_scale * controller_state["move_yaw"]

    for leg_name in LEG_ORDER:
        hip_name, thigh_name, calf_name = LEG_JOINTS[leg_name]
        leg_phase = (controller_state["phase"] + gait["phase_offsets"][leg_name]) % 1.0
        sweep, lift = stance_swing_value(leg_phase, gait["duty"])
        side_sign = 1.0 if leg_name in ("FL", "RL") else -1.0
        front_sign = 1.0 if leg_name in ("FR", "FL") else -1.0

        sagittal_sweep = sweep * step_length * controller_state["move_x"]
        hip_delta = 0.0
        hip_delta += side_sign * turn_amount * (0.55 if front_sign > 0.0 else 0.95)
        thigh_delta = -args.fore_aft_thigh_gain * sagittal_sweep
        thigh_delta += -0.55 * step_height * lift
        calf_delta = args.fore_aft_calf_gain * sagittal_sweep
        calf_delta += 1.1 * step_height * lift

        targets[hip_name] += hip_delta
        targets[thigh_name] += thigh_delta
        targets[calf_name] += calf_delta

    return targets


def compute_stability_threat(imu_state, polygon_margin, feet_in_contact, args):
    pitch_threat = abs(imu_state["pitch"]) / max(args.stability_pitch_trigger, 1e-6)
    roll_threat = abs(imu_state["roll"]) / max(args.stability_roll_trigger, 1e-6)
    contact_threat = 0.0
    if feet_in_contact < args.stability_min_contacts:
        contact_threat = (args.stability_min_contacts - feet_in_contact) / max(args.stability_min_contacts, 1)

    margin_threat = 0.0
    if args.stability_margin_trigger > 0.0:
        margin_deficit = max(0.0, args.stability_margin_trigger - polygon_margin)
        margin_threat = margin_deficit / args.stability_margin_trigger

    return max(pitch_threat, roll_threat, contact_threat, margin_threat)


def balance_override_scale(threat, args):
    if threat <= args.stability_threat_threshold:
        return 0.0
    excess = threat - args.stability_threat_threshold
    ramp = excess / max(args.stability_threat_full_scale - args.stability_threat_threshold, 1e-6)
    return args.balance_blend * min(1.0, ramp)


def blend_balance_offsets(gait_targets, imu_state, com_position, com_velocity, support_center, balance_scale, args):
    balance_targets, pitch_correction, roll_correction, height_correction, _ = stand.build_balance_targets(
        imu_state,
        com_position,
        com_velocity,
        support_center,
        args,
    )
    for joint_name, stand_value in stand.STAND_POSE.items():
        gait_targets[joint_name] += balance_scale * (balance_targets[joint_name] - stand_value)
    return gait_targets, pitch_correction, roll_correction, height_correction, balance_scale


def build_status_text(gait_name, controller_state, imu_state, com_position, polygon_margin, stability_threat, balance_scale, pitch_correction, roll_correction, height_correction, feet_in_contact):
    base_position = imu_state["position"]
    return "\n".join(
        (
            f"gait: {gait_name}",
            f"cmd x/yaw: {controller_state['move_x']:+.2f}, {controller_state['move_yaw']:+.2f}",
            f"phase: {controller_state['phase']:.2f}",
            f"base z: {base_position[2]:.3f} m",
            f"com xy: ({com_position[0]:+.3f}, {com_position[1]:+.3f}) m",
            f"support margin: {polygon_margin:+.3f} m",
            f"stability threat: {stability_threat:.2f}",
            f"balance override: {balance_scale:.2f}",
            f"roll/pitch: {imu_state['roll']:+.3f}, {imu_state['pitch']:+.3f} rad",
            f"feedback pitch: {pitch_correction:+.3f}",
            f"feedback roll: {roll_correction:+.3f}",
            f"feedback height: {height_correction:+.3f}",
            f"feet in contact: {feet_in_contact}/4",
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Unitree Go2 in PyBullet with a gait generator and debug stability overlays.")
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
    parser.add_argument("--time-step", type=float, default=1.0 / 240.0, help="Physics step size in seconds.")
    parser.add_argument("--max-fps", type=float, default=240.0, help="GUI pacing limit. Set to 0 to run uncapped.")
    parser.add_argument("--max-steps", type=int, default=0, help="Optional simulation-step limit for testing. Set to 0 to run until closed.")
    parser.add_argument("--render-mode", choices=("full", "minimal", "off"), default="full", help="Amount of debug rendering.")
    parser.add_argument("--debug-update-every", type=int, default=4, help="Only refresh debug overlays every N physics steps.")
    parser.add_argument("--contact-links", type=str, default="feet", help="Contact force visualization target: 'feet', 'all', or a comma-separated list of link names.")
    parser.add_argument("--contact-force-scale", type=float, default=0.0025, help="Meters per Newton for contact-force arrows.")
    parser.add_argument("--support-polygon-height", type=float, default=0.005, help="Height offset for the support polygon debug overlay.")
    parser.add_argument("--spawn-foot-clearance", type=float, default=0.02, help="Initial vertical foot clearance above the plane.")
    parser.add_argument("--settle-steps", type=int, default=120, help="Physics steps used to settle into the standing pose before gaiting.")
    parser.add_argument("--gait", choices=tuple(GAITS.keys()), default="Walk", help="Gait pattern from go2_gait_control_qt.py.")
    parser.add_argument("--move-x", type=float, default=0.35, help="Normalized forward command in [-1, 1].")
    parser.add_argument("--move-yaw", type=float, default=0.0, help="Normalized yaw command in [-1, 1].")
    parser.add_argument("--command-gradient", type=float, default=0.8, help="How quickly gait commands ramp toward their targets.")
    parser.add_argument("--step-scale", type=float, default=0.32, help="Scale factor applied to gait step length and height for PyBullet stability.")
    parser.add_argument("--turn-scale", type=float, default=0.06, help="Scale factor applied to the yaw-turn contribution.")
    parser.add_argument("--fore-aft-thigh-gain", type=float, default=0.32, help="How strongly forward sweep drives the thigh joints.")
    parser.add_argument("--fore-aft-calf-gain", type=float, default=0.58, help="How strongly forward sweep drives the calf joints.")
    parser.add_argument("--balance-blend", type=float, default=0.28, help="Blend factor for the standing stabilizer offsets on top of the gait targets.")
    parser.add_argument("--stability-threat-threshold", type=float, default=1.0, help="Balance override stays inactive until the threat score exceeds this threshold.")
    parser.add_argument("--stability-threat-full-scale", type=float, default=1.6, help="Threat score at which the balance override reaches the full configured blend.")
    parser.add_argument("--stability-roll-trigger", type=float, default=0.12, help="Absolute roll angle in radians corresponding to a threat score of 1.0.")
    parser.add_argument("--stability-pitch-trigger", type=float, default=0.12, help="Absolute pitch angle in radians corresponding to a threat score of 1.0.")
    parser.add_argument("--stability-min-contacts", type=int, default=3, help="Minimum desired number of feet in contact before the threat score rises.")
    parser.add_argument("--stability-margin-trigger", type=float, default=0.0, help="Optional support-polygon margin threshold. Set to 0 to ignore polygon margin in the threat score.")
    parser.add_argument("--target-base-height", type=float, default=0.343, help="Nominal floating-base height during walking.")
    parser.add_argument("--pitch-kp", type=float, default=0.18, help="Pitch proportional gain.")
    parser.add_argument("--pitch-kd", type=float, default=0.03, help="Pitch derivative gain.")
    parser.add_argument("--roll-kp", type=float, default=0.16, help="Roll proportional gain.")
    parser.add_argument("--roll-kd", type=float, default=0.03, help="Roll derivative gain.")
    parser.add_argument("--height-kp", type=float, default=0.0, help="Base-height proportional gain.")
    parser.add_argument("--height-kd", type=float, default=0.0, help="Base-height derivative gain.")
    parser.add_argument("--pitch-limit", type=float, default=0.045, help="Clamp for pitch feedback.")
    parser.add_argument("--roll-limit", type=float, default=0.045, help="Clamp for roll feedback.")
    parser.add_argument("--height-limit", type=float, default=0.0, help="Clamp for height feedback.")
    parser.add_argument("--hip-roll-gain", type=float, default=0.08, help="How strongly roll feedback drives the hip joints.")
    parser.add_argument("--thigh-pitch-gain", type=float, default=0.06, help="How strongly pitch feedback drives the thigh joints.")
    parser.add_argument("--calf-pitch-gain", type=float, default=0.10, help="How strongly pitch feedback drives the calf joints.")
    parser.add_argument("--position-gain", type=float, default=0.18, help="PyBullet position gain for the joint motors.")
    parser.add_argument("--velocity-gain", type=float, default=0.5, help="PyBullet velocity gain for the joint motors.")
    parser.add_argument("--force-scale", type=float, default=1.8, help="Extra scale applied on top of the URDF joint effort limits.")
    parser.add_argument("--com-balance-kp", type=float, default=0.0, help="COM-to-support-center proportional gain in the sagittal plane.")
    parser.add_argument("--com-balance-kd", type=float, default=0.0, help="COM velocity gain in the sagittal plane.")
    parser.add_argument("--com-balance-roll-kp", type=float, default=0.0, help="COM-to-support-center proportional gain in the lateral plane.")
    parser.add_argument("--com-balance-roll-kd", type=float, default=0.0, help="COM velocity gain in the lateral plane.")
    return parser.parse_args()


def main():
    args = parse_args()
    urdf_path = args.urdf.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")
    if not DEFAULT_PLANE_URDF.is_file():
        raise FileNotFoundError(f"PyBullet plane URDF not found: {DEFAULT_PLANE_URDF}")

    package_root = args.package_root.resolve() if args.package_root is not None else stand.resolve_package_root(urdf_path)
    if not package_root.is_dir():
        raise FileNotFoundError(f"Package root not found: {package_root}")

    resolved_urdf_path = stand.build_resolved_urdf(urdf_path, package_root)
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

    actuators = stand.build_joint_actuator_map(body_id)
    link_name_map = stand.build_link_name_map(body_id)
    foot_link_indices = stand.get_default_foot_link_indices(link_name_map)
    stand.reset_joints_to_pose(body_id, actuators, stand.STAND_POSE)
    stand.place_base_so_feet_touch_ground(body_id, foot_link_indices, args.spawn_foot_clearance)

    for joint_index in range(-1, p.getNumJoints(body_id)):
        p.changeDynamics(body_id, joint_index, lateralFriction=1.0, spinningFriction=0.04, rollingFriction=0.02)

    mass_properties = stand.build_mass_properties(body_id)
    total_weight = stand.compute_total_mass(body_id) * 9.81
    contact_link_indices, include_inactive_contact_links = stand.build_contact_link_map(link_name_map, args.contact_links)
    contact_debug_items = {}
    support_polygon_items = {}
    status_item = -1
    controller_state = {"phase": 0.0, "move_x": 0.0, "move_yaw": 0.0}
    stand.configure_camera(body_id)

    for _ in range(max(0, args.settle_steps)):
        stand.apply_joint_targets(body_id, actuators, stand.STAND_POSE, args.position_gain, args.velocity_gain, args.force_scale)
        p.stepSimulation()

    step_index = 0
    last_wall_time = time.perf_counter()
    try:
        while p.isConnected():
            imu_state = stand.read_dummy_imu(body_id)
            com_position, com_velocity = stand.compute_center_of_mass(body_id, mass_properties)
            active_foot_supports = stand.get_active_foot_supports(body_id, plane_id, foot_link_indices)
            support_polygon = stand.build_support_polygon(body_id, foot_link_indices, active_foot_supports, args.support_polygon_height)
            support_center = stand.polygon_centroid_2d(support_polygon)
            polygon_margin = stand.support_polygon_margin((com_position[0], com_position[1]), support_polygon)
            feet_in_contact = len(active_foot_supports)
            if support_center is None:
                support_center = (com_position[0], com_position[1], args.support_polygon_height)

            gait_targets = build_gait_targets(controller_state, args)
            stability_threat = compute_stability_threat(imu_state, polygon_margin, feet_in_contact, args)
            targets, pitch_correction, roll_correction, height_correction, balance_scale = blend_balance_offsets(
                gait_targets,
                imu_state,
                com_position,
                com_velocity,
                support_center,
                balance_override_scale(stability_threat, args),
                args,
            )
            stand.apply_joint_targets(body_id, actuators, targets, args.position_gain, args.velocity_gain, args.force_scale)
            p.stepSimulation()

            if not args.headless and step_index % max(1, args.debug_update_every) == 0:
                feet_in_contact = stand.count_feet_in_contact(body_id, plane_id, link_name_map)
                status_item = p.addUserDebugText(
                    build_status_text(
                        args.gait,
                        controller_state,
                        imu_state,
                        com_position,
                        polygon_margin,
                        stability_threat,
                        balance_scale,
                        pitch_correction,
                        roll_correction,
                        height_correction,
                        feet_in_contact,
                    ),
                    (-0.50, -0.28, 0.55),
                    textColorRGB=(0.95, 0.95, 0.95),
                    textSize=1.05,
                    replaceItemUniqueId=status_item,
                )
                if args.render_mode in ("full", "minimal") and support_polygon:
                    stand.update_support_polygon_visuals(
                        support_polygon,
                        com_position,
                        support_center,
                        support_polygon_items,
                        show_text=args.render_mode == "full",
                    )
                elif support_polygon_items:
                    stand.clear_debug_item_group(support_polygon_items)

                if args.render_mode in ("full", "minimal") and contact_link_indices:
                    contact_summaries = stand.summarize_contact_forces(
                        body_id,
                        plane_id,
                        contact_link_indices,
                        include_inactive=include_inactive_contact_links,
                    )
                    if not include_inactive_contact_links:
                        stand.clear_contact_visuals(contact_debug_items, contact_summaries.keys())
                    stand.update_contact_visuals(
                        contact_summaries,
                        contact_debug_items,
                        total_weight=total_weight,
                        force_scale=args.contact_force_scale,
                        show_text=args.render_mode == "full",
                    )
                stand.configure_camera(body_id)

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
        stand.clear_debug_item_group(contact_debug_items)
        stand.clear_debug_item_group(support_polygon_items)
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    atexit.register(stand.cleanup_temp_urdfs)
    main()
