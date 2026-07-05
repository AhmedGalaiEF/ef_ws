# G1 Geometry & Kinematics Reference

Source: `g1_29dof_with_hand_rev_1_0_pkg.urdf`, found at
`~/ef_ws/install/g1_description/share/g1_description/urdf/g1_29dof_with_hand_rev_1_0_pkg.urdf`
(duplicated under `src/install/...` and various `log/build_*` colcon
artifacts; an older variant also exists at
`~/ef_ws/go2/sim/unitree_description-master/model/g1/g1.urdf`).

Not present in this repo's `./sim` or `./WBC` — those only contain motion
scripts and IK/WBC code that load the model from the external
`g1_description` ROS package. Confirm which exact URDF path `./WBC/wbc.py`
and the IK CLI scripts load before treating these numbers as authoritative
for what's actually driving the robot.

## Link masses

49 links, sum ≈ 34.4 kg (Unitree's spec sheet lists ~35 kg for the full
G1, consistent once battery/electronics lumped elsewhere are counted).

| Link | Mass (kg) |
|---|---|
| pelvis | 3.813 |
| pelvis_contour_link | 0.001 |
| left/right_hip_pitch_link | 1.350 each |
| left/right_hip_roll_link | 1.520 each |
| left/right_hip_yaw_link | 1.702 each |
| left/right_knee_link | 1.932 each |
| left/right_ankle_pitch_link | 0.074 each |
| left/right_ankle_roll_link | 0.608 each |
| waist_yaw_link | 0.214 |
| waist_roll_link | 0.086 |
| torso_link | 6.780 |
| logo_link | 0.001 |
| head_link | 1.036 |
| left/right_shoulder_pitch_link | 0.718 each |
| left/right_shoulder_roll_link | 0.643 each |
| left/right_shoulder_yaw_link | 0.734 each |
| left/right_elbow_link | 0.600 each |
| left/right_wrist_roll_link | 0.0854 each |
| left/right_wrist_pitch_link | 0.484 each |
| left/right_wrist_yaw_link | 0.0846 each |
| left/right_hand_palm_link | 0.3728 each |
| left/right_hand_thumb_0_link | 0.0862 each |
| left/right_hand_thumb_1_link | 0.0589 each |
| left/right_hand_thumb_2_link | 0.0203 each |
| left/right_hand_middle_0_link | 0.0589 each |
| left/right_hand_middle_1_link | 0.0203 each |
| left/right_hand_index_0_link | 0.0589 each |
| left/right_hand_index_1_link | 0.0203 each |

## Joint limits

54 joints total: 41 revolute (with position/effort/velocity limits),
12 fixed, 1 floating (base). Units: **rad, N·m, rad/s**.

### Legs (12 joints, mirrored L/R)

| Joint | Lower | Upper | Effort | Velocity |
|---|---|---|---|---|
| hip_pitch | -2.5307 | 2.8798 | 88 | 32 |
| hip_roll (left) | -0.5236 | 2.9671 | 139 | 20 |
| hip_roll (right) | -2.9671 | 0.5236 | 139 | 20 |
| hip_yaw | -2.7576 | 2.7576 | 88 | 32 |
| knee | -0.0873 | 2.8798 | 139 | 20 |
| ankle_pitch | -0.8727 | 0.5236 | 35 | 30 |
| ankle_roll | -0.2618 | 0.2618 | 35 | 30 |

hip_roll ranges are mirrored (left positive-outward, right
negative-outward) — asymmetric about zero, not a symmetric ±.

### Waist (3 joints)

| Joint | Lower | Upper | Effort | Velocity |
|---|---|---|---|---|
| waist_yaw | -2.618 | 2.618 | 88 | 32 |
| waist_roll | -0.52 | 0.52 | 35 | 30 |
| waist_pitch | -0.52 | 0.52 | 35 | 30 |

### Arms (7 joints/side, mirrored)

| Joint | Lower | Upper | Effort | Velocity |
|---|---|---|---|---|
| shoulder_pitch | -3.0892 | 2.6704 | 25 | 37 |
| shoulder_roll (left) | -1.5882 | 2.2515 | 25 | 37 |
| shoulder_roll (right) | -2.2515 | 1.5882 | 25 | 37 |
| shoulder_yaw | -2.618 | 2.618 | 25 | 37 |
| elbow | -1.0472 | 2.0944 | 25 | 37 |
| wrist_roll | -1.9722 | 1.9722 | 25 | 37 |
| wrist_pitch | -1.6144 | 1.6144 | 5 | 22 |
| wrist_yaw | -1.6144 | 1.6144 | 5 | 22 |

### Hand (7 joints/side)

Thumb×3, index×2, middle×2. Effort 1.4–2.45 N·m, velocity 3.14–12 rad/s,
small ranges (~0–1.75 rad). Not simple mirrors — exact per-joint values:

| Joint | Lower | Upper | Effort | Velocity |
|---|---|---|---|---|
| left_hand_thumb_0 | -1.0472 | 1.0472 | 2.45 | 3.14 |
| left_hand_thumb_1 | -0.6109 | 1.0472 | 1.4 | 12 |
| left_hand_thumb_2 | 0 | 1.7453 | 1.4 | 12 |
| left_hand_middle_0 | -1.5708 | 0 | 1.4 | 12 |
| left_hand_middle_1 | -1.7453 | 0 | 1.4 | 12 |
| left_hand_index_0 | -1.5708 | 0 | 1.4 | 12 |
| left_hand_index_1 | -1.7453 | 0 | 1.4 | 12 |
| right_hand_thumb_0 | -1.0472 | 1.0472 | 2.45 | 3.14 |
| right_hand_thumb_1 | -1.0472 | 0.6109 | 1.4 | 12 |
| right_hand_thumb_2 | -1.7453 | 0 | 1.4 | 12 |
| right_hand_middle_0 | 0 | 1.5708 | 1.4 | 12 |
| right_hand_middle_1 | 0 | 1.7453 | 1.4 | 12 |
| right_hand_index_0 | 0 | 1.5708 | 1.4 | 12 |
| right_hand_index_1 | 0 | 1.7453 | 1.4 | 12 |

### Fixed joints (no limits, rigid mounts)

base_link_to_pelvis, pelvis_contour, logo, head, imu_in_torso,
imu_in_pelvis, d435 (camera), mid360 (lidar), left/right_hand_palm.

### floating_base_joint

Type `floating` — the free-flying pelvis DOF, no limits (6-DOF virtual
joint for the base in the world frame).
