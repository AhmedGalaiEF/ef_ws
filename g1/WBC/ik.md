# IK Implementation Notes

## Overview

`ik_pose_cli.py` controls the 6D Cartesian pose of the G1 arm end-effector(s) by running **damped least-squares (DLS) IK** over the 7 arm DOFs (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw).

The IK pipeline lives in `hand_pose_navigation_copy/arm_ik.py`, which depends on `arm_fk.py` for forward kinematics.

---

## Forward Kinematics (`arm_fk.py`)

**Entry point:** `ArmFK.compute_arm(q_arm)` → 4×4 `T_base_hand`

`ik_pose_cli` always uses the `urdf` backend, which calls `_fk_urdf`:

```python
T = np.eye(4)
T[:3, 3] = _TORSO_IN_BASE        # [-0.003964, 0, 0.044]
for i, (xyz, rpy, axis) in enumerate(_URDF_CHAIN[arm]):
    T = T @ _T_from_xyz_rpy(xyz, rpy) @ _T_from_axis_q(axis, q_arm[i])
```

Each of the 7 joints contributes two transforms:

1. **`_T_from_xyz_rpy(xyz, rpy)`** — fixed rigid offset from the URDF (translate + constant rotation `Rz@Ry@Rx`). Link geometry hardcoded from `g1_29dof_with_hand_rev_1_0_pkg.urdf`.
2. **`_T_from_axis_q(axis, q)`** — rotation by joint angle `q` about the joint axis via Rodrigues' formula: `R = I + sin(q)K + (1-cos(q))K²`, where `axis ∈ {[1,0,0], [0,1,0], [0,0,1]}`.

Full chain:
```
T_base_hand = T_torso · (T_link1 · R_j1) · (T_link2 · R_j2) · … · (T_link7 · R_j7)
```

---

## Jacobian (`arm_ik.py:_numerical_jacobian`)

**Shape:** `6×7` — 6 task-space DOFs × 7 arm joints.

**Method:** Forward finite difference with `eps = 1e-5` rad.

For each joint `i`:

```python
q1 = q.copy(); q1[i] += eps
T1 = fk.compute_arm(q1)

# Translational columns
J[:3, i] = (T1[:3, 3] - p0) / eps

# Rotational columns
dR = T1[:3, :3] @ R0.T
J[3:, i] = [dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]] / (2 * eps)
```

- **Position:** simple Cartesian difference of the two FK-computed EE positions, both in `base_link` frame.
- **Rotation:** axis-angle approximation extracted from the rotation error matrix `dR = R1 @ R0.T` (skew-symmetric part).

---

## DLS IK Solver (`arm_ik.py:_solve_dls`)

**Per iteration:**

1. Run FK: `T_cur = fk.compute_arm(q)`
2. Compute 6D error: `err = [pos_err(3), rot_err(3)]`
   - `pos_err = T_des[:3,3] - T_cur[:3,3]`
   - `rot_err = 0.5 * skew(R_des @ R_cur.T)` (axis-angle, valid near solution)
3. Compute Jacobian: `J = _numerical_jacobian(q, fk)`  — `6×7`
4. DLS update:
   ```
   dq = J^T @ (J J^T + λ²I)^-1 @ err
   ```
   with damping `λ = 0.05`.
5. Clamp `‖dq‖ ≤ 0.3` rad, then clamp joints to URDF limits.

Converges when `‖pos_err‖ < tol_pos_m` **and** `‖rot_err‖ < tol_rot_rad`.

**Parameters used in `ik_pose_cli`:**
| Parameter | Value |
|---|---|
| `max_iter` | 10 |
| `tol_pos_m` | 0.005 m |
| `tol_rot_rad` | 0.02 rad |
| `damping λ` | 0.05 |

---

## Per-keypress Flow (`ik_pose_cli.py`)

1. Increment EE target `target_T[arm]` by the configured step (position or rotation).
2. Warm-start DLS IK from previous `desired_targets`.
3. Clamp each joint delta to `±max_dq` (default 0.2 rad).
4. On success: write clamped solution into `desired_targets`.
5. On failure: roll back `target_T[arm]` — joints do not move.
6. Control loop ramps `current_targets → desired_targets` at `max_speed` r/s.
