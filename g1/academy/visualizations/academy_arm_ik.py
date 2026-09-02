"""
Step 7 — Inverse kinematics: desired pose -> q_arm_desired
============================================================
Solves for the 7-DOF arm joint angles that achieve a desired
end-effector pose T_base_hand_desired.

Two solvers are available:
    "pin"   — pinocchio Levenberg-Marquardt IK (preferred)
    "num"   — pure-numpy damped least-squares Jacobian iteration

Both respect joint limits defined in arm_fk.JOINT_LIMITS.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from academy_arm_fk import ArmFK, JOINT_LIMITS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, _fk_dh, _DH_RIGHT


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _finite_scalar(value, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return parsed


def _finite_vector(values, size: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if vector.size != size:
        raise ValueError(f"{name} must contain exactly {size} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _validate_transform(value, name: str = "T_base_desired") -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 4x4 transform") from exc
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError(f"{name} rotation must be orthonormal with determinant +1")
    return transform.copy()


def _clamp(q: np.ndarray, limits: List[Tuple[float, float]]) -> np.ndarray:
    lo = np.array([lim[0] for lim in limits])
    hi = np.array([lim[1] for lim in limits])
    return np.clip(q, lo, hi)


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Return the SO(3) logarithm as an axis-angle vector."""
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-7:
        return 0.5 * skew
    if np.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, index])
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            raise ValueError("Could not determine axis for 180-degree rotation error")
        return axis / norm * angle
    return skew * (angle / (2.0 * np.sin(angle)))


def _pose_error(T_desired: np.ndarray, T_current: np.ndarray) -> np.ndarray:
    """
    6-D error vector [pos_err (3), rot_err (3)] in base frame.
    Rotation error uses the skew-symmetric approach (small-angle valid near solution).
    """
    pos_err = T_desired[:3, 3] - T_current[:3, 3]
    R_err = T_desired[:3, :3] @ T_current[:3, :3].T
    rot_err = _rotation_log(R_err)
    return np.concatenate([pos_err, rot_err])


def _numerical_jacobian(
    q: np.ndarray,
    fk: ArmFK,
    eps: float = 1e-5,
) -> np.ndarray:
    """Finite-difference 6×7 Jacobian for DH-based FK."""
    J = np.zeros((6, 7), dtype=np.float64)
    T0 = fk.compute_arm(q)
    p0 = T0[:3, 3]
    R0 = T0[:3, :3]
    for i in range(7):
        q1 = q.copy()
        q1[i] += eps
        T1 = fk.compute_arm(q1)
        J[:3, i] = (T1[:3, 3] - p0) / eps
        dR = T1[:3, :3] @ R0.T
        J[3:, i] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * eps)
    return J


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

class ArmIK:
    """
    Step 7: Solve IK for the G1 arm.

    Args:
        arm:          "left" | "right"
        solver:       "dls"
        max_iter:     maximum iterations for iterative solvers
        tol_pos_m:    position convergence tolerance (metres)
        tol_rot_rad:  rotation convergence tolerance (radians)
        damping:      DLS damping factor λ
    """

    def __init__(
        self,
        arm: str = "right",
        solver: str = "dls",
        max_iter: int = 200,
        tol_pos_m: float = 0.003,
        tol_rot_rad: float = 0.01,
        damping: float = 0.05,
    ) -> None:
        if solver != "dls":
            raise ValueError("The academy IK module supports only solver='dls'")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or not 1 <= max_iter <= 5000:
            raise ValueError("max_iter must be an integer between 1 and 5000")
        self.arm = arm
        self.solver = solver
        self.max_iter = max_iter
        self.tol_pos_m = _finite_scalar(tol_pos_m, "tol_pos_m", minimum=1e-6, maximum=0.1)
        self.tol_rot_rad = _finite_scalar(tol_rot_rad, "tol_rot_rad", minimum=1e-6, maximum=1.0)
        self.damping = _finite_scalar(damping, "damping", minimum=1e-6, maximum=10.0)
        self._fk = ArmFK(arm=arm, backend="urdf")
        self._limits = JOINT_LIMITS[arm]
        self._joint_indices = LEFT_ARM_JOINTS if arm == "left" else RIGHT_ARM_JOINTS

    # ------------------------------------------------------------------
    def solve(
        self,
        T_base_desired: np.ndarray,
        q_init: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Solve IK.

        Args:
            T_base_desired: 4×4 desired end-effector pose in base frame
            q_init:         7-element initial arm joint angles (radians);
                            if None, uses zeros

        Returns:
            (q_arm, info) where q_arm is length-7 or None on failure.
            info dict contains: "success", "error_pos_m", "error_rot_rad", "iterations"
        """
        T_base_desired = _validate_transform(T_base_desired)
        if q_init is None:
            q_init = np.zeros(7, dtype=np.float64)
        q = _clamp(_finite_vector(q_init, 7, "q_init"), self._limits)

        q_sol, info = self._solve_dls(T_base_desired, q)
        if q_sol is not None:
            return q_sol, info
        q_pos, pos_info = self._solve_position_dls(T_base_desired, q)
        if q_pos is not None:
            pos_info["fallback"] = "position_dls"
            pos_info["orientation_relaxed"] = True
            return q_pos, pos_info
        info["fallback_error_pos_m"] = pos_info["error_pos_m"]
        return None, info

    # ------------------------------------------------------------------
    # Damped Least Squares (primary solver — fast, no external deps)
    # ------------------------------------------------------------------

    def _solve_dls(
        self, T_des: np.ndarray, q: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict]:
        lam = self.damping
        for iteration in range(self.max_iter):
            T_cur = self._fk.compute_arm(q)
            err = _pose_error(T_des, T_cur)

            err_pos = float(np.linalg.norm(err[:3]))
            err_rot = float(np.linalg.norm(err[3:]))

            if err_pos < self.tol_pos_m and err_rot < self.tol_rot_rad:
                return q, {
                    "success": True,
                    "error_pos_m": err_pos,
                    "error_rot_rad": err_rot,
                    "iterations": iteration,
                }

            J = _numerical_jacobian(q, self._fk)
            # DLS: dq = J^T (J J^T + λ²I)^-1 err
            JJT = J @ J.T
            dq = J.T @ np.linalg.solve(JJT + lam**2 * np.eye(6), err)

            norm_dq = float(np.linalg.norm(dq))
            if norm_dq > 0.3:
                dq *= 0.3 / norm_dq
            q = _clamp(q + dq, self._limits)

        T_cur = self._fk.compute_arm(q)
        err = _pose_error(T_des, T_cur)
        return None, {
            "success": False,
            "error_pos_m": float(np.linalg.norm(err[:3])),
            "error_rot_rad": float(np.linalg.norm(err[3:])),
            "iterations": self.max_iter,
        }

    def _solve_position_dls(
        self, T_des: np.ndarray, q: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """DLS fallback that reaches the wrist position and ignores orientation.

        Vision detections only provide a coarse PCA orientation. Requiring the
        wrist to match that full 6-DoF pose can reject reachable object
        positions, so this fallback is intentionally position-first.
        """
        lam = self.damping
        target_pos = T_des[:3, 3]
        for iteration in range(self.max_iter):
            T_cur = self._fk.compute_arm(q)
            pos_err = target_pos - T_cur[:3, 3]
            err_pos = float(np.linalg.norm(pos_err))
            if err_pos < self.tol_pos_m:
                full_error = _pose_error(T_des, T_cur)
                return q, {
                    "success": True,
                    "error_pos_m": err_pos,
                    "error_rot_rad": float(np.linalg.norm(full_error[3:])),
                    "iterations": iteration,
                }

            J_pos = _numerical_jacobian(q, self._fk)[:3, :]
            JJT = J_pos @ J_pos.T
            dq = J_pos.T @ np.linalg.solve(JJT + lam**2 * np.eye(3), pos_err)

            norm_dq = float(np.linalg.norm(dq))
            if norm_dq > 0.3:
                dq *= 0.3 / norm_dq
            q = _clamp(q + dq, self._limits)

        T_cur = self._fk.compute_arm(q)
        err_pos = float(np.linalg.norm(target_pos - T_cur[:3, 3]))
        full_error = _pose_error(T_des, T_cur)
        return None, {
            "success": False,
            "error_pos_m": err_pos,
            "error_rot_rad": float(np.linalg.norm(full_error[3:])),
            "iterations": self.max_iter,
        }

    # ------------------------------------------------------------------
    # Compatibility stubs for callers that explicitly selected optional solvers.
    # ------------------------------------------------------------------

    def _solve_scipy(
        self, T_des: np.ndarray, q: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict]:
        raise ValueError("The academy IK module supports only the local DLS solver")

    # ------------------------------------------------------------------
    # Pinocchio IK (best accuracy, requires pip install pin)
    # ------------------------------------------------------------------

    def _solve_pin(
        self, T_des: np.ndarray, q: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Dict]:
        raise ValueError("The academy IK module supports only the local DLS solver")

    # ------------------------------------------------------------------
    def extract_arm_q(self, q_full: np.ndarray) -> np.ndarray:
        """Extract 7-element arm-only q from 30-element full joint array."""
        q_full = _finite_vector(q_full, 30, "q_full")
        return q_full[self._joint_indices].copy()

    def inject_arm_q(self, q_full: np.ndarray, q_arm: np.ndarray) -> np.ndarray:
        """Write 7-element q_arm back into q_full at correct indices."""
        q_out = _finite_vector(q_full, 30, "q_full").copy()
        q_arm = _finite_vector(q_arm, 7, "q_arm")
        q_out[self._joint_indices] = q_arm
        return q_out
