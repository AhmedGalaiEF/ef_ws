from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


VISUALIZATIONS_DIR = Path(__file__).resolve().parents[1] / "g1" / "academy" / "visualizations"
if str(VISUALIZATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZATIONS_DIR))

from academy_arm_fk import ArmFK, _T_from_axis_q  # noqa: E402
from academy_arm_ik import ArmIK, _pose_error  # noqa: E402


def test_fk_returns_finite_homogeneous_transform() -> None:
    transform = ArmFK("right", backend="urdf").compute_arm(np.zeros(7))

    assert transform.shape == (4, 4)
    assert np.all(np.isfinite(transform))
    assert np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-7)


@pytest.mark.parametrize("backend", ["unknown", "pin"])
def test_fk_rejects_unknown_backends(backend: str) -> None:
    with pytest.raises(ValueError, match="backend"):
        ArmFK("right", backend=backend)


def test_fk_rejects_bad_joint_vectors_and_partial_counts() -> None:
    fk = ArmFK("left")

    with pytest.raises(ValueError, match="exactly 7"):
        fk.compute_arm(np.zeros(6))
    with pytest.raises(ValueError, match="finite"):
        fk.compute_arm(np.array([0.0] * 6 + [np.nan]))
    with pytest.raises(ValueError, match="between 0 and 7"):
        fk.compute_arm_partial(np.zeros(7), 8)


def test_axis_rotation_normalizes_axis_and_rejects_zero() -> None:
    transform = _T_from_axis_q([0.0, 2.0, 0.0], 0.5)
    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-7)

    with pytest.raises(ValueError, match="non-zero"):
        _T_from_axis_q([0.0, 0.0, 0.0], 0.5)


def test_rotation_error_handles_180_degrees() -> None:
    desired = np.eye(4)
    desired[:3, :3] = np.diag([1.0, -1.0, -1.0])

    error = _pose_error(desired, np.eye(4))

    assert np.linalg.norm(error[3:]) == pytest.approx(np.pi)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver": "pin"},
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"tol_pos_m": float("nan")},
        {"damping": 0.0},
    ],
)
def test_ik_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ArmIK(**kwargs)


def test_ik_validates_target_and_initial_joint_vector() -> None:
    ik = ArmIK("right")

    with pytest.raises(ValueError, match="shape"):
        ik.solve(np.eye(3))
    invalid_rotation = np.eye(4)
    invalid_rotation[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        ik.solve(invalid_rotation)
    with pytest.raises(ValueError, match="finite"):
        ik.solve(np.eye(4), np.array([0.0] * 6 + [np.inf]))


def test_ik_accepts_a_pose_already_reached_by_initial_state() -> None:
    q = np.array([0.1, -0.1, 0.05, 0.2, 0.0, 0.1, -0.1])
    target = ArmFK("right").compute_arm(q)

    solution, info = ArmIK("right").solve(target, q_init=q)

    assert info["success"] is True
    assert np.allclose(solution, q)
