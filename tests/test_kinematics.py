"""Phase 1 tests: forward kinematics, analytical Jacobian, and singularities.

The load-bearing test is ``test_jacobian_matches_finite_difference`` -- it pins
the hand-derived Jacobian to the forward kinematics, so every later phase can
trust ``jacobian()`` without re-deriving it.
"""

import numpy as np
import pytest

from armsim import (
    RobotArm,
    det_jacobian,
    forward_kinematics,
    jacobian,
    joint_positions,
    manipulability,
)

UNIT_ARM = RobotArm(1.0, 1.0, 1.0)


def _finite_difference_jacobian(arm: RobotArm, q, eps: float = 1e-6) -> np.ndarray:
    """Central-difference approximation of d(pose)/d(q), for validating jacobian()."""
    q = np.asarray(q, dtype=float)
    cols = []
    for i in range(3):
        dq = np.zeros(3)
        dq[i] = eps
        forward = forward_kinematics(arm, q + dq)
        backward = forward_kinematics(arm, q - dq)
        cols.append((forward - backward) / (2.0 * eps))
    return np.column_stack(cols)


@pytest.mark.parametrize(
    "q, expected_pose",
    [
        ([0.0, 0.0, 0.0], [3.0, 0.0, 0.0]),          # fully extended along +x
        ([np.pi / 2, 0.0, 0.0], [0.0, 3.0, np.pi / 2]),  # fully extended along +y
        ([0.0, np.pi / 2, 0.0], [1.0, 2.0, np.pi / 2]),  # right angle at joint 2
        ([0.0, 0.0, np.pi / 2], [2.0, 1.0, np.pi / 2]),  # right angle at joint 3
    ],
)
def test_forward_kinematics_spot_checks(q, expected_pose):
    assert np.allclose(forward_kinematics(UNIT_ARM, q), expected_pose, atol=1e-12)


def test_joint_positions_endpoint_matches_fk():
    q = [0.3, -0.7, 1.1]
    pos = joint_positions(UNIT_ARM, q)
    assert pos.shape == (4, 2)
    assert np.allclose(pos[0], [0.0, 0.0])
    assert np.allclose(pos[-1], forward_kinematics(UNIT_ARM, q)[:2], atol=1e-12)


def test_jacobian_matches_finite_difference():
    arm = RobotArm(0.9, 1.3, 0.7)
    rng = np.random.default_rng(42)
    for _ in range(200):
        q = rng.uniform(-np.pi, np.pi, 3)
        assert np.allclose(jacobian(arm, q), _finite_difference_jacobian(arm, q), atol=1e-6)


def test_det_closed_form_matches_numpy():
    arm = RobotArm(0.9, 1.3, 0.7)
    rng = np.random.default_rng(1)
    for _ in range(500):
        q = rng.uniform(-np.pi, np.pi, 3)
        assert det_jacobian(arm, q) == pytest.approx(np.linalg.det(jacobian(arm, q)), abs=1e-9)


def test_det_depends_only_on_l1_l2_q2():
    # det(J) = l1*l2*sin(q2): varying q1, q3, or l3 must not move it.
    base = RobotArm(1.1, 0.8, 0.6)
    bigger_l3 = RobotArm(1.1, 0.8, 2.5)
    expected = base.l1 * base.l2 * np.sin(0.5)
    assert det_jacobian(base, [0.0, 0.5, 0.0]) == pytest.approx(expected)
    assert det_jacobian(base, [2.0, 0.5, -1.3]) == pytest.approx(expected)
    assert det_jacobian(bigger_l3, [2.0, 0.5, -1.3]) == pytest.approx(expected)


def test_manipulability_equals_abs_det():
    # w = sqrt(det(J J^T)) == |det(J)| for a square Jacobian. We skip near-singular
    # configs: forming the Gram matrix squares the condition number, so its det has
    # a roundoff floor whose sqrt is ~1e-8 -- see test_singular_locus below.
    arm = RobotArm(1.2, 0.9, 1.4)
    rng = np.random.default_rng(7)
    for _ in range(300):
        q = rng.uniform(-np.pi, np.pi, 3)
        if abs(np.sin(q[1])) < 0.05:
            continue
        assert manipulability(arm, q) == pytest.approx(abs(det_jacobian(arm, q)), abs=1e-9)


@pytest.mark.parametrize("q2", [0.0, np.pi, -np.pi])
def test_singular_locus_is_rank_deficient(q2):
    # On q2 in {0, pi} the arm is singular for any q1, q3. Manipulability via the
    # Gram determinant bottoms out near 1e-8 (sqrt of a ~1e-16 near-zero value), so
    # we assert it is negligible rather than bit-exactly zero; the rank check via
    # SVD confirms the true rank drop.
    arm = RobotArm(1.0, 1.0, 1.0)
    rng = np.random.default_rng(3)
    for _ in range(20):
        q1, q3 = rng.uniform(-np.pi, np.pi, 2)
        q = [q1, q2, q3]
        assert manipulability(arm, q) < 1e-6
        assert abs(det_jacobian(arm, q)) < 1e-9
        assert np.linalg.matrix_rank(jacobian(arm, q), tol=1e-9) < 3


def test_full_rank_away_from_singularity():
    arm = RobotArm(1.0, 1.0, 1.0)
    q = [0.2, np.pi / 2, -0.4]  # q2 = 90 deg -> maximally conditioned in q2
    assert manipulability(arm, q) > 0.5
    assert np.linalg.matrix_rank(jacobian(arm, q), tol=1e-9) == 3


def test_link_length_validation():
    with pytest.raises(ValueError):
        RobotArm(1.0, -0.5, 1.0)
    with pytest.raises(ValueError):
        RobotArm(0.0, 1.0, 1.0)
