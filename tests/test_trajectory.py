"""Tests for the task-space references, including the trap the first design fell into."""

from __future__ import annotations

import numpy as np
import pytest

from armsim.kinematics import (
    RobotArm,
    det_jacobian,
    forward_kinematics,
    jacobian,
    manipulability,
)
from armsim.trajectory import (
    quintic,
    skim_singularity,
    through_singularity,
    well_conditioned,
    wrap_angle,
)

ARM = RobotArm()


# ------------------------------------------------------------------- scaling
def test_quintic_endpoints_and_rest():
    s0, d0 = quintic(0.0)
    s1, d1 = quintic(1.0)
    assert (s0, d0) == pytest.approx((0.0, 0.0))
    assert (s1, d1) == pytest.approx((1.0, 0.0))


def test_quintic_peak_speed_is_at_the_midpoint():
    """The crossing is placed at s = 0.5, so this is where the arm is fastest."""
    taus = np.linspace(0.0, 1.0, 1001)
    speeds = np.array([quintic(t)[1] for t in taus])
    assert taus[int(np.argmax(speeds))] == pytest.approx(0.5, abs=1e-3)


def test_quintic_is_monotone():
    s = np.array([quintic(t)[0] for t in np.linspace(0, 1, 501)])
    assert np.all(np.diff(s) >= -1e-15)


def test_wrap_angle_range():
    a = wrap_angle(np.array([0.0, np.pi, -np.pi, 3 * np.pi, -3 * np.pi + 0.1]))
    assert np.all(a > -np.pi - 1e-12) and np.all(a <= np.pi + 1e-12)


# ---------------------------------------------------------------- continuity
@pytest.mark.parametrize(
    "make", [through_singularity, skim_singularity, well_conditioned]
)
def test_twist_matches_finite_differenced_pose(make):
    """The analytic twist must be the derivative of the pose it claims to be."""
    traj = make(ARM)
    h = 1e-6
    for t in np.linspace(0.2, traj.duration - 0.2, 9):
        fd = (traj.pose(t + h) - traj.pose(t - h)) / (2.0 * h)
        assert np.allclose(traj.twist(t), fd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "make", [through_singularity, skim_singularity, well_conditioned]
)
def test_starts_at_the_arms_actual_pose(make):
    """Zero initial tracking error, so the run measures control, not a step input."""
    traj = make(ARM)
    assert np.allclose(forward_kinematics(ARM, traj.q_start()), traj.pose(0.0), atol=1e-12)


@pytest.mark.parametrize(
    "make", [through_singularity, skim_singularity, well_conditioned]
)
def test_endpoints_are_at_rest(make):
    traj = make(ARM)
    assert np.allclose(traj.twist(0.0), 0.0, atol=1e-12)
    assert np.allclose(traj.twist(traj.duration), 0.0, atol=1e-12)


# ------------------------------------------------------------- the geometry
def test_through_passes_exactly_through_the_singular_pose():
    traj = through_singularity(ARM)
    p_mid = traj.pose(traj.duration / 2.0)          # quintic: s = 0.5 at t = T/2
    assert np.allclose(p_mid, traj.meta["p_singular"], atol=1e-12)

    q_sing = traj.meta["q_singular"]
    # sqrt(det(J J^T)) only reaches its ~1e-8 roundoff floor.
    assert det_jacobian(ARM, q_sing) == 0.0
    assert manipulability(ARM, q_sing) < 1e-7


def test_through_commands_an_infeasible_twist_at_the_singularity():
    """At the singularity the commanded twist must leave range(J).

    Guards against building the reference as FK of a joint path (docs/lessons.md).
    """
    traj = through_singularity(ARM)
    q_sing = traj.meta["q_singular"]
    j = jacobian(ARM, q_sing)
    xdot = traj.twist(traj.duration / 2.0)

    assert np.linalg.norm(xdot) > 1e-3, "need a nonzero command to say anything"

    # Least-squares projection onto range(J); residual is the unachievable part.
    resid = xdot - j @ np.linalg.lstsq(j, xdot, rcond=None)[0]
    frac = float(np.linalg.norm(resid) / np.linalg.norm(xdot))
    assert frac > 0.05, (
        f"commanded twist is nearly feasible ({frac:.3%}) -- no singularity to fight"
    )


def test_skim_gets_close_without_crossing():
    traj = skim_singularity(ARM, q2_far=0.8, q2_near=0.02)
    assert traj.meta["q_closest"][1] == pytest.approx(0.02)
    # returns to where it started -- the turnaround is real
    assert np.allclose(traj.pose(0.0), traj.pose(traj.duration), atol=1e-12)


def test_skim_rejects_a_crossing():
    with pytest.raises(ValueError, match="must be positive"):
        skim_singularity(ARM, q2_near=-0.1)
    with pytest.raises(ValueError, match="closer to the singularity"):
        skim_singularity(ARM, q2_far=0.1, q2_near=0.5)


def test_well_conditioned_stays_well_conditioned():
    """Sanity: the control scenario must never dip near the singular locus."""
    traj = well_conditioned(ARM)
    # w along the reference's own anchor sweep
    q2s = np.linspace(traj.q_start()[1], traj.meta["q_end"][1], 200)
    ws = [manipulability(ARM, np.array([0.3, q2, np.pi / 2])) for q2 in q2s]
    assert min(ws) > 0.9


def test_through_rejects_zero_amplitude():
    with pytest.raises(ValueError, match="q2_amplitude must be positive"):
        through_singularity(ARM, q2_amplitude=0.0)
