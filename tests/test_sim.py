"""Closed-loop tests: the plant, the metrics, and the headline result as a regression."""

from __future__ import annotations

import numpy as np
import pytest

from armsim.controllers import AdaptiveDLS, DampedLeastSquares, NaiveInverse
from armsim.kinematics import RobotArm, forward_kinematics
from armsim.sim import Limits, metrics, pose_error, simulate
from armsim.trajectory import skim_singularity, through_singularity, well_conditioned

ARM = RobotArm()
DT = 2e-3          # coarser than the 1 ms used for figures, to keep tests quick


# ------------------------------------------------------------------ plumbing
def test_pose_error_wraps_orientation():
    e = pose_error(np.array([0.0, 0.0, 3.1]), np.array([0.0, 0.0, -3.1]))
    assert abs(e[2]) < 0.1      # not ~6.2
    assert np.allclose(e[:2], 0.0)


def test_limits_reject_nonpositive():
    with pytest.raises(ValueError, match="qdot_max must be positive"):
        Limits(qdot_max=0.0)


def test_logs_are_index_aligned_and_start_at_the_reference():
    traj = well_conditioned(ARM)
    res = simulate(ARM, traj, NaiveInverse(), dt=DT)
    n = len(res.t)
    for arr in (res.q, res.qdot_cmd, res.qdot_applied, res.p, res.p_ref):
        assert arr.shape == (n, 3)
    for arr in (res.w, res.lam, res.cond, res.saturated):
        assert arr.shape == (n,)
    assert np.allclose(res.p[0], res.p_ref[0], atol=1e-12)
    assert np.allclose(res.q[0], traj.q_start(), atol=1e-12)


def test_applied_velocity_never_exceeds_the_limit():
    """Saturation is the plant's job; the log must prove the plant did it."""
    lim = Limits(qdot_max=1.0)
    res = simulate(ARM, through_singularity(ARM), NaiveInverse(), dt=DT, limits=lim)
    assert np.max(np.abs(res.qdot_applied)) <= 1.0 + 1e-12
    assert np.max(np.abs(res.qdot_cmd)) > 1.0        # and the command did exceed it


def test_forward_kinematics_log_is_consistent_with_q():
    res = simulate(ARM, well_conditioned(ARM), AdaptiveDLS(), dt=DT)
    for i in (0, len(res.t) // 2, len(res.t) - 1):
        assert np.allclose(res.p[i], forward_kinematics(ARM, res.q[i]), atol=1e-12)


# --------------------------------------------------------------- the findings
def test_naive_controller_blows_up_through_the_singularity():
    """Phase 3's result, pinned so it cannot silently regress.

    The naive inverse is not merely worse here -- it commands joint speeds three
    orders of magnitude past the actuator limit and spends most of the run
    saturated.
    """
    m = metrics(simulate(ARM, through_singularity(ARM), NaiveInverse(), dt=DT))
    assert m["peak_qdot_cmd"] > 100.0
    assert m["pct_saturated"] > 20.0
    assert m["pos_err_max"] > 0.5


def test_dls_bounds_velocity_through_the_same_singularity():
    """Phase 4's result: same trajectory, same limits, no saturation at all."""
    traj = through_singularity(ARM)
    for ctl in (DampedLeastSquares(lam=0.1), AdaptiveDLS()):
        m = metrics(simulate(ARM, traj, ctl, dt=DT))
        assert m["pct_saturated"] == 0.0, ctl.name
        assert m["peak_qdot_cmd"] < 3.0, ctl.name
        assert m["pos_err_max"] < 0.3, ctl.name


def test_dls_beats_naive_by_orders_of_magnitude_on_peak_velocity():
    traj = through_singularity(ARM)
    naive = metrics(simulate(ARM, traj, NaiveInverse(), dt=DT))
    dls = metrics(simulate(ARM, traj, AdaptiveDLS(), dt=DT))
    assert naive["peak_qdot_cmd"] / dls["peak_qdot_cmd"] > 100.0


def test_adaptive_costs_nothing_when_well_conditioned():
    """The reason to prefer adaptive over fixed damping, as a test.

    On a trajectory that never approaches a singularity the schedule holds
    ``lam = 0``, so adaptive DLS reproduces the naive inverse exactly, while a
    fixed ``lam`` pays tracking error for protection nothing needed.
    """
    traj = well_conditioned(ARM)
    naive = metrics(simulate(ARM, traj, NaiveInverse(), dt=DT))
    fixed = metrics(simulate(ARM, traj, DampedLeastSquares(lam=0.1), dt=DT))
    adaptive = metrics(simulate(ARM, traj, AdaptiveDLS(), dt=DT))

    assert adaptive["lam_max_used"] == 0.0
    assert adaptive["pos_err_rms"] == pytest.approx(naive["pos_err_rms"], abs=1e-12)
    assert fixed["pos_err_rms"] > 20.0 * adaptive["pos_err_rms"] + 1e-6


def test_adaptive_tracks_better_than_fixed_lambda_when_skimming():
    """Near-but-not-at a singularity, scheduled damping is the more accurate law."""
    traj = skim_singularity(ARM)
    fixed = metrics(simulate(ARM, traj, DampedLeastSquares(lam=0.1), dt=DT))
    adaptive = metrics(simulate(ARM, traj, AdaptiveDLS(), dt=DT))
    assert adaptive["pos_err_rms"] < fixed["pos_err_rms"]
    assert adaptive["pct_saturated"] == 0.0


def test_no_controller_produces_nonfinite_commands_except_naive_at_singularity():
    traj = through_singularity(ARM)
    for ctl in (DampedLeastSquares(lam=0.1), AdaptiveDLS()):
        res = simulate(ARM, traj, ctl, dt=DT)
        assert res.n_nonfinite == 0, ctl.name
        assert np.all(np.isfinite(res.qdot_cmd)), ctl.name


# ------------------------------------------------------------------- metrics
def test_metrics_keys_and_types():
    m = metrics(simulate(ARM, well_conditioned(ARM), AdaptiveDLS(), dt=DT))
    expected = {
        "controller", "trajectory", "label", "peak_qdot_cmd", "peak_qdot_applied",
        "pos_err_rms", "pos_err_max", "ori_err_rms", "ori_err_max", "accel_rms",
        "pct_saturated", "w_min", "cond_max", "lam_max_used", "n_nonfinite",
    }
    assert set(m) == expected
    assert m["pos_err_rms"] <= m["pos_err_max"] + 1e-12
    assert m["ori_err_rms"] <= m["ori_err_max"] + 1e-12
    assert m["peak_qdot_applied"] <= m["peak_qdot_cmd"] + 1e-12


def test_dt_refinement_does_not_change_the_conclusion():
    """The headline gap must be a property of the control law, not the step size."""
    traj = through_singularity(ARM)
    ratios = []
    for dt in (4e-3, 2e-3, 1e-3):
        naive = metrics(simulate(ARM, traj, NaiveInverse(), dt=dt))
        dls = metrics(simulate(ARM, traj, AdaptiveDLS(), dt=dt))
        ratios.append(naive["peak_qdot_cmd"] / dls["peak_qdot_cmd"])
    assert all(r > 100.0 for r in ratios)
