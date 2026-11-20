"""Singularity maps checked against their closed forms."""

import numpy as np
import pytest

from armsim import RobotArm, manipulability
from armsim.singularity import (
    SINGULAR_Q2,
    joint_space_grid,
    manipulability_of_q2,
    max_manipulability_at_radius,
    radial_profile_swept,
    singular_radius,
    workspace_grid,
)

UNIT_ARM = RobotArm(1.0, 1.0, 1.0)
ODD_ARM = RobotArm(1.3, 0.7, 0.5)
#: |l1 - l2| = 0.8 > l3, so the wrist can never fold close enough to the base for
#: the tip to reach it: this arm has a genuine inner hole of radius 0.6.
HOLED_ARM = RobotArm(1.3, 0.5, 0.2)


def _bin_upper_bound(arm: RobotArm, r_lo, r_hi, n_sub: int = 17):
    """Largest closed-form ``w`` anywhere in ``[r_lo, r_hi]``."""
    t = np.linspace(0.0, 1.0, n_sub)
    r = np.asarray(r_lo)[..., None] + (np.asarray(r_hi) - np.asarray(r_lo))[..., None] * t
    return max_manipulability_at_radius(arm, r).max(axis=-1)


# --- the closed form vs the Jacobian it is supposed to summarise ------------


def test_manipulability_of_q2_matches_jacobian_route():
    rng = np.random.default_rng(11)
    for arm in (UNIT_ARM, ODD_ARM):
        for _ in range(200):
            q = rng.uniform(-np.pi, np.pi, 3)
            if abs(np.sin(q[1])) < 0.05:
                continue  # Gram-determinant roundoff floor, see Phase 1 tests
            assert manipulability_of_q2(arm, q[1]) == pytest.approx(
                manipulability(arm, q), abs=1e-9
            )


# --- joint-space map --------------------------------------------------------


def test_joint_space_grid_is_flat_in_q3():
    # Every row identical is the visual claim the heatmap makes; assert it exactly.
    _, _, w = joint_space_grid(ODD_ARM, n_q2=181, n_q3=91)
    assert w.shape == (91, 181)
    assert np.allclose(w, w[0], atol=0.0)


def test_joint_space_grid_zeros_sit_on_the_locus():
    q2, _, w = joint_space_grid(UNIT_ARM, n_q2=721, n_q3=5)
    row = w[0]
    for q2_sing in (-np.pi, *SINGULAR_Q2):
        nearest = np.argmin(np.abs(q2 - q2_sing))
        assert row[nearest] < 1e-12
    # ...and the peaks are exactly at q2 = +-pi/2, where w = l1*l2.
    assert row.max() == pytest.approx(UNIT_ARM.l1 * UNIT_ARM.l2)
    assert np.abs(np.abs(q2[np.argmax(row)]) - np.pi / 2) < 1e-9


# --- workspace map: closed form vs brute force ------------------------------


def test_radial_profile_never_beats_the_closed_form():
    # No swept configuration may beat the closed-form w_max at its own radius.
    for arm in (UNIT_ARM, ODD_ARM):
        _, w_swept, r_at = radial_profile_swept(arm, n_r=200, n_q2=1400, n_q3=1400)
        seen = np.isfinite(w_swept)
        assert seen.sum() > 100
        allowed = max_manipulability_at_radius(arm, r_at[seen])
        assert np.all(w_swept[seen] <= allowed + 1e-9)


def test_radial_profile_converges_to_the_closed_form():
    # Stronger than any single tolerance: refining the sweep must close the gap,
    # which is what "the sweep is sampling this curve" actually means.
    gaps = []
    for n in (500, 1500):
        _, w_swept, r_at = radial_profile_swept(UNIT_ARM, n_r=200, n_q2=n, n_q3=n)
        seen = np.isfinite(w_swept)
        allowed = max_manipulability_at_radius(UNIT_ARM, r_at[seen])
        gaps.append(np.max(np.abs(w_swept[seen] - allowed)))
    assert gaps[1] < 0.6 * gaps[0]


def test_max_manipulability_collapses_at_full_reach():
    for arm in (UNIT_ARM, ODD_ARM):
        assert max_manipulability_at_radius(arm, arm.reach) == pytest.approx(0.0, abs=1e-9)
        assert max_manipulability_at_radius(arm, arm.reach * 0.999) < 0.2 * arm.l1 * arm.l2
        assert max_manipulability_at_radius(arm, arm.reach * 1.01) == 0.0
        assert singular_radius(arm) == pytest.approx(arm.reach)


def test_max_manipulability_peaks_in_the_interior():
    # Somewhere mid-workspace the arm can pick q2 = +-pi/2 and be maximally
    # conditioned; that is the "danger is a thin outer shell" claim.
    r = np.linspace(0.0, UNIT_ARM.reach, 400)
    w = max_manipulability_at_radius(UNIT_ARM, r)
    assert w.max() == pytest.approx(UNIT_ARM.l1 * UNIT_ARM.l2)
    assert w[-1] < 1e-9


def test_max_manipulability_is_monotone_near_the_boundary():
    # Past the peak, reaching further can only cost conditioning.
    r = np.linspace(UNIT_ARM.reach * 0.6, UNIT_ARM.reach, 300)
    w = max_manipulability_at_radius(UNIT_ARM, r)
    assert np.all(np.diff(w) <= 1e-9)


def test_unreachable_radii_report_zero():
    # Equal inner links can fold the wrist onto the base, so RobotArm(1, 1, l3) has
    # no hole at all -- the hole needs |l1 - l2| > l3.
    assert max_manipulability_at_radius(RobotArm(1.0, 1.0, 0.1), 0.0) > 0.0

    assert max_manipulability_at_radius(HOLED_ARM, 0.0) == 0.0
    assert max_manipulability_at_radius(HOLED_ARM, 0.59) == 0.0
    assert max_manipulability_at_radius(HOLED_ARM, 0.61) > 0.0
    assert max_manipulability_at_radius(HOLED_ARM, HOLED_ARM.reach + 0.5) == 0.0


def test_workspace_grid_is_radially_symmetric():
    # q1 is swept explicitly, so matching the radius-only closed form is a real
    # check that nothing but q2 drives manipulability.
    x, y, w = workspace_grid(UNIT_ARM, n_bins=120, n_q1=72, n_q2=140, n_q3=140)
    assert w.shape == (120, 120)
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.hypot(X, Y)
    seen = np.isfinite(w)
    assert seen.sum() > 5000

    # A cell spans a range of radii; bound by the best radius anywhere in it.
    half_diag = 0.5 * np.hypot(x[1] - x[0], y[1] - y[0])
    ceiling = _bin_upper_bound(UNIT_ARM, np.maximum(0.0, r - half_diag), r + half_diag)
    assert np.all(w[seen] <= ceiling[seen] + 1e-9)

    interior = seen & (r < 0.9 * UNIT_ARM.reach)
    expected = max_manipulability_at_radius(UNIT_ARM, r)
    assert np.median(np.abs(w[interior] - expected[interior])) < 5e-3


def test_workspace_grid_has_a_hole_when_inner_links_are_uneven():
    x, y, w = workspace_grid(HOLED_ARM, n_bins=100, n_q1=48, n_q2=120, n_q3=120)
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.hypot(X, Y)
    assert np.all(~np.isfinite(w[r < 0.55]))   # inside the hole
    assert np.any(np.isfinite(w[(r > 0.7) & (r < 1.5)]))
    assert np.all(~np.isfinite(w[r > HOLED_ARM.reach + 0.05]))
