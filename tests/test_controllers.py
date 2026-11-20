"""Tests for the three control laws, including the DLS boundedness theorem."""

from __future__ import annotations

import numpy as np
import pytest

from armsim.controllers import (
    AdaptiveDLS,
    DampedLeastSquares,
    NaiveInverse,
    det3,
    inv3,
)
from armsim.kinematics import RobotArm, det_jacobian, jacobian, manipulability

ARM = RobotArm()
RNG = np.random.default_rng(20260730)


def _random_q(n: int = 40, *, min_w: float = 0.2) -> list[np.ndarray]:
    """Well-conditioned configurations only (so J^-1 is a fair reference)."""
    out = []
    while len(out) < n:
        q = RNG.uniform(-np.pi, np.pi, size=3)
        if manipulability(ARM, q) >= min_w:
            out.append(q)
    return out


# --------------------------------------------------------------------- inv3
def test_det3_matches_numpy():
    for q in _random_q(15):
        j = jacobian(ARM, q)
        assert det3(j) == pytest.approx(float(np.linalg.det(j)), rel=1e-10, abs=1e-12)


def test_det3_matches_closed_form():
    """det(J) = l1*l2*sin(q2) -- the cofactor expansion must agree with the algebra."""
    for q in _random_q(15, min_w=0.0):
        assert det3(jacobian(ARM, q)) == pytest.approx(det_jacobian(ARM, q), abs=1e-12)


def test_inv3_is_an_inverse():
    for q in _random_q(15):
        j = jacobian(ARM, q)
        assert np.allclose(inv3(j) @ j, np.eye(3), atol=1e-9)


def test_inv3_of_singular_blows_up():
    """At the singularity the naive inverse is huge but finite (det lands near 3e-16)."""
    q_sing = np.array([0.3, 0.0, 0.7])             # q2 = 0
    assert det_jacobian(ARM, q_sing) == 0.0        # closed form: exactly zero
    j = jacobian(ARM, q_sing)
    assert abs(det3(j)) < 1e-14                    # cofactor: zero to roundoff
    assert np.max(np.abs(inv3(j))) > 1e12          # ...which is enough to explode


# ------------------------------------------------------------------ naive/DLS
def test_dls_with_zero_lambda_equals_naive():
    """From docs/PLAN.md: lam = 0 must reduce exactly to the plain inverse."""
    naive, dls = NaiveInverse(), DampedLeastSquares(lam=0.0)
    for q in _random_q(20):
        xdot = RNG.normal(size=3)
        a, _ = naive.compute(ARM, q, xdot)
        b, _ = dls.compute(ARM, q, xdot)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-11)


def test_adaptive_is_identical_to_naive_when_well_conditioned():
    """Above the threshold lam = 0, so adaptive DLS is the naive law exactly."""
    naive, adaptive = NaiveInverse(), AdaptiveDLS(w0=0.15, lam_max=0.3)
    for q in _random_q(20, min_w=0.25):     # comfortably above w0
        xdot = RNG.normal(size=3)
        a, lam_a = naive.compute(ARM, q, xdot)
        b, lam_b = adaptive.compute(ARM, q, xdot)
        assert lam_a == 0.0 and lam_b == 0.0
        assert np.allclose(a, b, rtol=1e-12, atol=1e-14)


def test_dls_output_is_bounded_at_the_singularity():
    """DLS bound ``||qdot|| <= ||xdot|| / (2*lam)`` holds even for an exactly singular J."""
    lam = 0.1
    dls = DampedLeastSquares(lam=lam)
    q_sing = np.array([0.3, 0.0, 0.7])          # exactly singular
    assert manipulability(ARM, q_sing) == pytest.approx(0.0, abs=1e-15)

    for _ in range(30):
        xdot = RNG.normal(size=3)
        qdot, _ = dls.compute(ARM, q_sing, xdot)
        assert np.all(np.isfinite(qdot))
        assert np.linalg.norm(qdot) <= np.linalg.norm(xdot) / (2.0 * lam) + 1e-9


def test_dls_stays_bounded_as_w_goes_to_zero():
    """Sweeping q2 -> 0, the naive command diverges and the DLS command does not."""
    dls = DampedLeastSquares(lam=0.05)
    naive = NaiveInverse()
    xdot = np.array([1.0, 0.0, 0.0])

    naive_norms, dls_norms = [], []
    for q2 in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        q = np.array([0.3, q2, 0.7])
        naive_norms.append(np.linalg.norm(naive.compute(ARM, q, xdot)[0]))
        dls_norms.append(np.linalg.norm(dls.compute(ARM, q, xdot)[0]))

    # naive grows without bound; DLS is capped by 1/(2 lam) = 10
    assert naive_norms[-1] > 100.0 * naive_norms[0]
    assert max(dls_norms) <= 1.0 / (2.0 * 0.05) + 1e-9


# ------------------------------------------------------------------- schedule
def test_adaptive_lambda_schedule_shape():
    a = AdaptiveDLS(w0=0.2, lam_max=0.5)
    assert a.damping(ARM, np.array([0.0, np.pi / 2, 0.0])) == 0.0      # w = 1 > w0
    assert a.damping(ARM, np.array([0.0, 0.0, 0.0])) == pytest.approx(0.5)  # w = 0


def test_adaptive_lambda_is_continuous_at_the_threshold():
    """A step in lambda would show up as a joint-acceleration spike."""
    a = AdaptiveDLS(w0=0.2, lam_max=0.5)
    # w = |sin(q2)|, so q2 = asin(w) walks w across the threshold.
    below = a.damping(ARM, np.array([0.0, float(np.arcsin(0.2 - 1e-6)), 0.0]))
    above = a.damping(ARM, np.array([0.0, float(np.arcsin(0.2 + 1e-6)), 0.0]))
    assert above == 0.0
    assert below == pytest.approx(0.0, abs=1e-4)


def test_adaptive_lambda_is_monotone_in_w():
    a = AdaptiveDLS(w0=0.3, lam_max=0.4)
    ws = np.linspace(0.0, 0.3, 25)
    lams = [a.damping(ARM, np.array([0.0, float(np.arcsin(w)), 0.0])) for w in ws]
    assert np.all(np.diff(lams) <= 1e-12)      # more manipulability -> less damping


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"w0": 0.0}, "w0 must be positive"),
        ({"lam_max": -1.0}, "lam_max must be non-negative"),
    ],
)
def test_adaptive_rejects_bad_gains(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AdaptiveDLS(**kwargs)


def test_fixed_dls_rejects_negative_lambda():
    with pytest.raises(ValueError, match="lam must be non-negative"):
        DampedLeastSquares(lam=-0.1)
