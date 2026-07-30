"""Velocity-level controllers: the naive inverse, and the damped-least-squares fix.

Three laws, one interface, so the experiment harness can swap them and the only
difference in the resulting plots is the control law itself:

* :class:`NaiveInverse` -- ``qdot = J^-1 xdot``. Phase 3's baseline, included to
  fail.
* :class:`DampedLeastSquares` -- ``qdot = J^T (J J^T + lam^2 I)^-1 xdot`` with a
  constant ``lam``. Bounded near the singularity, but pays damping error
  *everywhere*, including where the arm is perfectly well-conditioned.
* :class:`AdaptiveDLS` -- the same law with ``lam`` scheduled off manipulability,
  so damping is spent only where it is needed. Phase 4's contribution.

Everything is closed-form and fixed-size: an explicit cofactor inverse, no
``np.linalg.solve``, no SVD. That is not a micro-optimisation, it is the Phase 5
constraint -- this file has to become dependency-free C++ running inside a 1 ms
budget, and a 3x3 cofactor inverse is a fixed instruction sequence with no
allocation and no iteration.

Note that :class:`DampedLeastSquares` inverts ``J J^T + lam^2 I``, not ``J``.
That matrix is symmetric positive-definite for any ``lam > 0``, so the cofactor
inverse is well-conditioned by construction even when ``J`` itself is exactly
singular. Avoiding the ill-conditioned inverse is the entire point of the method,
so it would be self-defeating to reach for a general solver here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from armsim.kinematics import Array, RobotArm, jacobian

__all__ = [
    "AdaptiveDLS",
    "DampedLeastSquares",
    "NaiveInverse",
    "det3",
    "inv3",
]

def det3(m: Array) -> float:
    """Determinant of a 3x3 by cofactor expansion (the form that ports to C++)."""
    return float(
        m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
    )


def inv3(m: Array) -> Array:
    """Explicit 3x3 inverse via the adjugate. Blows up when ``m`` is singular.

    Deliberately does not guard against a small determinant: the caller decides
    what a singular Jacobian means. :class:`NaiveInverse` wants the blow-up,
    because that blow-up is the finding.

    In practice the blow-up is enormous but *finite*. At an analytically singular
    configuration the closed form gives ``det(J) = l1*l2*sin(0) = 0`` exactly, but
    the cofactor expansion of a numerically-evaluated ``J`` lands near 3e-16
    instead, so this returns entries around 1e15 rather than ``inf``. That is the
    honest failure mode of the naive law on real hardware -- a huge finite command
    the actuator cannot honour, not a tidy NaN someone would have caught.
    """
    d = det3(m)
    adj = np.array(
        [
            [
                m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1],
                m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2],
                m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1],
            ],
            [
                m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2],
                m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0],
                m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2],
            ],
            [
                m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0],
                m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1],
                m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0],
            ],
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return adj / d


@dataclass(frozen=True)
class NaiveInverse:
    """``qdot = J^-1 xdot``. The baseline that fails.

    Kept as a real implementation rather than a straw man: this is exactly what a
    textbook resolved-rate controller does, and it is correct and well-behaved
    everywhere except near a singularity. The point of Phase 3 is to measure
    *how badly* that exception bites, not to invent a bad controller.
    """

    name: str = "naive"
    label: str = r"naive $J^{-1}$"

    def compute(self, arm: RobotArm, q: Array, xdot: Array) -> tuple[Array, float]:
        j = jacobian(arm, q)
        return inv3(j) @ np.asarray(xdot, dtype=float), 0.0

    def damping(self, arm: RobotArm, q: Array) -> float:  # noqa: ARG002 - interface
        return 0.0


@dataclass(frozen=True)
class DampedLeastSquares:
    """``qdot = J^T (J J^T + lam^2 I)^-1 xdot`` with fixed ``lam``.

    Solves the regularised least-squares problem
    ``min ||J qdot - xdot||^2 + lam^2 ||qdot||^2``: it trades task accuracy for
    joint-velocity norm at a fixed exchange rate. Bounded everywhere, but the
    fixed rate is the flaw -- the damping error is paid in well-conditioned
    regions where no damping was needed. That is what :class:`AdaptiveDLS` fixes,
    and why this class is here as the intermediate step rather than the answer.
    """

    lam: float = 0.1
    name: str = "dls_fixed"

    def __post_init__(self) -> None:
        if self.lam < 0.0:
            raise ValueError("lam must be non-negative")

    @property
    def label(self) -> str:
        return rf"fixed-$\lambda$ DLS ($\lambda$={self.lam:g})"

    def compute(self, arm: RobotArm, q: Array, xdot: Array) -> tuple[Array, float]:
        return _dls(arm, q, xdot, self.lam), self.lam

    def damping(self, arm: RobotArm, q: Array) -> float:  # noqa: ARG002 - interface
        return self.lam


@dataclass(frozen=True)
class AdaptiveDLS:
    """DLS with damping scheduled off manipulability -- the Phase 4 contribution.

    Nakamura/Chiaverini scheduling::

        lam^2 = lam_max^2 * (1 - w/w0)^2   for w < w0
        lam   = 0                          otherwise

    so ``lam`` ramps continuously from 0 at the threshold ``w0`` to ``lam_max``
    at the singularity. Continuity matters: a step change in ``lam`` would show
    up as a joint-acceleration spike, which is exactly the kind of artefact the
    smoothness metric is there to catch.

    Away from the singularity this *is* the naive controller (``lam = 0``,
    identical output), so it inherits the baseline's accuracy where the baseline
    is fine and only diverges where the baseline breaks. That property is
    asserted in the tests.
    """

    # Defaults chosen from the (w0, lam_max) sweep in run_experiment.py, not by
    # taste: this is the cheapest corner of the grid that reaches 0% saturation
    # on the through-singularity run. Damping earlier turns out to be nearly
    # free -- pushing w0 from 0.05 to 0.30 cuts peak commanded velocity ~6x while
    # tracking error *improves*, because the error was coming from saturation
    # rather than from the damping.
    w0: float = 0.30
    lam_max: float = 0.20
    name: str = "dls_adaptive"

    def __post_init__(self) -> None:
        if self.w0 <= 0.0:
            raise ValueError("w0 must be positive")
        if self.lam_max < 0.0:
            raise ValueError("lam_max must be non-negative")

    @property
    def label(self) -> str:
        return rf"adaptive-$\lambda$ DLS ($w_0$={self.w0:g}, $\lambda_{{max}}$={self.lam_max:g})"

    def damping(self, arm: RobotArm, q: Array) -> float:
        """Scheduled ``lambda`` from the manipulability at ``q``.

        Uses ``|det3(J)|`` rather than ``kinematics.manipulability()``. For a
        square Jacobian the two are the same quantity -- ``w = sqrt(det(J J^T)) =
        |det(J)|`` -- but they are not numerically equal near a singularity.
        Forming the Gram matrix squares the condition number, so
        ``sqrt(det(J J^T))`` bottoms out around 1e-8 (documented in
        ``tests/test_kinematics.py``) while the cofactor determinant stays
        accurate to roundoff, ~3e-16.

        The difference is immaterial to the output -- a 1e-8 error in ``w``
        perturbs ``lambda`` by ~1e-7 -- but it is free, and it removes a numerical
        floor from the one input the scheduler reads closest to zero. It also
        matters for Phase 5: ``det3`` is a fixed arithmetic sequence that ports to
        dependency-free C++ untouched, whereas ``manipulability()`` calls
        ``np.linalg.det``.
        """
        w = abs(det3(jacobian(arm, q)))
        if w >= self.w0:
            return 0.0
        return float(self.lam_max * (1.0 - w / self.w0))

    def compute(self, arm: RobotArm, q: Array, xdot: Array) -> tuple[Array, float]:
        lam = self.damping(arm, q)
        return _dls(arm, q, xdot, lam), lam


def _dls(arm: RobotArm, q: Array, xdot: Array, lam: float) -> Array:
    """Shared DLS kernel. ``lam == 0`` reduces exactly to ``J^-1 xdot``."""
    j = jacobian(arm, q)
    xdot = np.asarray(xdot, dtype=float)
    if lam == 0.0:
        return inv3(j) @ xdot
    a = j @ j.T + (lam * lam) * np.eye(3)
    return j.T @ (inv3(a) @ xdot)
