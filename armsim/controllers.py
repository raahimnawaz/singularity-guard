"""Velocity-level control laws: the naive inverse and two damped-least-squares variants.

All three use an explicit 3x3 cofactor inverse (no solver, no SVD) so the same
math ports to allocation-free C++.
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
    """Explicit 3x3 inverse via the adjugate.

    Unguarded on purpose: near a singularity it returns huge finite values,
    which is the naive controller's real failure mode.
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
    """``qdot = J^-1 xdot``: the textbook resolved-rate controller."""

    name: str = "naive"
    label: str = r"naive $J^{-1}$"

    def compute(self, arm: RobotArm, q: Array, xdot: Array) -> tuple[Array, float]:
        j = jacobian(arm, q)
        return inv3(j) @ np.asarray(xdot, dtype=float), 0.0

    def damping(self, arm: RobotArm, q: Array) -> float:  # noqa: ARG002 - interface
        return 0.0


@dataclass(frozen=True)
class DampedLeastSquares:
    """``qdot = J^T (J J^T + lam^2 I)^-1 xdot`` with a constant ``lam``.

    Minimises ``||J qdot - xdot||^2 + lam^2 ||qdot||^2``.
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
    """DLS with damping scheduled on manipulability (Nakamura/Chiaverini)::

        lam = lam_max * (1 - w/w0)   for w < w0, else 0

    Continuous in ``w``, and identical to the naive law when ``w >= w0``.
    """

    # Defaults from the (w0, lam_max) sweep in run_experiment.py: the cheapest
    # setting with 0% saturation on the through-singularity run.
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
        """Scheduled ``lambda`` at ``q``.

        Uses ``|det3(J)|``, which stays accurate near zero where
        ``sqrt(det(J J^T))`` bottoms out around 1e-8.
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
