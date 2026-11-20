"""Task-space reference trajectories that cross, skim or avoid a singularity.

References are straight lines in task space with no feasibility check, as a
real controller would receive. A reference built as ``FK`` of a joint path
would only ever ask for achievable velocities and hide the singularity
(see docs/lessons.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from armsim.kinematics import Array, RobotArm, forward_kinematics

__all__ = [
    "Trajectory",
    "quintic",
    "skim_singularity",
    "through_singularity",
    "well_conditioned",
    "wrap_angle",
]


def wrap_angle(a: Array | float) -> Array | float:
    """Wrap angle(s) into ``(-pi, pi]``."""
    return np.mod(np.asarray(a) + np.pi, 2.0 * np.pi) - np.pi


def quintic(tau: float) -> tuple[float, float]:
    """Quintic time scaling ``s = 10t^3 - 15t^4 + 6t^5`` and ``ds/dtau``.

    Zero velocity and acceleration at both ends; peak speed at ``tau = 0.5``, where
    the crossing is placed.
    """
    tau = float(np.clip(tau, 0.0, 1.0))
    s = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
    ds = 30.0 * tau**2 * (1.0 - 2.0 * tau + tau**2)
    return s, ds


@dataclass(frozen=True)
class Trajectory:
    """A straight-line (or out-and-back) task-space reference.

    ``q0`` satisfies ``FK(q0) == p_ref(0)`` and puts the arm on the branch that
    leads to the singularity.
    """

    arm: RobotArm
    duration: float
    name: str
    q0: Array
    _p_of_s: object       # Callable[[float], Array]
    _dp_ds: object        # Callable[[float], Array]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.duration <= 0.0:
            raise ValueError("duration must be positive")

    def pose(self, t: float) -> Array:
        s, _ = quintic(t / self.duration)
        return np.asarray(self._p_of_s(s), dtype=float)

    def twist(self, t: float) -> Array:
        """``dp/dt = (dp/ds) * (ds/dtau) * (dtau/dt)``."""
        s, ds_dtau = quintic(t / self.duration)
        return np.asarray(self._dp_ds(s), dtype=float) * ds_dtau / self.duration

    def q_start(self) -> Array:
        return np.asarray(self.q0, dtype=float).copy()


def through_singularity(
    arm: RobotArm,
    *,
    q1: float = 0.3,
    q3: float = np.pi / 2.0,
    q2_amplitude: float = 0.8,
    duration: float = 2.0,
) -> Trajectory:
    """Straight line passing exactly through a singular pose (``q2 = 0``) at ``s = 0.5``.

    The two IK branches coincide at ``q2 = 0``, so a tracking arm must pass through it.
    """
    a = float(q2_amplitude)
    if a <= 0.0:
        raise ValueError("q2_amplitude must be positive")

    q_start = np.array([q1, a, q3])
    q_sing = np.array([q1, 0.0, q3])
    p0 = forward_kinematics(arm, q_start)
    p_sing = forward_kinematics(arm, q_sing)
    d = p_sing - p0                     # half the line

    def p_of_s(s: float) -> Array:
        return p0 + 2.0 * s * d

    def dp_ds(s: float) -> Array:  # noqa: ARG001 - constant; signature is the contract
        return 2.0 * d

    return Trajectory(
        arm,
        duration,
        "through",
        q_start,
        p_of_s,
        dp_ds,
        meta={"q_singular": q_sing, "p_singular": p_sing, "s_singular": 0.5},
    )


def well_conditioned(
    arm: RobotArm,
    *,
    q1: float = 0.3,
    q3: float = np.pi / 2.0,
    q2_from: float = np.pi / 2.0 - 0.35,
    q2_to: float = np.pi / 2.0 + 0.35,
    duration: float = 2.0,
) -> Trajectory:
    """A straight line that stays near maximum manipulability: the control case."""
    q_start = np.array([q1, float(q2_from), q3])
    q_end = np.array([q1, float(q2_to), q3])
    p0 = forward_kinematics(arm, q_start)
    p1 = forward_kinematics(arm, q_end)
    d = p1 - p0

    def p_of_s(s: float) -> Array:
        return p0 + s * d

    def dp_ds(s: float) -> Array:  # noqa: ARG001 - constant; signature is the contract
        return d

    return Trajectory(
        arm, duration, "well_cond", q_start, p_of_s, dp_ds, meta={"q_end": q_end}
    )


def skim_singularity(
    arm: RobotArm,
    *,
    q1: float = 0.3,
    q3: float = np.pi / 2.0,
    q2_far: float = 0.8,
    q2_near: float = 0.02,
    duration: float = 2.0,
) -> Trajectory:
    """Approach a near-singular pose and retreat without crossing.

    ``w`` gets small but never zero, so a bounded exact solution always exists.
    """
    far, near = float(q2_far), float(q2_near)
    if near <= 0.0 or far <= 0.0:
        raise ValueError("q2_far and q2_near must be positive (stay on one branch)")
    if near >= far:
        raise ValueError("q2_near must be closer to the singularity than q2_far")

    q_start = np.array([q1, far, q3])
    q_near = np.array([q1, near, q3])
    p0 = forward_kinematics(arm, q_start)
    p_near = forward_kinematics(arm, q_near)
    d = p_near - p0

    def p_of_s(s: float) -> Array:
        return p0 + np.sin(np.pi * s) * d

    def dp_ds(s: float) -> Array:
        return np.pi * np.cos(np.pi * s) * d

    return Trajectory(
        arm,
        duration,
        "skim",
        q_start,
        p_of_s,
        dp_ds,
        meta={"q_closest": q_near, "p_closest": p_near, "s_closest": 0.5},
    )
