"""Task-space reference trajectories that provoke (or skim) a singularity.

Phase 2 established that bad conditioning is a thin outer shell: the inner 90% of
the reach radius stays above ``0.9 * w_max``. Drifting around the workspace will
not produce a singularity, so a reference has to *aim* at one.

**The reference is a straight line in task space, and that is the whole point.**

The first attempt at this module built the reference the other way round -- pick a
joint path through ``q2 = 0``, then hand the controller its forward-kinematic
image, ``p_ref = FK(q_ref)`` with ``pdot_ref = J(q_ref) qdot_ref``. That looks
safer: the reference is reachable by construction and provably crosses the
singularity.

It also silently destroys the experiment. If ``pdot_ref = J qdot_ref`` exactly,
then the commanded twist is *always* in the range of ``J``, and a bounded joint
velocity achieving it exists at every instant -- namely ``qdot_ref`` itself. The
naive controller then sails through the singularity with near-zero error and
damping only makes things worse. Measured, before the rewrite: naive peaked at
1.5 rad/s with no tracking error, adaptive DLS at 7.4 rad/s with 0.058 m RMS.
Exactly backwards, and for a real reason.

A singularity is not "the arm is in a funny pose". It is "certain task velocities
are no longer achievable". Handing the controller only achievable velocities
removes the phenomenon while appearing to test it.

So the reference here is a straight line through the singular pose, at whatever
speed the time scaling dictates, with **no feasibility guarantee at all** -- which
is also what a real controller gets. An operator jogging a Cartesian direction,
or a planner handing down a twist, does not first check it against the range of
the current Jacobian. That is the controller's problem, and how it handles the
impossible request is the finding.

Anchor configurations are still used, but only to *locate* poses via FK, never to
define the motion.
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
    """Wrap angle(s) into ``(-pi, pi]``.

    Orientation error must be wrapped or a controller will take the long way
    round a +-pi boundary.
    """
    return np.mod(np.asarray(a) + np.pi, 2.0 * np.pi) - np.pi


def quintic(tau: float) -> tuple[float, float]:
    """Quintic time scaling ``s(tau)`` and ``ds/dtau`` on ``tau in [0, 1]``.

    ``s = 10t^3 - 15t^4 + 6t^5`` has zero velocity *and* acceleration at both
    ends, so the arm starts and stops smoothly and the run is not dominated by a
    step-input transient. Its peak speed is at ``tau = 0.5`` -- which is where
    the crossing is placed, so the arm arrives at the worst configuration while
    moving fastest. That is deliberate.
    """
    tau = float(np.clip(tau, 0.0, 1.0))
    s = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
    ds = 30.0 * tau**2 * (1.0 - 2.0 * tau + tau**2)
    return s, ds


@dataclass(frozen=True)
class Trajectory:
    """A straight-line (or out-and-back) task-space reference.

    ``_p_of_s`` and ``_dp_ds`` are pure task-space functions of a normalized path
    parameter ``s in [0, 1]``. Time enters only through the quintic scaling, so
    geometry and speed stay independent -- the same path can be run faster
    without redesigning it.

    ``q0`` is where the arm is initialised. It is chosen so ``FK(q0) == p_ref(0)``,
    which starts the run with zero tracking error and puts the arm on the
    inverse-kinematic branch that leads to the singularity rather than away from
    it. That is the only role joint space plays here.
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
    """Straight task-space line passing **exactly through** a singular pose.

    ``det(J) = l1*l2*sin(q2)``, so ``q2 = 0`` is singular regardless of ``q1``,
    ``q3``, or ``l3``. The line runs from ``FK([q1, +a, q3])`` through
    ``FK([q1, 0, q3])`` and on to a mirrored endpoint, so the singular pose sits
    at ``s = 0.5`` -- the quintic's peak speed.

    Why the crossing is guaranteed even though the reference is task-space: for
    this arm a target pose fixes ``phi = q1+q2+q3``, and the wrist position admits
    the usual two 2R solutions (elbow up / elbow down). Those two solutions
    *coincide* exactly when ``q2 = 0`` -- that coincidence is the singularity. So
    the singular pose has a unique preimage, and an arm tracking the line has
    nowhere else to be when it gets there.
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
    """A straight line that never approaches a singularity: the control case.

    ``w = |sin(q2)|`` peaks at ``q2 = pi/2``, so sweeping ``q2`` symmetrically
    about it keeps the arm near maximum manipulability throughout. Every
    controller should track this well, which is exactly why it belongs in the
    comparison: it is where fixed-lambda damping pays a bill for nothing, and
    where adaptive damping should be *bit-identical* to the naive inverse because
    its schedule returns ``lambda = 0``. Without this scenario the metrics table
    only shows what damping buys, never what it costs.
    """
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
    """Approach a near-singular pose and retreat, never crossing.

    Runs out along the same line as :func:`through_singularity` but stops at
    ``FK([q1, q2_near, q3])`` and comes back, on a half-sine in ``s``. ``w`` gets
    small but never reaches zero, so a bounded exact solution exists at *every*
    instant -- the naive controller has no excuse here, and still spikes.

    The turnaround is unavoidable, not a shortcut: the singular set is a 2-D
    surface in the 3-D task space, so a 1-D path with both endpoints on the same
    side of it cannot avoid crossing without reversing. A consequence worth
    stating plainly is that commanded speed passes through zero at closest
    approach, so the naive controller's worst moment is not at minimum ``w`` but
    slightly before and after it, where small ``w`` and nonzero speed overlap.
    That is also what decelerating into a tight corner looks like in practice.
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
