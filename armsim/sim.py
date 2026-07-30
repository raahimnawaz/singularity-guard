"""Closed-loop simulation of the arm under a velocity controller, and the metrics.

Kinematic plant: the state is ``q`` and the controller commands ``qdot`` directly.
No torques, no inertia. That is the honest scope of this project -- it is about
*kinematic* singularities and the velocity-level laws that stay well-behaved
through them, and adding a dynamic model would obscure the one effect being
measured.

Two decisions in here shape every number the README reports.

**Saturation lives in the plant, not the controller.** Real joints have speed
limits, and clamping there is what makes the naive controller fail *realistically*
-- it saturates and loses tracking, rather than reporting ``1e16 rad/s`` and
making the comparison a strawman about floating point. Putting the clamp in the
controller instead would be handing the baseline a safety feature it does not
have, which is the opposite error.

**Control is resolved-rate with proportional feedback**::

    xdot_cmd = pdot_ref + kp * (p_ref - p_actual)

Pure feedforward (``kp = 0``) is available and is the cleaner isolation of the
Jacobian-inversion effect, but it is not a controller anyone ships: integration
error accumulates with nothing to correct it, so every law drifts and the plots
show drift rather than singularity behaviour. With feedback the interesting thing
becomes visible -- whether a law *recovers* after the singularity, or has thrown
the arm somewhere it cannot come back from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from armsim.kinematics import (
    Array,
    RobotArm,
    forward_kinematics,
    jacobian,
    manipulability,
)
from armsim.trajectory import Trajectory, wrap_angle

__all__ = [
    "Limits",
    "SimResult",
    "metrics",
    "pose_error",
    "simulate",
]


@dataclass(frozen=True)
class Limits:
    """Plant-side actuator limits.

    ``qdot_max`` is per-joint and generous for a small arm; the point is that it
    is finite, so an unbounded command has a visible consequence.
    """

    qdot_max: float = 3.0     # rad/s, per joint

    def __post_init__(self) -> None:
        if self.qdot_max <= 0.0:
            raise ValueError("qdot_max must be positive")


def pose_error(p_ref: Array, p: Array) -> Array:
    """``p_ref - p`` with the orientation component wrapped into ``(-pi, pi]``."""
    e = np.asarray(p_ref, dtype=float) - np.asarray(p, dtype=float)
    e[2] = wrap_angle(e[2])
    return e


@dataclass
class SimResult:
    """Per-step logs. Every array is length ``n`` and index-aligned on ``t``."""

    name: str
    trajectory: str
    dt: float
    t: Array
    q: Array                  # (n, 3)
    qdot_cmd: Array           # (n, 3) what the controller asked for
    qdot_applied: Array       # (n, 3) what the plant allowed
    p: Array                  # (n, 3) actual pose
    p_ref: Array              # (n, 3) reference pose
    w: Array                  # (n,)   manipulability
    lam: Array                # (n,)   damping actually used
    cond: Array               # (n,)   condition number of J (diagnostic)
    saturated: Array          # (n,)   bool, any joint clamped this step
    n_nonfinite: int = 0      # steps where the control law produced NaN/inf
    meta: dict = field(default_factory=dict)

    @property
    def err(self) -> Array:
        """(n, 3) pose error, orientation wrapped."""
        e = self.p_ref - self.p
        e[:, 2] = wrap_angle(e[:, 2])
        return e


def simulate(
    arm: RobotArm,
    traj: Trajectory,
    controller,
    *,
    dt: float = 1e-3,
    kp: float = 5.0,
    limits: Limits | None = None,
    q0: Array | None = None,
) -> SimResult:
    """Integrate the closed loop with fixed-step RK4.

    RK4 evaluates the control law four times per step, which treats it as a
    continuous-time law rather than a zero-order-hold digital controller. That is
    the plan's choice and it keeps the Python and C++ integrators comparable
    step-for-step; a ZOH model would be the next refinement if this were driving
    real hardware at a fixed service rate.

    The saturating clamp makes the closed-loop vector field non-smooth, so RK4's
    formal fourth-order accuracy does not hold across a saturation boundary. It
    stays a stable, consistent integrator there, and only the naive controller
    spends meaningful time saturated -- so this costs accuracy exactly where the
    trajectory is already a failure case, not in the comparisons that matter.
    """
    limits = limits or Limits()
    q = np.asarray(traj.q_start() if q0 is None else q0, dtype=float).copy()

    n = int(round(traj.duration / dt)) + 1
    ts = np.arange(n) * dt

    log_q = np.zeros((n, 3))
    log_cmd = np.zeros((n, 3))
    log_app = np.zeros((n, 3))
    log_p = np.zeros((n, 3))
    log_pref = np.zeros((n, 3))
    log_w = np.zeros(n)
    log_lam = np.zeros(n)
    log_cond = np.zeros(n)
    log_sat = np.zeros(n, dtype=bool)
    n_nonfinite = 0

    lim = limits.qdot_max

    def control(t: float, qq: Array) -> tuple[Array, Array, float, bool, bool]:
        """Returns (qdot_applied, qdot_cmd, lam, saturated, nonfinite)."""
        p_now = forward_kinematics(arm, qq)
        xdot_cmd = traj.twist(t) + kp * pose_error(traj.pose(t), p_now)
        qdot_cmd, lam = controller.compute(arm, qq, xdot_cmd)
        qdot_cmd = np.asarray(qdot_cmd, dtype=float)

        nonfinite = not bool(np.all(np.isfinite(qdot_cmd)))
        if nonfinite:
            # A singular J has no inverse, so the naive law has no answer here.
            # Commanding zero (rather than a large number) is the conservative
            # reading and keeps the metrics finite; the step is counted and
            # reported so this never silently flatters the baseline.
            qdot_cmd = np.nan_to_num(qdot_cmd, nan=0.0, posinf=lim, neginf=-lim)

        qdot_app = np.clip(qdot_cmd, -lim, lim)
        sat = bool(np.any(np.abs(qdot_cmd) > lim))
        return qdot_app, qdot_cmd, lam, sat, nonfinite

    for i, t in enumerate(ts):
        qdot_app, qdot_cmd, lam, sat, nonfinite = control(t, q)

        j = jacobian(arm, q)
        log_q[i] = q
        log_cmd[i] = qdot_cmd
        log_app[i] = qdot_app
        log_p[i] = forward_kinematics(arm, q)
        log_pref[i] = traj.pose(t)
        log_w[i] = manipulability(arm, q)
        log_lam[i] = lam
        log_cond[i] = float(np.linalg.cond(j))
        log_sat[i] = sat
        n_nonfinite += int(nonfinite)

        if i == n - 1:
            break

        # RK4 on qdot = f(t, q), f being the saturated closed-loop field.
        k1 = qdot_app
        k2 = control(t + 0.5 * dt, q + 0.5 * dt * k1)[0]
        k3 = control(t + 0.5 * dt, q + 0.5 * dt * k2)[0]
        k4 = control(t + dt, q + dt * k3)[0]
        q = q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return SimResult(
        name=controller.name,
        trajectory=traj.name,
        dt=dt,
        t=ts,
        q=log_q,
        qdot_cmd=log_cmd,
        qdot_applied=log_app,
        p=log_p,
        p_ref=log_pref,
        w=log_w,
        lam=log_lam,
        cond=log_cond,
        saturated=log_sat,
        n_nonfinite=n_nonfinite,
        meta={"kp": kp, "qdot_max": limits.qdot_max, "label": getattr(controller, "label", "")},
    )


def metrics(res: SimResult) -> dict:
    """The metric set defined once in docs/PLAN.md, computed the same way for every run.

    Position and orientation error are reported separately on purpose: they have
    different units, and averaging metres against radians into one number would
    hide which one the damping is actually costing.
    """
    err = res.err
    pos_err = np.linalg.norm(err[:, :2], axis=1)
    ori_err = np.abs(err[:, 2])

    # Smoothness: RMS joint acceleration, finite-differenced from the velocity
    # the plant actually applied (not the command -- the arm never felt the
    # clamped part, and differencing a saturated command reports a jerk that
    # never physically happened).
    if len(res.t) > 1:
        accel = np.diff(res.qdot_applied, axis=0) / res.dt
        accel_rms = float(np.sqrt(np.mean(accel**2)))
    else:
        accel_rms = 0.0

    return {
        "controller": res.name,
        "trajectory": res.trajectory,
        "label": res.meta.get("label", ""),
        "peak_qdot_cmd": float(np.max(np.abs(res.qdot_cmd))),
        "peak_qdot_applied": float(np.max(np.abs(res.qdot_applied))),
        "pos_err_rms": float(np.sqrt(np.mean(pos_err**2))),
        "pos_err_max": float(np.max(pos_err)),
        "ori_err_rms": float(np.sqrt(np.mean(ori_err**2))),
        "ori_err_max": float(np.max(ori_err)),
        "accel_rms": accel_rms,
        "pct_saturated": float(100.0 * np.mean(res.saturated)),
        "w_min": float(np.min(res.w)),
        "cond_max": float(np.max(res.cond)),
        "lam_max_used": float(np.max(res.lam)),
        "n_nonfinite": int(res.n_nonfinite),
    }
