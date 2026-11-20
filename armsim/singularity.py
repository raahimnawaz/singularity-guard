"""Where the singularities are, in joint space and in the workspace.

Manipulability ``w = l1 * l2 * |sin(q2)|`` depends on ``q2`` alone. Each map is
computed by brute-force sweep and checked against a closed form:

* joint space ``(q2, q3)``: ``w`` directly, flat in ``q3``, zero at ``q2 in {0, pi}``;
* workspace: the best ``w`` reachable at each radius, ``w_max(r)``, which only
  falls to zero in a thin shell near full reach.
"""

from __future__ import annotations

import numpy as np

from armsim.kinematics import RobotArm

Array = np.ndarray

#: Values of ``q2`` (mod 2*pi) at which the arm is singular, from ``det(J) = 0``.
SINGULAR_Q2: tuple[float, float] = (0.0, np.pi)


def manipulability_of_q2(arm: RobotArm, q2: Array | float) -> Array:
    """``w = l1 * l2 * |sin(q2)|``, vectorised over ``q2``."""
    q2 = np.asarray(q2, dtype=float)
    return arm.l1 * arm.l2 * np.abs(np.sin(q2))


def joint_space_grid(
    arm: RobotArm,
    n_q2: int = 361,
    n_q3: int = 361,
) -> tuple[Array, Array, Array]:
    """Manipulability over ``(q2, q3)`` in ``[-pi, pi]^2``.

    Returns ``(q2, q3, W)`` with ``W[i, j] = w(q2[j], q3[i])``.
    """
    q2 = np.linspace(-np.pi, np.pi, n_q2)
    q3 = np.linspace(-np.pi, np.pi, n_q3)
    w_row = manipulability_of_q2(arm, q2)
    return q2, q3, np.tile(w_row, (n_q3, 1))


def max_manipulability_at_radius(arm: RobotArm, r: Array | float) -> Array:
    """Best manipulability with the end-effector at radius ``r``, in closed form.

    Returns 0 outside the reachable annulus.
    """
    r = np.asarray(r, dtype=float)
    l1, l2, l3 = arm.links

    # Wrist radius must land in [|r - l3|, r + l3] to put the tip at radius r, and
    # is itself confined to [|l1 - l2|, l1 + l2] by the inner two links.
    r2_lo = np.maximum(np.abs(r - l3), abs(l1 - l2))
    r2_hi = np.minimum(r + l3, l1 + l2)
    reachable = r2_lo <= r2_hi + 1e-12

    # w is twice the base-joint2-wrist triangle's area, so by Heron
    #     w(r2) = 0.5 * sqrt(((l1+l2)^2 - r2^2) * (r2^2 - (l1-l2)^2)),
    # which is exactly zero when the arm is straight (sqrt(1 - cos^2) is not).
    span_hi, span_lo = (l1 + l2) ** 2, (l1 - l2) ** 2
    s = np.clip(l1**2 + l2**2, r2_lo**2, r2_hi**2)
    w = 0.5 * np.sqrt(np.maximum(0.0, (span_hi - s) * (s - span_lo)))
    return np.where(reachable, w, 0.0)


def radial_profile_swept(
    arm: RobotArm,
    n_r: int = 400,
    n_q2: int = 1400,
    n_q3: int = 1400,
) -> tuple[Array, Array]:
    """Brute-force ``w_max`` versus radius, to validate the closed form.

    Returns ``(r_centres, w_max, r_at_max)``, NaN in empty bins. ``r_at_max`` is the
    radius that achieved each bin's maximum, which avoids a bin-centre artefact
    near full reach.
    """
    l1, l2, l3 = arm.links
    q2 = np.linspace(-np.pi, np.pi, n_q2)
    q3 = np.linspace(-np.pi, np.pi, n_q3)

    # tip radius with q1 = 0: position of joint 3 plus link 3, as a function of
    # (q2, q3) only.
    Q2, Q3 = np.meshgrid(q2, q3, indexing="xy")
    x = l1 + l2 * np.cos(Q2) + l3 * np.cos(Q2 + Q3)
    y = l2 * np.sin(Q2) + l3 * np.sin(Q2 + Q3)
    r = np.hypot(x, y).ravel()
    w = np.broadcast_to(manipulability_of_q2(arm, q2), Q2.shape).ravel()

    edges = np.linspace(0.0, arm.reach, n_r + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, n_r - 1)
    best = np.full(n_r, -np.inf)
    np.maximum.at(best, idx, w)

    # Ties are common (w depends only on q2); any winner will do.
    won = w >= best[idx] - 1e-15
    r_at = np.full(n_r, np.nan)
    r_at[idx[won]] = r[won]

    centres = 0.5 * (edges[:-1] + edges[1:])
    seen = np.isfinite(best)
    return centres, np.where(seen, best, np.nan), r_at


def workspace_grid(
    arm: RobotArm,
    n_bins: int = 320,
    n_q1: int = 180,
    n_q2: int = 260,
    n_q3: int = 260,
) -> tuple[Array, Array, Array]:
    """Brute-force ``w_max(x, y)`` over the workspace, sweeping all three joints.

    Returns ``(x_centres, y_centres, W)`` with ``W[i, j]`` at ``(x[j], y[i])``.
    """
    l1, l2, l3 = arm.links
    reach = arm.reach
    edges = np.linspace(-reach, reach, n_bins + 1)
    best = np.full(n_bins * n_bins, -np.inf)

    q1 = np.linspace(-np.pi, np.pi, n_q1, endpoint=False)
    q2 = np.linspace(-np.pi, np.pi, n_q2)
    q3 = np.linspace(-np.pi, np.pi, n_q3)
    Q2, Q3 = np.meshgrid(q2, q3, indexing="xy")
    w_flat = np.broadcast_to(manipulability_of_q2(arm, q2), Q2.shape).ravel()

    # Tip position at q1 = 0; each q1 slice is this rotated about the base.
    x0 = (l1 + l2 * np.cos(Q2) + l3 * np.cos(Q2 + Q3)).ravel()
    y0 = (l2 * np.sin(Q2) + l3 * np.sin(Q2 + Q3)).ravel()

    for a in q1:  # chunked over q1 to keep peak memory to one slice
        ca, sa = np.cos(a), np.sin(a)
        xi = np.clip(np.digitize(ca * x0 - sa * y0, edges) - 1, 0, n_bins - 1)
        yi = np.clip(np.digitize(sa * x0 + ca * y0, edges) - 1, 0, n_bins - 1)
        np.maximum.at(best, yi * n_bins + xi, w_flat)

    grid = np.where(np.isfinite(best), best, np.nan).reshape(n_bins, n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, centres, grid


def singular_radius(arm: RobotArm) -> float:
    """Full reach: the radius at which the arm is forced singular (``q2 = 0``)."""
    return arm.reach
