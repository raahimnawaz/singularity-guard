"""Phase 2: where the singularities are, in joint space and in the workspace.

Phase 1 gave the closed form ``det(J) = l1 * l2 * sin(q2)``, so manipulability is::

    w(q) = |det(J)| = l1 * l2 * |sin(q2)|

which depends on *q2 alone*.  This module turns that scalar into the two maps the
controller work is aimed at, and — more importantly — into something falsifiable:
each map is computed by brute-force sweep and checked against a closed form
derived independently of the sweep.

Two views, because they answer different questions:

**Joint space** ``(q2, q3)``.  Direct evaluation of ``w``.  The map comes out in
vertical bands — perfectly flat in ``q3`` — with zeros exactly on
``q2 in {0, pi}``.  That flatness *is* the visual proof of the derivation, and it
is why the earlier guess that ``q3`` causes a singularity was wrong.

**Workspace** ``(x, y)``.  Here the natural question is not "what is ``w``" (a
point can be reached by several configurations with different ``w``) but "how
well-conditioned *can* I be while standing here?"  So we map the best case::

    w_max(x, y) = max { w(q) : forward_kinematics(q)[:2] == (x, y) }

Because ``q1`` only rotates the whole arm about the base and ``w`` does not depend
on it, ``w_max`` is a function of the radius ``r = hypot(x, y)`` alone, and it has
a closed form.  Write the wrist (joint-3) radius as::

    r2(q2) = sqrt(l1^2 + l2^2 + 2*l1*l2*cos(q2))

With ``q1`` and ``q3`` free, the end-effector can sit anywhere in the annulus
``|r2 - l3| <= r <= r2 + l3``.  Inverting that, radius ``r`` is reachable with a
given ``q2`` iff ``r2(q2)`` lies in ``[|r - l3|, r + l3]``.  Substituting
``u = cos(q2)``, which is increasing in ``r2``, turns the constraint into an
interval ``u in [u_lo, u_hi]``, and since ``|sin(q2)| = sqrt(1 - u^2)`` is largest
at ``u = 0`` the optimum is just the point of that interval closest to zero.

The upshot is the headline of this phase: **reaching far costs conditioning**.
Near the outer boundary the arm must straighten, which drives ``q2 -> 0``, which
drives ``w -> 0``.  Everywhere else the arm can usually pick a configuration with
``q2`` near ``+-pi/2`` and be maximally conditioned, so the danger zone is a thin
outer shell rather than a broad region — which is exactly why Phase 3 has to
*aim* at a singularity to hit one.
"""

from __future__ import annotations

import numpy as np

from armsim.kinematics import RobotArm

Array = np.ndarray

#: Values of ``q2`` (mod 2*pi) at which the arm is singular, from ``det(J) = 0``.
SINGULAR_Q2: tuple[float, float] = (0.0, np.pi)


def manipulability_of_q2(arm: RobotArm, q2: Array | float) -> Array:
    """``w = l1 * l2 * |sin(q2)|``, vectorised over ``q2``.

    The closed form from Phase 1.  Present separately from
    :func:`armsim.kinematics.manipulability` because that one is scalar-config and
    goes through the Jacobian; this one is the analytical shortcut used to paint
    whole grids at once.  The tests assert the two agree.
    """
    q2 = np.asarray(q2, dtype=float)
    return arm.l1 * arm.l2 * np.abs(np.sin(q2))


def joint_space_grid(
    arm: RobotArm,
    n_q2: int = 361,
    n_q3: int = 361,
) -> tuple[Array, Array, Array]:
    """Manipulability over the ``(q2, q3)`` square ``[-pi, pi]^2``.

    Returns ``(q2, q3, W)`` where ``W[i, j] = w(q2[j], q3[i])`` — row index is
    ``q3`` and column index is ``q2``, i.e. the layout ``imshow`` wants for
    ``q2`` on the x-axis.  Every row is identical; that is the point.
    """
    q2 = np.linspace(-np.pi, np.pi, n_q2)
    q3 = np.linspace(-np.pi, np.pi, n_q3)
    w_row = manipulability_of_q2(arm, q2)
    return q2, q3, np.tile(w_row, (n_q3, 1))


def max_manipulability_at_radius(arm: RobotArm, r: Array | float) -> Array:
    """Best manipulability achievable with the end-effector at radius ``r``.

    Closed form (see the module docstring).  Returns ``0`` for radii outside the
    reachable annulus, so it can be evaluated on a full grid without masking.
    """
    r = np.asarray(r, dtype=float)
    l1, l2, l3 = arm.links

    # Wrist radius must land in [|r - l3|, r + l3] to put the tip at radius r, and
    # is itself confined to [|l1 - l2|, l1 + l2] by the inner two links.
    r2_lo = np.maximum(np.abs(r - l3), abs(l1 - l2))
    r2_hi = np.minimum(r + l3, l1 + l2)
    reachable = r2_lo <= r2_hi + 1e-12

    # w = l1*l2*|sin(q2)| is twice the area of the base-joint2-wrist triangle, so
    # Heron in terms of the wrist radius r2 gives
    #     w(r2) = 0.5 * sqrt( ((l1+l2)^2 - r2^2) * (r2^2 - (l1-l2)^2) ).
    # Preferred over the sqrt(1 - cos^2) route: that one cancels catastrophically
    # exactly where we care most (arm straight, cos(q2) -> 1), reporting ~1e-8
    # instead of 0.  This form is exactly zero when the triangle degenerates.
    # In s = r2^2 it is a downward parabola peaking at s = l1^2 + l2^2 (that is
    # q2 = +-pi/2), so the best feasible s is the peak clamped to the feasible span.
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
    """Brute-force ``w_max`` versus radius, for validating the closed form.

    Sweeps ``(q2, q3)`` (``q1`` is fixed at zero — it only rotates the arm and
    leaves both ``r`` and ``w`` untouched), bins the resulting tip radii, and keeps
    the largest ``w`` in each bin.

    Returns ``(r_centres, w_max, r_at_max)``, each of length ``n_r`` with ``nan`` in
    bins no sample landed in.  ``r_at_max`` is the radius of the configuration that
    actually achieved ``w_max`` in that bin, and it is *not* the bin centre — the
    difference matters, because ``w_max(r)`` falls off a cliff near full reach, so a
    bin maximum plotted at its centre looks like it beats the closed form when it is
    really just sitting at the low-``r`` end of its bin.  Comparing at ``r_at_max``
    removes that artefact rather than budgeting a tolerance for it.
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

    # Recover where each bin's winner actually sat.  Samples tie often (w depends
    # only on q2), and any winner is equally valid, so an arbitrary one will do.
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
    """Brute-force ``w_max(x, y)`` over the workspace.

    Sweeps all three joints and keeps, per ``(x, y)`` bin, the largest ``w`` of any
    configuration landing in it.  ``q1`` is swept explicitly rather than assuming
    the radial symmetry, so the resulting picture is an independent check that the
    map really is radially symmetric.

    Returns ``(x_centres, y_centres, W)`` with ``W[i, j]`` at ``(x[j], y[i])`` and
    ``nan`` in unreached bins.
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
    """Outer radius at which the arm is forced singular — its full reach.

    At ``r = l1 + l2 + l3`` the only configuration that gets there is the fully
    extended one, which has ``q2 = 0``.  This is the workspace-side face of the
    joint-space locus, and the target for the Phase 3 trajectory.
    """
    return arm.reach
