"""Plotting for the singularity maps (Phase 2) and the arm itself (reused later).

Figures are written as PNGs into ``results/`` by ``armsim.run_heatmap``.  Nothing
here is imported by the control path -- keep matplotlib out of anything that has
to stay portable to C++ in Phase 5.

Colour choices follow one rule: manipulability is a *magnitude*, so it gets a
single-hue sequential ramp (light = near zero, dark = well conditioned), never a
rainbow.  The singular locus is drawn in the reserved "critical" red and always
carries a text label, so the meaning never rests on colour alone.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from armsim.kinematics import RobotArm, joint_positions
from armsim.singularity import (
    SINGULAR_Q2,
    joint_space_grid,
    max_manipulability_at_radius,
    radial_profile_swept,
    workspace_grid,
)

# --- palette ---------------------------------------------------------------
# Sequential blue ramp, steps 100 -> 700, for magnitude.
_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_BLUE = LinearSegmentedColormap.from_list("flux_blue", _BLUE_STEPS)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
CRITICAL = "#d03b3b"      # reserved status colour: the singular locus
SERIES_1 = "#2a78d6"      # categorical slot 1
SERIES_2 = "#eb6834"      # categorical slot 2

_PI_TICKS = ([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
             ["$-\\pi$", "$-\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])


def apply_style() -> None:
    """Recessive chrome, thin marks -- applied by the runner before plotting."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 140,
    })


def draw_arm(ax, arm: RobotArm, q, color: str = INK, alpha: float = 1.0, lw: float = 2.0):
    """Draw the arm's links and joints at configuration ``q``. Reused by Phase 3/4."""
    pts = joint_positions(arm, np.asarray(q, dtype=float))
    (line,) = ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=lw, alpha=alpha,
                      solid_capstyle="round", zorder=5)
    ax.plot(pts[1:3, 0], pts[1:3, 1], "o", color=color, ms=5, alpha=alpha, zorder=6)
    ax.plot(pts[0, 0], pts[0, 1], "s", color=color, ms=6, alpha=alpha, zorder=6)
    ax.plot(pts[3, 0], pts[3, 1], "o", color=color, ms=7, alpha=alpha,
            mfc=SURFACE, mew=2.0, zorder=6)
    return line


# --- the three Phase 2 figures ---------------------------------------------


def plot_joint_space(arm: RobotArm, n: int = 481):
    """``w`` over ``(q2, q3)``: vertical bands, zero exactly on ``q2 in {0, +-pi}``."""
    import matplotlib.pyplot as plt

    q2, q3, w = joint_space_grid(arm, n_q2=n, n_q3=n)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(w, origin="lower", extent=[q2[0], q2[-1], q3[0], q3[-1]],
                   cmap=SEQ_BLUE, aspect="auto", vmin=0.0, interpolation="nearest")

    for k, q2_sing in enumerate((-np.pi, *SINGULAR_Q2)):
        ax.axvline(q2_sing, color=CRITICAL, lw=2.0, ls="--", zorder=3,
                   label="singular locus  $q_2 \\in \\{0, \\pm\\pi\\}$" if k == 0 else None)

    ax.set_xlabel("$q_2$  (rad)")
    ax.set_ylabel("$q_3$  (rad)")
    ax.set_xticks(*_PI_TICKS)
    ax.set_yticks(*_PI_TICKS)
    ax.set_title("Manipulability in joint space", pad=26)
    ax.text(0.5, 1.02, "flat in $q_3$ — only $q_2$ moves $w$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, color=INK_MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9, labelcolor=INK)
    fig.colorbar(im, ax=ax, label="manipulability  $w = |\\det J|$", pad=0.02)
    fig.tight_layout()
    return fig


def plot_workspace(arm: RobotArm, n_bins: int = 320, n_q1: int = 180, n_q: int = 260):
    """Best achievable ``w`` at each ``(x, y)`` -- the "how safe can I be here" map."""
    import matplotlib.pyplot as plt

    x, y, w = workspace_grid(arm, n_bins=n_bins, n_q1=n_q1, n_q2=n_q, n_q3=n_q)
    cmap = SEQ_BLUE.with_extremes(bad=SURFACE)  # unreachable cells drop to the surface

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    im = ax.imshow(np.ma.masked_invalid(w), origin="lower",
                   extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap, vmin=0.0,
                   interpolation="nearest")

    ax.add_patch(plt.Circle((0, 0), arm.reach, fill=False, color=CRITICAL,
                            lw=2.0, ls="--", zorder=4))
    ax.annotate("fully extended → $q_2 = 0$ → singular",
                xy=(0.0, arm.reach), xytext=(0.0, arm.reach * 1.16),
                ha="center", fontsize=9, color=CRITICAL,
                arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1.4))

    # The headline of the phase, computed rather than asserted: how far out you can
    # go before conditioning starts to cost you.
    rr = np.linspace(0.0, arm.reach, 20001)
    safe = max_manipulability_at_radius(arm, rr) >= 0.9 * arm.l1 * arm.l2
    pct = 100.0 * rr[safe][-1] / arm.reach if safe.any() else 0.0

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.set_title("Best achievable manipulability across the workspace", pad=26)
    ax.text(0.5, 1.02, f"danger is a thin outer shell — the inner {pct:.0f}% of the "
            "reach stays above $0.9\\,w_{\\max}$",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=INK_MUTED)
    ax.set_xlim(-arm.reach * 1.08, arm.reach * 1.08)
    ax.set_ylim(-arm.reach * 1.08, arm.reach * 1.30)
    fig.colorbar(im, ax=ax, label="$w_{\\max}(x, y)$", pad=0.02, shrink=0.86)
    fig.tight_layout()
    return fig


def plot_radial_validation(arm: RobotArm, n_r: int = 300, n_q: int = 1200):
    """Swept ``w_max`` vs the closed form -- the figure that makes Phase 2 falsifiable."""
    import matplotlib.pyplot as plt

    # Plot each bin's winner at the radius it actually occurred, not the bin centre
    # -- otherwise the boundary points ride visibly above the curve for no reason
    # other than bin width.
    _, w_swept, r_swept = radial_profile_swept(arm, n_r=n_r, n_q2=n_q, n_q3=n_q)
    r_fine = np.linspace(0.0, arm.reach, 1200)
    w_closed = max_manipulability_at_radius(arm, r_fine)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.grid(True, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    # Markers under the line, so the curve is visibly running *through* them. With
    # the line on top the near-vertical stretch by the boundary reads honestly
    # instead of looking like the dots float above it.
    seen = np.isfinite(w_swept)
    ax.plot(r_swept[seen], w_swept[seen], "o", color=SERIES_2, ms=5.0, zorder=3,
            mfc=SERIES_2, mec=SURFACE, mew=0.8, label="brute-force sweep")
    ax.plot(r_fine, w_closed, "-", color=SERIES_1, lw=2.0, zorder=4,
            label="closed form  $w_{\\max}(r)$")

    ax.axvline(arm.reach, color=CRITICAL, lw=2.0, ls="--", zorder=2)
    ax.annotate("reach — forced singular", xy=(arm.reach, arm.l1 * arm.l2 * 0.62),
                xytext=(-24, 0), textcoords="offset points", ha="right",
                fontsize=9, color=CRITICAL)

    ax.set_xlabel("end-effector radius  $r = \\sqrt{x^2 + y^2}$")
    ax.set_ylabel("$w_{\\max}$")
    ax.set_title("The sweep never beats the algebra")
    ax.set_xlim(0.0, arm.reach * 1.04)
    ax.set_ylim(0.0, arm.l1 * arm.l2 * 1.12)
    ax.legend(loc="lower left", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    return fig
