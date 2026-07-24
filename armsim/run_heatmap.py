"""Render the Phase 2 singularity maps.

    python -m armsim.run_heatmap                 # full resolution into results/
    python -m armsim.run_heatmap --quick         # coarse, for a fast look
    python -m armsim.run_heatmap --out /tmp/figs --l1 1.3 --l2 0.7 --l3 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs in CI

import numpy as np  # noqa: E402

from armsim.kinematics import RobotArm  # noqa: E402
from armsim.singularity import max_manipulability_at_radius  # noqa: E402
from armsim.viz import (  # noqa: E402
    apply_style,
    plot_joint_space,
    plot_radial_validation,
    plot_workspace,
)


def _summary(arm: RobotArm) -> dict:
    """The few numbers worth committing next to the figures."""
    r = np.linspace(0.0, arm.reach, 20001)
    w = max_manipulability_at_radius(arm, r)
    safe = w >= 0.9 * arm.l1 * arm.l2
    reachable = w > 0.0
    return {
        "links": {"l1": arm.l1, "l2": arm.l2, "l3": arm.l3},
        "reach": arm.reach,
        "w_max_global": float(arm.l1 * arm.l2),
        "singular_q2_rad": [0.0, float(np.pi)],
        "min_reachable_radius": float(r[reachable][0]) if reachable.any() else None,
        "largest_radius_at_90pct_conditioning": float(r[safe][-1]) if safe.any() else None,
        "fraction_of_reach_well_conditioned": float(r[safe][-1] / arm.reach) if safe.any() else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Render Phase 2 singularity maps.")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--l1", type=float, default=1.0)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--l3", type=float, default=1.0)
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--quick", action="store_true", help="coarse sweeps, ~10x faster")
    args = p.parse_args()

    arm = RobotArm(args.l1, args.l2, args.l3)
    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()

    if args.quick:
        joint_n, ws_bins, ws_q1, ws_q, rad_r, rad_q = 181, 120, 60, 120, 70, 500
    else:
        joint_n, ws_bins, ws_q1, ws_q, rad_r, rad_q = 481, 320, 180, 260, 140, 1400

    figures = {
        "joint_space_manipulability.png": plot_joint_space(arm, n=joint_n),
        "workspace_manipulability.png": plot_workspace(
            arm, n_bins=ws_bins, n_q1=ws_q1, n_q=ws_q
        ),
        "radial_validation.png": plot_radial_validation(arm, n_r=rad_r, n_q=rad_q),
    }
    for name, fig in figures.items():
        path = args.out / name
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {path}")

    summary = _summary(arm)
    (args.out / "singularity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out / 'singularity_summary.json'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
