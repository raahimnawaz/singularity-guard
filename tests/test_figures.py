"""The figure pipeline has to keep running, so CI runs it at coarse resolution.

These are smoke tests, not image comparisons -- they catch import errors, bad
kwargs, and shape bugs in the plotting path without pinning pixels.
"""

import json
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from armsim import RobotArm  # noqa: E402
from armsim.viz import (  # noqa: E402
    apply_style,
    draw_arm,
    plot_joint_space,
    plot_radial_validation,
    plot_workspace,
)

ARM = RobotArm(1.0, 1.0, 1.0)


def test_each_figure_builds(tmp_path):
    apply_style()
    figs = [
        plot_joint_space(ARM, n=61),
        plot_workspace(ARM, n_bins=48, n_q1=24, n_q=48),
        plot_radial_validation(ARM, n_r=40, n_q=200),
    ]
    for i, fig in enumerate(figs):
        out = tmp_path / f"fig{i}.png"
        fig.savefig(out, dpi=60)
        assert out.stat().st_size > 2000
        plt.close(fig)


def test_draw_arm_places_the_end_effector():
    fig, ax = plt.subplots()
    line = draw_arm(ax, ARM, [0.0, 0.0, 0.0])
    xs, ys = line.get_data()
    assert len(xs) == 4
    assert (xs[-1], ys[-1]) == (3.0, 0.0)  # fully extended along +x
    plt.close(fig)


def test_runner_writes_figures_and_summary(tmp_path):
    out = tmp_path / "figs"
    subprocess.run(
        [sys.executable, "-m", "armsim.run_heatmap", "--quick", "--out", str(out)],
        check=True,
        capture_output=True,
    )
    for name in (
        "joint_space_manipulability.png",
        "workspace_manipulability.png",
        "radial_validation.png",
    ):
        assert (out / name).stat().st_size > 5000

    summary = json.loads((out / "singularity_summary.json").read_text())
    assert summary["reach"] == 3.0
    assert summary["w_max_global"] == 1.0
    # the phase's headline: the badly-conditioned region is a thin outer shell
    assert summary["fraction_of_reach_well_conditioned"] > 0.85
