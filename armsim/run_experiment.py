"""Phase 3 + 4 experiment: run every controller on every scenario and write the results.

    python -m armsim.run_experiment              # figures + metrics.json
    python -m armsim.run_experiment --sweep      # also the (w0, lam_max) trade-off
    python -m armsim.run_experiment --coarse     # fast, for CI

Writes into ``results/``:

* ``phase3_naive_failure.png``  -- the baseline blowing up
* ``phase4_comparison.png``     -- all three laws, all three scenarios
* ``phase4_lambda_schedule.png``-- what the scheduler actually did
* ``dls_tradeoff.png``          -- (w0, lam_max) sweep, ``--sweep`` only
* ``metrics.json``              -- every number the README quotes
* ``metrics.md``                -- the same table, ready to paste
* ``golden_through.csv``        -- Phase 5 C++ parity reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from armsim.controllers import AdaptiveDLS, DampedLeastSquares, NaiveInverse  # noqa: E402
from armsim.kinematics import RobotArm  # noqa: E402
from armsim.sim import Limits, metrics, simulate  # noqa: E402
from armsim.trajectory import (  # noqa: E402
    skim_singularity,
    through_singularity,
    well_conditioned,
)
from armsim.viz import INK, apply_style  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

# One colour per control law, used consistently in every figure.
COLORS = {"naive": "#c1121f", "dls_fixed": "#8d99ae", "dls_adaptive": "#0353a4"}
SCENARIO_TITLES = {
    "well_cond": "well-conditioned (no singularity)",
    "skim": "skim: approach and retreat",
    "through": "through: exact crossing",
}


def _controllers() -> list:
    return [NaiveInverse(), DampedLeastSquares(lam=0.1), AdaptiveDLS()]


def _scenarios(arm: RobotArm) -> list:
    return [well_conditioned(arm), skim_singularity(arm), through_singularity(arm)]


def run_all(arm: RobotArm, dt: float, limits: Limits) -> dict:
    """Every (scenario, controller) pair. Returns ``{scenario: {controller: SimResult}}``."""
    out: dict = {}
    for traj in _scenarios(arm):
        out[traj.name] = {
            c.name: simulate(arm, traj, c, dt=dt, limits=limits) for c in _controllers()
        }
    return out


# --------------------------------------------------------------------- figures
def fig_naive_failure(runs: dict, limits: Limits, path: Path) -> None:
    """Phase 3: the baseline, on its own, failing."""
    res = runs["through"]["naive"]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True)

    ax = axes[0]
    for k in range(3):
        ax.plot(res.t, res.qdot_cmd[:, k], lw=1.4, label=rf"$\dot q_{k+1}$ commanded")
    ax.axhline(limits.qdot_max, color=INK, ls=":", lw=1.1, alpha=0.7)
    ax.axhline(-limits.qdot_max, color=INK, ls=":", lw=1.1, alpha=0.7)
    # Signed here (individual joints, not a norm), so the symmetric symlog is
    # correct -- unlike the comparison figure, these values really do go negative.
    ax.set_yscale("symlog", linthresh=limits.qdot_max, linscale=0.6)
    ax.set_ylabel("joint velocity [rad/s]\n(symlog)")
    ax.set_title(
        r"Naive $\dot q = J^{-1}\dot x$ through an exact singularity"
        f"\npeak command {np.max(np.abs(res.qdot_cmd)):,.0f} rad/s against a "
        f"{limits.qdot_max:g} rad/s limit",
        loc="left",
    )
    ax.legend(fontsize=8, ncol=3, loc="upper left")

    ax = axes[1]
    ax.plot(res.t, res.w, color="#0353a4", lw=1.6)
    ax.set_ylabel(r"manipulability $w$")
    ax.set_yscale("log")
    ax.axvline(res.t[int(np.argmin(res.w))], color=INK, ls=":", lw=1.0)
    ax.annotate(
        f"$w_{{min}}$ = {np.min(res.w):.1e}",
        xy=(res.t[int(np.argmin(res.w))], np.min(res.w)),
        xytext=(6, 14), textcoords="offset points", fontsize=8,
    )

    ax = axes[2]
    err = np.linalg.norm(res.err[:, :2], axis=1)
    ax.plot(res.t, err, color="#c1121f", lw=1.6, label="position error")
    ax.fill_between(res.t, 0, err, where=res.saturated, color="#c1121f", alpha=0.15,
                    label=f"saturated ({100 * np.mean(res.saturated):.0f}% of run)")
    ax.set_ylabel("position error [m]")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Naive drawn first and thick, adaptive last and dashed, so identical curves stay visible.
_STYLE = {
    "naive": {"lw": 2.8, "alpha": 0.85, "zorder": 2},
    "dls_fixed": {"lw": 1.7, "alpha": 1.0, "zorder": 3},
    "dls_adaptive": {"lw": 1.5, "ls": (0, (5, 2.5)), "zorder": 4},
}
_ERR_FLOOR = 1e-7      # below this is integrator roundoff, not tracking behaviour


def fig_comparison(runs: dict, limits: Limits, path: Path) -> None:
    """Phase 4: three laws x three scenarios, velocity on top, error below."""
    names = ["well_cond", "skim", "through"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6))

    for col, scen in enumerate(names):
        ax = axes[0, col]
        for cname in ("naive", "dls_fixed", "dls_adaptive"):
            res = runs[scen][cname]
            ax.plot(res.t, np.max(np.abs(res.qdot_cmd), axis=1),
                    color=COLORS[cname], label=res.meta["label"], **_STYLE[cname])
        ax.axhline(limits.qdot_max, color=INK, ls=":", lw=1.1, alpha=0.7,
                   label=f"actuator limit {limits.qdot_max:g} rad/s" if col == 0 else None)
        # symlog: the norm touches zero at the skim turnaround, so a log axis cannot show it.
        ax.set_yscale("symlog", linthresh=0.1, linscale=0.4)
        ax.set_ylim(bottom=0.0)
        ax.set_title(SCENARIO_TITLES[scen], loc="left", fontsize=10)
        if col == 0:
            ax.set_ylabel(r"$\|\dot{q}\|_\infty$ commanded" "\n[rad/s, symlog]")

        ax = axes[1, col]
        for cname in ("naive", "dls_fixed", "dls_adaptive"):
            res = runs[scen][cname]
            err = np.maximum(np.linalg.norm(res.err[:, :2], axis=1), _ERR_FLOOR)
            ax.plot(res.t, err, color=COLORS[cname], **_STYLE[cname])
        ax.set_yscale("log")
        # Floor the axis so 1e-14 roundoff does not squash the differences that matter.
        ax.set_ylim(bottom=_ERR_FLOOR / 2.0)
        ax.set_xlabel("time [s]")
        if col == 0:
            ax.set_ylabel("position error [m]\n(log)")
            ax.annotate(
                f"axis floored at {_ERR_FLOOR:.0e} m\n(below is integrator roundoff)",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top",
                fontsize=7, color=INK, alpha=0.65,
            )

    fig.suptitle(
        "Damping costs accuracy where it is not needed, and saves the arm where it is",
        x=0.01, ha="left", fontsize=12,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.045, 1, 0.955))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_lambda_schedule(runs: dict, path: Path) -> None:
    """What the scheduler did: lambda tracking 1/w, and the resulting velocity."""
    res_a = runs["through"]["dls_adaptive"]
    res_f = runs["through"]["dls_fixed"]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True)

    ax = axes[0]
    ax.plot(res_a.t, res_a.w, color="#0353a4", lw=1.6, label=r"manipulability $w$")
    ax.axhline(AdaptiveDLS().w0, color=INK, ls="--", lw=1.0, alpha=0.7,
               label=rf"threshold $w_0$ = {AdaptiveDLS().w0:g}")
    ax.set_yscale("log")
    ax.set_ylabel(r"$w$")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title(r"Adaptive $\lambda$ spends damping only where conditioning is poor",
                 loc="left")

    ax = axes[1]
    ax.plot(res_a.t, res_a.lam, color="#0353a4", lw=1.8, label=r"adaptive $\lambda$")
    ax.plot(res_f.t, res_f.lam, color="#8d99ae", lw=1.4, ls="--",
            label=rf"fixed $\lambda$ = {res_f.lam[0]:g}")
    ax.fill_between(res_a.t, 0, res_a.lam, color="#0353a4", alpha=0.12)
    ax.set_ylabel(r"damping $\lambda$")
    ax.set_xlabel("time [s]")
    ax.legend(fontsize=8, loc="upper left")
    frac = 100.0 * np.mean(res_a.lam > 0)
    ax.annotate(
        f"adaptive damping active {frac:.0f}% of the run;\n"
        f"fixed damping active 100%",
        xy=(0.99, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(arm: RobotArm, dt: float, limits: Limits, path: Path) -> list[dict]:
    """The (w0, lam_max) sweep. The trade-off curve *is* the design argument."""
    traj = through_singularity(arm)
    w0s = [0.05, 0.10, 0.20, 0.30, 0.50]
    lams = [0.05, 0.10, 0.20, 0.40]

    rows = []
    for w0 in w0s:
        for lm in lams:
            m = metrics(simulate(arm, traj, AdaptiveDLS(w0=w0, lam_max=lm),
                                 dt=dt, limits=limits))
            m["w0"], m["lam_max"] = w0, lm
            rows.append(m)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    cmap = plt.get_cmap("viridis")

    ax = axes[0]
    for i, lm in enumerate(lams):
        sel = [r for r in rows if r["lam_max"] == lm]
        ax.plot([r["peak_qdot_cmd"] for r in sel], [r["pos_err_rms"] for r in sel],
                "o-", ms=4, lw=1.3, color=cmap(i / max(len(lams) - 1, 1)),
                label=rf"$\lambda_{{max}}$={lm:g}")
        for r in sel:
            ax.annotate(f"{r['w0']:g}", (r["peak_qdot_cmd"], r["pos_err_rms"]),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.axvline(limits.qdot_max, color="#c1121f", ls="--", lw=1.0,
               label=f"actuator limit {limits.qdot_max:g}")
    ax.set_xscale("log")
    ax.set_xlabel(r"peak commanded $\|\dot q\|_\infty$ [rad/s]")
    ax.set_ylabel("position error RMS [m]")
    ax.set_title(r"Trade-off (labels are $w_0$)", loc="left", fontsize=10)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    grid = np.array([[next(r["pct_saturated"] for r in rows
                           if r["w0"] == w0 and r["lam_max"] == lm)
                      for lm in lams] for w0 in w0s])
    im = ax.imshow(grid, cmap="rocket_r" if "rocket_r" in plt.colormaps() else "magma_r",
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(lams)), [f"{v:g}" for v in lams])
    ax.set_yticks(range(len(w0s)), [f"{v:g}" for v in w0s])
    ax.set_xlabel(r"$\lambda_{max}$")
    ax.set_ylabel(r"$w_0$")
    ax.set_title("% of run saturated (0 is the goal)", loc="left", fontsize=10)
    for i in range(len(w0s)):
        for j in range(len(lams)):
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if grid[i, j] > grid.max() / 2 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rows


# --------------------------------------------------------------------- outputs
def write_golden(res, path: Path) -> None:
    """Golden CSV for the Phase 5 C++ parity test."""
    cols = ["t", "q1", "q2", "q3", "qdot1", "qdot2", "qdot3", "x", "y", "phi", "w", "lam"]
    data = np.column_stack([res.t, res.q, res.qdot_applied, res.p, res.w, res.lam])
    np.savetxt(path, data, delimiter=",", header=",".join(cols), comments="",
               fmt="%.12e")


_TABLE_COLS = [
    ("label", "controller", "{}"),
    ("peak_qdot_cmd", "peak &#124;q̇&#124;∞ [rad/s]", "{:.2f}"),
    ("pct_saturated", "saturated [%]", "{:.1f}"),
    ("pos_err_rms", "pos RMS [m]", "{:.5f}"),
    ("pos_err_max", "pos max [m]", "{:.4f}"),
    ("ori_err_rms", "ori RMS [rad]", "{:.5f}"),
    ("accel_rms", "accel RMS [rad/s²]", "{:.1f}"),
    ("w_min", "w min", "{:.2e}"),
]


def write_markdown(all_metrics: dict, path: Path) -> None:
    lines = []
    for scen in ("well_cond", "skim", "through"):
        lines.append(f"### {SCENARIO_TITLES[scen]}\n")
        lines.append("| " + " | ".join(h for _, h, _ in _TABLE_COLS) + " |")
        lines.append("|" + "|".join(["---"] * len(_TABLE_COLS)) + "|")
        for m in all_metrics[scen].values():
            cells = [fmt.format(m[key]) for key, _, fmt in _TABLE_COLS]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true", help="also run the (w0, lam_max) sweep")
    ap.add_argument("--coarse", action="store_true", help="dt=4ms, coarse sweep (for CI)")
    ap.add_argument("--dt", type=float, default=None, help="integrator step [s]")
    ap.add_argument("--qdot-max", type=float, default=3.0, help="joint speed limit [rad/s]")
    args = ap.parse_args(argv)

    dt = args.dt if args.dt is not None else (4e-3 if args.coarse else 1e-3)
    limits = Limits(qdot_max=args.qdot_max)
    arm = RobotArm()
    apply_style()
    RESULTS.mkdir(exist_ok=True)

    runs = run_all(arm, dt, limits)
    all_metrics = {s: {c: metrics(r) for c, r in d.items()} for s, d in runs.items()}

    fig_naive_failure(runs, limits, RESULTS / "phase3_naive_failure.png")
    fig_comparison(runs, limits, RESULTS / "phase4_comparison.png")
    fig_lambda_schedule(runs, RESULTS / "phase4_lambda_schedule.png")
    write_golden(runs["through"]["dls_adaptive"], RESULTS / "golden_through.csv")

    payload = {
        "config": {"dt": dt, "qdot_max": limits.qdot_max, "links": list(arm.links)},
        "scenarios": all_metrics,
    }
    if args.sweep:
        payload["sweep"] = fig_tradeoff(arm, dt, limits, RESULTS / "dls_tradeoff.png")

    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(all_metrics, RESULTS / "metrics.md")

    # Console summary: the two numbers the README leads with.
    naive = all_metrics["through"]["naive"]
    adap = all_metrics["through"]["dls_adaptive"]
    print(f"wrote {RESULTS}")
    print(f"  through/naive        peak {naive['peak_qdot_cmd']:>12,.1f} rad/s  "
          f"saturated {naive['pct_saturated']:.1f}%  pos RMS {naive['pos_err_rms']:.4f} m")
    print(f"  through/adaptive     peak {adap['peak_qdot_cmd']:>12,.1f} rad/s  "
          f"saturated {adap['pct_saturated']:.1f}%  pos RMS {adap['pos_err_rms']:.4f} m")
    print(f"  peak-velocity ratio  {naive['peak_qdot_cmd'] / adap['peak_qdot_cmd']:,.0f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
