"""Score the recorded Webots runs with the same metrics as the kinematic study.

    uv run python -m armsim.webots_metrics

Reads ``results/webots/{naive,fixed,adaptive}.csv`` from ``ros2/scripts/record_run.py``.
Velocity columns: ``qdc`` is the law's raw output (comparable to the kinematic
``peak_qdot_cmd``); ``qd`` is what the simulated joint actually reached.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from armsim.kinematics import RobotArm, forward_kinematics, manipulability
from armsim.trajectory import through_singularity, wrap_angle

REPO = Path(__file__).resolve().parent.parent
WEBOTS_DIR = REPO / "results" / "webots"
QDOT_MAX = 3.0
LAWS = ("naive", "fixed", "adaptive")
PRETTY = {
    "naive": "naive $J^{-1}$",
    "fixed": r"fixed-$\lambda$ DLS ($\lambda$=0.1)",
    "adaptive": r"adaptive-$\lambda$ DLS ($w_0$=0.3, $\lambda_{max}$=0.2)",
}


def load(path: Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return cols


def compute(name: str, c: dict[str, np.ndarray], arm: RobotArm, traj) -> dict:
    t = c["t"]
    q = np.stack([c["q1"], c["q2"], c["q3"]], axis=1)
    qd = np.stack([c["qd1"], c["qd2"], c["qd3"]], axis=1)
    qdc = np.stack([c["qdc1"], c["qdc2"], c["qdc3"]], axis=1)

    # Tracking error against the same reference the offline runs used. Sampled
    # at the recorded sim time, so a slow or fast simulator cannot shift it.
    p_ref = np.stack([traj.pose(min(float(ti), traj.duration)) for ti in t])
    p_act = np.stack([forward_kinematics(arm, qi) for qi in q])
    err = p_ref - p_act
    err[:, 2] = wrap_angle(err[:, 2])
    pos_err = np.linalg.norm(err[:, :2], axis=1)
    ori_err = np.abs(err[:, 2])

    # inf/nan are expected from naive J^-1 at the singularity; keep them out of
    # the aggregates but count them, exactly as armsim.sim does.
    finite = np.isfinite(qdc).all(axis=1)
    n_nonfinite = int((~finite).sum())
    qdc_f = qdc[finite] if finite.any() else np.zeros((1, 3))

    applied = np.clip(qdc_f, -QDOT_MAX, QDOT_MAX)
    saturated = (np.abs(qdc_f) >= QDOT_MAX - 1e-9).any(axis=1)

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    if len(t) > 1 and dt > 0:
        accel_rms = float(np.sqrt(np.mean((np.diff(qd, axis=0) / dt) ** 2)))
    else:
        accel_rms = 0.0

    w = c["w"][np.isfinite(c["w"])]
    if w.size == 0:
        w = np.array([manipulability(arm, qi) for qi in q])

    return {
        "controller": name,
        "n": int(len(t)),
        "dt": dt,
        "peak_qdot_cmd": float(np.max(np.abs(qdc_f))),
        "peak_qdot_applied": float(np.max(np.abs(applied))),
        "peak_qdot_achieved": float(np.max(np.abs(qd))),
        "pct_saturated": float(100.0 * np.mean(saturated)),
        "pos_err_rms": float(np.sqrt(np.mean(pos_err**2))),
        "pos_err_max": float(np.max(pos_err)),
        "ori_err_rms": float(np.sqrt(np.mean(ori_err**2))),
        "accel_rms": accel_rms,
        "w_min": float(np.min(w)),
        "lam_max_used": float(np.nanmax(c["lam"])) if np.isfinite(c["lam"]).any() else 0.0,
        "n_nonfinite": n_nonfinite,
    }


def main() -> int:
    arm = RobotArm(l1=1.0, l2=1.0, l3=1.0)
    traj = through_singularity(arm)

    missing = [law for law in LAWS if not (WEBOTS_DIR / f"{law}.csv").exists()]
    if missing:
        print(f"missing CSVs for {missing} in {WEBOTS_DIR}", file=sys.stderr)
        print("run ros2/scripts/compare_laws.sh first", file=sys.stderr)
        return 1

    out = [compute(law, load(WEBOTS_DIR / f"{law}.csv"), arm, traj) for law in LAWS]

    hdr = ("| controller | peak cmd &#124;q̇&#124;∞ | peak achieved | saturated [%] "
           "| pos RMS [m] | pos max [m] | accel RMS | w min | non-finite |")
    print("\n### Webots plant — through-singularity crossing\n")
    print(hdr)
    print("|" + "---|" * 9)
    for m in out:
        print(
            f"| {PRETTY[m['controller']]} | {m['peak_qdot_cmd']:.2f} | "
            f"{m['peak_qdot_achieved']:.2f} | {m['pct_saturated']:.1f} | "
            f"{m['pos_err_rms']:.5f} | {m['pos_err_max']:.4f} | "
            f"{m['accel_rms']:.1f} | {m['w_min']:.2e} | {m['n_nonfinite']} |"
        )

    dest = WEBOTS_DIR / "webots_metrics.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
