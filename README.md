# singularity-robust-control

Detect an approaching kinematic singularity in a manipulator and switch to a
damped, singularity-robust velocity control law **before joint velocities blow
up** — then prove quantitatively that it beats a naive inverse-Jacobian
controller.

The testbed is a planar **3R arm** tracking its full end-effector pose
`(x, y, phi)`, so the Jacobian is square and the singularity is exact and
hand-derivable:

> **det(J) = l1 · l2 · sin(q2)**  →  singular exactly when `q2 ∈ {0, π}`
> (the first two links collinear), independent of `q1`, `q3`, and `l3`.

Manipulability is then `w = |det(J)| = l1·l2·|sin(q2)|`, and that scalar is what
the robust controller uses to schedule its damping.

## Status

| Phase | What | State |
|------|------|-------|
| 1 | Kinematics + analytical Jacobian + manipulability | ✅ done |
| 2 | Singularity map (workspace / joint-space heatmaps) | ✅ done |
| 3 | Naive `qdot = J⁻¹·xdot` controller — let it fail | ⬜ next |
| 4 | Adaptive damped-least-squares controller — the contribution | ⬜ |
| 5 | Dependency-free C++ port at a 1 kHz control rate | ⬜ |
| 6 | *(stretch)* learned manipulability estimator | ⬜ |

See [`docs/PLAN.md`](docs/PLAN.md) for the full roadmap and
[`docs/derivation.md`](docs/derivation.md) for the hand derivation.

## Quickstart

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest -q
```

```python
import numpy as np
from armsim import RobotArm, forward_kinematics, jacobian, manipulability

arm = RobotArm(l1=1.0, l2=1.0, l3=1.0)
q = np.array([0.2, np.pi / 2, -0.4])

forward_kinematics(arm, q)   # end-effector pose [x, y, phi]
jacobian(arm, q)             # 3x3 analytical task Jacobian
manipulability(arm, q)       # w -> 0 near a singularity
```

## Where the singularities are (Phase 2)

```bash
uv run python -m armsim.run_heatmap        # writes results/
```

**In joint space** the map is flat in `q3` and zero exactly on `q2 ∈ {0, ±π}`.
That flatness is the visual proof of the derivation — folding link 3 does
*nothing* to conditioning, which is the opposite of an earlier guess about this
arm.

![Manipulability in joint space](results/joint_space_manipulability.png)

**In the workspace** the question isn't "what is `w` here" — a point can be
reached by several configurations with different `w` — but *how well-conditioned
can I be while standing here?* So the map shows the best case,
`w_max(x, y) = max{ w(q) : FK(q) = (x, y) }`.

![Best achievable manipulability across the workspace](results/workspace_manipulability.png)

The result is the headline of this phase, and it sets up Phase 3: **reaching far
costs conditioning, but only right at the end.** The inner **90%** of the reach
radius stays above `0.9·w_max`; the arm only has to straighten — driving
`q2 → 0` — in a thin outer shell. Danger is not spread across the workspace, so
the Phase 3 trajectory has to *aim* at a singularity to hit one.

**And it's checked, not asserted.** Because `q1` only rotates the arm and `w`
doesn't depend on it, `w_max` is a function of radius alone and has a closed
form — twice the area of the base–joint2–wrist triangle, by Heron:

```
w_max(r) = ½·√( ((l1+l2)² − r2²)·(r2² − (l1−l2)²) )   maximised over the
                                                     feasible wrist radius r2
```

A brute-force sweep of a few million configurations is then held to it. It never
wins, and the gap closes as the sweep refines — which is a test, not just a
picture:

![Sweep versus closed form](results/radial_validation.png)

## What's here

- `armsim/kinematics.py` — forward kinematics, the hand-derived analytical
  Jacobian, `det(J)`, and the manipulability index. Scalar and closed-form on
  purpose: this is the exact math that ports to C++ in Phase 5.
- `armsim/singularity.py` — the joint-space and workspace sweeps, plus the
  closed-form `w_max(r)` they are validated against.
- `armsim/viz.py`, `armsim/run_heatmap.py` — the figures above and
  `results/singularity_summary.json`.
- `docs/derivation.md` — the full FK/Jacobian/determinant derivation and
  singularity analysis.
- `tests/` — the safety net. Two loads bear: one central-differences the forward
  kinematics against the analytical Jacobian, so every later phase can trust
  `jacobian()` without re-deriving it; the other pits the brute-force sweep
  against the closed form, so the maps can't quietly drift from the algebra.

## Scope

The model is **kinematic**, not dynamic — we integrate `qdot` directly, with no
torques or inertia. That is the honest scope: this project is about *kinematic*
singularities and the velocity-level control laws that stay well-behaved through
them.

## License

MIT — see [`LICENSE`](LICENSE).
