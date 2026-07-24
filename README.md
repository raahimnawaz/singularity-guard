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
| 2 | Singularity map (workspace / joint-space heatmaps) | ⬜ next |
| 3 | Naive `qdot = J⁻¹·xdot` controller — let it fail | ⬜ |
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

## What's here (Phase 1)

- `armsim/kinematics.py` — forward kinematics, the hand-derived analytical
  Jacobian, `det(J)`, and the manipulability index. Scalar and closed-form on
  purpose: this is the exact math that ports to C++ in Phase 5.
- `docs/derivation.md` — the full FK/Jacobian/determinant derivation and
  singularity analysis.
- `tests/test_kinematics.py` — the safety net. The key test central-differences
  the forward kinematics and checks it against the analytical Jacobian, so every
  later phase can trust `jacobian()` without re-deriving it.

## Scope

The model is **kinematic**, not dynamic — we integrate `qdot` directly, with no
torques or inertia. That is the honest scope: this project is about *kinematic*
singularities and the velocity-level control laws that stay well-behaved through
them.

## License

MIT — see [`LICENSE`](LICENSE).
