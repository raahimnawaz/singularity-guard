# Plan: singularity-robust-control

Build a manipulator controller that detects an approaching kinematic singularity
(via manipulability `w = sqrt(det(J·Jᵀ))`) and switches to a damped,
singularity-robust control law before velocities blow up — then prove
quantitatively that it outperforms a naive inverse-Jacobian controller.

## Key design decisions

**Task space is the full pose `(x, y, phi)`.** A 3R planar arm tracking only
`(x, y)` gives a `2x3` Jacobian — no determinant, and the arm is redundant, which
muddies the story. Tracking `(x, y, phi)` makes `J` square (`3x3`), so `det(J)`
is well defined and the naive controller is literally `qdot = J⁻¹·xdot`.

**Where the singularities are.** Hand derivation (see `docs/derivation.md`)
gives the closed form:

```
det(J) = l1 * l2 * sin(q2)        w = |det(J)| = l1 * l2 * |sin(q2)|
```

so the arm is singular exactly at **`q2 ∈ {0, π}`** — the inner two links
collinear — independent of `q1`, `q3`, and `l3`. (Note: an earlier sketch of this
project guessed the singularity was at `q3 = 0/π`; the derivation shows that is
wrong — folding link 3 does not create a singularity for this task. Knowing the
locus exactly gives the heatmap a ground truth and lets us design a trajectory
guaranteed to cross it.)

**Simulation is kinematic, not dynamic.** Integrate `qdot` directly (no
torques/inertia) with fixed-step RK4 at `dt = 1 ms`, so the Python and C++ sims
are bit-comparable. Saturation policy: clamp `|qdot|` at a realistic joint-speed
limit **in the plant**, not the controller — that is how the naive controller
"fails" realistically (it saturates and loses tracking) instead of emitting
`1e16 rad/s` nonsense.

**λ scheduling (Phase 4).** Use the Nakamura/Chiaverini scheme:
`λ² = λ_max²·(1 − w/w₀)²` when `w < w₀`, else `λ = 0`. Two tunables: threshold
`w₀` and `λ_max`. Do a small `(w₀, λ_max)` sweep and plot the
tracking-error-vs-peak-velocity trade-off — that trade-off *is* the core insight
(DLS deliberately sacrifices tracking accuracy near the singularity for bounded
velocity). Show the cost, not just the win.

**Metrics (defined once, used everywhere):**
- peak joint speed `max‖qdot‖∞` over the trajectory,
- RMS and max task-space tracking error (position and orientation separately),
- smoothness = RMS joint acceleration (finite-differenced `qdot`),
- % of time saturated,
- condition number of `J` along the path (diagnostic).

## Repo structure

```
singularity-robust-control/
├── README.md
├── LICENSE
├── pyproject.toml            # uv + hatchling + ruff + pytest
├── docs/
│   ├── PLAN.md               # this file
│   └── derivation.md         # hand-derived FK + Jacobian + singularity analysis
├── armsim/                   # flat package (matches synfuel-control convention)
│   ├── kinematics.py         # ✅ FK, analytical J, manipulability
│   ├── controllers.py        # ⬜ naive inverse-J, fixed-λ DLS, adaptive-λ DLS
│   ├── trajectory.py         # ⬜ task-space reference generators
│   ├── sim.py                # ⬜ RK4 loop, plant-side saturation, logging
│   ├── viz.py                # ⬜ heatmaps, arm animation, comparison plots
│   ├── run_heatmap.py        # ⬜ python -m armsim.run_heatmap
│   └── run_experiment.py     # ⬜ python -m armsim.run_experiment
├── cpp/                      # ⬜ Phase 5: dependency-free real-time port
│   ├── CMakeLists.txt
│   ├── include/armctl/       # header-only mat3 / kinematics / dls
│   ├── src/main.cpp          # timed 1 kHz loop, CSV out
│   └── tests/                # parity vs Python golden CSVs
├── tests/                    # ✅ pytest
├── results/                  # committed figures + metrics.json (curated, small)
└── .github/workflows/ci.yml  # ✅ lint + test (grows with C++ build later)
```

## Phase-by-phase

**Phase 1 — Kinematics + Jacobian.** ✅ Hand-derive FK and `J`; implement in
`kinematics.py`; write the finite-difference Jacobian test as the safety net for
everything downstream.

**Phase 2 — Singularity map.** Sweep the workspace and joint space; render `w(q)`
as heatmaps. Deliverables: a workspace `(x, y)` "danger map" colored by
manipulability, and a joint-space `(q2, q3)` map with the analytical singular
locus `q2 ∈ {0, π}` overlaid — computed map matching the analytical curve is the
first validation.

**Phase 3 — Naive controller, let it fail.** `qdot = J⁻¹·xdot` via an explicit
`3x3` inverse (the exact code that ports to C++). Run a trajectory straight
through a singular configuration at nonzero commanded speed; log joint
velocities; show the spike / saturation. Save the run as a golden CSV for the
C++ parity test.

**Phase 4 — DLS controller (the contribution).** `qdot = Jᵀ(JJᵀ + λ²I)⁻¹·xdot`,
fixed-λ first, then adaptive λ. Rerun the identical trajectories. Deliverables:
overlay plots, the metrics table (README centerpiece), the `(w₀, λ_max)`
trade-off plot, side-by-side animations. Also test a "skim near, don't cross"
trajectory — DLS shines there too and it is the more realistic scenario.

**Phase 5 — C++ real-time port.** Header-only fixed-size `Mat3`/`Vec3`, explicit
`3x3` inverse via cofactors (`JJᵀ + λ²I` is symmetric and well-conditioned by
construction — that is the point of DLS). Measure per-iteration latency with
`std::chrono`, report median/p99/max, demonstrate headroom vs a 1 ms budget.
Parity test: C++ vs Python golden CSV within `1e-9`.

**Phase 6 — (stretch) learned manipulability estimator.** Small MLP
`(q2, q3) → w`. Honest framing: for a `3x3` `J`, computing the determinant is
already nanoseconds, so the pitch is a proof-of-concept for high-DOF arms where
the SVD is the bottleneck. Only after 1–5 ship.

## Testing plan

- FK spot checks against hand-computed poses. *(Phase 1 ✅)*
- Analytical `J` vs central-difference `J` at random configs. *(Phase 1 ✅)*
- `|det(J)| == sqrt(det(JJᵀ))`. *(Phase 1 ✅)*
- DLS with `λ = 0` equals the plain inverse away from singularity. *(Phase 4)*
- DLS output norm bounded as `w → 0` — the actual theorem, as a test. *(Phase 4)*
- Trajectory generator continuity and endpoint accuracy. *(Phase 2/3)*
- C++/Python parity on the golden trajectory. *(Phase 5)*

## References

- Yoshikawa 1985 (manipulability); Nakamura & Hanafusa 1986 and Wampler 1986
  (damped least squares); Chiaverini 1997 (singularity-robust redundancy).
