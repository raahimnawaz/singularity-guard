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
│   ├── singularity.py        # ✅ joint/workspace sweeps + closed-form w_max(r)
│   ├── controllers.py        # ⬜ naive inverse-J, fixed-λ DLS, adaptive-λ DLS
│   ├── trajectory.py         # ⬜ task-space reference generators
│   ├── sim.py                # ⬜ RK4 loop, plant-side saturation, logging
│   ├── viz.py                # ✅ heatmaps + draw_arm; animation lands in Phase 3
│   ├── run_heatmap.py        # ✅ python -m armsim.run_heatmap
│   └── run_experiment.py     # ⬜ python -m armsim.run_experiment
├── cpp/                      # ✅ Phase 5: dependency-free real-time port
│   ├── CMakeLists.txt
│   ├── include/armctl/       # header-only mat3 / kinematics / controllers / dls
│   ├── src/main.cpp          # timed 1 kHz loop, CSV out
│   └── tests/                # ✅ parity vs Python golden CSV (parity_check.py)
├── tests/                    # ✅ pytest
├── results/                  # committed figures + metrics.json (curated, small)
└── .github/workflows/ci.yml  # ✅ lint + test, plus a cmake + ctest job for cpp/
```

## Phase-by-phase

**Phase 1 — Kinematics + Jacobian.** ✅ Hand-derive FK and `J`; implement in
`kinematics.py`; write the finite-difference Jacobian test as the safety net for
everything downstream.

**Phase 2 — Singularity map.** ✅ Joint-space `(q2, q3)` map with the singular
locus overlaid (flat in `q3`, zero on `q2 ∈ {0, π}` — the derivation, visible),
and a workspace "danger map" of the *best achievable* `w` at each `(x, y)`.

Two things came out of it that the plan hadn't anticipated:

- **The analysis needed its own module.** `singularity.py` holds the sweeps and
  the closed form; `kinematics.py` stays scalar-config and closed-form so it can
  port to C++ untouched, and `viz.py` stays pure plotting.
- **`w_max(x, y)` has a closed form**, which turned the "computed map matches the
  analytical curve" validation from a picture into a test. Since `q1` only
  rotates the arm, `w_max` depends on radius alone, and maximising over the
  feasible wrist radius `r2` gives `w_max(r) = ½·√(((l1+l2)²−r2²)(r2²−(l1−l2)²))`
  — twice the area of the base–joint2–wrist triangle, by Heron. Worth writing it
  this way rather than via `√(1−cos²q2)`: that form cancels catastrophically
  exactly at full reach, reporting `1e-8` where the answer is exactly `0`.

**Result that shapes Phase 3:** the inner **90%** of the reach radius stays above
`0.9·w_max`. Bad conditioning is a thin outer shell, not a broad region, so the
Phase 3 trajectory has to deliberately drive at `q2 → 0` to provoke a failure —
drifting around the workspace will not do it.

**Phase 3 — Naive controller, let it fail.** ✅ `qdot = J⁻¹·xdot` via an explicit
`3x3` inverse (the exact code that ports to C++). Result: **2,434 rad/s** peak
command against a 3 rad/s limit, **52.2%** of the run saturated, 0.740 m position
RMS, and the arm never recovers — it gets stuck at the singular configuration.
Golden CSV written for the Phase 5 parity test.

**One thing the plan got wrong, and it mattered.** The plan says only "run a
trajectory straight through a singular configuration", without saying in which
space to define it. The first implementation designed it in *joint* space and
handed the controller the forward-kinematic image, `p_ref = FK(q_ref)` with
`ṗ_ref = J(q_ref) q̇_ref` — reachable by construction, provably crossing the
singular locus, and completely inert as an experiment. A twist built that way is
always in `range(J)`, so a bounded solution exists at every instant and the naive
controller sails through: it measured 1.5 rad/s peak with zero error, while
adaptive DLS came out *worse* at 7.4 rad/s.

A singularity is not a pose, it is a set of task velocities that stop being
achievable. The reference must therefore be a straight line in **task space**,
with no feasibility guarantee — which is also what a real controller receives.
`tests/test_trajectory.py` now projects the commanded twist onto `range(J)` and
fails if the unachievable residual drops below 5%.

**Phase 4 — DLS controller (the contribution).** ✅ Fixed-λ and adaptive-λ, both
via `qdot = Jᵀ(JJᵀ + λ²I)⁻¹·xdot`. Adaptive defaults `w₀ = 0.30`, `λ_max = 0.20`
come from the sweep. Three scenarios rather than two — `well_conditioned` was
added because without it the table only shows what damping *buys* and never what
it *costs*.

Results: adaptive DLS is bit-identical to naive when well-conditioned (`λ = 0`,
zero cost), 4.6× more accurate than fixed-λ when skimming, and bounded at 2.8
rad/s with 0% saturation through the exact crossing.

**Not a clean sweep, and worth saying so:** fixed-λ actually edges adaptive *at*
the singularity (1.86 vs 2.78 rad/s peak, 0.1025 vs 0.1036 m RMS). Adaptive's
claim is not dominance — it is being the only law that is free when
well-conditioned *and* protected when singular.

**Second finding: the trade-off is weak.** Raising `w₀` from 0.05 to 0.30 cuts
peak velocity ~6× while tracking error slightly *improves*, because the error was
coming from saturation rather than damping. The real cost only shows up at
`λ_max = 0.4`. Damping earlier is close to free in this regime.

Still open from the original plan: side-by-side animations (the sim-based video is
a better use of the effort — see the Webots note below).

**Phase 5 — C++ real-time port.** ✅ Header-only fixed-size `Mat3`/`Vec3`, explicit
`3x3` inverse via cofactors (`JJᵀ + λ²I` is symmetric and well-conditioned by
construction — that is the point of DLS). Measure per-iteration latency with
`std::chrono`, report median/p99/max, demonstrate headroom vs a 1 ms budget.
Parity test: C++ vs Python golden CSV within `1e-9`.

Result: `cpp/tests/parity_check.py` holds every field of the through-singularity/
adaptive-DLS run to `results/golden_through.csv` within `1e-9` — measured max
abs diff `1.2e-10` across all 2001 rows x 12 columns, an order of magnitude of
margin from roundoff, not algorithmic drift (both implementations do the same
cofactor arithmetic in the same order). `step_control()` — the one control-law
evaluation a real 1 kHz loop repeats per tick, timed in isolation over a
representative run — measured **~0.1 µs median / ~0.2 µs p99 / low-single-digit
µs max** against the 1,000 µs budget: roughly **5,000x** headroom at p99. (The
max is noisy run-to-run, as any wall-clock measurement of a sub-microsecond
call is; median and p99 are the numbers that matter here.)

**Worth being honest about what that headroom is and isn't.** It is not
evidence this project stress-tested a 1 ms budget; a `3x3` cofactor solve was
never going to threaten it on any modern core. It *is* the number Phase 6 (if
attempted) has to justify beating: the pitch for a learned manipulability
estimator only makes sense once `det(J)` stops being nanoseconds, i.e. on a
high-DOF arm where the exact Jacobian solve is no longer free. Measuring it
here, on the arm where it's expected to be trivial, is what makes that later
claim checkable instead of asserted.

**One deliberate scope cut.** `simulate()`'s 4-stage RK4 evaluates the control
law four times per integration step — a Python/C++ parity choice (see
`armsim/sim.py`'s docstring), not something a real digital controller does. A
real loop calls the control law once per tick under zero-order hold, so
`step_control()` alone is what gets timed, not the RK4 wrapper around it. Timing
the wrapper would have reported a number roughly 4x too pessimistic for what
Phase 5 is actually trying to demonstrate.

**Phase 6 — (stretch) learned manipulability estimator.** Small MLP
`(q2, q3) → w`. Honest framing: for a `3x3` `J`, computing the determinant is
already nanoseconds, so the pitch is a proof-of-concept for high-DOF arms where
the SVD is the bottleneck. Only after 1–5 ship.

## Testing plan

- FK spot checks against hand-computed poses. *(Phase 1 ✅)*
- Analytical `J` vs central-difference `J` at random configs. *(Phase 1 ✅)*
- `|det(J)| == sqrt(det(JJᵀ))`. *(Phase 1 ✅)*
- Brute-force sweep never beats the closed-form `w_max(r)`, and converges to it
  as the sweep refines. *(Phase 2 ✅)*
- Workspace map is radially symmetric with `q1` swept explicitly — an independent
  check that nothing but `q2` drives `w`. *(Phase 2 ✅)*
- Figure pipeline runs headless at coarse resolution. *(Phase 2 ✅)*
- DLS with `λ = 0` equals the plain inverse away from singularity. *(Phase 4)*
- DLS output norm bounded as `w → 0` — the actual theorem, as a test. *(Phase 4)*
- Trajectory generator continuity and endpoint accuracy. *(Phase 3)*
- C++/Python parity on the golden trajectory. *(Phase 5)*

## References

- Yoshikawa 1985 (manipulability); Nakamura & Hanafusa 1986 and Wampler 1986
  (damped least squares); Chiaverini 1997 (singularity-robust redundancy).
