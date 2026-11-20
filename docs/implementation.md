# Implementation

How the control law is built and run outside the Python simulation: the header-only C++ port, the `ros2_control` plugin on a Webots arm, and a map of the code.

## The C++ port (Phase 5)

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
cpp/build/armctl_main out.csv     # regenerate + check by eye
ctest --test-dir cpp/build --output-on-failure   # parity vs the Python golden CSV
```

`cpp/include/armctl/` is a header-only, dependency-free port of the exact math
above — `mat3.hpp` (fixed 3×3 arithmetic, the same cofactor `det3`/`inv3`),
`kinematics.hpp`, `controllers.hpp`, `trajectory.hpp`, and `sim.hpp` (the RK4
closed loop). `cpp/src/main.cpp` runs the through-singularity / adaptive-DLS
scenario — the run that produced `results/golden_through.csv` — and
`cpp/tests/parity_check.py` (stdlib-only, no numpy) holds every field of its
output to that CSV within `1e-9`.

**Why time `step_control()` alone, not the full RK4 step.** The simulation
evaluates the control law four times per step because RK4 is a four-stage
integrator; that is a Python/C++ parity choice (see `armsim/sim.py`), not
something a real digital controller does. A real 1 kHz loop calls the control
law exactly once per tick, so that is the one call timed:

| | median | p99 | max | budget | headroom at p99 |
|---|---:|---:|---:|---:|---:|
| `step_control()` | ~0.1 µs | ~0.2 µs | low single-digit µs | 1,000 µs (1 kHz) | ~5,000× |

(Numbers from a representative run — `max` is noisy at this timescale, as any
wall-clock measurement of a sub-microsecond call is; median and p99 are the
numbers that matter. Run `cpp/build/armctl_main` yourself for a fresh sample.)

That headroom is expected, not a finding — for a `3×3` `J`, a cofactor
determinant and adjugate inverse are a fixed handful of FLOPs, nanoseconds on
any modern core. The honest framing (also in `docs/PLAN.md`, Phase 6) is that
this result is the *baseline* a learned or SVD-based estimator would have to
beat on a high-DOF arm, not evidence that this particular 3×3 solve was ever
at risk of missing a 1 ms budget.

## On a real dynamic plant (Phase 7)

Everything above runs against a **kinematic** plant: `armsim/sim.py` integrates
`q̇` straight from the control law, with saturation applied plant-side. No mass,
no inertia, no torque limit, no actuator bandwidth. Those numbers describe a
*control law*, not a robot.

`ros2/` closes that gap. The same `armctl` headers — unmodified, included from
`../cpp/include` rather than copied — are wrapped in a `ros2_control` controller
plugin and pointed at a Webots model of the same 3R arm:

```bash
docker build -f ros2/Dockerfile -t singularity-arm-webots .
docker run --rm --shm-size=1g singularity-arm-webots            # headless, Xvfb
docker run --rm --shm-size=1g -e HEADLESS=0 -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix singularity-arm-webots       # with the GUI
```

**`--shm-size=1g` is required, not optional.** Docker's default 64 MB `/dev/shm`
is too small for FastDDS's shared-memory transport, which fails with
`RTPS_TRANSPORT_SHM ... open_and_lock_file failed`. The symptom is misleading:
topics keep working while *services* time out, so the controllers appear to hang
during spawning rather than reporting a transport error.

The image bakes ROS 2 Jazzy, Webots R2025a, and the built workspace, so the demo
needs no network access and no local ROS install.

The controller is a `controller_interface::ControllerInterface` claiming
`<joint>/velocity` command interfaces and `<joint>/position` state interfaces,
with the naive / fixed-λ / adaptive-λ choice exposed as a `controller_law`
parameter — so the three-way comparison is a config change, not a rebuild. It
publishes `~/manipulability` and `~/damping` through `RealtimePublisher`, which
is what makes `w` and `λ` observable live rather than only in post-processing.

**It runs, and the published values check out exactly.** Both controllers reach
`active`, and `~/manipulability` and `~/damping` can be verified by hand against
the joint state they were computed from. From one live sample:

```
joint2                  -0.045884852089259416
~/manipulability         0.045868752639277255   = |l1·l2·sin(q2)|
~/damping                0.16942083157381516    = λ_max(1 − w/w₀), λ_max=0.20, w₀=0.30
```

Both match to every published digit, so the closed-form manipulability and the
adaptive damping schedule are demonstrably the ones running on the plant — not
merely something the node emits.

**What is still not claimed: anything about dynamics.** The three-law comparison
has not been re-run against Webots, so `results/metrics.md` remains a purely
kinematic result. The prediction being tested — and it is written down
in [`PLAN.md`](PLAN.md) so it can be wrong — is that dynamics will
*shrink* the naive controller's 2,434 rad/s headline (real motors cannot slew
that fast; divergence turns into tracking failure instead) while *strengthening*
the smoothness case for adaptive-λ (torque limits punish acceleration spikes the
kinematic model treats as free).

## What's here

- `armsim/kinematics.py` — forward kinematics, the hand-derived analytical
  Jacobian, `det(J)`, and the manipulability index. Scalar and closed-form on
  purpose: this is the exact math that ports to C++ in Phase 5.
- `armsim/singularity.py` — the joint-space and workspace sweeps, plus the
  closed-form `w_max(r)` they are validated against.
- `armsim/trajectory.py` — task-space references: `through_singularity`,
  `skim_singularity`, `well_conditioned`.
- `armsim/controllers.py` — the three laws above, plus `det3`/`inv3`.
- `armsim/sim.py` — RK4 closed loop, plant-side saturation, and the metric set.
- `armsim/viz.py`, `armsim/run_heatmap.py` — the Phase 2 figures.
- `armsim/run_experiment.py` — regenerates every figure and number above.
- `cpp/include/armctl/` — the header-only C++ port (Phase 5): `mat3.hpp`,
  `kinematics.hpp`, `controllers.hpp`, `trajectory.hpp`, `sim.hpp`.
- `cpp/src/main.cpp` — runs the through-singularity/adaptive-DLS scenario,
  writes the golden-format CSV, and reports control-law latency.
- `cpp/tests/parity_check.py` — holds the C++ output to
  `results/golden_through.csv` within `1e-9`, wired into `ctest`.
- `ros2/` — the `ros2_control` plugin (Phase 7). `src/` + `include/` are the
  `ControllerInterface` wrapper and the runtime law switch; `urdf/` and
  `worlds/` are the same 3R arm as `RobotArm{1,1,1}`; `config/controllers.yaml`
  carries the gains at 125 Hz; `launch/webots_sim.launch.py` brings up Webots,
  the driver, and the spawners; `Dockerfile` makes the whole thing one command.
- `docs/derivation.md` — the full FK/Jacobian/determinant derivation and
  singularity analysis.
- `tests/` — 74 tests. The ones that carry weight: central-differencing the
  analytical Jacobian, holding the brute-force sweep to the closed-form `w_max`,
  the DLS boundedness theorem `‖q̇‖ ≤ ‖ẋ‖/(2λ)` asserted at an exactly singular
  configuration, adaptive-equals-naive when well-conditioned, and the headline
  875× gap pinned as a regression at three different step sizes.

## Reproduce

```bash
uv run python -m armsim.run_experiment --sweep
```

Writes every figure, `results/metrics.json`, `results/metrics.md`, and
`results/golden_through.csv` — the reference trajectory the Phase 5 C++ port will
be held to.
