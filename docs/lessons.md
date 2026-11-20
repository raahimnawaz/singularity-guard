# Lessons

Mistakes that changed the design, kept here so they are not repeated.

## One mistake worth recording

The first version of `trajectory.py` built the reference the safe-looking way: pick
a joint path through `q2 = 0`, then hand the controller its forward-kinematic image,
`p_ref = FK(q_ref)` with `ṗ_ref = J(q_ref) q̇_ref`. Reachable by construction, and
provably crossing the singularity.

It also quietly destroyed the experiment. If `ṗ_ref = J q̇_ref` exactly, the
commanded twist is *always* in the range of `J`, so a bounded joint velocity
achieving it exists at every instant — namely `q̇_ref` itself. Measured with that
reference: naive peaked at 1.5 rad/s with zero tracking error, and adaptive DLS
came out **worse** at 7.4 rad/s. Exactly backwards, and for a real reason.

A singularity is not "the arm is in an awkward pose". It is "certain task
velocities are no longer achievable". Handing the controller only achievable
velocities removes the phenomenon while appearing to test it.

The fix is the current design: a straight line in task space, with no feasibility
guarantee — which is also what a real controller gets, since an operator jogging a
Cartesian direction does not first check it against the range of the current
Jacobian. `tests/test_trajectory.py::test_through_commands_an_infeasible_twist_at_the_singularity`
now projects the commanded twist onto `range(J)` and fails if the unachievable
residual is under 5%, so this specific mistake cannot come back.
