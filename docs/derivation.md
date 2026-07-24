# Derivation: Planar 3R Arm — FK, Jacobian, and Singularities

This is the hand derivation behind `armsim/kinematics.py`. It is done manually,
not by autodiff, because deriving the Jacobian by hand is what makes the
singularity structure fall out in closed form.

## 1. Conventions

A planar arm with three revolute joints in the x–y plane. Link lengths
`l1, l2, l3`. The joint variables `q = (q1, q2, q3)` are **relative** angles —
each measured from the previous link — so the absolute (cumulative) direction of
each link is:

```
theta1   = q1
theta12  = q1 + q2
theta123 = q1 + q2 + q3
```

Shorthand: `s1 = sin(theta1)`, `c1 = cos(theta1)`, `s12 = sin(theta12)`, etc.

The task space is the full planar pose `p = (x, y, phi)`: end-effector position
**and** orientation. Because `dim(task) = dim(joints) = 3`, the Jacobian is
square, `det(J)` exists, and the naive controller is literally `qdot = J^-1 xdot`.
If we tracked position only, `J` would be `2x3`, there would be no determinant,
and the arm would be kinematically redundant — which muddies the singularity
story. Tracking `phi` is a deliberate design choice.

## 2. Forward kinematics

Summing the link vectors down the chain:

```
x   = l1*c1 + l2*c12 + l3*c123
y   = l1*s1 + l2*s12 + l3*s123
phi = q1 + q2 + q3 = theta123
```

## 3. Analytical Jacobian

`J = d p / d q`. Joint `j` rotates every link at or beyond it, so:

```
        d/dq1                         d/dq2                    d/dq3
x   [ -l1*s1 - l2*s12 - l3*s123   -l2*s12 - l3*s123   -l3*s123 ]
y   [  l1*c1 + l2*c12 + l3*c123    l2*c12 + l3*c123    l3*c123 ]
phi [  1                           1                   1       ]
```

Column `j` is joint `j`'s contribution to `[xdot, ydot, phidot]`. The all-ones
bottom row is because `phi = q1 + q2 + q3`, so every joint changes orientation
one-for-one.

## 4. Determinant — the key result

We derive `det(J)` two independent ways and get the same clean answer.

### 4a. Column reduction (fast)

The bottom row is all ones. Subtracting **column 3 from columns 1 and 2** leaves
the determinant unchanged. The `phi`-row entries of the new columns 1 and 2
become `0`, and their `(x, y)` parts become:

```
col1' = (-l1*s1 - l2*s12,  l1*c1 + l2*c12,  0)
col2' = (-l2*s12,          l2*c12,          0)
col3  = (-l3*s123,         l3*c123,         1)
```

Expanding along the bottom row `(0, 0, 1)`, only the `(3,3)` cofactor survives,
so `det(J)` equals the `2x2` determinant of the `(x, y)` parts of `col1'` and
`col2'`. But those two columns are exactly the **position Jacobian of the inner
2R sub-arm** (base → wrist point, links `l1, l2`). Its determinant is the
classic 2R result:

```
det(J) = l1 * l2 * sin(q2)
```

### 4b. Cofactor expansion (full check)

Expanding along the all-ones bottom row directly, the three `2x2` cofactors are:

```
A =  l2*l3*sin(q3)
B =  l3*( l1*sin(q2+q3) + l2*sin(q3) )
C =  l1*l2*sin(q2) + l1*l3*sin(q2+q3)

det(J) = A - B + C = l1*l2*sin(q2)
```

Every `l3` term cancels identically. Confirmed numerically to ~`3e-15` over
200k random configurations.

## 5. Manipulability

The Yoshikawa manipulability index is `w = sqrt(det(J J^T))`. For a **square**
Jacobian, `det(J J^T) = det(J)^2`, so:

```
w = |det(J)| = l1 * l2 * |sin(q2)|
```

`w` is a scalar "distance from singularity" in velocity-transmission units; it is
what the adaptive damping in Phase 4 uses to schedule `lambda`.

## 6. Singularity analysis

```
det(J) = 0   <=>   sin(q2) = 0   <=>   q2 in {0, pi}
```

independent of `q1`, `q3`, and `l3`.

- **`q2 = 0`** — link 2 is collinear with and extended past link 1.
- **`q2 = pi`** — link 2 is folded back onto link 1.

In both cases the inner 2R sub-arm (base → wrist point) collapses: its two
columns become parallel, the wrist point can only move along one line, and the
arm loses a translational DOF.

**Why joint 3 never causes it (for this task):** the orientation row lets joint
3's column act as an independent pivot for `phi`, so rank deficiency can only
come from the inner 2R position sub-Jacobian. Folding link 3 (`q3 = 0` or `pi`)
does *not* create a singularity here — a common misconception worth stating
explicitly, since for other task definitions (e.g. position-only redundant arms)
the story is different.

## 7. Validation (see `tests/test_kinematics.py`)

- Analytical `J` vs central finite differences at 200 random configs (`atol 1e-6`).
- Closed-form `det` vs `numpy.linalg.det` at 500 random configs.
- `w = |det(J)|` identity at 300 random configs.
- `rank(J) < 3` exactly on `q2 in {0, pi}`; full rank away from it.
- Forward-kinematics spot checks against hand-computed poses.

## References

- T. Yoshikawa, "Manipulability of Robotic Mechanisms," *IJRR*, 1985.
- Y. Nakamura & H. Hanafusa, "Inverse Kinematic Solutions with Singularity
  Robustness for Robot Manipulator Control," *ASME J. Dyn. Sys.*, 1986.
- C. W. Wampler, "Manipulator Inverse Kinematic Solutions Based on Vector
  Formulations and Damped Least-Squares Methods," *IEEE SMC*, 1986.
- S. Chiaverini, "Singularity-Robust Task-Priority Redundancy Resolution,"
  *IEEE T-RA*, 1997.
