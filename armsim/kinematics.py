"""Forward kinematics, analytical Jacobian, and manipulability for a planar 3R arm.

The task space is the full end-effector pose ``p = [x, y, phi]`` (position +
orientation), so the Jacobian ``J = dp/dq`` is square (3x3) and ``det(J)`` is well
defined -- that is the whole reason we track orientation instead of position only.

Every function here is scalar-config (one ``q`` at a time) and closed-form: no
SVD, no linear solves.  This is deliberate -- it is the exact math that gets
ported to dependency-free C++ in Phase 5, so it stays trivially portable.

See ``docs/derivation.md`` for the hand derivation.  Headline result::

    det(J) = l1 * l2 * sin(q2)

so the arm is singular exactly when the first two links are collinear
(``q2 = 0`` or ``q2 = pi``), independent of ``q1``, ``q3``, and ``l3``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class RobotArm:
    """A planar 3R (three-revolute) arm lying in the x-y plane.

    Attributes are the three link lengths.  Joint angles are supplied per-call as
    ``q = [q1, q2, q3]`` and are *relative* angles (each measured from the previous
    link), so the cumulative link directions are ``q1``, ``q1+q2``, ``q1+q2+q3``.
    """

    l1: float = 1.0
    l2: float = 1.0
    l3: float = 1.0

    def __post_init__(self) -> None:
        for name in ("l1", "l2", "l3"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"link length {name} must be positive")

    @property
    def links(self) -> tuple[float, float, float]:
        return (self.l1, self.l2, self.l3)

    @property
    def reach(self) -> float:
        """Farthest the end-effector can get from the base (arm fully extended)."""
        return self.l1 + self.l2 + self.l3


def _cumulative_angles(q: Array) -> tuple[float, float, float]:
    """Absolute link directions (theta1, theta12, theta123) from relative q."""
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
    return q1, q1 + q2, q1 + q2 + q3


def forward_kinematics(arm: RobotArm, q: Array) -> Array:
    """End-effector pose ``[x, y, phi]`` for configuration ``q = [q1, q2, q3]``.

    ``phi`` is returned unwrapped (the raw sum ``q1 + q2 + q3``); wrap it into
    ``(-pi, pi]`` yourself if you need that.
    """
    q = np.asarray(q, dtype=float)
    t1, t12, t123 = _cumulative_angles(q)
    l1, l2, l3 = arm.links
    x = l1 * np.cos(t1) + l2 * np.cos(t12) + l3 * np.cos(t123)
    y = l1 * np.sin(t1) + l2 * np.sin(t12) + l3 * np.sin(t123)
    phi = t123
    return np.array([x, y, phi])


def joint_positions(arm: RobotArm, q: Array) -> Array:
    """``(x, y)`` of ``[base, joint2, joint3, end-effector]`` as a ``(4, 2)`` array.

    Used for drawing the arm; the last row equals ``forward_kinematics(...)[:2]``.
    """
    q = np.asarray(q, dtype=float)
    t1, t12, t123 = _cumulative_angles(q)
    l1, l2, l3 = arm.links
    p0 = np.array([0.0, 0.0])
    p1 = p0 + l1 * np.array([np.cos(t1), np.sin(t1)])
    p2 = p1 + l2 * np.array([np.cos(t12), np.sin(t12)])
    p3 = p2 + l3 * np.array([np.cos(t123), np.sin(t123)])
    return np.vstack([p0, p1, p2, p3])


def jacobian(arm: RobotArm, q: Array) -> Array:
    """Analytical 3x3 task Jacobian ``J = d[x, y, phi] / d[q1, q2, q3]``.

    Derived by hand rather than by autodiff -- doing it manually is what exposes
    the singularity structure.  Column ``j`` is joint ``j``'s contribution to
    ``[xdot, ydot, phidot]``; the all-ones bottom row is because every joint adds
    one-for-one to the end-effector orientation.
    """
    q = np.asarray(q, dtype=float)
    t1, t12, t123 = _cumulative_angles(q)
    l1, l2, l3 = arm.links

    s1, c1 = np.sin(t1), np.cos(t1)
    s12, c12 = np.sin(t12), np.cos(t12)
    s123, c123 = np.sin(t123), np.cos(t123)

    return np.array(
        [
            [-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123],
            [l1 * c1 + l2 * c12 + l3 * c123, l2 * c12 + l3 * c123, l3 * c123],
            [1.0, 1.0, 1.0],
        ]
    )


def det_jacobian(arm: RobotArm, q: Array) -> float:
    """Closed-form Jacobian determinant: ``det(J) = l1 * l2 * sin(q2)``.

    The third joint and the orientation row factor out of the determinant, leaving
    the position-Jacobian determinant of the inner 2R sub-arm (base -> wrist).
    See ``docs/derivation.md``.
    """
    q = np.asarray(q, dtype=float)
    return float(arm.l1 * arm.l2 * np.sin(q[1]))


def manipulability(arm: RobotArm, q: Array) -> float:
    """Yoshikawa manipulability index ``w = sqrt(det(J @ J.T))``.

    For a square Jacobian this equals ``|det(J)| = l1 * l2 * |sin(q2)|``.  ``w -> 0``
    at a singularity; it is the scalar the adaptive damping (Phase 4) keys off of.
    """
    j = jacobian(arm, q)
    gram_det = float(np.linalg.det(j @ j.T))
    return float(np.sqrt(max(gram_det, 0.0)))
