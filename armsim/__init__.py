"""armsim -- kinematics and singularity-robust control for a planar 3R arm."""

from armsim.kinematics import (
    RobotArm,
    det_jacobian,
    forward_kinematics,
    jacobian,
    joint_positions,
    manipulability,
)

__all__ = [
    "RobotArm",
    "det_jacobian",
    "forward_kinematics",
    "jacobian",
    "joint_positions",
    "manipulability",
]
