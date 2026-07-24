"""armsim -- kinematics and singularity-robust control for a planar 3R arm."""

from armsim.kinematics import (
    RobotArm,
    det_jacobian,
    forward_kinematics,
    jacobian,
    joint_positions,
    manipulability,
)
from armsim.singularity import (
    SINGULAR_Q2,
    joint_space_grid,
    manipulability_of_q2,
    max_manipulability_at_radius,
    radial_profile_swept,
    singular_radius,
    workspace_grid,
)

__all__ = [
    # kinematics (Phase 1)
    "RobotArm",
    "det_jacobian",
    "forward_kinematics",
    "jacobian",
    "joint_positions",
    "manipulability",
    # singularity analysis (Phase 2)
    "SINGULAR_Q2",
    "joint_space_grid",
    "manipulability_of_q2",
    "max_manipulability_at_radius",
    "radial_profile_swept",
    "singular_radius",
    "workspace_grid",
]
