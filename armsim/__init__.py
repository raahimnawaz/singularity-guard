"""armsim -- kinematics and singularity-robust control for a planar 3R arm."""

from armsim.controllers import (
    AdaptiveDLS,
    DampedLeastSquares,
    NaiveInverse,
    det3,
    inv3,
)
from armsim.kinematics import (
    RobotArm,
    det_jacobian,
    forward_kinematics,
    jacobian,
    joint_positions,
    manipulability,
)
from armsim.sim import Limits, SimResult, metrics, pose_error, simulate
from armsim.singularity import (
    SINGULAR_Q2,
    joint_space_grid,
    manipulability_of_q2,
    max_manipulability_at_radius,
    radial_profile_swept,
    singular_radius,
    workspace_grid,
)
from armsim.trajectory import (
    Trajectory,
    quintic,
    skim_singularity,
    through_singularity,
    well_conditioned,
    wrap_angle,
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
    # task-space references (Phase 3)
    "Trajectory",
    "quintic",
    "skim_singularity",
    "through_singularity",
    "well_conditioned",
    "wrap_angle",
    # control laws (Phases 3-4)
    "AdaptiveDLS",
    "DampedLeastSquares",
    "NaiveInverse",
    "det3",
    "inv3",
    # closed loop + metrics
    "Limits",
    "SimResult",
    "metrics",
    "pose_error",
    "simulate",
]
