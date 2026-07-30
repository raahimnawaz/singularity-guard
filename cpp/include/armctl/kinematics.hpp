// Forward kinematics, analytical Jacobian, and manipulability for the planar
// 3R arm -- the direct C++ port of armsim/kinematics.py. Scalar-config and
// closed-form, same as the Python: no autodiff, no numerical Jacobian.
//
// det(J) = l1 * l2 * sin(q2), so the arm is singular exactly at
// q2 in {0, pi}, independent of q1, q3, and l3. See docs/derivation.md.
#pragma once

#include <cmath>

#include "mat3.hpp"

namespace armctl {

struct RobotArm {
    double l1 = 1.0;
    double l2 = 1.0;
    double l3 = 1.0;
};

// Absolute link directions (theta1, theta12, theta123) from relative q.
inline void cumulative_angles(const Vec3& q, double& t1, double& t12, double& t123) {
    t1 = q[0];
    t12 = q[0] + q[1];
    t123 = q[0] + q[1] + q[2];
}

// End-effector pose [x, y, phi]. phi is unwrapped (the raw sum q1+q2+q3);
// wrap it into (-pi, pi] at the call site if needed.
inline Vec3 forward_kinematics(const RobotArm& arm, const Vec3& q) {
    double t1, t12, t123;
    cumulative_angles(q, t1, t12, t123);
    const double x = arm.l1 * std::cos(t1) + arm.l2 * std::cos(t12) + arm.l3 * std::cos(t123);
    const double y = arm.l1 * std::sin(t1) + arm.l2 * std::sin(t12) + arm.l3 * std::sin(t123);
    return {x, y, t123};
}

// Analytical 3x3 task Jacobian d[x, y, phi] / d[q1, q2, q3].
inline Mat3 jacobian(const RobotArm& arm, const Vec3& q) {
    double t1, t12, t123;
    cumulative_angles(q, t1, t12, t123);
    const double s1 = std::sin(t1), c1 = std::cos(t1);
    const double s12 = std::sin(t12), c12 = std::cos(t12);
    const double s123 = std::sin(t123), c123 = std::cos(t123);
    const double l1 = arm.l1, l2 = arm.l2, l3 = arm.l3;

    return {{
        {-l1 * s1 - l2 * s12 - l3 * s123, -l2 * s12 - l3 * s123, -l3 * s123},
        {l1 * c1 + l2 * c12 + l3 * c123, l2 * c12 + l3 * c123, l3 * c123},
        {1.0, 1.0, 1.0},
    }};
}

// Closed-form det(J) = l1 * l2 * sin(q2). Not used by the control laws (they
// use det3(jacobian(...)) directly, same as the Python), but kept for parity
// with kinematics.py and for anything that wants the ground truth.
inline double det_jacobian(const RobotArm& arm, const Vec3& q) {
    return arm.l1 * arm.l2 * std::sin(q[1]);
}

// Yoshikawa manipulability w = sqrt(det(J J^T)). For a square J this equals
// |det(J)|, but this goes through the Gram determinant on purpose -- it is
// only ever used for logging (the w column), never for control, so it is
// fine that it is not bit-identical to det3(J).
inline double manipulability(const RobotArm& arm, const Vec3& q) {
    const Mat3 j = jacobian(arm, q);
    const double gram_det = det3(matmul(j, transpose(j)));
    return std::sqrt(gram_det > 0.0 ? gram_det : 0.0);
}

}  // namespace armctl
