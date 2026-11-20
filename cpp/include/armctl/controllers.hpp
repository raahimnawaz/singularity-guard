// The three control laws, ported term for term from armsim/controllers.py:
//   naive     qdot = J^-1 xdot
//   fixed     qdot = J^T (J J^T + lam^2 I)^-1 xdot
//   adaptive  same, with lam scheduled on manipulability
#pragma once

#include <cmath>

#include "kinematics.hpp"
#include "mat3.hpp"

namespace armctl {

struct ControlResult {
    Vec3 qdot;
    double lam;
};

// Shared DLS kernel; lam == 0 reduces exactly to J^-1 xdot.
inline Vec3 dls_kernel(const RobotArm& arm, const Vec3& q, const Vec3& xdot, double lam) {
    const Mat3 j = jacobian(arm, q);
    if (lam == 0.0) {
        return matvec(inv3(j), xdot);
    }
    Mat3 a = matmul(j, transpose(j));
    const double lam2 = lam * lam;
    a[0][0] += lam2;
    a[1][1] += lam2;
    a[2][2] += lam2;
    return matvec(transpose(j), matvec(inv3(a), xdot));
}

struct NaiveInverse {
    ControlResult compute(const RobotArm& arm, const Vec3& q, const Vec3& xdot) const {
        return {matvec(inv3(jacobian(arm, q)), xdot), 0.0};
    }
};

struct DampedLeastSquares {
    double lam = 0.1;

    ControlResult compute(const RobotArm& arm, const Vec3& q, const Vec3& xdot) const {
        return {dls_kernel(arm, q, xdot, lam), lam};
    }
};

struct AdaptiveDLS {
    // Defaults from the (w0, lam_max) sweep in armsim/run_experiment.py.
    double w0 = 0.30;
    double lam_max = 0.20;

    // lam = lam_max (1 - w/w0) for w < w0, else 0. Uses |det3(J)|, which stays
    // accurate near zero.
    double damping(const RobotArm& arm, const Vec3& q) const {
        const double w = std::abs(det3(jacobian(arm, q)));
        if (w >= w0) {
            return 0.0;
        }
        return lam_max * (1.0 - w / w0);
    }

    ControlResult compute(const RobotArm& arm, const Vec3& q, const Vec3& xdot) const {
        const double lam = damping(arm, q);
        return {dls_kernel(arm, q, xdot, lam), lam};
    }
};

}  // namespace armctl
