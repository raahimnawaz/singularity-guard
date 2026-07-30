// The three control laws, ported term-for-term from armsim/controllers.py:
//
//   NaiveInverse    qdot = J^-1 xdot                         (Phase 3 baseline)
//   DampedLeastSquares  qdot = J^T (J J^T + lam^2 I)^-1 xdot, lam fixed
//   AdaptiveDLS         same law, lam scheduled off manipulability (Phase 4)
//
// All three share dls_kernel()/inv3(), the one explicit cofactor inverse --
// no general solver, because JJ^T + lam^2 I is symmetric positive-definite by
// construction for any lam > 0, which is the entire point of DLS: avoiding
// the ill-conditioned inverse of J itself.
#pragma once

#include <cmath>

#include "kinematics.hpp"
#include "mat3.hpp"

namespace armctl {

struct ControlResult {
    Vec3 qdot;
    double lam;
};

// Shared DLS kernel. lam == 0 reduces exactly to J^-1 xdot, matching
// armsim/controllers.py::_dls (which special-cases lam == 0 the same way
// rather than relying on J J^T + 0*I to be well-conditioned, since it is not
// -- it is exactly as singular as J itself there).
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
    // Defaults from the (w0, lam_max) sweep in armsim/run_experiment.py --
    // the cheapest corner of the grid that reaches 0% saturation on the
    // through-singularity run.
    double w0 = 0.30;
    double lam_max = 0.20;

    // Scheduled lambda: lam^2 = lam_max^2 (1 - w/w0)^2 for w < w0, else 0.
    // Uses |det3(J)| rather than manipulability() for the same reason the
    // Python does: the Gram determinant squares the condition number and
    // bottoms out around 1e-8 near a singularity, while the cofactor
    // determinant stays accurate to roundoff.
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
