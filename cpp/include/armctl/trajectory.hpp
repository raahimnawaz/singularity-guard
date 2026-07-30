// Task-space reference trajectory, ported from armsim/trajectory.py.
//
// Only through_singularity is needed here: it is the scenario
// results/golden_through.csv was generated from, and the one Phase 5 has to
// hold parity on. See trajectory.py's module docstring for why the reference
// is a straight line in *task* space with no feasibility guarantee -- that
// design mistake (and its fix) is one of the more important things this
// project got right on the second try, and the C++ port inherits the fix.
#pragma once

#include <algorithm>
#include <cmath>

#include "kinematics.hpp"
#include "mat3.hpp"

namespace armctl {

inline constexpr double kPi = 3.14159265358979323846;

// Wrap angle(s) into (-pi, pi].
inline double wrap_angle(double a) {
    double m = std::fmod(a + kPi, 2.0 * kPi);
    if (m < 0.0) {
        m += 2.0 * kPi;
    }
    return m - kPi;
}

// Quintic time scaling s(tau) and ds/dtau on tau in [0, 1]. Zero velocity and
// acceleration at both ends; peak speed at tau = 0.5.
inline void quintic(double tau, double& s, double& ds) {
    tau = std::clamp(tau, 0.0, 1.0);
    const double tau2 = tau * tau;
    s = tau2 * tau * (10.0 - 15.0 * tau + 6.0 * tau2);
    ds = 30.0 * tau2 * (1.0 - 2.0 * tau + tau2);
}

// Straight task-space line through an exactly singular pose at s = 0.5.
// p(s) = p0 + 2*s*d, dp/ds = 2*d (constant), matching
// armsim/trajectory.py::through_singularity's closed form.
struct ThroughSingularity {
    double duration;
    Vec3 q0;
    Vec3 p0;
    Vec3 d;  // half the line: p_singular - p0

    Vec3 pose(double t) const {
        double s, ds;
        quintic(t / duration, s, ds);
        return p0 + (2.0 * s) * d;
    }

    Vec3 twist(double t) const {
        double s, ds;
        quintic(t / duration, s, ds);
        return ((2.0 * ds) / duration) * d;
    }
};

inline ThroughSingularity make_through_singularity(const RobotArm& arm, double q1 = 0.3,
                                                     double q3 = kPi / 2.0,
                                                     double q2_amplitude = 0.8,
                                                     double duration = 2.0) {
    const Vec3 q_start{q1, q2_amplitude, q3};
    const Vec3 q_sing{q1, 0.0, q3};
    const Vec3 p0 = forward_kinematics(arm, q_start);
    const Vec3 p_sing = forward_kinematics(arm, q_sing);
    return ThroughSingularity{duration, q_start, p0, p_sing - p0};
}

}  // namespace armctl
