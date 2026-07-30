// Closed-loop simulation, ported from armsim/sim.py: fixed-step RK4 on
// qdot = f(t, q), with plant-side saturation and the same nonfinite guard.
//
// step_control() is the one control-law evaluation a real 1 kHz digital
// controller repeats every tick -- forward kinematics, the reference twist,
// the controller's command, the saturation clamp. simulate() calls it four
// times per step (RK4's stages) purely for offline-integration accuracy, the
// same choice armsim/sim.py makes to keep the Python and C++ integrators
// comparable step-for-step. src/main.cpp times step_control() alone, not the
// RK4 wrapper, because the 4x evaluation is a simulation artifact, not
// something real hardware does.
#pragma once

#include <algorithm>
#include <cmath>

#include "controllers.hpp"
#include "kinematics.hpp"
#include "mat3.hpp"
#include "trajectory.hpp"

namespace armctl {

struct Limits {
    double qdot_max = 3.0;  // rad/s, per joint
};

// p_ref - p with the orientation component wrapped into (-pi, pi].
inline Vec3 pose_error(const Vec3& p_ref, const Vec3& p) {
    Vec3 e = p_ref - p;
    e[2] = wrap_angle(e[2]);
    return e;
}

struct StepResult {
    Vec3 applied;  // what the plant allowed (saturation-clamped)
    Vec3 cmd;      // what the controller asked for
    double lam;
    bool saturated;
};

template <typename Controller>
inline StepResult step_control(const RobotArm& arm, const ThroughSingularity& traj,
                                const Controller& ctrl, const Limits& limits, double kp,
                                double t, const Vec3& q) {
    const Vec3 p_now = forward_kinematics(arm, q);
    const Vec3 xdot_cmd = traj.twist(t) + kp * pose_error(traj.pose(t), p_now);
    const ControlResult res = ctrl.compute(arm, q, xdot_cmd);
    Vec3 cmd = res.qdot;

    bool nonfinite = false;
    for (int i = 0; i < 3; ++i) {
        if (!std::isfinite(cmd[i])) {
            nonfinite = true;
            break;
        }
    }
    if (nonfinite) {
        // A singular J has no inverse, so the naive law has no answer here.
        // Commanding zero (not a huge number) is the conservative reading,
        // matching armsim/sim.py's np.nan_to_num call.
        for (int i = 0; i < 3; ++i) {
            if (std::isnan(cmd[i])) {
                cmd[i] = 0.0;
            } else if (std::isinf(cmd[i])) {
                cmd[i] = cmd[i] > 0.0 ? limits.qdot_max : -limits.qdot_max;
            }
        }
    }

    bool saturated = false;
    Vec3 applied{};
    for (int i = 0; i < 3; ++i) {
        if (std::abs(cmd[i]) > limits.qdot_max) {
            saturated = true;
        }
        applied[i] = std::min(std::max(cmd[i], -limits.qdot_max), limits.qdot_max);
    }
    return {applied, cmd, res.lam, saturated};
}

struct StepLog {
    double t;
    Vec3 q;
    Vec3 qdot_applied;
    Vec3 p;
    double w;
    double lam;
};

// Integrates the closed loop with fixed-step RK4 and calls on_step(StepLog)
// once per logged sample, in order -- mirroring armsim.sim.simulate()'s
// per-step logging (q, applied qdot, pose, manipulability, lambda) exactly.
template <typename Controller, typename OnStep>
inline void simulate(const RobotArm& arm, const ThroughSingularity& traj, const Controller& ctrl,
                      double dt, double kp, const Limits& limits, OnStep on_step) {
    Vec3 q = traj.q0;
    const int n = static_cast<int>(std::llround(traj.duration / dt)) + 1;

    for (int i = 0; i < n; ++i) {
        const double t = i * dt;
        const StepResult ev = step_control(arm, traj, ctrl, limits, kp, t, q);

        on_step(StepLog{t, q, ev.applied, forward_kinematics(arm, q), manipulability(arm, q),
                         ev.lam});

        if (i == n - 1) {
            break;
        }

        const Vec3 k1 = ev.applied;
        const Vec3 k2 =
            step_control(arm, traj, ctrl, limits, kp, t + 0.5 * dt, q + (0.5 * dt) * k1).applied;
        const Vec3 k3 =
            step_control(arm, traj, ctrl, limits, kp, t + 0.5 * dt, q + (0.5 * dt) * k2).applied;
        const Vec3 k4 = step_control(arm, traj, ctrl, limits, kp, t + dt, q + dt * k3).applied;
        q = q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
}

}  // namespace armctl
