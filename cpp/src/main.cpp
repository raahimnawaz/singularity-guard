// Phase 5: dependency-free real-time port.
//
// Runs the same closed loop as armsim.sim.simulate() on the through-
// singularity scenario with the adaptive-DLS controller (the exact run that
// produced results/golden_through.csv), and does two things with it:
//
//   1. Writes a CSV in that file's format. tests/parity_check.py holds it to
//      the Python reference within 1e-9.
//   2. Times step_control() -- the one control-law evaluation a real 1 kHz
//      digital controller repeats every tick -- in isolation over every
//      (t, q) the closed loop actually visited, and reports median/p99/max
//      against a 1 ms budget. The 4-stage RK4 used to produce the CSV is an
//      offline-integration artifact and is deliberately not what gets timed.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "armctl/controllers.hpp"
#include "armctl/kinematics.hpp"
#include "armctl/mat3.hpp"
#include "armctl/sim.hpp"
#include "armctl/trajectory.hpp"

using namespace armctl;

namespace {

std::string fmt_e12(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.12e", v);
    return std::string(buf);
}

// Linear-interpolated percentile over a pre-sorted vector.
double percentile(const std::vector<double>& sorted_vals, double p) {
    if (sorted_vals.empty()) {
        return 0.0;
    }
    const double idx = p * (static_cast<double>(sorted_vals.size()) - 1.0);
    const auto lo = static_cast<std::size_t>(idx);
    const std::size_t hi = std::min(lo + 1, sorted_vals.size() - 1);
    const double frac = idx - static_cast<double>(lo);
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string out_path = argc > 1 ? argv[1] : "golden_through_cpp.csv";

    const RobotArm arm{};
    const ThroughSingularity traj = make_through_singularity(arm);
    const AdaptiveDLS controller{};
    const Limits limits{3.0};
    constexpr double dt = 1e-3;
    constexpr double kp = 5.0;

    std::vector<StepLog> log;
    log.reserve(2001);
    simulate(arm, traj, controller, dt, kp, limits,
              [&](const StepLog& s) { log.push_back(s); });

    std::ofstream out(out_path);
    if (!out) {
        std::cerr << "failed to open " << out_path << " for writing\n";
        return 1;
    }
    out << "t,q1,q2,q3,qdot1,qdot2,qdot3,x,y,phi,w,lam\n";
    for (const auto& s : log) {
        out << fmt_e12(s.t) << ',' << fmt_e12(s.q[0]) << ',' << fmt_e12(s.q[1]) << ','
            << fmt_e12(s.q[2]) << ',' << fmt_e12(s.qdot_applied[0]) << ','
            << fmt_e12(s.qdot_applied[1]) << ',' << fmt_e12(s.qdot_applied[2]) << ','
            << fmt_e12(s.p[0]) << ',' << fmt_e12(s.p[1]) << ',' << fmt_e12(s.p[2]) << ','
            << fmt_e12(s.w) << ',' << fmt_e12(s.lam) << '\n';
    }
    out.close();
    std::cout << "wrote " << out_path << " (" << log.size() << " rows)\n";

    // Real-time headroom. checksum forces the compiler to keep every call --
    // without it, a sufficiently aggressive optimizer could see the loop
    // results are otherwise unused and remove the very thing being timed.
    std::vector<double> latency_us;
    latency_us.reserve(log.size());
    double checksum = 0.0;
    for (const auto& s : log) {
        const auto t0 = std::chrono::steady_clock::now();
        const StepResult r = step_control(arm, traj, controller, limits, kp, s.t, s.q);
        const auto t1 = std::chrono::steady_clock::now();
        latency_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        checksum += r.applied[0] + r.applied[1] + r.applied[2] + r.lam;
    }

    std::vector<double> sorted_lat = latency_us;
    std::sort(sorted_lat.begin(), sorted_lat.end());
    const double median = percentile(sorted_lat, 0.50);
    const double p99 = percentile(sorted_lat, 0.99);
    const double max_lat = sorted_lat.back();
    constexpr double kBudgetUs = 1000.0;  // 1 kHz

    std::cout << "\ncontrol-law latency over " << sorted_lat.size()
              << " ticks (step_control: FK + reference + DLS solve)\n"
              << "  median  " << median << " us\n"
              << "  p99     " << p99 << " us\n"
              << "  max     " << max_lat << " us\n"
              << "  budget  " << kBudgetUs << " us (1 kHz)\n"
              << "  headroom at p99: " << (kBudgetUs / p99) << "x\n"
              << "  [checksum " << checksum << ", ignore -- keeps the loop from being "
                 "optimized away]\n";

    return 0;
}
