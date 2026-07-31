#!/usr/bin/env python3
"""Record one Webots run to CSV, for the Phase 7 three-law comparison.

Subscribes to /joint_states plus the controller's own ~/manipulability and
~/damping diagnostics, and writes one row per joint-state message:

    t,q1,q2,q3,qd1,qd2,qd3,w,lam,qdc1,qdc2,qdc3

Only raw signals are recorded. Every derived quantity -- tracking error,
saturation fraction, RMS acceleration -- is computed afterwards by
armsim.sim.metrics on the host, so the Webots numbers and the kinematic numbers
in results/metrics.md come out of exactly the same function rather than two
implementations that are meant to agree.

Runs on **sim time**: the trajectory is defined against the simulator clock, and
Webots in --mode=fast does not advance in step with wall time. Sampling on wall
time would stretch or compress the trajectory depending on how fast the
simulator happened to run.

    ros2 run armctl_ros2_control record_run.py --out /out/run.csv --duration 2.0
"""

import argparse
import csv
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray


class Recorder(Node):
    def __init__(self, out_path, duration, joints):
        super().__init__("record_run")
        # Declared before anything else reads the clock: with use_sim_time the
        # node must not latch a wall-clock t0 first.
        self.set_parameters(
            [rclpy.parameter.Parameter("use_sim_time", value=True)]
        )

        self.joints = joints
        self.duration = duration
        self.rows = []
        self.t0 = None
        self._w = float("nan")
        self._lam = float("nan")
        self._qdc = [float("nan")] * 3
        self.out_path = out_path

        self.create_subscription(
            JointState, "/joint_states", self._on_js, qos_profile_sensor_data
        )
        self.create_subscription(
            Float64, "/singularity_controller/manipulability", self._on_w, 10
        )
        self.create_subscription(
            Float64, "/singularity_controller/damping", self._on_lam, 10
        )
        self.create_subscription(
            Float64MultiArray, "/singularity_controller/qdot_cmd", self._on_qdc, 10
        )
        self.get_logger().info(f"recording {duration}s of sim time -> {out_path}")

    def _on_w(self, msg):
        self._w = msg.data

    def _on_lam(self, msg):
        self._lam = msg.data

    def _on_qdc(self, msg):
        if len(msg.data) >= 3:
            self._qdc = list(msg.data[:3])

    def _on_js(self, msg: JointState):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        rel = t - self.t0
        if rel > self.duration:
            self._finish()
            return

        # Joint order in the message is not guaranteed, so index by name rather
        # than assuming the broadcaster preserves URDF order.
        try:
            idx = [msg.name.index(j) for j in self.joints]
        except ValueError:
            self.get_logger().warn(f"expected {self.joints}, got {list(msg.name)}")
            return

        q = [msg.position[i] for i in idx]
        qd = [msg.velocity[i] for i in idx] if len(msg.velocity) >= 3 else [0.0] * 3
        self.rows.append([rel, *q, *qd, self._w, self._lam, *self._qdc])

    def _finish(self):
        with open(self.out_path, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow("t q1 q2 q3 qd1 qd2 qd3 w lam qdc1 qdc2 qdc3".split())
            wtr.writerows(self.rows)
        self.get_logger().info(f"wrote {len(self.rows)} rows -> {self.out_path}")
        raise SystemExit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--joints", default="joint1,joint2,joint3")
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = Recorder(args.out, args.duration, args.joints.split(","))
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
