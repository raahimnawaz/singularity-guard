// ros2_control wrapper around the armctl headers: interfaces, parameters,
// diagnostics, and a runtime switch between the three control laws.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include "armctl/controllers.hpp"
#include "armctl/kinematics.hpp"
#include "armctl/sim.hpp"
#include "armctl/trajectory.hpp"

namespace armctl_ros2_control
{

class SingularityController : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  // The laws share a signature but no base class, so dispatch is a switch.
  armctl::ControlResult compute_command(const armctl::Vec3 & q, const armctl::Vec3 & xdot) const;

  // Declared in on_init() so YAML overrides apply before they are read.
  std::vector<std::string> joint_names_;
  std::string controller_law_ = "adaptive";  // naive | fixed | adaptive
  double w0_ = 0.30;
  double lam_max_ = 0.20;
  double fixed_lambda_ = 0.10;
  double kp_ = 5.0;
  double qdot_max_ = 3.0;
  double l1_ = 1.0;
  double l2_ = 1.0;
  double l3_ = 1.0;
  double traj_q1_ = 0.3;
  double traj_q3_ = armctl::kPi / 2.0;
  double traj_q2_amplitude_ = 0.8;
  double traj_duration_ = 2.0;
  bool loop_ = true;

  // Derived state, rebuilt in on_configure() from the parameters above.
  armctl::RobotArm arm_{};
  armctl::ThroughSingularity traj_{2.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  bool has_start_time_ = false;
  rclcpp::Time start_time_;

  std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float64>> w_pub_raw_;
  std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float64>> lam_pub_raw_;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64>> w_pub_;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64>> lam_pub_;

  // The law's raw output, before the clamp: the joint never sees it, but it is
  // what results/metrics.md reports as peak_qdot_cmd.
  std::shared_ptr<rclcpp::Publisher<std_msgs::msg::Float64MultiArray>> qdot_cmd_pub_raw_;
  std::unique_ptr<realtime_tools::RealtimePublisher<std_msgs::msg::Float64MultiArray>>
    qdot_cmd_pub_;
};

}  // namespace armctl_ros2_control
