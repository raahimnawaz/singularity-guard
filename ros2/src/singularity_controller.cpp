#include "armctl_ros2_control/singularity_controller.hpp"

#include <algorithm>
#include <cmath>

#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/rclcpp.hpp"

// Vec3 is an alias for std::array<double, 3>, not a distinct class type, so
// armctl's operator+/operator* aren't found by ADL outside namespace armctl
// (ADL follows the aliased type's namespace, i.e. std, not armctl). Bring
// them in explicitly rather than qualifying every infix expression.
using armctl::operator+;
using armctl::operator-;
using armctl::operator*;

namespace armctl_ros2_control
{

controller_interface::CallbackReturn SingularityController::on_init()
{
  try {
    joint_names_ = auto_declare<std::vector<std::string>>(
      "joints", std::vector<std::string>{"joint1", "joint2", "joint3"});
    controller_law_ = auto_declare<std::string>("controller_law", controller_law_);
    w0_ = auto_declare<double>("w0", w0_);
    lam_max_ = auto_declare<double>("lam_max", lam_max_);
    fixed_lambda_ = auto_declare<double>("fixed_lambda", fixed_lambda_);
    kp_ = auto_declare<double>("kp", kp_);
    qdot_max_ = auto_declare<double>("qdot_max", qdot_max_);
    l1_ = auto_declare<double>("l1", l1_);
    l2_ = auto_declare<double>("l2", l2_);
    l3_ = auto_declare<double>("l3", l3_);
    traj_q1_ = auto_declare<double>("trajectory.q1", traj_q1_);
    traj_q3_ = auto_declare<double>("trajectory.q3", traj_q3_);
    traj_q2_amplitude_ = auto_declare<double>("trajectory.q2_amplitude", traj_q2_amplitude_);
    traj_duration_ = auto_declare<double>("trajectory.duration", traj_duration_);
    loop_ = auto_declare<bool>("loop", loop_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "on_init failed: %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }

  if (joint_names_.size() != 3) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "'joints' parameter must list exactly 3 joint names, got %zu",
      joint_names_.size());
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
SingularityController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : joint_names_) {
    config.names.push_back(joint + "/velocity");
  }
  return config;
}

controller_interface::InterfaceConfiguration
SingularityController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : joint_names_) {
    config.names.push_back(joint + "/position");
  }
  return config;
}

controller_interface::CallbackReturn SingularityController::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  if (controller_law_ != "naive" && controller_law_ != "fixed" && controller_law_ != "adaptive") {
    RCLCPP_ERROR(
      get_node()->get_logger(), "controller_law must be one of naive|fixed|adaptive, got '%s'",
      controller_law_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  arm_ = armctl::RobotArm{l1_, l2_, l3_};
  traj_ =
    armctl::make_through_singularity(arm_, traj_q1_, traj_q3_, traj_q2_amplitude_, traj_duration_);
  has_start_time_ = false;

  w_pub_raw_ =
    get_node()->create_publisher<std_msgs::msg::Float64>("~/manipulability", rclcpp::SystemDefaultsQoS());
  lam_pub_raw_ =
    get_node()->create_publisher<std_msgs::msg::Float64>("~/damping", rclcpp::SystemDefaultsQoS());
  w_pub_ =
    std::make_unique<realtime_tools::RealtimePublisher<std_msgs::msg::Float64>>(w_pub_raw_);
  lam_pub_ =
    std::make_unique<realtime_tools::RealtimePublisher<std_msgs::msg::Float64>>(lam_pub_raw_);

  RCLCPP_INFO(
    get_node()->get_logger(),
    "SingularityController configured: law=%s joints=[%s, %s, %s] loop=%s duration=%.2fs",
    controller_law_.c_str(), joint_names_[0].c_str(), joint_names_[1].c_str(),
    joint_names_[2].c_str(), loop_ ? "true" : "false", traj_.duration);

  return controller_interface::CallbackReturn::SUCCESS;
}

armctl::ControlResult SingularityController::compute_command(
  const armctl::Vec3 & q, const armctl::Vec3 & xdot) const
{
  if (controller_law_ == "naive") {
    return armctl::NaiveInverse{}.compute(arm_, q, xdot);
  }
  if (controller_law_ == "fixed") {
    return armctl::DampedLeastSquares{fixed_lambda_}.compute(arm_, q, xdot);
  }
  return armctl::AdaptiveDLS{w0_, lam_max_}.compute(arm_, q, xdot);
}

controller_interface::return_type SingularityController::update(
  const rclcpp::Time & time, const rclcpp::Duration & /*period*/)
{
  if (!has_start_time_) {
    start_time_ = time;
    has_start_time_ = true;
  }

  const double elapsed = std::max((time - start_time_).seconds(), 0.0);
  const double t = loop_ ? std::fmod(elapsed, traj_.duration) : std::min(elapsed, traj_.duration);

  armctl::Vec3 q{};
  for (std::size_t i = 0; i < 3; ++i) {
    const auto value = state_interfaces_[i].get_optional();
    if (!value.has_value()) {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "SingularityController: failed to read state interface '%s'",
        state_interfaces_[i].get_name().c_str());
      return controller_interface::return_type::OK;
    }
    q[i] = *value;
  }

  const armctl::Vec3 p_now = armctl::forward_kinematics(arm_, q);
  const armctl::Vec3 xdot_cmd = traj_.twist(t) + kp_ * armctl::pose_error(traj_.pose(t), p_now);

  const armctl::ControlResult res = compute_command(q, xdot_cmd);
  armctl::Vec3 cmd = res.qdot;
  for (int i = 0; i < 3; ++i) {
    if (!std::isfinite(cmd[i])) {
      cmd[i] = 0.0;
    }
  }

  for (std::size_t i = 0; i < 3; ++i) {
    const double applied = std::min(std::max(cmd[i], -qdot_max_), qdot_max_);
    if (!command_interfaces_[i].set_value(applied)) {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "SingularityController: failed to write command interface '%s'",
        command_interfaces_[i].get_name().c_str());
    }
  }

  const double w = std::abs(armctl::det3(armctl::jacobian(arm_, q)));
  if (w_pub_ && w_pub_->trylock()) {
    w_pub_->msg_.data = w;
    w_pub_->unlockAndPublish();
  }
  if (lam_pub_ && lam_pub_->trylock()) {
    lam_pub_->msg_.data = res.lam;
    lam_pub_->unlockAndPublish();
  }

  return controller_interface::return_type::OK;
}

}  // namespace armctl_ros2_control

PLUGINLIB_EXPORT_CLASS(
  armctl_ros2_control::SingularityController, controller_interface::ControllerInterface)
