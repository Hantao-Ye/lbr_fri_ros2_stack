#include "lbr_fri_ros2/filters.hpp"

namespace lbr_fri_ros2 {
ExponentialFilter::ExponentialFilter() : ExponentialFilter::ExponentialFilter(0, 0.0) {}

ExponentialFilter::ExponentialFilter(const double &cutoff_frequency, const double &sample_time) {
  set_cutoff_frequency(cutoff_frequency, sample_time);
}

void ExponentialFilter::set_cutoff_frequency(const double &cutoff_frequency,
                                             const double &sample_time) {
  cutoff_frequency_ = cutoff_frequency;
  if (cutoff_frequency_ > (1. / sample_time)) {
    cutoff_frequency_ = (1. / sample_time);
  }
  sample_time_ = sample_time;
  alpha_ = compute_alpha_(cutoff_frequency, sample_time);
  if (!validate_alpha_(alpha_)) {
    throw std::runtime_error("Alpha is not within [0, 1]");
  }
}

double ExponentialFilter::compute_alpha_(const double &cutoff_frequency,
                                         const double &sample_time) {
  double omega_3db = 2.0 * M_PI * sample_time * cutoff_frequency;
  return std::cos(omega_3db) - 1 +
         std::sqrt(std::pow(std::cos(omega_3db), 2) - 4 * std::cos(omega_3db) + 3);
}

bool ExponentialFilter::validate_alpha_(const double &alpha) { return alpha <= 1. && alpha >= 0.; }

void JointExponentialFilterArray::compute(const double *const current, value_array_t &previous) {
  std::for_each(current, current + KUKA::FRI::LBRState::NUMBER_OF_JOINTS,
                [&, i = 0](const auto &current_i) mutable {
                  previous[i] = exponential_filter_.compute(current_i, previous[i]);
                  ++i;
                });
}

void JointExponentialFilterArray::initialize(const double &cutoff_frequency,
                                             const double &sample_time) {
  exponential_filter_.set_cutoff_frequency(cutoff_frequency, sample_time);
  initialized_ = true;
}

JointPIDArray::JointPIDArray(const PIDParameters &pid_parameters)
    : pid_parameters_(pid_parameters) // keep local copy of parameters since
                                      // controller_toolbox::Pid::getGains is not const correct
                                      // (i.e. can't be called in this->log_info)
{
  std::for_each(pid_controllers_.begin(), pid_controllers_.end(), [&](auto &pid) {
    pid.initPid(pid_parameters_.p, pid_parameters_.i, pid_parameters_.d, pid_parameters_.i_max,
                pid_parameters_.i_min, pid_parameters_.antiwindup);
  });
}

void JointPIDArray::compute(const value_array_t &command_target, const value_array_t &state,
                            const std::chrono::nanoseconds &dt, value_array_t &command) {
  std::for_each(command.begin(), command.end(), [&, i = 0](double &command_i) mutable {
    command_i += pid_controllers_[i].computeCommand(command_target[i] - state[i], dt.count());
    ++i;
  });
}

void JointPIDArray::compute(const value_array_t &command_target, const double *state,
                            const std::chrono::nanoseconds &dt, value_array_t &command) {
  std::for_each(command.begin(), command.end(), [&, i = 0](double &command_i) mutable {
    command_i += pid_controllers_[i].computeCommand(command_target[i] - state[i], dt.count());
    ++i;
  });
}

void JointPIDArray::log_info() const {
  RCLCPP_INFO(rclcpp::get_logger(LOGGER_NAME), "*** Parameters:");
  RCLCPP_INFO(rclcpp::get_logger(LOGGER_NAME), "*   p: %.1f", pid_parameters_.p);
  RCLCPP_INFO(rclcpp::get_logger(LOGGER_NAME), "*   i: %.1f", pid_parameters_.i);
  RCLCPP_INFO(rclcpp::get_logger(LOGGER_NAME), "*   d: %.1f", pid_parameters_.d);
  RCLCPP_INFO(rclcpp::get_logger(LOGGER_NAME), "*   i_max: %.1f", pid_parameters_.i_max);
  RCLCPP_INFO(rclcpp::get_logger(LOGGER_NAME), "*   i_min: %.1f", pid_parameters_.i_min);
  RCLCPP_INFO(rclcpp::get_logger(LOGGER_NAME), "*   antiwindup: %s",
              pid_parameters_.antiwindup ? "true" : "false");
};
} // end of namespace lbr_fri_ros2
