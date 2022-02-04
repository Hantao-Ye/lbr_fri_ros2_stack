#pragma once

#include <string>
#include <vector>
#include <memory>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <hardware_interface/base_interface.hpp>
#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/handle.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <hardware_interface/types/hardware_interface_status_values.hpp>
#include <fri/friLBRState.h>
#include <fri/friLBRClient.h>
#include <fri/friUdpConnection.h>
#include <fri/friClientApplication.h>
#include <controller_manager/controller_manager.hpp>

// add thread?
// load basic plugin
// pluginlib https://docs.ros.org/en/foxy/Tutorials/Pluginlib.html

// todo: visibility control for windows compilation
// see #include <hardware_interface/system.hpp> -> HARDWARE_INTERFACE_PUBLIC
// and https://github.com/ros-controls/ros2_control_demos/blob/master/ros2_control_demo_hardware/include/ros2_control_demo_hardware/rrbot_system_multi_interface.hpp
// https://jeffzzq.medium.com/designing-a-ros2-robot-7c31a62c535a
// https://ros-controls.github.io/control.ros.org/getting_started.html#hardware-components

// galactic: http://control.ros.org/ros2_control/hardware_interface/doc/hardware_components_userdoc.html

namespace LBR {

class FRIHardwareInterface : public hardware_interface::BaseInterface<hardware_interface::SystemInterface>, public KUKA::FRI::LBRClient {

    public:
        FRIHardwareInterface() : app_(connection_, *this) { };
        ~FRIHardwareInterface() = default;

        // hardware interface
        hardware_interface::return_type configure(const hardware_interface::HardwareInfo& system_info) override;  // check ros2 control and set status
        std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
        std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

        hardware_interface::return_type prepare_command_mode_switch(const std::vector<std::string>& start_interfaces, const std::vector<std::string>& stop_interfaces) override;  // not supported in FRI

        hardware_interface::return_type start() override;
        hardware_interface::return_type stop() override;

        hardware_interface::return_type read() override;
        hardware_interface::return_type write() override;

        // FRI
        void onStateChange(KUKA::FRI::ESessionState old_state, KUKA::FRI::ESessionState new_state) override;
        // void monitor() override; // possibly publish full state to topic?
        // void waitForCommand() override;
        void command() override;
        void step();

    private:
        std::string FRI_HW_LOGGER = "FRIHardwareInterface";

        // exposed states
        std::vector<double> hw_position_;      // accessible through FRI
        std::vector<double> hw_effort_;        // accessible through FRI

        // commands
        std::vector<double> hw_position_command_;  // supported by FRI
        std::vector<double> hw_effort_command_;    // supported by FRI

        // FRI
        KUKA::FRI::UdpConnection connection_;
        KUKA::FRI::ClientApplication app_;

        std::string hw_operation_mode_;
        std::uint16_t hw_port_;
        const char* hw_remote_host_;

        // track command mode as FRI does not support switches
        bool command_mode_init_;

        // Communication thread
        std::thread fri_thread_;

        std::string fri_e_session_state_to_string_(const KUKA::FRI::ESessionState& state);
        std::string fri_e_operation_mode_to_string_(const KUKA::FRI::EOperationMode& mode);
};

} // end of name space LBR
