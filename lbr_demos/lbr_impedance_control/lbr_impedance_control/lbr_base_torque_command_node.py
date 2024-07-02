import rclpy
import rclpy.parameter
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node

from lbr_fri_idl.msg import LBRTorqueCommand, LBRState


class LBRBaseTorqueCommandNode(Node):

    _update_rate: int
    _dt: float
    _robot_description: str

    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)

        self._robot_description = self._retrieve_parameter(
            "robot_state_publisher/get_parameters", "robot_description"
        ).string_value
        self._update_rate = self._retrieve_parameter(
            "controller_manager/get_parameters", "update_rate"
        ).integer_value
        self._dt = 1.0 / float(self._update_rate)

        # self._lbr_state_sub = self.create_subscription(
        #     LBRState, "state", self._on_lbr_state, 1
        # )
        self._lbr_state_sub = None
        self._lbr_torque_command_pub = self.create_publisher(
            LBRTorqueCommand,
            "command/torque",
            1,
        )
    
    def _retrieve_parameter(self, service: str, parameter_name: str) -> ParameterValue:
        parameter_client = self.create_client(GetParameters, service)
        while not parameter_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                self.get_logger().error(
                    "Interrupted while waiting for the service. Exiting."
                )
                return None
            self.get_logger().info(f"Waiting for '{service}' service...")
        request = GetParameters.Request(names=[parameter_name])
        future = parameter_client.call_async(request=request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error(f"Failed to retrieve '{parameter_name}'.")
            return None
        self.get_logger().info(f"Received '{parameter_name}' from '{service}'.")
        return future.result().values[0]
    
    def _on_lbr_state(self, lbr_state: LBRState) -> None:
        raise NotImplementedError