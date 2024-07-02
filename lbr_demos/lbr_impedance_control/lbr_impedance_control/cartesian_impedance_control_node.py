import os
import collections
import threading
import time

import numpy as np
import pinocchio

from scipy.spatial.transform import Rotation

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor

from ament_index_python.packages import get_package_share_path

from geometry_msgs.msg import Pose
from lbr_fri_idl.msg import LBRState
from action_interface.action import CartesianImpedanceCommand

from .cartesian_impedance_controller import CartesianImpedanceController
from .lbr_base_torque_command_node import LBRBaseTorqueCommandNode


class CartesianImpedanceControlNode(LBRBaseTorqueCommandNode):
    def __init__(self, node_name: str = "cartesian_impedance_control") -> None:
        super().__init__(node_name=node_name)

        self._action_client = ActionServer(
            self,
            CartesianImpedanceCommand,
            "cartesian_impedance_control",
            execute_callback=self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback,
        )

        self._robot_type = "iiwa14"
        self._goal_queue = collections.deque()
        self._goal_queue_lock = threading.Lock()
        self._goal_handle = None

        self._robot_model = pinocchio.RobotWrapper.BuildFromURDF(
            os.path.join(
                get_package_share_path("lbr_description"),
                "urdf",
                self._robot_type,
                f"{self._robot_type}.xacro",
            )
        )
        self._robot_data = self._robot_model.createData()

    def destroy(self):
        self._action_client.destroy()
        super().destroy()

    def goal_callback(self, goal_request: CartesianImpedanceCommand.Goal):
        valid_damping = all(x > 0 and x <= 1.0 for x in goal_request.damping)
        valid_stiffness = all(x > 100.0 for x in goal_request.stiffness)
        valid_ik = self.get_ik(goal_request.pose) is not None

        if valid_damping and valid_stiffness and valid_ik:
            return GoalResponse.ACCEPT
        else:
            return GoalResponse.REJECT

    def handle_accepted_callback(self, goal_handle: ServerGoalHandle):
        if goal_handle.request.abort_previous:
            if self._goal_handle is not None and self._goal_handle.is_active:
                self._goal_handle.abort()
            with self._goal_queue_lock:
                self._goal_queue.clear()
            
            self._goal_handle = goal_handle
            goal_handle.execute()
        else:
            with self._goal_queue_lock:
                if self._goal_handle is not None:
                    self._goal_queue.append(goal_handle)
                else:
                    self._goal_handle = goal_handle
                    goal_handle.execute()

    async def execute_callback(self, goal_handle):
        pass

    def cancel_callback(self, goal_handle: ServerGoalHandle):
        pass

    def get_ik(self, pose: Pose) -> np.ndarray | None:
        model = self._robot_model
        data = self._robot_data

        JOINT_ID = 8

        oMdes = pinocchio.SE3(
            Rotation.from_quat(
                np.array(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                )
            ).as_matrix(),
            np.array([pose.position.x, pose.position.y, pose.position.z]),
        )

        q = pinocchio.neutral(model)
        eps = 1e-4
        IT_MAX = 1000
        DT = 1e-1
        damp = 1e-12

        i = 0
        success = False

        while i < IT_MAX:
            pinocchio.forwardKinematics(model, data, q)
            iMd = data.oMi[JOINT_ID].actInv(oMdes)
            err = pinocchio.log(iMd).vector  # in joint frame

            if np.linalg.norm(err) < eps:
                success = True
                break

            J = pinocchio.computeJointJacobian(
                model, data, q, JOINT_ID
            )  # in joint frame
            J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pinocchio.integrate(model, q, v * DT)

            i += 1

        return q if success else None
