import numpy as np
import pinocchio

from lbr_fri_idl.msg import LBRTorqueCommand, LBRState


class CartesianImpedanceController(object):
    def __init__(
            self,
            robot_description: str,
            impedance_pos: np.ndarray = np.asarray([3000.0, 3000.0, 3000.0]),
            impedance_ori: np.ndarray = np.asarray([50.0, 50.0, 50.0]),
            Kp_null: np.ndarray = np.asarray([75.0, 75.0, 50.0, 50.0, 40.0, 25.0, 25.0]),
            damping_ratio: float = 0.7,
            Kpos: float = 0.95,
            Kori: float = 0.95,
            integration_dt: float = 1.0
    ) -> None:
        self._lbr_torque_command = LBRTorqueCommand()
        self._robot = pinocchio.RobotWrapper.BuildFromURDF(robot_description)

        self._dof = self._robot.nv
        self._jacobian = np.zeros((6, self._dof))
        self._jacobian_inv = np.zeros((self._dof, 6))
        self._q = np.zeros(self._dof)
        self._q_commanded = np.zeros(self._dof)

        self._damping_pos = 2.0 * np.sqrt(impedance_pos) * damping_ratio
        self._damping_ori = 2.0 * np.sqrt(impedance_ori) * damping_ratio

        self._Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
        self._Kd = np.concatenate([self._damping_pos, self._damping_ori], axis=0)

        self._Kp_null = Kp_null
        self._Kd_null = 2.0 * np.sqrt(Kp_null) * damping_ratio

        self._Kpos = Kpos
        self._Kori = Kori
        self._integration_dt = integration_dt

        self._twist = np.zeros(6)

    def __call__(self, lbr_state: LBRState, dt: float) -> LBRTorqueCommand:
        self._q = np.array(lbr_state.measured_joint_position.tolist())
        self._q_commanded = np.array(lbr_state.commanded_joint_position.tolist())
        
        self._robot.forwardKinematics(self._q)

        self._jacobian = self._robot.computeJointJacobians(self._q)
        self._mass_inertia_matrix = self._robot.mass(self._q)

        M_inv = np.linalg.inv(self._mass_inertia_matrix)
        Mx_inv = self._jacobian @ M_inv @ self._jacobian.T

        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)



    

