import warnings
from typing import Optional

import numpy as np
from airo_robots.grippers.base import ParallelPositionGripper
from airo_robots.manipulators.base import ManipulatorSpecs, PositionManipulator
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, JointConfigurationType
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

RotVecPoseType = np.ndarray
""" a 6D pose [tx,ty,tz,rotvecx,rotvecy,rotvecz]"""

UR3e_config = ManipulatorSpecs(6, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], 1.0)


class UR_RTDE(PositionManipulator):
    def __init__(
        self, ip_address: str, manipulator_specs: ManipulatorSpecs, gripper: Optional[ParallelPositionGripper] = None
    ) -> None:
        super().__init__(manipulator_specs, gripper)
        self.ip_address = ip_address
        try:
            self.rtde_control = RTDEControlInterface(self.ip_address)
            self.rtde_receive = RTDEReceiveInterface(self.ip_address)

        except TimeoutError:
            raise TimeoutError(
                "Could not connect to the robot. Is the robot in remote control? Is the IP correct? Is the network connection ok?"
            )

        self.default_linear_speed = 0.1  # m/s
        self.default_linear_acceleration = 1.2  # m/s^2
        self.default_leading_axis_joint_speed = 1  # rad/s
        self.default_leading_axis_joint_acceleration = 1.2  # rad/s^2

        self.servo_proportional_gain = 300
        self.servo_lookahead_time = 0.05

    def get_joint_configuration(self) -> JointConfigurationType:
        return self.rtde_receive.getActualQ()

    def get_tcp_pose(self) -> HomogeneousMatrixType:
        tpc_rotvec_pose = self.rtde_receive.getActualTCPPose()
        return self._convert_rotvec_pose_to_homogeneous_pose(tpc_rotvec_pose)

    def move_linear_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None):
        if not self.is_tcp_pose_kinematically_reachable(tcp_pose):
            warnings.warn(f"Pose is not reachable :\n{tcp_pose}")
            return
        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)
        linear_speed = linear_speed or self.default_linear_speed
        linear_speed = np.clip(linear_speed, 0.0, self.manipulator_specs.max_linear_speed)
        return self.rtde_control.moveL(tcp_rotvec_pose, linear_speed, self.default_linear_acceleration)

    def move_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, joint_speed: Optional[float] = None):
        if not self.is_tcp_pose_kinematically_reachable(tcp_pose):
            warnings.warn(f"Pose is not reachable :\n{tcp_pose}")
            return

        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)
        joint_speed = joint_speed or self.default_leading_axis_joint_speed
        # don't know what leading axis is atm, so take min of all max joint speeds
        joint_speed = np.clip(joint_speed, 0.0, min(self.manipulator_specs.max_joint_speeds))
        self.rtde_control.moveJ_IK(tcp_rotvec_pose, joint_speed, self.default_leading_axis_joint_acceleration)

    def move_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, joint_speed: Optional[float] = None
    ):
        # TODO: check for joint limits.
        joint_speed = joint_speed or self.default_leading_axis_joint_speed
        # don't know what leading axis is atm, so take min of all max joint speeds
        joint_speed = np.clip(joint_speed, 0.0, min(self.manipulator_specs.max_joint_speeds))
        self.rtde_control.moveJ(joint_configuration, joint_speed, self.default_leading_axis_joint_acceleration)
        return

    def servo_linear_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, time: float):
        if not self.is_tcp_pose_kinematically_reachable(tcp_pose):
            warnings.warn(f"Pose is not reachable :\n{tcp_pose}")
            return

        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)

        self.rtde_control.servoL(
            tcp_rotvec_pose, 0.0, 0.0, time, self.servo_lookahead_time, self.servo_proportional_gain
        )

    def servo_to_joint_configuration(self, joint_configuration: JointConfigurationType, time: float):
        pass

    def inverse_kinematics(
        self, tcp_pose: HomogeneousMatrixType, joint_configuration_guess: JointConfigurationType
    ) -> JointConfigurationType:
        return super().inverse_kinematics(tcp_pose, joint_configuration_guess)

    def forward_kinematics(self, joint_configuration: JointConfigurationType) -> HomogeneousMatrixType:
        return super().forward_kinematics(joint_configuration)

    def is_tcp_pose_kinematically_reachable(self, tcp_pose: HomogeneousMatrixType) -> bool:
        tcp_rotvec_pose = self._convert_homegeneous_pose_to_rotvec_pose(tcp_pose)
        return self.rtde_control.isPoseWithinSafetyLimits(tcp_rotvec_pose)

    @staticmethod
    def _convert_rotvec_pose_to_homogeneous_pose(ur_pose: RotVecPoseType) -> HomogeneousMatrixType:
        return SE3Container.from_rotation_vector_and_translation(ur_pose[3:], ur_pose[:3]).homogeneous_matrix

    @staticmethod
    def _convert_homegeneous_pose_to_rotvec_pose(homogeneous_pose: HomogeneousMatrixType) -> RotVecPoseType:
        se3 = SE3Container.from_homogeneous_matrix(homogeneous_pose)
        rotation = se3.orientation_as_rotation_vector
        translation = se3.translation
        return np.concatenate([translation, rotation])


if __name__ == "__main__":
    ip = "10.42.0.162"
    ur3e = UR_RTDE(ip, UR3e_config)
    print(ur3e.get_joint_configuration())
    print(ur3e.get_tcp_pose())
    pose = SE3Container.from_euler_angles_and_translation(
        np.array([0, np.pi, 0.0001]), np.array([-0.2, -0.3, 0.1])
    ).homogeneous_matrix
    ur3e.move_linear_to_tcp_pose(pose)
    pose = SE3Container.from_euler_angles_and_translation(
        np.array([0, np.pi, 0.0001]), np.array([0.2, -0.3, 0.1])
    ).homogeneous_matrix
    ur3e.servo_linear_to_tcp_pose(pose, 1)
