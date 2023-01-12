from dataclasses import dataclass
from typing import Optional

from airo_typing import HomogeneousMatrixType, JointConfigurationType


class JointTrajectory:
    pass


@dataclass
class ManipulatorSpecs:
    max_joint_speed: float


class PositionManipulator:
    def get_tcp_pose() -> HomogeneousMatrixType:
        pass

    def get_joint_configuration() -> JointConfigurationType:
        pass

    def move_to_tcp_pose(tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None):
        """move to desired pose (synchronous)"""

    def move_linear_to_tcp_pose(tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None):
        """move to desired pose in a straight line (synchronous)"""

    def move_to_joint_configuration(joint_configuration: JointConfigurationType, joint_speed: Optional[float] = None):
        """move to a desired joint configuration (synchronous)"""

    def servo_to_tcp_pose(tcp_pose: HomogeneousMatrixType, time: float):
        """servo to the desired tcp pose for the specified time (and the function blocks for this time)"""

    def servo_to_joint_configuration(joint_configuration: JointConfigurationType, time: float):
        """servo to the desired joint configuration for the specified time (and the function blocks for this time)"""

    def is_tcp_pose_kinematically_reachable(tcp_pose: HomogeneousMatrixType) -> bool:
        pass

    def execute_joint_trajectory(joint_trajectory: JointTrajectory):
        """executes a joint trajectory (synchronously)"""


class FTSensor:
    pass
