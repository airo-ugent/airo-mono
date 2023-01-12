from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from airo_typing import HomogeneousMatrixType, JointConfigurationType


class JointTrajectory:
    # TODO: this class has a dict of waypoints and timesteps on which to reach those waypoints.
    pass


@dataclass
class ManipulatorSpecs:
    dof: int
    max_joint_speed: float


class PositionManipulator:
    """(Synchronous) interface for position-controlled manipulators"""

    def __init__(self, manipulator_specs: ManipulatorSpecs) -> None:
        self.manipulator_specs = manipulator_specs

    @abstractmethod
    def get_tcp_pose() -> HomogeneousMatrixType:
        pass

    @abstractmethod
    def get_joint_configuration() -> JointConfigurationType:
        pass

    @abstractmethod
    def move_to_tcp_pose(tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None):
        """move to desired pose (synchronous)"""

    @abstractmethod
    def move_linear_to_tcp_pose(tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None):
        """move to desired pose in a straight line (synchronous)"""

    @abstractmethod
    def move_to_joint_configuration(joint_configuration: JointConfigurationType, joint_speed: Optional[float] = None):
        """move to a desired joint configuration (synchronous)"""

    @abstractmethod
    def servo_linear_to_tcp_pose(tcp_pose: HomogeneousMatrixType, time: float):
        """servo to the desired tcp pose with a linear EEF motion for the specified time (the function blocks for this time)
        The interface does not enforce an interpolation profile to interpolate between the current tcp pose and the target tcp pose.
        """

    @abstractmethod
    def servo_to_joint_configuration(joint_configuration: JointConfigurationType, time: float):
        """servo to the desired joint configuration for the specified time (the function blocks for this time)
        The interface does not enforce an interpolation profile to interpolate between the current joint configuration and the target joint configuration
        """

    def is_tcp_pose_kinematically_reachable(tcp_pose: HomogeneousMatrixType) -> bool:
        raise NotImplementedError

    def execute_joint_trajectory(joint_trajectory: JointTrajectory):
        """executes a joint trajectory (synchronously)."""
        raise NotImplementedError


class FTSensor:
    pass
