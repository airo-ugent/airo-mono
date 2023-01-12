from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from airo_robots.grippers.base import ParallelPositionGripper
from airo_typing import HomogeneousMatrixType, JointConfigurationType, WrenchType


class JointTrajectory:
    # TODO: this class has a dict of waypoints and timesteps on which to reach those waypoints.
    pass


@dataclass
class ManipulatorSpecs:
    """
    dof: the Degrees of freedom of the robot, can be used to verify the shape of joint configurations
    max_joint_speeds: list of max joint speeds in [rad/s]
    max_linear_speed: an (approximate) maximal linear speed [m/s], since it is hard to test for joint speed limitations on each interpolation step
    """

    dof: int
    max_joint_speeds: List[float]
    max_linear_speed: float


class PositionManipulator:
    """(Synchronous) base class for position-controlled manipulators

    To use this asynchronous, the recommended way would be to start a thread and have a Future object to see when it is finished.
    More support (i.e. creating this thread and simply returning the future object) for async control could (should?) be added later
    """

    def __init__(self, manipulator_specs: ManipulatorSpecs, gripper: Optional[ParallelPositionGripper] = None) -> None:
        self.manipulator_specs = manipulator_specs
        self.gripper = gripper

    @abstractmethod
    def get_tcp_pose(self) -> HomogeneousMatrixType:
        pass

    @abstractmethod
    def get_joint_configuration(self) -> JointConfigurationType:
        pass

    @abstractmethod
    def move_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, joint_speed: Optional[float] = None):
        """move to desired pose (synchronous)"""

    @abstractmethod
    def move_linear_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None):
        """move to desired pose in a straight line (synchronous)"""

    @abstractmethod
    def move_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, joint_speed: Optional[float] = None
    ):
        """move to a desired joint configuration (synchronous)"""

    @abstractmethod
    def servo_linear_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, time: float):
        """servo to the desired tcp pose with a linear EEF motion for the specified time (the function blocks for this time)
        The interface does not enforce an interpolation profile to interpolate between the current tcp pose and the target tcp pose.
        """

    @abstractmethod
    def servo_to_joint_configuration(self, joint_configuration: JointConfigurationType, time: float):
        """servo to the desired joint configuration for the specified time (the function blocks for this time)
        The interface does not enforce an interpolation profile to interpolate between the current joint configuration and the target joint configuration
        """

    @abstractmethod
    def inverse_kinematics(
        self, tcp_pose: HomogeneousMatrixType, joint_configuration_guess: JointConfigurationType
    ) -> JointConfigurationType:
        pass

    @abstractmethod
    def forward_kinematics(self, joint_configuration: JointConfigurationType) -> HomogeneousMatrixType:
        pass

    def is_tcp_pose_kinematically_reachable(self, tcp_pose: HomogeneousMatrixType) -> bool:
        raise NotImplementedError

    def execute_joint_trajectory(self, joint_trajectory: JointTrajectory):
        """executes a joint trajectory (synchronously)."""
        raise NotImplementedError


class BimanualPositionManipulator:
    """
    (synchronous) base class for bimanual position-controlled manipulators. This could be a bimanual robot or a combination of 2 unimanual arms
    """


class FTSensor:
    """Interface for FT sensor, this can be an internal FT sensor (such as with the UR e-series) or an external sensor."""

    @abstractmethod
    def get_wrench(self) -> WrenchType:
        """Returns the wrench on the TCP frame, so any frame conversions should be done internally."""

    @property
    def wrench_in_tcp_pose(self):
        """Returns the (fixed) transform between the FT sensor frame and the TCP frame

        Raises:
            NotImplementedError: This function this not need to be implemented, as you sometimes don't know (nor need) this transform.
        """
        raise NotImplementedError
