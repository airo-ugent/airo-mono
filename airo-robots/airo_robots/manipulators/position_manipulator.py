from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper
from airo_typing import HomogeneousMatrixType, JointConfigurationType


@dataclass
class ManipulatorSpecs:
    """
    dof: the Degrees of freedom of the robot, can be used to verify the shape of joint configurations
    max_joint_speeds: list of max joint speeds in [rad/s]
    max_linear_speed: an (approximate) maximal linear speed [m/s], since it is hard to test for joint speed limitations on each interpolation step
    """

    max_joint_speeds: List[float]
    max_linear_speed: float

    @property
    def dof(self) -> int:
        """degrees of freedom of the robot"""
        return len(self.max_joint_speeds)


class PositionManipulator(ABC):
    """base class for position-controlled manipulators.

    Commands are asynchronous (i.e. the method returns once the command has been sent to the robot)
    and return an AwaitableAction that can be waited for.

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
    def move_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        """move to a desired pose. This function should be used for 'open-loop' movements as it will
        enforce an open-loop trajectory with a speed profile on the trajectory.

        E.g. you want to grasp an object:
        >move_to_tcp_pose(<pregrasp_pose>)
        >move_to_tcp_pose(<grasp_pose>,<low_speed>)

        Args:
            tcp_pose (HomogeneousMatrixType): desired tcp pose
            joint_speed (Optional[float], optional): speed to use for the movement. Defaults to None, in which case the maximum speed is used.

        Returns:
            AwaitableAction: with termination condition that the robot has reached the desired pose.
        """

    @abstractmethod
    def move_linear_to_tcp_pose(
        self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None
    ) -> AwaitableAction:
        """move to desired pose in a straight line (synchronous). This function should be used for 'open-loop' movements as it will
        enforce an open-loop trajectory with a speed profile on the trajectory.

        Args:
            tcp_pose (HomogeneousMatrixType): desired tcp pose
            linear_speed (Optional[float], optional): speed to use for the movement. Defaults to None, in which case the maximum speed is used.

        Returns:
            AwaitableAction: with termination condition that the robot has reached the desired pose.
        """

    @abstractmethod
    def move_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, joint_speed: Optional[float] = None
    ) -> AwaitableAction:
        """move to a desired joint configuration (synchronous). This function should be used for 'open-loop' movements as it will
        enforce an open-loop trajectory with a speed profile on the trajectory.

        Args:
            joint_configuration (JointConfigurationType): desired joint configuration
            joint_speed (Optional[float], optional): speed to use for the movement. Defaults to None, in which case the maximum speed is used.

        Returns:
            AwaitableAction: with termination condition that the robot has reached the desired joint configuration.
        """

    @abstractmethod
    def servo_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, time: float) -> AwaitableAction:
        """servo to the desired tcp pose for the specified time (the function blocks for this time). Servoing implies 'best-effort' movements towards the target pose instead of
        open-loop trajectories with a velocity profile that brings the robot to zero. So this function can be used for 'closed-loop'/higher-frequency control.
        Note that the motion is not guaranteed to be a straight line in EEF space.

        E.g. for visual servoing or a learning-based policy you could send commands at a certain frequency f
        > servo_to_joint_configuration(<tcp_pose>, 1/f)
        and the robot would do its best to track the tcp pose within the kinematic and dynamical constraints of the
        robot and low-level controllers that are used.

        Be aware that providing 'unrealistic' target poses that the robot cannot track (too far away) could result in very jerky motions.

        Args:
            tcp_pose (HomogeneousMatrixType): desired tcp pose
            time (float): time to reach the desired pose

        Returns:
            AwaitableAction: with termination condition that the time has passed. Waiting on this action has limited accuracy on non real-time OS, cf the airo-robots Readme.

        """

    @abstractmethod
    def servo_to_joint_configuration(self, joint_configuration: JointConfigurationType, time: float) -> None:
        """servo to the desired joint  pose for the specified time (the function blocks for this time). Servoing implies 'best-effort' movements towards the target pose instead of
        open-loop trajectories with a velocity profile that brings the robot to zero. So this function can be used for 'closed-loop'/higher-frequency control.

        E.g. for visual servoing or a learning-based policy you could send commands at a certain frequency f
        > servo_to_joint_configuration(<joint_config>, 1/f)
        and the robot would do its best to track the joint configurations within the kinematic and dynamical constraints of the
        robot and low-level controllers that are used.

        Be aware that providing 'unrealistic' target poses that the robot cannot track (too far away) could result in very jerky motions.

        Args:
            joint_configuration (JointConfigurationType): desired joint configuration
            time (float): time to reach the desired joint configuration

        Returns:
            AwaitableAction: with termination condition that the time has passed. Waiting on this action has limited accuracy on non real-time OS, cf the airo-robots Readme.
        """

    @abstractmethod
    def inverse_kinematics(
        self, tcp_pose: HomogeneousMatrixType, joint_configuration_near: Optional[JointConfigurationType] = None
    ) -> Optional[JointConfigurationType]:
        """Solve inverse kinematics.

        Args:
            tcp_pose (HomogeneousMatrixType): desired tcp pose
            joint_configuration_near (Optional[JointConfigurationType], optional):joint configuration to filter to closest solution,
              since there are usually multiple solutions.
            Defaults to None, in which case the current joint configuration is used.

        Returns:
            Optional[JointConfigurationType]: joint configuration that is closest to the desired tcp pose. None if no solution is found.
        """

    @abstractmethod
    def forward_kinematics(self, joint_configuration: JointConfigurationType) -> HomogeneousMatrixType:
        pass

    def is_tcp_pose_kinematically_reachable(self, tcp_pose: HomogeneousMatrixType) -> bool:
        raise NotImplementedError
