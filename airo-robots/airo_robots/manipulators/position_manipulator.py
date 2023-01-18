from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper
from airo_typing import HomogeneousMatrixType, JointConfigurationType


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


class PositionManipulator(ABC):
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
        """move to a desired pose (synchronous). This function should be used for 'open-loop' movements as it will
        enforce an open-loop trajectory with a speed profile on the trajectory.

        E.g. you want to grasp an object:
        >move_to_tcp_pose(<pregrasp_pose>)
        >move_to_tcp_pose(<grasp_pose>,<low_speed>)
        """

    @abstractmethod
    def move_linear_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None):
        """move to desired pose in a straight line (synchronous). This function should be used for 'open-loop' movements as it will
        enforce an open-loop trajectory with a speed profile on the trajectory."""

    @abstractmethod
    def move_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, joint_speed: Optional[float] = None
    ):
        """move to a desired joint configuration (synchronous). This function should be used for 'open-loop' movements as it will
        enforce an open-loop trajectory with a speed profile on the trajectory."""

    @abstractmethod
    def servo_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, time: float):
        """servo to the desired tcp pose for the specified time (the function blocks for this time). Servoing implies 'best-effort' movements towards the target pose instead of
        open-loop trajectories, so this function can be used for 'closed-loop'/higher-frequency control.
        Note that the motion is not guaranteed to be a straight line in EEF space.

        E.g. for visual servoing or a learning-based policy you could send commands at a certain frequency f
        > servo_to_joint_configuration(<tcp_pose>, 1/f)
        and the robot would do its best to track the tcp pose within the kinematic and dynamical constraints of the
        robot and low-level controllers that are used.

        Be aware that providing 'unrealistic' target poses that the robot cannot track (too far away) could result in very jerky motions.
        """

    @abstractmethod
    def servo_to_joint_configuration(self, joint_configuration: JointConfigurationType, time: float):
        """servo to the desired joint  pose for the specified time (the function blocks for this time). Servoing implies 'best-effort' movements towards the target pose instead of
        open-loop trajectories, so this function can be used for 'closed-loop'/higher-frequency control.

        E.g. for visual servoing or a learning-based policy you could send commands at a certain frequency f
        > servo_to_joint_configuration(<joint_config>, 1/f)
        and the robot would do its best to track the joint configurations within the kinematic and dynamical constraints of the
        robot and low-level controllers that are used.

        Be aware that providing 'unrealistic' target poses that the robot cannot track (too far away) could result in very jerky motions.
        """

    @abstractmethod
    def inverse_kinematics(
        self, tcp_pose: HomogeneousMatrixType, joint_configuration_near: Optional[JointConfigurationType] = None
    ) -> JointConfigurationType:
        """do inverse kinematics for the TCP pose.
        As there are often multiple solutions to the IK problem, an optional joint configuration can be passed to filter to closest solution to the configuration.
        If none is provided, the closest solution to the current joint configuration will be returned.

        """

    @abstractmethod
    def forward_kinematics(self, joint_configuration: JointConfigurationType) -> HomogeneousMatrixType:
        pass

    def is_tcp_pose_kinematically_reachable(self, tcp_pose: HomogeneousMatrixType) -> bool:
        raise NotImplementedError

    def execute_joint_trajectory(self, joint_trajectory: JointTrajectory):
        """executes a joint trajectory (synchronously) by trying to reach the waypoints at their target times (without coming to a halt at each waypoint)"""
        raise NotImplementedError


class PositionManipulatorDecorator(PositionManipulator):
    """A base class for decorator classes for the Position manipulator class.
    Decorators should inherit from this class and alter the behaviour of the wrapped manipulator instance while adhering to the same interface.
    cf. https://refactoring.guru/design-patterns/decorator for more info on the decorator design pattern.
    Args:
        PositionManipulator (_type_): _description_
    """

    def __init__(self, manipulator: PositionManipulator) -> None:
        super().__init__(manipulator_specs=manipulator.manipulator_specs, gripper=manipulator.gripper)
        self.wrapped_manipulator = manipulator

    def get_tcp_pose(self) -> HomogeneousMatrixType:
        return self.wrapped_manipulator.get_tcp_pose()

    def get_joint_configuration(self) -> JointConfigurationType:
        return super().get_joint_configuration()

    def move_linear_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, linear_speed: Optional[float] = None):
        return self.wrapped_manipulator.move_linear_to_tcp_pose(tcp_pose, linear_speed)

    def move_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, joint_speed: Optional[float] = None):
        return self.wrapped_manipulator.move_to_tcp_pose(tcp_pose, joint_speed)

    def move_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, joint_speed: Optional[float] = None
    ):
        return self.wrapped_manipulator().move_to_joint_configuration(joint_configuration, joint_speed)

    def servo_to_joint_configuration(self, joint_configuration: JointConfigurationType, time: float):
        return self.wrapped_manipulator.servo_to_joint_configuration(joint_configuration, time)

    def servo_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, time: float):
        return self.wrapped_manipulator.servo_to_tcp_pose(tcp_pose, time)

    def forward_kinematics(self, joint_configuration: JointConfigurationType) -> HomogeneousMatrixType:
        return self.wrapped_manipulator.forward_kinematics(joint_configuration)

    def inverse_kinematics(
        self, tcp_pose: HomogeneousMatrixType, joint_configuration_near: Optional[JointConfigurationType] = None
    ) -> JointConfigurationType:
        return self.wrapped_manipulator.inverse_kinematics(tcp_pose, joint_configuration_near)
