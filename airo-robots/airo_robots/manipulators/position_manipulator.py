import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper
from airo_typing import HomogeneousMatrixType, JointConfigurationType, SingleArmTrajectory


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
        self._manipulator_specs = manipulator_specs
        self._gripper = gripper
        self._default_linear_speed = 0.1  # m/s often a good default value
        self._default_joint_speed = min(manipulator_specs.max_joint_speeds) / 4

    @property
    def manipulator_specs(self) -> ManipulatorSpecs:
        return self._manipulator_specs

    @property
    def gripper(self) -> Optional[ParallelPositionGripper]:
        return self._gripper

    @gripper.setter
    def gripper(self, gripper: ParallelPositionGripper) -> None:
        self._gripper = gripper

    @property
    def default_linear_speed(self) -> float:
        """the linear speed to use in move_linear_to_tcp_pose if no speed is specified."""
        return self._default_linear_speed

    @default_linear_speed.setter
    def default_linear_speed(self, speed: float) -> None:
        assert speed <= self._manipulator_specs.max_linear_speed
        self._default_linear_speed = speed

    @property
    def default_joint_speed(self) -> float:
        """the leading-axis joint speed to use in move_to_joint_configuration or move_to_tcp_pose if no speed is specified."""
        return self._default_joint_speed

    @default_joint_speed.setter
    def default_joint_speed(self, speed: float) -> None:
        assert speed <= min(self._manipulator_specs.max_joint_speeds)
        self._default_joint_speed = speed

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
            joint_speed (Optional[float], optional): speed to use for the movement. Defaults to None, in which case the default linear speed is used.

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
            linear_speed (Optional[float], optional): speed to use for the movement. Defaults to None, in which case the default speed is used.

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
            joint_speed (Optional[float], optional): speed to use for the movement. Defaults to None, in which case the default speed is used.

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
    def servo_to_joint_configuration(
        self, joint_configuration: JointConfigurationType, time: float
    ) -> AwaitableAction:
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

    @abstractmethod
    def _is_joint_configuration_reachable(self, joint_configuration: JointConfigurationType) -> bool:
        """Is the joint configuration reachable by the robot? Usually comes down to checking joint limits,
        but additional constraints could be set in the controller (such as safety planes)"""

    def is_tcp_pose_reachable(self, tcp_pose: HomogeneousMatrixType) -> bool:
        """Is the TCP pose reachable by the robot?
        Default implementation uses inverse kinematics to check if the joint configuration is reachable. But this could be overridden in hardware implementations
        if a more suitable method is offered by the robot controller."""
        joint_configuration = self.inverse_kinematics(tcp_pose)
        if joint_configuration is None:
            return False
        return self._is_joint_configuration_reachable(joint_configuration)

    def execute_trajectory(self, joint_trajectory: SingleArmTrajectory) -> None:
        """Execute a joint trajectory. This function will interpolate the trajectory and send the commands to the robot.

        Args:
            joint_trajectory: the joint trajectory to execute."""
        self._assert_joint_trajectory_is_executable(joint_trajectory)

        period = 0.005  # Time per servo, approximately. This may be slightly changed because of rounding errors.
        # The period determines the times at which we sample the trajectory that was time-parameterized.
        duration = (joint_trajectory.times[-1] - joint_trajectory.times[0]).item()

        n_servos = int(np.ceil(duration / period))
        period_adjusted = duration / n_servos  # can be slightly different from period due to rounding

        for t in np.linspace(0, duration, n_servos):
            # Find the two joint configurations that are closest to time t.
            i0, i1 = 0, 0
            for i, trajectory_time in enumerate(joint_trajectory.times):
                if trajectory_time >= t:
                    i0 = i - 1
                    i1 = i
                    break

            # Interpolate between the two joint configurations.
            q0 = joint_trajectory.path.positions[i0]
            q1 = joint_trajectory.path.positions[i1]
            q_interp = q0 + (q1 - q0) * (t - joint_trajectory.times[i0]) / (
                joint_trajectory.times[i1] - joint_trajectory.times[i0]
            )
            self.servo_to_joint_configuration(q_interp, period_adjusted)
            # We do not wait for the servo to finish, because we want to sample the trajectory at a fixed rate and avoid lagging.

            if joint_trajectory.gripper_path is not None:
                gripper_q0 = joint_trajectory.gripper_path.positions[i0]
                gripper_q1 = joint_trajectory.gripper_path.positions[i1]
                gripper_q_interp = gripper_q0 + (gripper_q1 - gripper_q0) * (t - joint_trajectory.times[i0]) / (
                    joint_trajectory.times[i1] - joint_trajectory.times[i0]
                )
                self.gripper.move(gripper_q_interp)

            time.sleep(period_adjusted)

        # This avoids the abrupt stop and "thunk" sounds at the end of paths that end with non-zero velocity
        # However, I believe these functions are blocking, so right only stops after left has stopped.
        self.rtde_control.servoStop(2.0)

        # Servo can overshoot. Do a final move to the last configuration.
        self._assert_joint_configuration_nearby(joint_trajectory.path.positions[-1])
        self.move_to_joint_configuration(joint_trajectory.path.positions[-1]).wait()

        # TODO: BimanualPositionManipulator: how can we assert code reuse?

    ###################################
    # util functions to validate inputs
    ###################################
    def _assert_linear_speed_is_valid(self, linear_speed: float) -> None:
        if not linear_speed <= self.manipulator_specs.max_linear_speed:
            raise ValueError(
                f"linear speed {linear_speed} is too high. Max linear speed is {self.manipulator_specs.max_linear_speed}"
            )

    def _assert_joint_speed_is_valid(self, joint_speed: float) -> None:
        if not joint_speed <= min(self.manipulator_specs.max_joint_speeds):
            raise ValueError(
                f"joint speed {joint_speed} is too high. Max joint speeds are {self.manipulator_specs.max_joint_speeds}"
            )

    def _assert_pose_is_valid(self, pose: HomogeneousMatrixType) -> None:
        if not self.is_tcp_pose_reachable(pose):
            raise ValueError(
                f"pose {pose} is not reachable, could be because of kinematic constraints or safety constraints"
            )

    def _assert_joint_configuration_is_valid(self, joint_configuration: JointConfigurationType) -> None:
        if not self._is_joint_configuration_reachable(joint_configuration):
            raise ValueError(
                f"joint configuration {joint_configuration} is not reachable, could be because of kinematic constraints or safety constraints"
            )

    def _assert_joint_configuration_nearby(
        self, joint_configuration: JointConfigurationType, absolute_angle_tolerance=np.radians(1.0)
    ) -> None:
        """Assert that a joint configuration is nearby the current configuration.

        Args:
            joint_configuration: the configuration that should be nearby the current configuration.
            absolute_angle_tolerance: the absolute tolerance for the comparison.

        Raises:
            ValueError: If the joint configuration is not nearby the current configuration."""
        current_configuration = self.get_joint_configuration()
        if (
            not np.isclose(joint_configuration, current_configuration, atol=absolute_angle_tolerance, rtol=0.0)
            .all()
            .item()
        ):
            raise ValueError(
                f"joint configuration {joint_configuration} is not nearby the current configuration {current_configuration}"
            )

    def _assert_joint_trajectory_start_time_is_zero(self, joint_trajectory: SingleArmTrajectory) -> None:
        if joint_trajectory.times[0] != 0.0:
            raise ValueError("joint trajectory should start at time 0.0")

    def _assert_joint_trajectory_is_executable(self, joint_trajectory: SingleArmTrajectory) -> None:
        self._assert_joint_trajectory_start_time_is_zero(joint_trajectory)

        if joint_trajectory.path.positions is None:
            raise ValueError("joint trajectory should contain joint positions")

        self._assert_joint_configuration_nearby(joint_trajectory.path.positions[0])

        if joint_trajectory.path.velocities is not None:
            for velocity in joint_trajectory.path.velocities:
                self._assert_joint_speed_is_valid(velocity)
