from dataclasses import dataclass
from typing import List

import numpy as np
import pygame
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_spatial_algebra.se3 import SE3Container
from pygame import joystick
from spatialmath import SO3


@dataclass
class AxisConfig:
    axis_index: int
    twist_index: int
    revert: bool = False


@dataclass
class HatConfig:
    horizontal_twist_index: int
    vertical_twist_index: int


@dataclass
class JoystickConfig:
    axes: List[AxisConfig]
    hat: HatConfig


@dataclass
class ControllerLayout:
    left_joy_horizontal_index: int
    left_joy_vertical_axis_index: int
    right_joy_horizontal_axis_index: int
    right_joy_vertical_axis_index: int
    lt_axis_index: int
    rt_axis_index: int

    lb_button_index: int
    rb_button_index: int
    a_button_index: int
    b_button_index: int
    y_button_index: int
    x_button_index: int


XBox360Layout = ControllerLayout(
    left_joy_horizontal_index=0,
    left_joy_vertical_axis_index=1,
    right_joy_horizontal_axis_index=3,
    right_joy_vertical_axis_index=4,
    lt_axis_index=2,
    rt_axis_index=5,
    lb_button_index=4,
    rb_button_index=5,
    a_button_index=0,
    b_button_index=1,
    y_button_index=3,
    x_button_index=2,
)


class Teleop:
    def __init__(self, robot, control_rate) -> None:
        pass

    def get_twist(self):
        """using the 2 joystick buttons and the hat, determine the twist."""

    def get_gripper_delta(
        self,
    ):
        """using the R buttons, determine the relative opening of the gripper."""


class JoystickTeleop(Teleop):
    def __init__(
        self, robot: PositionManipulator, control_rate: int, config: JoystickConfig, joystick_id: int = 0
    ) -> None:
        pygame.init()
        joystick.init()
        assert joystick_id < joystick.get_count()
        self.controller = joystick.Joystick(joystick_id)
        self.config = config
        self.robot = robot
        self.control_rate = control_rate
        # TODO: validate the configuration

        self.linear_speed_scaling = 0.2  # m/s
        self.angular_speed_scaling = 0.6  # rad/s
        self.gripper_delta_step_size = 0.01  # mm/step

        self.controller_twist_bias = self.get_twist()

    def get_twist(self):
        # TODO: (optional) add ability to 'lock' dimensions to avoid drift

        self._get_pygame_events()
        twist = np.zeros(6)
        for axis in self.config.axes:
            twist[axis.twist_index] = self.controller.get_axis(axis.axis_index) * ((-1) ** (1 * axis.revert))

        hat = self.controller.get_hat(0)
        twist[self.config.hat.horizontal_twist_index] = hat[0]
        twist[self.config.hat.vertical_twist_index] = hat[1]

        # get linear & angular velocity by scaling the respective part of the 'twist' vector
        linear_velocity_in_base_frame = twist[:3] * self.linear_speed_scaling
        angular_velocity_in_tcp_frame = twist[3:] * self.angular_speed_scaling
        print(f"controller twist = {twist}")
        # convert rotations from the tcp frame to the base frame..
        # which requires the robot's TCP pose to compute the adjoint matrix to convert the 'expressed-in' frame of the twist.
        # since we want to convert angular velocity with zero linear velocity, we can simply multiply the angular velocity
        # with the rotation matrix of the transform between the base frame and the tcp frame.
        # see Modern Robotics, Lynch et Al., Ch 3.3.2
        tcp_pose_in_base_frame = self.robot.get_tcp_pose()
        angular_velocity_in_base_frame = (
            SE3Container.from_homogeneous_matrix(tcp_pose_in_base_frame).rotation_matrix
            @ angular_velocity_in_tcp_frame
        )
        tcp_twist_in_base_frame = np.concatenate([linear_velocity_in_base_frame, angular_velocity_in_base_frame])
        return tcp_twist_in_base_frame

    def get_gripper_delta(self):
        self._get_pygame_events()
        step = 0.0
        if self.controller.get_button(XBox360Layout.lb_button_index):
            step = -1.0
        elif self.controller.get_button(XBox360Layout.rb_button_index):
            step = 1.0
        return step * self.gripper_delta_step_size

    def calculate_new_target_position(self, tcp_twist_in_base_frame):
        tcp_pose_in_base_frame = self.robot.get_tcp_pose()
        # print(f"tcp pose = \n {tcp_pose_in_base_frame}")
        se3_tcp_pose_in_base_frame = SE3Container.from_homogeneous_matrix(tcp_pose_in_base_frame)
        # apply the delta translation
        target_translation = se3_tcp_pose_in_base_frame.translation + tcp_twist_in_base_frame[:3]
        # apply the delta rotation
        # note: cannot just add the angular velocity to the rotation vector!
        # TODO: explain why.

        twist_rotation_matrix = SO3.Exp(tcp_twist_in_base_frame[3:]).R
        target_orientation_as_rotation_matrix = twist_rotation_matrix @ se3_tcp_pose_in_base_frame.rotation_matrix

        return SE3Container.from_rotation_matrix_and_translation(
            target_orientation_as_rotation_matrix, target_translation
        ).homogeneous_matrix

    def read_twist_and_servo_to_target_position(self):
        """gets the twists, converts them to new target pose for the robot and servos to that pose in  1/control_rate seconds"""

        twist = self.get_twist()
        twist -= self.controller_twist_bias

        relative_motion = twist / self.control_rate
        # print(f"relative motion twist = {relative_motion}")
        tcp_target_pose = self.calculate_new_target_position(relative_motion)
        # print(f"tcp target pose = \n {tcp_target_pose}")
        self.robot.servo_to_tcp_pose(tcp_target_pose, 1 / self.control_rate)
        return relative_motion

    def read_gripper_delta_and_move_gripper(self):
        if not self.robot.gripper:
            return

        delta = self.get_gripper_delta()
        print(f"gripper delta = {delta}")
        self.robot.gripper.move(self.robot.gripper.get_current_width() + delta)
        return delta

    def teleoperate(self):

        while True:
            self.read_twist_and_servo_to_target_position()
            # TODO: control rate will be slower than target control rate due to gripper control.
            # TODO: this gripper movement takes variable amount of time so control rate is not constant
            self.read_gripper_delta_and_move_gripper()

    def _get_pygame_events(self):
        """update the pygame events, used to read out the current axis/button/hat values."""
        for _ in pygame.event.get():
            pass


if __name__ == "__main__":
    from airo_robots.grippers.hardware.robotiq_2f85_tcp import Robotiq2F85
    from airo_robots.manipulators.hardware.ur_rtde import UR3E_CONFIG, UR_RTDE

    robot_ip = "10.42.0.162"
    robot = UR_RTDE(robot_ip, UR3E_CONFIG)
    gripper = Robotiq2F85(robot_ip)
    robot.gripper = gripper

    xbox_config = JoystickConfig(
        [
            AxisConfig(XBox360Layout.left_joy_horizontal_index, 0, False),
            AxisConfig(XBox360Layout.left_joy_vertical_axis_index, 1, True),
            AxisConfig(XBox360Layout.right_joy_horizontal_axis_index, 5),
            AxisConfig(XBox360Layout.right_joy_vertical_axis_index, 2, True),
        ],
        HatConfig(3, 4),
    )

    joystick_teleop = JoystickTeleop(robot, 30, xbox_config)
    joystick_teleop.teleoperate()
