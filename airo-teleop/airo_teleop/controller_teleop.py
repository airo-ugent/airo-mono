from typing import Optional

import numpy as np
import pygame
from airo_robots.manipulators.position_manipulator import PositionManipulator
from airo_spatial_algebra.se3 import SE3Container
from airo_teleop.game_controller_mapping import GameControllerLayout
from airo_typing import HomogeneousMatrixType, TwistType
from pygame import joystick
from spatialmath import SO3


class GameControllerTeleop:
    # TODO: (optional) add ability to 'lock' dimensions to avoid drift

    def __init__(
        self,
        robot: PositionManipulator,
        control_rate: int,
        controller_layout: GameControllerLayout,
        joystick_id: int = 0,
    ) -> None:
        pygame.init()
        joystick.init()
        assert joystick_id < joystick.get_count()
        self.controller = joystick.Joystick(joystick_id)
        self.controller_layout = controller_layout
        self.robot = robot
        self.control_rate = control_rate

        # you can set these by addressing the attributes.
        self.linear_speed_scaling = 0.2  # m/s
        self.angular_speed_scaling = 0.6  # rad/s
        self.gripper_delta_step_size = 0.01  # m/step

        # get the 'bias' of the controller,
        # or the axes values when the controller is in 'rest'.
        # this is later used to avoid drift of the manipulator
        self.controller_twist_bias = self.get_twist()

    def get_twist(self) -> TwistType:
        """Get a twist from the game controller inputs. The 'meaning' of the different axes is fixed.
        The controller inputs are interpreted as a linear velocity in the robot base frame and an angular velocity as this
        is more intuitive to control.

        Return:
            A twist that represents the spatial velocity of the TCP in the robot base frame.
        """
        self._get_pygame_events()

        twist = np.zeros(6)
        twist[0] = self.controller.get_axis(self.controller_layout.left_joy_horizontal_index)
        twist[1] = self.controller.get_axis(self.controller_layout.left_joy_vertical_axis_index) * (-1)  # revert axis
        twist[2] = self.controller.get_axis(self.controller_layout.right_joy_vertical_axis_index) * (-1)  # revert axis
        twist[5] = self.controller.get_axis(self.controller_layout.right_joy_horizontal_axis_index)

        # read out the 'cross', which is called hat in pygame
        hat = self.controller.get_hat(0)
        twist[4] = hat[0]
        twist[3] = hat[1]

        # get linear & angular velocity by scaling the respective part of the 'twist' vector
        linear_velocity_in_base_frame = twist[:3] * self.linear_speed_scaling
        angular_velocity_in_tcp_frame = twist[3:] * self.angular_speed_scaling
        print(f"controller twist = {twist}")
        # convert rotations from the tcp frame to the base frame..
        # which requires the robot's TCP pose to compute the adjoint matrix to convert the 'expressed-in' frame of the twist.
        # since we want to convert angular velocity with zero linear velocity, we can simply multiply the angular velocity
        # with the rotation matrix of the transform between the base frame and the tcp frame instead of using its full Adjoint matrix
        # see Modern Robotics, Lynch et Al., Ch 3.3.2
        tcp_pose_in_base_frame = self.robot.get_tcp_pose()
        angular_velocity_in_base_frame = (
            SE3Container.from_homogeneous_matrix(tcp_pose_in_base_frame).rotation_matrix
            @ angular_velocity_in_tcp_frame
        )
        tcp_twist_in_base_frame = np.concatenate([linear_velocity_in_base_frame, angular_velocity_in_base_frame])
        return tcp_twist_in_base_frame

    def get_gripper_delta(self) -> float:
        """using the LB/RB buttons, create a delta step for the gripper."""
        self._get_pygame_events()
        step = 0.0
        if self.controller.get_button(self.controller_layout.lb_button_index):
            step = -1.0
        elif self.controller.get_button(self.controller_layout.rb_button_index):
            step = 1.0
        return step * self.gripper_delta_step_size

    def calculate_new_target_position(self, tcp_twist_in_base_frame: TwistType) -> HomogeneousMatrixType:
        """Takes the twist in the base  frame and uses that to compute the new target pose to servo to.
        This requires some Spatial (Lie) Algebra to convert a twist to a transform."""
        tcp_pose_in_base_frame = self.robot.get_tcp_pose()
        # print(f"tcp pose = \n {tcp_pose_in_base_frame}")
        se3_tcp_pose_in_base_frame = SE3Container.from_homogeneous_matrix(tcp_pose_in_base_frame)
        # apply the delta translation
        target_translation = se3_tcp_pose_in_base_frame.translation + tcp_twist_in_base_frame[:3]
        # apply the delta rotation
        # note: cannot just add the angular velocity to the rotation vector,
        # because that is not equivalent to applying the 'twist' to the current pose.
        # have to construct the rotation matrix from the angular velocity (using the matrix exponentional to convert from so2 to SO2)
        # and then left-multipy the pose's rotation matrix with this matrix.
        twist_rotation_matrix = SO3.Exp(tcp_twist_in_base_frame[3:]).R
        target_orientation_as_rotation_matrix = twist_rotation_matrix @ se3_tcp_pose_in_base_frame.rotation_matrix

        return SE3Container.from_rotation_matrix_and_translation(
            target_orientation_as_rotation_matrix, target_translation
        ).homogeneous_matrix

    def read_twist_and_servo_to_target_position(self) -> TwistType:
        """gets the twists, converts them to new target pose for the robot and servos to that pose in  (1/control_rate) seconds

        Return: relative motion that was applied.
        """

        twist = self.get_twist()
        twist -= self.controller_twist_bias
        relative_motion = twist / self.control_rate
        # print(f"relative motion twist = {relative_motion}")
        tcp_target_pose = self.calculate_new_target_position(relative_motion)
        # print(f"tcp target pose = \n {tcp_target_pose}")
        self.robot.servo_to_tcp_pose(tcp_target_pose, 1 / self.control_rate)
        return relative_motion

    def read_gripper_delta_and_move_gripper(self) -> Optional[float]:
        """gets the delta step for the gripper, and moves the gripper synchronously to the new opening width.
        Gripper will have shaky behaviour as its internal controller will come to a halt in each cycle (no servoing)."""
        if not self.robot.gripper:
            return None

        delta = self.get_gripper_delta()
        # print(f"gripper delta = {delta}")
        self.robot.gripper.move(self.robot.gripper.get_current_width() + delta)
        return delta

    def teleoperate(self):
        """Starts streaming servo commands based on the controller input, runs untill stopped with CTRL+C."""

        while True:
            self.read_twist_and_servo_to_target_position()
            # TODO: control rate will be slower than target control rate due to gripper control.
            # TODO: this gripper movement takes variable amount of time so control rate is not constant
            self.read_gripper_delta_and_move_gripper()

    def _get_pygame_events(self):
        """update the pygame events, used to read out the current axis/button/hat values"""
        for _ in pygame.event.get():
            pass
