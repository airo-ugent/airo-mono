import threading

import loguru
import numpy as np
from airo_robots.manipulators.force_torque_sensor import ForceTorqueSensor
from airo_robots.manipulators.position_manipulator import PositionManipulator, PositionManipulatorDecorator
from airo_spatial_algebra.se3 import SE3Container
from airo_typing import HomogeneousMatrixType
from spatialmath import SO3

logger = loguru.logger


class AdmittanceController(PositionManipulatorDecorator):
    def __init__(self, manipulator: PositionManipulator, ft_sensor: ForceTorqueSensor, control_rate: int):
        super().__init__(manipulator)
        self.robot = manipulator
        self.ft_sensor = ft_sensor
        self.control_rate = control_rate

        # mass spring damper parameter
        self.kp_position = 100
        self.mass_position = 5
        self.kd_position = 2 * np.sqrt(self.kp_position * self.mass_position)  # critical damping

        self.kp_orientation = 20
        self.mass_orientation = 1.0
        self.kd_orientation = 2 * np.sqrt(self.kp_orientation * self.mass_orientation)  # critical damping

        self.tcp_target_pose = None

        # state variables of the controller
        self.control_position = np.zeros(3)
        self.control_orientation = np.eye(3)
        self.control_linear_velocity = np.zeros(3)
        self.control_angular_velocity = np.zeros(3)

    def start(self):
        # start thread and servo at 500Hz
        # to the current setpoint whilst compensating
        # the wrench
        self.control_thread = threading.Thread(target=self.run)
        self.control_thread.start()

    def stop(self):
        # stop the servoing thread
        raise NotImplementedError

    def _update(self, wrench, dt):
        # print(self.tcp_target_pose)
        se3_target_pose = SE3Container.from_homogeneous_matrix(self.tcp_target_pose)
        position_error = self.control_position - se3_target_pose.translation
        # logger.debug(f"position error = {position_error}")
        orientation_error = SE3Container.from_rotation_matrix_and_translation(
            self.control_orientation @ se3_target_pose.rotation_matrix.T
        ).orientation_as_rotation_vector
        # print(orientation_error)
        linear_velocity_error = self.control_linear_velocity
        angular_velocity_error = self.control_angular_velocity

        linear_acc = (
            wrench[:3] - self.kp_position * position_error - self.kd_position * linear_velocity_error
        ) / self.mass_position
        angular_acc = (
            wrench[3:] - self.kp_orientation * orientation_error - self.kd_orientation * angular_velocity_error
        ) / self.mass_orientation

        self.control_linear_velocity += dt * linear_acc
        self.control_angular_velocity += dt * angular_acc

        self.control_position += dt * self.control_linear_velocity
        self.control_orientation = SO3.Exp(self.control_angular_velocity * dt).R @ self.control_orientation
        # logger.debug(f"control position = {self.control_position}")
        # logger.debug(f"control orientation =  \n {self.control_orientation}")

    def update_and_servo(self):
        wrench = self.ft_sensor.get_wrench()
        self._update(wrench, 1 / self.control_rate)
        control_pose = SE3Container.from_rotation_matrix_and_translation(
            self.control_orientation, self.control_position
        ).homogeneous_matrix
        # print(control_pose)
        self.robot.servo_to_tcp_pose(control_pose, 1 / self.control_rate)

    def run(self):
        self._initialize_control_variables()
        while True:
            self.update_and_servo()

    def _initialize_control_variables(self):
        se3 = SE3Container.from_homogeneous_matrix(self.tcp_target_pose)
        self.control_position = np.copy(se3.translation)
        self.control_orientation = np.copy(se3.rotation_matrix)

    def servo_to_tcp_pose(self, tcp_pose: HomogeneousMatrixType, time: float = None):
        del time
        print(f"target pose = {tcp_pose}")
        self.tcp_target_pose = tcp_pose


if __name__ == "__main__":
    from airo_robots.manipulators.hardware.ur_force_torque_rtde import UReForceTorqueSensor
    from airo_robots.manipulators.hardware.ur_rtde import UR_RTDE
    from airo_teleop.game_controller_mapping import LogitechF310Layout
    from airo_teleop.game_controller_teleop import GameControllerTeleop

    ip_address = "10.42.0.162"
    robot = UR_RTDE(ip_address, UR_RTDE.UR3E_CONFIG)
    print(f"robot joint configuration = {robot.get_joint_configuration()}")
    ft_sensor = UReForceTorqueSensor(ip_address)

    controller = AdmittanceController(robot, ft_sensor, 500)
    # controller.tcp_target_pose = robot.get_tcp_pose()
    # controller._initialize_control_variables()
    # for _ in range(5000):
    #     controller.update_and_servo()
    controller.servo_to_tcp_pose(robot.get_tcp_pose())
    controller.start()
    print("starting teleop")
    pose = robot.get_tcp_pose()
    controller.servo_to_tcp_pose(pose)
    game_controller = GameControllerTeleop(controller, 30, LogitechF310Layout)
    game_controller.angular_speed_scaling = 0.01
    game_controller.linear_speed_scaling = 0.01
    game_controller.teleoperate()
