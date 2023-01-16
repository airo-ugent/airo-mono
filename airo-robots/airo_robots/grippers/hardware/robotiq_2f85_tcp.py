import asyncio
import socket
import time
from typing import Optional

import numpy as np
from airo_robots.grippers.hardware.manual_gripper_testing import manual_test_gripper
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper, ParallelPositionGripperSpecs


def rescale_range(x, from_min, from_max, to_min, to_max):
    return to_min + (x - from_min) / (from_max - from_min) * (to_max - to_min)


class Robotiq2F85(ParallelPositionGripper):
    """
    This class implements the appropriate Gripper interface for a Robotiq 2F-85 gripper that is connected to a UR robot with the Robotiq URscript, so that it exposes the TCP API.
    This API is available at TCP port 63352 of the UR controller and wraps the Modbus registers of the gripper, as described in
    https://dof.robotiq.com/discussion/2420/control-robotiq-gripper-mounted-on-ur-robot-via-socket-communication-python.
    The control sequence is gripper motor <---- gripper registers<--ModbusSerial(rs485)-- UR controller <--TCP-- remote control

    This class does 2 things:
    - it communicates over TCP using the above mentioned API to read/write values from/to the gripper's registers.
    - it rescales all those register values into metric units, as required by the gripper interface

    For more info on how to install the URCap and connecting the gripper's RS-485 connection to a UR robot, see the manual, section 4.8
    https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf

    Make sure you can control the gripper using the robot controller (polyscope) before using this script.
    """

    # values obtained from the manual
    # see https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf
    ROBOTIQ_2F85_DEFAULT_SPECS = ParallelPositionGripperSpecs(0.085, 0.0, 220, 25, 0.15, 0.02)

    def __init__(self, host_ip: str, port: int = 63352, fingers_max_stroke: Optional[float] = None):
        """
        host_ip: the IP adress of the robot to which the gripper is connected.

        fingers_max_stroke:
        allow for custom max stroke width (if you have fingertips that are closer together).
        the robotiq will always calibrate such that max opening = 0 and min opening = 255 for its register.
        see manual p42
        """
        self.gripper_specs = self.ROBOTIQ_2F85_DEFAULT_SPECS
        if fingers_max_stroke:
            self.gripper_specs.max_width = fingers_max_stroke
        self.host_ip = host_ip
        self.port = port

        self._check_connection()
        self.activate_gripper()
        super().__init__(self.gripper_specs)

    def get_current_width(self) -> float:
        register_value = int(self._communicate("GET POS").split(" ")[1])
        width = rescale_range(register_value, 0, 230, self.gripper_specs.max_width, self.gripper_specs.min_width)
        return width

    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None):
        if speed:
            self.speed = speed
        if force:
            self.max_grasp_force = force
        self.set_target_width(width)

        while self.is_gripper_moving():
            time.sleep(0.01)

    def is_an_object_grasped(self) -> bool:
        return int(self._communicate("GET OBJ").split(" ")[1]) == 2

    @property
    def speed(self) -> float:
        speed_register_value = self._read_speed_register()
        return rescale_range(speed_register_value, 0, 255, self.gripper_specs.min_speed, self.gripper_specs.max_speed)

    @speed.setter
    def speed(self, value: float):
        speed = np.clip(value, self.gripper_specs.min_speed, self.gripper_specs.max_speed)
        speed_register_value = int(
            rescale_range(speed, self.gripper_specs.min_speed, self.gripper_specs.max_speed, 0, 255)
        )
        self._communicate(f"SET SPE {speed_register_value}")
        while not self._is_target_value_set(speed_register_value, self._read_speed_register()):
            time.sleep(0.01)

    @property
    def max_grasp_force(self) -> float:
        force_register_value = self._read_force_register()
        return rescale_range(force_register_value, 0, 255, self.gripper_specs.min_force, self.gripper_specs.max_force)

    @max_grasp_force.setter
    def max_grasp_force(self, value: float):
        pass

    ## non interface classes
    async def ansyncio_move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None):
        """moves the gripper asynchronously."""
        if speed:
            self.speed = speed
        if force:
            self.max_grasp_force = force
        self.set_target_width(width)

        while self.is_gripper_moving():
            await asyncio.sleep(0.05)

    def set_target_width(self, target_width_in_meters: float):
        """sets a target width asynchronously"""
        target_width_in_meters = np.clip(
            target_width_in_meters, self.gripper_specs.min_width, self.gripper_specs.max_width
        )
        # 230 is 'force closed', cf function below.
        target_width_register_value = int(
            rescale_range(target_width_in_meters, self.gripper_specs.min_width, self.gripper_specs.max_width, 230, 0)
        )
        self._write_target_width_to_register(target_width_register_value)

    def _write_target_width_to_register(self, target_width_register_value: int):
        """
        Takes values in range 0 -255
        3 is actually fully open, 230 is fully closed in operating mode (straight fingers) and 255 is fully closed in encompassed mode.
        For the default fingertips, the range 3 - 230 maps approximately to 0mm - 85mm with a quasi linear relation of 0.4mm / unit
        (experimental findings, but manual also mentions this 0.4mm/unit relation)

        For custom finger tips with a different stroke, it won't be 0.4mm/unit anymore,
        but 230 will still be fully closed due to the self-calibration of the gripper.

        """
        self._communicate(f"SET  POS {target_width_register_value}")
        while not self._is_target_value_set(target_width_register_value, self._read_target_width_register()):
            time.sleep(0.01)

    def _read_target_width_register(self):
        return int(self._communicate("GET PRE").split(" ")[1])

    def _read_speed_register(self):
        return int(self._communicate("GET SPE").split(" ")[1])

    def _read_force_register(self):
        return int()

    def is_gripper_moving(self) -> bool:
        # Moving == 0 => detected OR position reached
        return int(self._communicate("GET OBJ").split(" ")[1]) == 0

    def _check_connection(self):
        """validate communication with gripper is possible.
        Raises:
            ConnectionError
        """

        if not self._communicate("GET STA").startswith("STA"):
            raise ConnectionError("Could not connect to gripper")

    def _communicate(self, command: str) -> str:
        """Helper function to communicate with gripper over a tcp socket.
        Args:
            command (str): The GET/SET command string.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.host_ip, self.port))
                s.sendall(("" + str.strip(command) + "\n").encode())

                data = s.recv(2**10)
                return data.decode()[:-1]
            except Exception as e:
                raise (e)

    def activate_gripper(self):
        """Activates the gripper, sets target position to "Open" and sets GTO flag."""
        self._communicate("SET ACT 1")
        while not self._communicate("GET STA") == "STA 3":
            time.sleep(0.01)

        # initialize gripper
        self._communicate("SET GTO 1")  # enable Gripper
        self.speed = self.gripper_specs.max_speed
        self.force = self.gripper_specs.min_force

    def deactivate_gripper(self):
        self._communicate("SET ACT 0")
        while not self._communicate("GET STA") == "STA 0":
            time.sleep(0.01)

    @staticmethod
    def _is_target_value_set(target: int, value: int) -> bool:
        """helper to compare target value to current value and make the force / speed request synchronous"""
        return abs(target - value) < 5


def get_empirical_data_on_opening_angles(robot_ip: str):
    """used to verify the relation between finger width and register values."""
    gripper = Robotiq2F85(robot_ip)
    gripper._write_target_width_to_register(230)
    input("press")
    for position in [0, 50, 100, 150, 200, 250]:
        gripper._write_target_width_to_register(position)
        input(f"press key for moving to next position {position}")


if __name__ == "__main__":
    robot_ip = "10.42.0.162"  # hardcoded IP of Victor UR3e
    gripper = Robotiq2F85(robot_ip)
    manual_test_gripper(gripper, gripper.gripper_specs)
