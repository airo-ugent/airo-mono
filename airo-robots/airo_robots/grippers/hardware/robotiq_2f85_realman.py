"""Parallel-position-gripper implementation for a Robotiq 2F-85 connected to the tool RS-485 port of a RealMan arm."""

from typing import TYPE_CHECKING, Any

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers.hardware.robotiq_2f85_urcap import rescale_range
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper, ParallelPositionGripperSpecs
from airo_robots.hardware_interaction_utils import wait_for_condition_with_timeout
from loguru import logger

if TYPE_CHECKING:
    from airo_robots.manipulators.hardware.realman import RealmanControl

# Register map of the Robotiq 2F-85 Modbus RTU interface (see the gripper's instruction manual).
_OUTPUT_REGISTERS_ADDRESS = 0x03E8  # holding registers: action request, position request, speed, force
_INPUT_REGISTERS_ADDRESS = 0x07D0  # input registers: status, fault, position request echo, position, current
_ROBOTIQ_MODBUS_SLAVE_ID = 9
_ROBOTIQ_BAUDRATE = 115200
_TOOL_RS485_PORT = 1  # RealMan end-effector interface board RS-485 port
_TOOL_VOLTAGE_24V = 3

_ACTION_ACTIVATE = 0x01
_ACTION_GO_TO = 0x08
_REGISTER_FULLY_CLOSED = 230


class Robotiq2F85Realman(ParallelPositionGripper):
    """A Robotiq 2F-85 wired to the tool connector of a RealMan arm, driven over Modbus RTU through the arm.

    The RealMan controller acts as Modbus master on the end-effector RS-485 port and forwards register reads and
    writes to the gripper. The tool port is powered with 24V on construction.

    Usage:
        robot = RealmanControl("192.168.1.18")
        robot.gripper = Robotiq2F85Realman(robot)

    Args:
        robot: Connected RealMan manipulator whose tool port the gripper is plugged into.
        fingers_max_stroke: Max opening [m] if you use fingertips that are closer together than the default.
        slave_id: Modbus slave address of the gripper (Robotiq default is 9).
        baudrate: Baudrate of the gripper's RS-485 interface (Robotiq default is 115200).
    """

    ROBOTIQ_2F85_DEFAULT_SPECS = ParallelPositionGripperSpecs(0.085, 0.0, 220, 25, 0.15, 0.02)

    def __init__(
        self,
        robot: "RealmanControl",
        fingers_max_stroke: float | None = None,
        slave_id: int = _ROBOTIQ_MODBUS_SLAVE_ID,
        baudrate: int = _ROBOTIQ_BAUDRATE,
    ) -> None:
        specs = ParallelPositionGripperSpecs(**vars(self.ROBOTIQ_2F85_DEFAULT_SPECS))
        if fingers_max_stroke:
            specs.max_width = fingers_max_stroke
        super().__init__(specs)

        self._robot_api: Any = robot.robot
        self._api: Any = robot._api
        self._slave_id = slave_id

        self._raise_for_error(self._robot_api.rm_set_tool_voltage(_TOOL_VOLTAGE_24V), "rm_set_tool_voltage")
        # Timeout is in units of 100 ms.
        self._raise_for_error(self._robot_api.rm_set_modbus_mode(_TOOL_RS485_PORT, baudrate, 5), "rm_set_modbus_mode")

        self._target_register = 0
        self._speed_register = 255
        self._force_register = 1

        if not self.gripper_is_active():
            self._activate_gripper()
        else:
            self._write_action()

    def get_current_width(self) -> float:
        return rescale_range(
            self._read_position_register(), 0, _REGISTER_FULLY_CLOSED, self._gripper_specs.max_width, 0.0
        )

    def move(self, width: float, speed: float | None = None, force: float | None = None) -> AwaitableAction:
        if speed:
            self._speed_register = self._speed_to_register(speed)
        if force:
            self._force_register = self._force_to_register(force)
        width = float(np.clip(width, self._gripper_specs.min_width, self._gripper_specs.max_width))
        self._target_register = round(
            rescale_range(
                width, self._gripper_specs.min_width, self._gripper_specs.max_width, _REGISTER_FULLY_CLOSED, 0
            )
        )
        self._write_action()
        target = self._target_register

        def move_done_condition() -> bool:
            # The request echo guards against reading the object status of the previous move.
            return self._read_position_request_echo() == target and self._read_object_status() != 0

        return AwaitableAction(move_done_condition)

    def is_an_object_grasped(self) -> bool:
        # gOBJ 2: fingers stopped due to a contact while closing.
        return self._read_object_status() == 2

    @property
    def speed(self) -> float:
        return rescale_range(
            self._speed_register, 0, 255, self._gripper_specs.min_speed, self._gripper_specs.max_speed
        )

    @speed.setter
    def speed(self, value: float) -> None:
        self._speed_register = self._speed_to_register(value)
        self._write_action()

    @property
    def max_grasp_force(self) -> float:
        # Register value 0 has a special meaning (minimum force), so the force range starts at 1.
        return rescale_range(
            self._force_register, 1, 255, self._gripper_specs.min_force, self._gripper_specs.max_force
        )

    @max_grasp_force.setter
    def max_grasp_force(self, value: float) -> None:
        self._force_register = self._force_to_register(value)
        self._write_action()

    def gripper_is_active(self) -> bool:
        return (self._read_status_byte() & 0x30) == 0x30  # gSTA == 3: activation completed

    def close(self) -> None:
        """Release the tool RS-485 port from Modbus mode."""
        self._raise_for_error(self._robot_api.rm_close_modbus_mode(_TOOL_RS485_PORT), "rm_close_modbus_mode")

    ####################
    # Private methods #
    ####################

    def _speed_to_register(self, speed: float) -> int:
        speed = float(np.clip(speed, self._gripper_specs.min_speed, self._gripper_specs.max_speed))
        return int(rescale_range(speed, self._gripper_specs.min_speed, self._gripper_specs.max_speed, 0, 255))

    def _force_to_register(self, force: float) -> int:
        force = float(np.clip(force, self._gripper_specs.min_force, self._gripper_specs.max_force))
        return int(rescale_range(force, self._gripper_specs.min_force, self._gripper_specs.max_force, 1, 255))

    def _activate_gripper(self) -> None:
        logger.info("Activating Robotiq gripper, the fingers will open and close.")
        self._write_registers(_OUTPUT_REGISTERS_ADDRESS, [0, 0, 0, 0, 0, 0])  # reset
        self._write_registers(_OUTPUT_REGISTERS_ADDRESS, [_ACTION_ACTIVATE, 0, 0, 0, 0, 0])
        wait_for_condition_with_timeout(self.gripper_is_active, timeout=10)
        self._write_action()

    def _write_action(self) -> None:
        """Send the activate + go-to request with the cached target, speed and force."""
        self._write_registers(
            _OUTPUT_REGISTERS_ADDRESS,
            [
                _ACTION_ACTIVATE | _ACTION_GO_TO,
                0,
                0,
                self._target_register,
                self._speed_register,
                self._force_register,
            ],
        )

    def _read_status_byte(self) -> int:
        return self._read_input_register(_INPUT_REGISTERS_ADDRESS) >> 8

    def _read_object_status(self) -> int:
        return (self._read_status_byte() >> 6) & 0x03

    def _read_position_request_echo(self) -> int:
        return self._read_input_register(_INPUT_REGISTERS_ADDRESS + 1) & 0xFF

    def _read_position_register(self) -> int:
        return self._read_input_register(_INPUT_REGISTERS_ADDRESS + 2) >> 8

    def _params(self, address: int, num: int = 1) -> Any:
        return self._api.rm_peripheral_read_write_params_t(_TOOL_RS485_PORT, address, self._slave_id, num)

    def _read_input_register(self, address: int) -> int:
        error_code, value = self._robot_api.rm_read_input_registers(self._params(address))
        self._raise_for_error(error_code, "rm_read_input_registers")
        return int(value)

    def _write_registers(self, address: int, data: list[int]) -> None:
        """Write consecutive 16-bit registers; ``data`` holds the bytes (high byte first) of each register."""
        error_code = self._robot_api.rm_write_registers(self._params(address, len(data) // 2), data)
        self._raise_for_error(error_code, "rm_write_registers")

    @staticmethod
    def _raise_for_error(error_code: int, method_name: str) -> None:
        if error_code != 0:
            raise RuntimeError(f"RealMan API call {method_name} failed with error code {error_code}.")


if __name__ == "__main__":
    import click
    from airo_robots.grippers.hardware.manual_gripper_testing import manually_test_gripper_implementation
    from airo_robots.manipulators.hardware.realman import RealmanControl  # noqa: F811

    @click.command()
    @click.option("--robot_ip", default="192.168.1.18", help="IP of the RealMan arm the gripper is connected to.")
    def test_robotiq(robot_ip: str) -> None:
        robot = RealmanControl(robot_ip)
        gripper = Robotiq2F85Realman(robot)
        manually_test_gripper_implementation(gripper, gripper.gripper_specs)

    test_robotiq()
