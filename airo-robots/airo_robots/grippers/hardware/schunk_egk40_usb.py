import time
from enum import Enum
from typing import Optional

import numpy as np
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper, ParallelPositionGripperSpecs
from bkstools.bks_lib.bks_module import BKSModule
from bkstools.scripts.bks_grip import WaitGrippedOrError
from pyschunk.generated.generated_enums import eCmdCode


def rescale_range(x: float, from_min: float, from_max: float, to_min: float, to_max: float) -> float:
    return to_min + (x - from_min) / (from_max - from_min) * (to_max - to_min)


class SCHUNK_STROKE_OPTIONS(Enum):
    DEFAULT = 0
    CALIBRATE = 1


class SchunkEGK40_USB(ParallelPositionGripper):
    """
    README: The Schunk gripper differs from the Robotiq in a couple ways.
    (1) It does not have an automatic width calibration to account for custom fingertips. A custom
    calibrate_width() routine is provided in this class, but given the Schunk's high minimum gripping force, it is
    recommended to manually set the max_stroke_setting in the constructor appropriately (refer to docstring in
    constructor for further explanation).
    (2) The connection to the Schunk gripper requires constant synchronisation, meaning that the connection is lost if
    your main code executes any other code for more than a couple seconds. For example, time.sleep() would lose the
    connection. To account for this,
    bkstools.bks_lib.bks_base provides functions such as keep_communication_alive_sleep, as a wrapper around time.sleep.
    This class handles it differently, making use of the BKSModule.MakeReady() cmd. This command will "refresh" the
    connection to the Schunk. It is hence executed before every move() command. However, it takes about 30ms to complete,
    so in addition servo() commands are provided: first call servo_start(), which will execute MakeReady(), then use
    servo() in a loop. MakeReady() doesn't have to be executed in the loop, since the movement commands themselves
    keep the communication alive.
    (3) bks_modbus OVERWRITES Python's serial.Serial.read() function which is absolutely ludicrous, a first issue that
    was discovered is that when you do serial.Serial.read(size=num_bytes), i.e. passing size as a keyword argument,
    this is not handled properly. serial.Serial.read(num_bytes), i.e. no keyword argument, works.
    """

    # values obtained from https://schunk.com/be/nl/grijpsystemen/parallelgrijper/egk/egk-40-mb-m-b/p/000000000001491762
    SCHUNK_DEFAULT_SPECS = ParallelPositionGripperSpecs(0.083, 0.0, 150, 55, 0.0575, 0.0055)

    def __init__(
        self, usb_interface: str = "/dev/ttyUSB0", max_stroke_setting: Optional[SCHUNK_STROKE_OPTIONS | float] = None
    ) -> None:
        """
        :param usb_interface: the USB interface to which the gripper is connected.
        :param fingers_max_stroke: custom max stroke width; this is twice the distance traveled by each finger when
        moving from fully closed to fully opened, which is relevant for custom fingertips.
        If fingers_max_stroke is set to SCHUNK_STROKE_OPTIONS.CALIBRATE, the Schunk will calibrate itself.
        If SCHUNK_STROKE_OPTIONS.DEFAULT, the default values are kept.
        TODO: this class currently doesn't support fingertips that, when fully opened, would have a larger width than
        the Schunk's maximum width of 83mm. Such fingertips would not touch when the Schunk is "closed".
        """
        super().__init__(self.SCHUNK_DEFAULT_SPECS)
        if isinstance(max_stroke_setting, SCHUNK_STROKE_OPTIONS):
            if max_stroke_setting == SCHUNK_STROKE_OPTIONS.DEFAULT:
                # keep default width setting
                pass
            elif max_stroke_setting == SCHUNK_STROKE_OPTIONS.CALIBRATE:
                self.calibrate_width()
            else:
                raise ValueError("Incorrect stroke setting.")
        else:
            if not isinstance(max_stroke_setting, float):
                raise ValueError("Incorrect stroke setting: must be float if not in SCHUNK_STROKE_OPTIONS.")
            else:
                # overwrite width setting
                self._gripper_specs.max_width = max_stroke_setting
        self.width_loss_due_to_fingers = self.SCHUNK_DEFAULT_SPECS.max_width - self.gripper_specs.max_width

        self.bks = BKSModule(usb_interface, sleep_time=None, debug=False)

        # Prepare gripper: Acknowledge any pending error:
        self.bks.command_code = eCmdCode.CMD_ACK
        self.bks.MakeReady()
        time.sleep(0.1)

    @property
    def speed(self) -> float:
        """return speed setting [m/s]."""
        return self.bks.set_vel / 1000

    @speed.setter
    def speed(self, new_speed: float) -> None:
        """set the max speed [m/s]."""
        _new_speed = np.clip(new_speed, self.gripper_specs.min_speed, self.gripper_specs.max_speed)
        self.bks.set_vel = float(_new_speed * 1000)  # value must be set in mm/s for bkstools

    @property
    def current_speed(self) -> float:
        """return current speed [m/s]."""
        return self.bks.actual_vel / 1000

    @property
    def max_grasp_force(self) -> float:
        _force = rescale_range(self.bks.set_force, 0, 100, self.gripper_specs.min_force, self.gripper_specs.max_force)
        return _force

    @max_grasp_force.setter
    def max_grasp_force(self, new_force: float) -> None:
        """set the max grasping force [N]."""
        _new_force = np.clip(new_force, self.gripper_specs.min_force, self.gripper_specs.max_force)
        _new_force = rescale_range(_new_force, self.gripper_specs.min_force, self.gripper_specs.max_force, 0, 100)
        self.bks.set_force = _new_force

    @property
    def current_motor_current(self) -> float:
        """return current motor current usage [unit?], this is a proxy for force."""
        return self.bks.actual_cur

    def get_current_width(self) -> float:
        """get the current opening of the fingers in meters"""
        # Reasoning:
        # width_without_fingers = self.SCHUNK_DEFAULT_SPECS.max_width - self.bks.actual_pos / 1000
        # return width_without_fingers - self.width_loss_due_to_fingers
        # The above equates to:
        return self.gripper_specs.max_width - (self.bks.actual_pos / 1000)

    def move(
        self,
        width: float,
        speed: Optional[float] = SCHUNK_DEFAULT_SPECS.min_speed,
        force: Optional[float] = SCHUNK_DEFAULT_SPECS.min_force,
        set_speed_and_force: bool = True,
    ) -> AwaitableAction:
        """
        Move the gripper to a certain position at a certain speed with a certain force. This function is assumed to run
        when some time has passed since the last communication with the Schunk gripper, meaning self.bks.MakeReady()
        must be called.
        :param width: in m
        :param speed: in m/s
        :param force: in N
        :param set_speed_and_force: setting to false can improve control frequency as less transactions have to happen with the gripper
        """
        self.bks.MakeReady()
        if speed:
            self.speed = speed
        if force:
            self.max_grasp_force = force
        return self.servo(
            width=width, speed=self.speed, force=self.max_grasp_force, set_speed_and_force=set_speed_and_force
        )

    def move_relative(
        self,
        width_difference: float,
        speed: float = SCHUNK_DEFAULT_SPECS.min_speed,
        force: float = SCHUNK_DEFAULT_SPECS.min_force,
        set_speed_and_force: bool = True,
    ) -> AwaitableAction:
        """
        Move the gripper to a certain position at a certain speed with a certain force. This function is assumed to run
        when some time has passed since the last communication with the Schunk gripper, meaning self.bks.MakeReady()
        must be called.
        :param width_difference: in m,  a positive difference will make the gripper open, a negative difference makes it close
        :param speed: in m/s
        :param force: in N
        :param set_speed_and_force: setting to false can improve control frequency as less transactions have to happen with the gripper
        """
        self.bks.MakeReady()
        return self.servo_relative(
            width_difference=width_difference, speed=speed, force=force, set_speed_and_force=set_speed_and_force
        )

    def servo_start(self) -> None:
        """
        Necessary to run before entering a servo loop using the SchunkEGK40_USB.servo() or SchunkEGK40_USB.servo_relative()
        command. No servo_stop() is required.
        :return:
        """
        self.bks.MakeReady()

    def servo(
        self,
        width: float,
        speed: float = SCHUNK_DEFAULT_SPECS.min_speed,
        force: float = SCHUNK_DEFAULT_SPECS.min_force,
        set_speed_and_force: bool = True,
    ) -> AwaitableAction:
        """
        Move the gripper to a certain position at a certain speed with a certain force. This function is assumed to run
        in a loop, meaning repeated calls of self.bks.MakeReady() (taking 31 ms each) are not necessary. Call servo_start()
        before entering the loop.
        :param width: in m
        :param speed: in m/s
        :param force: in N
        :param set_speed_and_force: setting to false can improve control frequency as less transactions have to happen with the gripper
        """
        if set_speed_and_force:
            self.speed = speed
            self.max_grasp_force = force
        _width = np.clip(width, self.gripper_specs.min_width, self.gripper_specs.max_width)
        # Reasoning:
        # width_without_fingers = _width + self.width_loss_due_to_fingers
        # self.bks.set_pos = (self.SCHUNK_DEFAULT_SPECS.max_width - width_without_fingers) * 1000
        # The above equates to:
        self.bks.set_pos = (self.gripper_specs.max_width - _width) * 1000
        self.bks.command_code = eCmdCode.MOVE_POS

        return AwaitableAction(self._move_done_condition)

    def servo_relative(
        self,
        width_difference: float,
        speed: float = SCHUNK_DEFAULT_SPECS.min_speed,
        force: float = SCHUNK_DEFAULT_SPECS.min_force,
        set_speed_and_force: bool = True,
    ) -> AwaitableAction:
        """
        Move the gripper to a certain position at a certain speed with a certain force. This function is assumed to run
        in a loop, meaning repeated calls of self.bks.MakeReady() (taking 31 ms each) are not necessary. Call servo_start()
        before entering the loop.
        :param width_difference: in m,  a positive difference will make the gripper open, a negative difference makes it close
        :param speed: in m/s
        :param force: in N
        :param set_speed_and_force: setting to false can improve control frequency as less transactions have to happen
        with the gripper. The reason to do this with an extra argument is so that the speed and force settings are
        minimal by default, which is desirable for the strong Schunk gripper.
        """
        if set_speed_and_force:
            self.speed = speed
            self.max_grasp_force = force
        self.bks.set_pos = -width_difference * 1000
        self.bks.command_code = eCmdCode.MOVE_POS_REL

        return AwaitableAction(self._move_done_condition)

    def grip(self) -> AwaitableAction:
        """
        Move the gripper until object contact. When using MOVE_FORCE as below, it seems that the force and speed must
        be set to 50% and 0 mm/s respectively.
        """
        self.bks.set_force = 50  # target force to 50 % => BasicGrip
        self.bks.set_vel = 0.0  # target velocity 0 => BasicGrip
        self.bks.grp_dir = True  # grip from outside
        self.bks.command_code = (
            eCmdCode.MOVE_FORCE
        )  # (for historic reasons the actual grip command for simple gripping is called MOVE_FORCE...)
        WaitGrippedOrError(self.bks)

        return AwaitableAction(self._move_done_condition)

    def stop(self) -> None:
        """
        TODO: figure out difference with fast_stop()
        """
        self.bks.command_code = eCmdCode.CMD_STOP

    def fast_stop(self) -> None:
        """
        TODO: figure out difference with stop()
        """
        self.bks.command_code = eCmdCode.CMD_FAST_STOP

    def calibrate_width(self) -> None:
        """
        Robotiq-like calibration to detect the gripper position where the (custom) fingertips touch. Note that the
        minimum grasp force of 55N is perhaps already large for custom fingertips, preferably you find the allowable
        grasp with for yourself.
        TODO: Could monitoring motor current (self.current_motor_current) allow for a more delicate calibration?
        """
        self.move(
            width=self.gripper_specs.max_width, speed=self.gripper_specs.max_speed, force=self.gripper_specs.min_force
        ).wait()
        self.grip().wait()
        self._gripper_specs.max_width = self.SCHUNK_DEFAULT_SPECS.max_width - self.get_current_width()
        self.move(
            width=self.gripper_specs.max_width, speed=self.gripper_specs.max_speed, force=self.gripper_specs.min_force
        ).wait()

    def _move_done_condition(self) -> bool:
        return self.current_speed == 0
