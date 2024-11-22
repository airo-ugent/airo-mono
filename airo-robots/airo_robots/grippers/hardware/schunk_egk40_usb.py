import time

import numpy as np
from bkstools.bks_lib.bks_base import BKSBase, keep_communication_alive_input, keep_communication_alive_sleep, keep_communication_alive
from bkstools.bks_lib.bks_module import BKSModule
from bkstools.scripts.bks_grip import WaitGrippedOrError
from filelock import AsyncWindowsFileLock
from pyschunk.generated.generated_enums import eCmdCode
from typing import Optional
from airo_robots.awaitable_action import AwaitableAction
from airo_robots.grippers.parallel_position_gripper import ParallelPositionGripper, ParallelPositionGripperSpecs


def rescale_range(x: float, from_min: float, from_max: float, to_min: float, to_max: float) -> float:
    return to_min + (x - from_min) / (from_max - from_min) * (to_max - to_min)


class SchunkEGK40_USB(ParallelPositionGripper):
    # values obtained from https://schunk.com/be/nl/grijpsystemen/parallelgrijper/egk/egk-40-mb-m-b/p/000000000001491762
    SCHUNK_DEFAULT_SPECS = ParallelPositionGripperSpecs(0.083, 0.0, 150, 55, 0.0575, 0.0055)

    def __init__(self, usb_interface="/dev/ttyUSB0",
                 fingers_max_stroke: Optional[float] = None) -> None:
        """
        usb_interface: the USB interface to which the gripper is connected.

        fingers_max_stroke:
        allow for custom max stroke width (if you have fingertips that are closer together). If None, the Schunk will
        calibrate itself. If -1, the default values are kept.
        TODO: this class currently doesn't support fingertips that, when fully opened, would have a larger width than
        the Schunk's maximum width of 83mm. Such fingertips would not touch when the Schunk is "closed".
        """
        super().__init__(self.SCHUNK_DEFAULT_SPECS)
        if fingers_max_stroke == -1:
            # keep default width settings
            pass
        elif fingers_max_stroke:
            self._gripper_specs.max_width = fingers_max_stroke
        elif fingers_max_stroke is None:
            self.calibrate_width()
        self.width_loss_due_to_fingers = self.SCHUNK_DEFAULT_SPECS.max_width - self.gripper_specs.max_width

        self.bks = BKSModule(usb_interface, sleep_time=None, debug=False)
        
        # Prepare gripper: Acknowledge any pending error:
        self.bks.command_code = eCmdCode.CMD_ACK
        self.bks.MakeReady()
        time.sleep(0.1)

    @property
    def speed(self) -> float:
        """returns speed setting [m/s]."""
        return self.bks.set_vel / 1000

    @speed.setter
    def speed(self, new_speed: float) -> None:
        """sets the max speed [m/s]."""
        _new_speed = np.clip(new_speed, self.gripper_specs.min_speed, self.gripper_specs.max_speed)
        self.bks.set_vel = float(_new_speed * 1000)  # value must be set in mm/s for bkstools

    @property
    def current_speed(self) -> float:
        """returns current speed [m/s]."""
        return self.bks.actual_vel / 1000

    @property
    def max_grasp_force(self) -> float:
        _force = rescale_range(self.bks.set_force, 0, 100,
                               self.gripper_specs.min_force, self.gripper_specs.max_force)
        return _force

    @max_grasp_force.setter
    def max_grasp_force(self, new_force: float) -> None:
        """sets the max grasping force [N]."""
        _new_force = np.clip(new_force, self.gripper_specs.min_force, self.gripper_specs.max_force)
        _new_force = rescale_range(_new_force, self.gripper_specs.min_force, self.gripper_specs.max_force,
                                   0, 100)
        self.bks.set_force = _new_force

    @property
    def current_motor_current(self) -> float:
        """returns current motor current usage [unit?], this is a proxy for force."""
        return self.bks.actual_cur

    def get_current_width(self) -> float:
        """the current opening of the fingers in meters"""
        # Reasoning:
        # width_without_fingers = self.SCHUNK_DEFAULT_SPECS.max_width - self.bks.actual_pos / 1000
        # return width_without_fingers - self.width_loss_due_to_fingers
        # The above equates to:
        return self.gripper_specs.max_width - (self.bks.actual_pos / 1000)

    def move(self, width: float, speed: Optional[float] = SCHUNK_DEFAULT_SPECS.min_speed,
             force: Optional[float] = SCHUNK_DEFAULT_SPECS.min_force, set_speed_and_force=True) -> AwaitableAction:
        """
        Moves the gripper to a certain position at a certain speed with a certain force
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

    def move_relative(self, width_difference: float, speed: Optional[float] = SCHUNK_DEFAULT_SPECS.min_speed,
             force: Optional[float] = SCHUNK_DEFAULT_SPECS.min_force, set_speed_and_force=True) -> AwaitableAction:
        """
        Moves the gripper to a certain position at a certain speed with a certain force
        :param width_difference: in m,  a positive difference will make the gripper open, a negative difference makes it close
        :param speed: in m/s
        :param force: in N
        :param set_speed_and_force: setting to false can improve control frequency as less transactions have to happen with the gripper
        """
        if set_speed_and_force:
            self.speed = speed
            self.max_grasp_force = force
        self.bks.set_pos = -width_difference*1000
        self.bks.command_code = eCmdCode.MOVE_POS_REL

        return AwaitableAction(self._move_done_condition)

    def grip(self) -> AwaitableAction:
        """
        Moves the gripper until object contact. When using MOVE_FORCE as below, it seems that the force and speed must
        be set to 50% and 0 mm/s respectively.
        :param speed: in m/s
        :param force: in N
        """
        self.bks.set_force = 50  # target force to 50 % => BasicGrip
        self.bks.set_vel = 0.0  # target velocity 0 => BasicGrip
        self.bks.grp_dir = True  # grip from outside
        self.bks.command_code = eCmdCode.MOVE_FORCE  # (for historic reasons the actual grip command for simple gripping is called MOVE_FORCE...)
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
        self.move(width=self.gripper_specs.max_width, speed=self.gripper_specs.max_speed, force=self.gripper_specs.min_force).wait()
        self.grip().wait()
        self._gripper_specs.max_width = self.SCHUNK_DEFAULT_SPECS.max_width - self.get_current_width()
        self.move(width=self.gripper_specs.max_width, speed=self.gripper_specs.max_speed, force=self.gripper_specs.min_force).wait()

    def _move_done_condition(self) -> bool:
        return self.current_speed == 0
