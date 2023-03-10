"""base classes for parallel-finger position-controlled grippers"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from airo_robots.awaitable_action import AwaitableAction


@dataclass
class ParallelPositionGripperSpecs:
    """
    all values are in metric units:
    - the position of the gripper is expressed as the width between the fingers in meters
    - the speed in meters/second
    - the force in Newton
    """

    max_width: float
    min_width: float
    max_force: float
    min_force: float
    max_speed: float
    min_speed: float


class ParallelPositionGripper(ABC):
    """
    Base class for a position-controlled, 2 finger parallel gripper.

    These grippers typically allow to set a speed and maximum applied force before moving,
    and attempt to move to specified positions under these constraints.

    all values are in metric units:
    - the position of the gripper is expressed as the width between the fingers in meters
    - the speed in meters/second
    - the force in Newton
    """

    def __init__(self, gripper_specs: ParallelPositionGripperSpecs) -> None:
        self._gripper_specs = gripper_specs

    @property
    def gripper_specs(self) -> ParallelPositionGripperSpecs:
        return self._gripper_specs

    @property
    @abstractmethod
    def speed(self) -> float:
        """speed with which the fingers will move in m/s"""
        # no need to raise NotImplementedError thanks to ABC

    @speed.setter
    @abstractmethod
    def speed(self, new_speed: float) -> None:
        """sets the moving speed [m/s]."""
        # this function is delibarately not templated
        # as one always requires this to happen synchronously.

    @property
    @abstractmethod
    def max_grasp_force(self) -> float:
        """max force the fingers will apply in Newton"""

    @max_grasp_force.setter
    @abstractmethod
    def max_grasp_force(self, new_force: float) -> None:
        """sets the max grasping force [N]."""
        # this function is delibarately not templated
        # as one always requires this to happen synchronously.

    @abstractmethod
    def get_current_width(self) -> float:
        """the current opening of the fingers in meters"""

    @abstractmethod
    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> AwaitableAction:
        """
        move the fingers to the desired width between the fingers[m].
        Optionally provide a speed and/or force, that will be used from then on for all move commands.
        """

    def open(self) -> AwaitableAction:
        return self.move(self.gripper_specs.max_width)

    def close(self) -> AwaitableAction:
        return self.move(0.0)

    def is_an_object_grasped(self) -> bool:
        """
        Heuristics to check if an object is grasped, usually by looking at motor currents.
        """
        raise NotImplementedError
