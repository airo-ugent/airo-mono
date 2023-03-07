""" asynchronous and synchronous base classes and wrappers for parallel-jax position-controlled grippers"""


from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from airo_robots.async_executor import AsyncExecutor


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


T = TypeVar("T")


class ParallelPositionGripperTemplate(ABC, Generic[T]):
    """
    Template base class for a position-controlled, 2 finger parallel gripper.

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

    @gripper_specs.setter
    def gripper_specs(self, spec: ParallelPositionGripperSpecs) -> None:
        self._gripper_specs = spec

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
    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> T:
        """
        move the fingers to the desired width between the fingers[m].
        Optionally provide a speed and/or force, that will be used from then on for all move commands."""

    def open(self) -> T:
        return self.move(self.gripper_specs.max_width)

    def close(self) -> T:
        return self.move(0.0)

    def is_an_object_grasped(self) -> bool:
        """
        Some grippers have heuristics to check if an object is grasped, usually by looking at motor currents.
        This function returns this heuristic, if it exists.
        """
        raise NotImplementedError


class ParallelPositionGripper(ParallelPositionGripperTemplate[None]):
    """
    Synchronous base class for a position-controlled, 2 finger parallel gripper.
    Synchronous means that implementations of this class will block while executing hardware actions and only return once the
    action has finished.

    all values are in metric units:
    - the position of the gripper is expressed as the width between the fingers in meters
    - the speed in meters/second
    - the force in Newton
    """


class AsyncParallelPositionGripper(ParallelPositionGripperTemplate[Future]):
    """
    Asynchronous base class for a position-controlled, 2 finger parallel gripper.
    Async means that implementations of this class will not block while executing hardware actions, but they will return a Future object
    that can be waited for.

    all values are in metric units:
    - the position of the gripper is expressed as the width between the fingers in meters
    - the speed in meters/second
    - the force in Newton
    """


class SynchronousParallelPositionGripperAdapter(ParallelPositionGripper):
    """
    This is a default adapter to turn an asynchronous gripper implementation into a synchronous one.
    It waits for the future object of the async methods before returning the return value of the wrapped gripper's call.
    """

    def __init__(self, gripper: AsyncParallelPositionGripper) -> None:
        super().__init__(gripper.gripper_specs)
        self._gripper = gripper
        self.timeout = 10

    @property
    def speed(self) -> float:
        return self._gripper.speed

    @speed.setter
    def speed(self, new_speed: float) -> None:
        self._gripper.speed = new_speed

    @property
    def max_grasp_force(self) -> float:
        return self._gripper.max_grasp_force

    @max_grasp_force.setter
    def max_grasp_force(self, new_force: float) -> None:
        self._gripper.max_grasp_force = new_force

    def get_current_width(self) -> float:
        return self._gripper.get_current_width()

    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> None:
        return self._gripper.move(width, speed, force).result(self.timeout)

    def is_an_object_grasped(self) -> bool:
        return self._gripper.is_an_object_grasped()


class AsynchronousParallelPositionGripperAdapter(AsyncParallelPositionGripper):
    """
    This is a default adapter to turn a synchronous gripper implementation into an asynchronous one.
    It executes the async methods in a separate thread and returns a future object to query.
    """

    def __init__(self, gripper: ParallelPositionGripper) -> None:
        super().__init__(gripper.gripper_specs)
        self._gripper = gripper
        self.async_executor = AsyncExecutor()

    @property
    def speed(self) -> float:
        return self._gripper.speed

    @speed.setter
    def speed(self, new_speed: float) -> None:
        self._gripper.speed = new_speed

    @property
    def max_grasp_force(self) -> float:
        return self._gripper.max_grasp_force

    @max_grasp_force.setter
    def max_grasp_force(self, new_force: float) -> None:
        self._gripper.max_grasp_force = new_force

    def get_current_width(self) -> float:
        return self._gripper.get_current_width()

    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> Future:
        return self.async_executor(self._gripper.move, width, speed, force)

    def is_an_object_grasped(self) -> bool:
        return self._gripper.is_an_object_grasped()
