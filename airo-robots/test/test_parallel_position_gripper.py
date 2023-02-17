import time
from concurrent.futures import Future
from typing import Optional

from airo_robots.grippers.parallel_position_gripper import (
    AsynchronousParallelPositionGripperWrapper,
    ParallelPositionGripper,
    ParallelPositionGripperSpecs,
    SynchronousParallelPositionGripperWrapper,
)


class DummySyncParallelPositionGripper(ParallelPositionGripper):
    """'Straight-through' implementation of a parallel position gripper for testing purposes."""

    def __init__(self, gripper_specs: ParallelPositionGripperSpecs) -> None:
        super().__init__(gripper_specs)
        self.gripper_pos = 0
        self.gripper_speed = 0
        self.gripper_force = 0

    def move(self, width: float, speed: Optional[float] = None, force: Optional[float] = None) -> None:
        if speed:
            self.gripper_speed = speed
        if force:
            self.gripper_force = force
        # simulate time to reach HW
        # will make test slower though..
        time.sleep(1)
        self.gripper_pos = width

    @property
    def speed(self) -> float:
        return self.gripper_speed

    @speed.setter
    def speed(self, value: float):
        self.gripper_speed = value

    @property
    def max_grasp_force(self) -> float:
        return self.gripper_force

    @max_grasp_force.setter
    def max_grasp_force(self, value: float) -> float:
        self.max_grasp_force = value

    def get_current_width(self) -> float:
        return self.gripper_pos


def test_sync_async_wrappper_implementations():
    gripper = DummySyncParallelPositionGripper(None)
    target_pos = 0.01
    res = gripper.move(target_pos)
    assert res is None
    assert gripper.get_current_width() == target_pos

    target_pos = 0.02
    async_gripper = AsynchronousParallelPositionGripperWrapper(gripper)
    res = async_gripper.move(target_pos)
    assert isinstance(res, Future)
    res.result(10)
    assert async_gripper.get_current_width() == target_pos

    target_pos = 0.03
    sync_wrapped_gripper = SynchronousParallelPositionGripperWrapper(async_gripper)
    res = sync_wrapped_gripper.move(target_pos)
    assert res is None
    assert sync_wrapped_gripper.get_current_width() == target_pos
