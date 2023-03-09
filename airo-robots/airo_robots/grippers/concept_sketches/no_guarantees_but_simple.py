import enum
import time
import warnings
from typing import Callable


class ACTION_STATUS_ENUM(enum.Enum):
    EXECUTING = 1
    SUCCEEDED = 2
    TIMEOUT = 3


class GripperAction:
    def __init__(self, done_callback: Callable[..., bool]):
        """_summary_

        Args:
            done_callback (Callable[..., bool]): Any  callable that returns True when the action is completed.
            In the simplest case, it can be a lambda that returns True.
            It can also be true when the gripper is in the desired position.
            Or it could be true after a certain amount of time has passed since the action was started.

            Note that the scope of this action is to send 1 command, do some other things, then wait for the command to finish.
            If you send multiple commands to the gripper, there is no guarantee that the gripper will execute them in the order you send them,
             as the gripper might preempt intermediate commands after it has finished the current command.
            Such preemption will also not be detected by this action, and hence the wait of a preempted action will either timeout or succeed by accident.

        """
        self.status = ACTION_STATUS_ENUM.EXECUTING
        self.done_callback = done_callback

    def wait(self, timeout) -> ACTION_STATUS_ENUM:
        if not self.status == ACTION_STATUS_ENUM.EXECUTING:
            return self.status
        while True:
            time.sleep(0.1)
            timeout -= 0.1
            if self.done_callback():
                self.status = ACTION_STATUS_ENUM.SUCCEEDED
                return self.status
            if timeout < 0:
                warnings.warn("Action timed out")


class Gripper:
    def __init__(self):
        self.position = 0

    def move(self, position, speed) -> GripperAction:
        action = GripperAction(lambda: self.get_position() == position)
        # send command to the gripper
        self.position = position
        return action

    def move_for_time(self, position, time) -> GripperAction:
        start_time = time.time()
        action = GripperAction(lambda: time.time() - start_time > time)
        # send command to the gripper
        self.position = position
        return action

    def get_position(self):
        return self.position


if __name__ == "__main__":
    gripper = Gripper()
    action = gripper.move(1, 1)
    action.wait(10)
    print(action.status)
