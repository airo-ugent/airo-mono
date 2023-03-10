import enum
import time
import warnings
from typing import Callable


class ACTION_STATUS_ENUM(enum.Enum):
    EXECUTING = 1
    SUCCEEDED = 2
    TIMEOUT = 3


class AwaitableAction:
    def __init__(self, done_callback: Callable[..., bool]):
        """

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

    def wait(self, timeout: float = 10.0, sleep_resolution: float = 0.1) -> ACTION_STATUS_ENUM:
        # larger sleep resolution means less CPU usage, but also larger interval between checks
        assert (
            sleep_resolution > 0.0002
        ), "sleep resolution must be at least 0.2ms, otherwise the relative error of a sleep becomes too large to be meaningful"
        if not self.status == ACTION_STATUS_ENUM.EXECUTING:
            return self.status
        while True:
            time.sleep(sleep_resolution)
            timeout -= sleep_resolution
            if self.done_callback():
                self.status = ACTION_STATUS_ENUM.SUCCEEDED
                return self.status
            if timeout < 0:
                warnings.warn("Action timed out. Make sure this was expected.")
                return ACTION_STATUS_ENUM.TIMEOUT

    def is_done(self) -> bool:
        return self.status == ACTION_STATUS_ENUM.SUCCEEDED
