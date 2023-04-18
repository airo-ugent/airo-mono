import enum
import time
import warnings
from typing import Callable, Optional


class ACTION_STATUS_ENUM(enum.Enum):
    EXECUTING = 1
    SUCCEEDED = 2
    TIMEOUT = 3


class AwaitableAction:
    def __init__(
        self,
        termination_condition: Callable[..., bool],
        default_timeout: float = 30.0,
        default_sleep_resolution: float = 0.1,
    ):
        """

        Args:
            termination_condition (Callable[..., bool]): Any  callable that returns True when the action is completed.
            In the simplest case, it can be a lambda that returns True.
            It can also be true when the gripper is in the desired position.
            Or it could be true after a certain amount of time has passed since the action was started.

            default_timeout (float, optional): The max waiting time before the wait returns and raises a warning.
            Select an appropriate default value for the command you are creating this AwaitableAction for.

            default_sleep_resolution (float, optional): The length of the time.sleep() in each iteration.
            Select an appropriate default value for the command you are creating this AwaitableAction for.

            Note that the scope of this action is to send 1 command, do some other things, then wait for the command to finish.
            If you send multiple commands to the gripper, there is no guarantee that the gripper will execute them in the order you send them,
             as the gripper might preempt intermediate commands after it has finished the current command.
            Such preemption will also not be detected by this action, and hence the wait of a preempted action will either timeout or succeed by accident.

        """
        self.status = ACTION_STATUS_ENUM.EXECUTING
        self.is_action_done = termination_condition
        self._default_timeout = default_timeout
        self._default_sleep_resolution = default_sleep_resolution

    def wait(self, timeout: Optional[float] = None, sleep_resolution: Optional[float] = None) -> ACTION_STATUS_ENUM:
        """Busy waiting until the termination condition returns true, or until timeout.

        Args:
            timeout (float, optional): The max waiting time before the wait returns and raises a warning.
            This prevents infinite loops. Defaults to the value set during creation of the awaitable, which is usually an appropriate value for the command.

            sleep_resolution (float, optional): The length of the time.sleep() in each iteration.
            higher values will take up less CPU resources but will also cause more latency between the action finishing and
            this method to return. Defaults to the value set during creation of the awaitable, which is usually an appropriate value for the command.

            Keep in mind that the time.sleep() function has limited accuracy, so the actual sleeping time will be usually higher
            due to scheduling activities of the OS. Take a look [here](https://github.com/airo-ugent/airo-mono/pull/21#discussion_r1132520057)
            for some realistic numbers on the error. The error is independent of the sleep time and is in the order of 0.2ms.
            So the lower the sleep time, the higher the relative error becomes.


        Returns:
            ACTION_STATUS_ENUM: _description_
        """
        # see #airo-robots/scripts/measure_sleep_accuracy.py for a script that measures the sleep accuracy and
        # the result of some measurements.
        sleep_resolution = sleep_resolution or self._default_sleep_resolution
        timeout = timeout or self._default_timeout
        assert (
            sleep_resolution > 0.001
        ), "sleep resolution must be at least 1 ms, otherwise the relative error of a sleep becomes too large to be meaningful"
        if not self.status == ACTION_STATUS_ENUM.EXECUTING:
            return self.status
        while True:
            time.sleep(sleep_resolution)
            timeout -= sleep_resolution
            if self.is_action_done():
                self.status = ACTION_STATUS_ENUM.SUCCEEDED
                return self.status
            if timeout < 0:
                warnings.warn("Action timed out. Make sure this was expected.")
                return ACTION_STATUS_ENUM.TIMEOUT

    def is_done(self) -> bool:
        return self.status == ACTION_STATUS_ENUM.SUCCEEDED
