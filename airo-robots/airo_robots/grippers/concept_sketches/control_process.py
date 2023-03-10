import enum
import multiprocessing
import time
import warnings


class ACTION_STATUS_ENUM(enum.Enum):
    WAITING = 0
    EXECUTING = 1
    SUCCEEDED = 2
    TIMEOUT = 3
    PREEMPTED = 4


class Action:
    def __init__(
        self, command: callable, command_args, command_kwargs, done_callback: callable, callback_args, callback_kwargs
    ) -> None:
        self.command = command
        self.done_callback = done_callback
        self.command_args = command_args
        self.command_kwargs = command_kwargs
        self.callback_args = callback_args
        self.callback_kwargs = callback_kwargs
        self.status = ACTION_STATUS_ENUM.EXECUTING

    def wait(self, timeout) -> ACTION_STATUS_ENUM:
        while True:
            if self.status != ACTION_STATUS_ENUM.EXECUTING and self.status != ACTION_STATUS_ENUM.WAITING:
                return self.status
            time.sleep(0.1)
            timeout -= 0.1
            if timeout < 0:
                warnings.warn("Action timed out")


action_queue = multiprocessing.Queue()


def foo(*args, **kwargs):
    print("foo")
    print(args)
    print(kwargs)
    time.sleep(1)
    print("foo done")


class Gripper:
    def __init__(self) -> None:
        self.connection = None

    def start_control_process(self):
        self.control_process = multiprocessing.Process(target=self.control_loop, args=(action_queue,))
        self.control_process.start()

    @staticmethod
    def control_loop(queue):
        while True:
            # print("looping")
            if queue.empty():
                # print("queue empty")
                time.sleep(0.1)  # longer sleeping -> less CPU load, but slower response to new commands...
            else:
                # take last action in the queue
                # and mark others as preempted
                while not queue.empty():
                    action = queue.get()
                    if not queue.empty():
                        action.status = ACTION_STATUS_ENUM.PREEMPTED
                action.command(*action.command_args, **action.command_kwargs)
                while not action.done_callback(*action.callback_args, **action.callback_kwargs):
                    # print("waiting")
                    time.sleep(0.1)
                action.status = ACTION_STATUS_ENUM.SUCCEEDED

    def move(self, position, speed) -> Action:
        action = Action(foo, (position, speed), {}, foo, (position,), {})
        action_queue.put(action)
        return action

    def _actual_move(self, position, speed):
        # send to gripper
        pass

    def _is_position_reached(self, position):
        # check if gripper is in position
        print("checking if gripper is in position")
        return True


if __name__ == "__main__":
    gripper = Gripper()
    gripper.start_control_process()
    action = gripper.move(0.5, 0.1)
    action.wait(5)
    print(action.status)

    s = Gripper()
