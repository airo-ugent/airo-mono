import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable


class AsyncExecutor:
    """Helper class to mock async hardware interfaces, used for testing.

    Note that using this class, even though it has only one worker in the executor pool, is not necessarily thread safe if you do not
    ensure that all methods of the class you are using it on, make use of the threadpool or if you create multiple instances of that class.

    In those cases you would have to provide appropriate locking mechanisms to avoid any race conditions.
    """

    def __init__(self) -> None:
        self._thread_pool = ThreadPoolExecutor(max_workers=1)

    def _threadpool_execution(self, func: Callable, *args: Any, **kwargs: Any) -> Future:
        """helper function to execute a function call asynchronously in the threadpool.

        returns a future which can be waited for (cf. join on a thread) or can be polled to see if the function has finished
        see https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future
        """
        future = self._thread_pool.submit(func, *args, **kwargs)
        return future

    def __call__(self, func: Callable, *args: Any, **kwargs: Any) -> Future:
        """execute the function call asynchronously in the threadpool.

        returns a future which can be waited for (cf. join on a thread) or can be polled to see if the function has finished
        see https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future
        """
        return self._threadpool_execution(func, *args, **kwargs)


def wait_for_condition_with_timeout(
    check_condition: Callable[..., bool], timeout: float = 10, sleep_resolution: float = 0.1
) -> None:
    """helper function to wait on completion of hardware interaction with a timeout to avoid blocking forever."""

    while not check_condition():
        time.sleep(sleep_resolution)
        timeout -= sleep_resolution
        if timeout < 0:
            raise TimeoutError()
