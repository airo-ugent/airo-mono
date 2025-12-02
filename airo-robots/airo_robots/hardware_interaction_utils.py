import gc
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from loguru import logger


@contextmanager
def gc_disabled(*, verbose: bool = False) -> Iterator:
    """Temporarily suspend the Python garbage collector for performance-critical code.

    Limit the code inside this block to avoid memory accumulation.

    This context manager disables the cyclic garbage collector on entry and
    restores its original state on exit. If the collector was already
    disabled when the block is entered, it remains disabled after the block.
    If it was enabled, the collector is re-enabled and an explicit collection
    is performed to reclaim any cyclic objects that accumulated while the GC
    was off.

    Args:
        verbose: If `True`, will log debug messages when disabling/enabling gc."""
    was_enabled = gc.isenabled()
    if verbose:
        logger.debug("Disabling garbage collection.")
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            if verbose:
                logger.debug("Re-enabling garbage collection and collecting garbage.")
            gc.enable()
            gc.collect()
            if verbose:
                logger.debug("Garbage collection reenabled.")

from typing_extensions import deprecated


@deprecated("Use ThreadPoolExecutor instead.")
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
