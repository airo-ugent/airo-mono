import time
from concurrent.futures import Future

from airo_robots.hardware_interaction_utils import AsyncExecutor


def dummy_function(arg1: int, arg2: float, arg3: int = 4):
    time.sleep(1)
    return arg1**arg2 + arg3


def test_async_executor():
    executor = AsyncExecutor()
    a, b, c = 1, 2, 3
    res = executor(dummy_function, a, b, arg3=c)
    assert isinstance(res, Future)
    assert res.result(2) == a**b + c
