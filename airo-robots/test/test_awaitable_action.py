import time

import pytest
from airo_robots.awaitable_action import ACTION_STATUS_ENUM, AwaitableAction


def test_awaitable_action_done_after_waiting():
    call_time = time.time()
    action = AwaitableAction(lambda: time.time() > call_time + 1)
    assert not action.is_done()
    assert action.status == ACTION_STATUS_ENUM.EXECUTING
    action.wait()
    assert action.is_done()


def test_awaitable_action_timeout_raises_warning():
    time.time()
    action = AwaitableAction(lambda: False)
    assert not action.is_done()
    with pytest.warns(UserWarning) as warnings:
        action.wait(timeout=0.1)
    assert len(warnings) == 1
