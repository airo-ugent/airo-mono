"""Tests for ZenohWriter's non-SHM publishing path."""

import threading
import time

import numpy as np
import pytest

pytest.importorskip("zenoh", reason="eclipse-zenoh not installed")

import zenoh  # noqa: E402
from airo_camera_toolkit.cameras.multiprocess.base_publisher import _make_zenoh_config  # noqa: E402
from airo_camera_toolkit.cameras.multiprocess.frame_data import RGBFrameBuffer  # noqa: E402
from airo_camera_toolkit.cameras.multiprocess.zenoh_writer import ZenohWriter  # noqa: E402

_KEY_EXPR = "airo_test/zenoh_writer_buffer_reuse"
_WIDTH, _HEIGHT = 640, 480
_NUM_FRAMES = 20


def _make_frame(frame_id: int) -> RGBFrameBuffer:
    return RGBFrameBuffer(
        frame_id=np.array([frame_id], dtype=np.uint64),
        frame_timestamp=np.array([float(frame_id)], dtype=np.float64),
        rgb=np.full((_HEIGHT, _WIDTH, 3), frame_id % 256, dtype=np.uint8),
        intrinsics=np.eye(3, dtype=np.float64) * frame_id,
    )


def test_back_to_back_frames_do_not_overwrite_each_other():
    """The writer reuses one buffer, so Zenoh must copy it when publishing.

    If ZBytes only referenced the buffer, a frame still in flight would be overwritten by the next one.
    """
    layout = RGBFrameBuffer.layout(_WIDTH, _HEIGHT)
    publisher_session = zenoh.open(_make_zenoh_config(shm=False))
    subscriber_session = zenoh.open(_make_zenoh_config(shm=False))
    received: list[bytes] = []
    all_received = threading.Event()

    def callback(sample: zenoh.Sample) -> None:
        received.append(sample.payload.to_bytes())
        if len(received) == _NUM_FRAMES:
            all_received.set()

    subscriber = subscriber_session.declare_subscriber(_KEY_EXPR, callback)
    writer = ZenohWriter(publisher_session, _KEY_EXPR, layout, shm=False)
    try:
        deadline = time.monotonic() + 10.0
        while not writer.has_subscribers:
            if time.monotonic() > deadline:
                pytest.fail("The subscriber never matched the writer.")
            time.sleep(0.05)

        for frame_id in range(_NUM_FRAMES):
            writer(_make_frame(frame_id))

        assert all_received.wait(timeout=10.0), f"Only {len(received)} of {_NUM_FRAMES} frames arrived."
    finally:
        subscriber.undeclare()
        writer.stop()
        subscriber_session.close()
        publisher_session.close()

    for frame_id, data in enumerate(received):
        frame = layout.deserialize(data)
        assert frame.frame_id.item() == frame_id
        assert np.all(frame.rgb == frame_id % 256), f"Frame {frame_id} was overwritten by a later frame."
