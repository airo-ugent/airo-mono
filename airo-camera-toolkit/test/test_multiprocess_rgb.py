"""Integration tests for the full multiprocess camera pipeline using a mock camera.

These tests spawn a real MultiprocessRGBPublisher (in a subprocess) backed by a
MockRGBCamera and verify that a MultiprocessRGBReceiver in the main process
receives frames correctly via Zenoh.

No real camera hardware is required.  Requires ``eclipse-zenoh`` to be installed;
tests are skipped automatically when it is not available.
"""

import multiprocessing
import time

import pytest

pytest.importorskip("zenoh", reason="eclipse-zenoh not installed")

import numpy as np
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import (
    MultiprocessRGBPublisher,
    MultiprocessRGBReceiver,
)
from airo_camera_toolkit.interfaces import RGBCamera
from airo_typing import CameraIntrinsicsMatrixType, CameraResolutionType, NumpyFloatImageType, NumpyIntImageType

# ---------------------------------------------------------------------------
# Mock camera
# ---------------------------------------------------------------------------

_MOCK_RESOLUTION = (64, 48)  # (width, height) — small for fast tests
_MOCK_FPS = 30
_MOCK_RGB_VALUE = 128  # constant pixel value so we can assert equality


class MockRGBCamera(RGBCamera):
    """Minimal RGB camera that returns fixed images without any hardware."""

    @property
    def resolution(self) -> CameraResolutionType:
        return _MOCK_RESOLUTION

    @property
    def fps(self) -> float:
        return _MOCK_FPS

    def grab_images(self) -> None:
        # Simulate a small capture delay so the publisher doesn't spin too fast
        time.sleep(1.0 / _MOCK_FPS)

    def retrieve_rgb_image(self) -> NumpyFloatImageType:
        return self.retrieve_rgb_image_as_int().astype(np.float32) / 255.0

    def retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        w, h = _MOCK_RESOLUTION
        return np.full((h, w, 3), _MOCK_RGB_VALUE, dtype=np.uint8)

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return np.eye(3, dtype=np.float64) * 500.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NAMESPACE = "test_camera_mock"
_STARTUP_TIMEOUT = 15  # seconds to wait for publisher to be ready


@pytest.fixture()
def publisher():
    """Start a MultiprocessRGBPublisher and stop it after the test."""
    multiprocessing.set_start_method("spawn", force=True)
    pub = MultiprocessRGBPublisher(
        camera_cls=MockRGBCamera,
        shared_memory_namespace=_NAMESPACE,
    )
    pub.start()
    yield pub
    pub.stop()
    pub.join(timeout=5)


@pytest.fixture()
def receiver_factory(publisher):
    """Create receivers that time out (rather than hang) and are stopped afterwards."""
    receivers = []

    def make() -> MultiprocessRGBReceiver:
        receiver = MultiprocessRGBReceiver(_NAMESPACE, timeout=_STARTUP_TIMEOUT)
        receivers.append(receiver)
        return receiver

    yield make
    for receiver in receivers:
        receiver.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_receiver_reads_resolution_and_fps(receiver_factory):
    """Receiver should report the same resolution and fps as the mock camera."""
    receiver = receiver_factory()
    assert receiver.resolution == _MOCK_RESOLUTION
    assert receiver.fps == _MOCK_FPS


def test_receiver_gets_rgb_image(receiver_factory):
    """Receiver should return an RGB image with the expected shape and values."""
    receiver = receiver_factory()
    receiver.grab_images()

    image = receiver.retrieve_rgb_image_as_int()
    w, h = _MOCK_RESOLUTION
    assert image.shape == (h, w, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image, _MOCK_RGB_VALUE)


def test_receiver_frame_timestamp_advances(receiver_factory):
    """Consecutive grab_images() calls should yield strictly increasing timestamps."""
    receiver = receiver_factory()

    receiver.grab_images()
    t0 = receiver.get_current_timestamp()

    receiver.grab_images()
    t1 = receiver.get_current_timestamp()

    assert t1 > t0, f"Timestamp did not advance: {t0} -> {t1}"


def test_receiver_frame_id_advances(receiver_factory):
    """Frame IDs should be monotonically increasing."""
    receiver = receiver_factory()

    receiver.grab_images()
    id0 = receiver.get_current_frame_id()

    receiver.grab_images()
    id1 = receiver.get_current_frame_id()

    assert id1 > id0, f"Frame ID did not advance: {id0} -> {id1}"


def test_multiple_receivers_same_namespace(receiver_factory):
    """Multiple receivers on the same namespace should all get valid frames."""
    r1 = receiver_factory()
    r2 = receiver_factory()

    r1.grab_images()
    r2.grab_images()

    np.testing.assert_array_equal(r1.retrieve_rgb_image_as_int(), _MOCK_RGB_VALUE)
    np.testing.assert_array_equal(r2.retrieve_rgb_image_as_int(), _MOCK_RGB_VALUE)


def test_grab_images_returns_buffered_frame_without_waiting(receiver_factory):
    """A frame that arrived while the caller was busy should be returned immediately.

    Regression test: comparing against the message count at entry (instead of the
    count of the frame we last returned) made every grab_images() wait for the
    *next* publish, so the frame returned was always captured after the call
    started, adding up to a full frame period of latency.
    """
    receiver = receiver_factory()
    receiver.grab_images()

    # Let at least one new frame arrive while we are not grabbing.
    time.sleep(3.0 / _MOCK_FPS)

    t_call = time.time()
    receiver.grab_images()

    # The frame we get must be one that was already buffered, i.e. captured
    # before we called grab_images() — not one published after the call.
    assert receiver.get_current_timestamp() < t_call
