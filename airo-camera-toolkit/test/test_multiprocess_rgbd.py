"""Integration tests for the RGBD multiprocess pipeline, including its point cloud stream, using a mock camera."""

import multiprocessing
import os
import time
import uuid

import numpy as np
import pytest
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgbd_camera import (
    MultiprocessRGBDPublisher,
    MultiprocessRGBDReceiver,
)
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
    PointCloud,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="On the CI runners only the first publisher process started in a pytest process is discovered.",
)

_RESOLUTION = (64, 48)
_FPS = 30
_TIMEOUT = 15


class MockRGBDCamera(RGBDCamera):
    """RGBD camera whose depth map and point cloud both hold the number of the current capture.

    That lets a test check that a received point cloud belongs to the frame it was received with.
    """

    def __init__(self) -> None:
        self._capture = 0

    @property
    def resolution(self) -> CameraResolutionType:
        return _RESOLUTION

    @property
    def fps(self) -> float:
        return _FPS

    def grab_images(self) -> None:
        time.sleep(1.0 / _FPS)
        self._capture += 1

    def retrieve_rgb_image(self) -> NumpyFloatImageType:
        return self.retrieve_rgb_image_as_int().astype(np.float32) / 255.0

    def retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        w, h = _RESOLUTION
        return np.zeros((h, w, 3), dtype=np.uint8)

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return np.eye(3, dtype=np.float64)

    def retrieve_depth_map(self) -> NumpyDepthMapType:
        w, h = _RESOLUTION
        return np.full((h, w), self._capture, dtype=np.float32)

    def retrieve_depth_image(self) -> NumpyIntImageType:
        w, h = _RESOLUTION
        return np.zeros((h, w, 3), dtype=np.uint8)

    def retrieve_colored_point_cloud(self) -> PointCloud:
        # A sparse point cloud, to check that only the valid points are returned.
        num_points = 100
        positions = np.full((num_points, 3), self._capture, dtype=np.float32)
        return PointCloud(positions, np.zeros((num_points, 3), dtype=np.uint8))


@pytest.fixture(scope="module")
def namespace():
    multiprocessing.set_start_method("spawn", force=True)
    namespace = f"test_rgbd_mock_{uuid.uuid4().hex[:8]}"
    publisher = MultiprocessRGBDPublisher(camera_cls=MockRGBDCamera, shared_memory_namespace=namespace)
    publisher.start()
    yield namespace
    publisher.stop()
    publisher.join(timeout=5)
    if publisher.is_alive():
        publisher.kill()
        publisher.join(timeout=5)


def test_point_clouds_match_their_frames(namespace):
    with MultiprocessRGBDReceiver(namespace, timeout=_TIMEOUT) as receiver:
        for _ in range(10):
            receiver.grab_images()
            point_cloud = receiver.retrieve_colored_point_cloud()
            assert point_cloud.points.shape == (100, 3)
            np.testing.assert_array_equal(point_cloud.points, receiver.retrieve_depth_map()[0, 0])


def test_receiver_without_point_cloud_receives_frames(namespace):
    with MultiprocessRGBDReceiver(namespace, enable_pointcloud=False, timeout=_TIMEOUT) as receiver:
        receiver.grab_images()
        assert receiver.retrieve_depth_map().shape == (_RESOLUTION[1], _RESOLUTION[0])
        with pytest.raises(RuntimeError, match="not enabled"):
            receiver.retrieve_colored_point_cloud()
