"""Unit tests for frame_data serialization round-trips."""

import numpy as np
import pytest
from airo_camera_toolkit.cameras.multiprocess.frame_data import (
    FpsIdl,
    PointCloudBuffer,
    ResolutionIdl,
    RGBDFrameBuffer,
    RGBDFrameBufferWithPointCloud,
    RGBFrameBuffer,
    SpatialMapBuffer,
    StereoRGBDFrameBuffer,
    StereoRGBDFrameBufferWithPointCloud,
    ZedFrameBuffer,
    deserialize_frame,
    serialize_frame,
)

W, H = 64, 48  # small resolution for fast tests


def _assert_frame_equal(a, b) -> None:
    import dataclasses

    for f in dataclasses.fields(a):
        np.testing.assert_array_equal(getattr(a, f.name), getattr(b, f.name), err_msg=f"field '{f.name}' mismatch")


# ---------------------------------------------------------------------------
# Metadata buffers
# ---------------------------------------------------------------------------


def test_fps_idl_round_trip():
    obj = FpsIdl(fps=np.array([42.0], dtype=np.float64))
    result = deserialize_frame(FpsIdl.template(), serialize_frame(obj))
    assert result.fps.item() == pytest.approx(42.0)


def test_resolution_idl_round_trip():
    obj = ResolutionIdl(resolution=np.array([1920, 1080], dtype=np.int32))
    result = deserialize_frame(ResolutionIdl.template(), serialize_frame(obj))
    np.testing.assert_array_equal(result.resolution, [1920, 1080])


# ---------------------------------------------------------------------------
# Frame buffers
# ---------------------------------------------------------------------------


def test_rgb_frame_buffer_round_trip():
    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    intrinsics = np.eye(3, dtype=np.float64) * 500.0
    obj = RGBFrameBuffer(
        frame_id=np.array([7], dtype=np.uint64),
        frame_timestamp=np.array([1.23456789], dtype=np.float64),
        rgb=rgb,
        intrinsics=intrinsics,
    )
    result = deserialize_frame(RGBFrameBuffer.template(W, H), serialize_frame(obj))
    _assert_frame_equal(obj, result)


def test_rgbd_frame_buffer_round_trip():
    obj = RGBDFrameBuffer(
        frame_id=np.array([1], dtype=np.uint64),
        frame_timestamp=np.array([0.5], dtype=np.float64),
        rgb=np.zeros((H, W, 3), dtype=np.uint8),
        intrinsics=np.eye(3, dtype=np.float64),
        depth_image=np.zeros((H, W, 3), dtype=np.uint8),
        depth=np.ones((H, W), dtype=np.float32) * 2.5,
    )
    result = deserialize_frame(RGBDFrameBuffer.template(W, H), serialize_frame(obj))
    _assert_frame_equal(obj, result)


def test_rgbd_frame_buffer_with_pointcloud_round_trip():
    n = H * W
    obj = RGBDFrameBufferWithPointCloud(
        frame_id=np.array([2], dtype=np.uint64),
        frame_timestamp=np.array([1.0], dtype=np.float64),
        rgb=np.zeros((H, W, 3), dtype=np.uint8),
        intrinsics=np.eye(3, dtype=np.float64),
        depth_image=np.zeros((H, W, 3), dtype=np.uint8),
        depth=np.zeros((H, W), dtype=np.float32),
        point_cloud_positions=np.random.rand(n, 3).astype(np.float32),
        point_cloud_colors=np.zeros((n, 3), dtype=np.uint8),
        num_valid_points=np.array([n // 2], dtype=np.int32),
    )
    result = deserialize_frame(RGBDFrameBufferWithPointCloud.template(W, H), serialize_frame(obj))
    _assert_frame_equal(obj, result)


def test_stereo_rgbd_frame_buffer_round_trip():
    obj = StereoRGBDFrameBuffer(
        frame_id=np.array([3], dtype=np.uint64),
        frame_timestamp=np.array([2.0], dtype=np.float64),
        rgb=np.zeros((H, W, 3), dtype=np.uint8),
        intrinsics=np.eye(3, dtype=np.float64),
        depth_image=np.zeros((H, W, 3), dtype=np.uint8),
        depth=np.zeros((H, W), dtype=np.float32),
        rgb_right=np.ones((H, W, 3), dtype=np.uint8) * 128,
        intrinsics_right=np.eye(3, dtype=np.float64) * 2.0,
        pose_right_in_left=np.eye(4, dtype=np.float64),
    )
    result = deserialize_frame(StereoRGBDFrameBuffer.template(W, H), serialize_frame(obj))
    _assert_frame_equal(obj, result)


def test_stereo_rgbd_frame_buffer_with_pointcloud_round_trip():
    n = H * W
    obj = StereoRGBDFrameBufferWithPointCloud(
        frame_id=np.array([4], dtype=np.uint64),
        frame_timestamp=np.array([3.0], dtype=np.float64),
        rgb=np.zeros((H, W, 3), dtype=np.uint8),
        intrinsics=np.eye(3, dtype=np.float64),
        depth_image=np.zeros((H, W, 3), dtype=np.uint8),
        depth=np.zeros((H, W), dtype=np.float32),
        rgb_right=np.zeros((H, W, 3), dtype=np.uint8),
        intrinsics_right=np.eye(3, dtype=np.float64),
        pose_right_in_left=np.eye(4, dtype=np.float64),
        point_cloud_positions=np.random.rand(n, 3).astype(np.float32),
        point_cloud_colors=np.zeros((n, 3), dtype=np.uint8),
        num_valid_points=np.array([n], dtype=np.int32),
    )
    result = deserialize_frame(StereoRGBDFrameBufferWithPointCloud.template(W, H), serialize_frame(obj))
    _assert_frame_equal(obj, result)


def test_zed_frame_buffer_round_trip():
    obj = ZedFrameBuffer(
        frame_id=np.array([5], dtype=np.uint64),
        frame_timestamp=np.array([4.0], dtype=np.float64),
        rgb=np.zeros((H, W, 3), dtype=np.uint8),
        intrinsics=np.eye(3, dtype=np.float64),
        depth_image=np.zeros((H, W, 3), dtype=np.uint8),
        depth=np.zeros((H, W), dtype=np.float32),
        rgb_right=np.zeros((H, W, 3), dtype=np.uint8),
        intrinsics_right=np.eye(3, dtype=np.float64),
        pose_right_in_left=np.eye(4, dtype=np.float64),
        camera_pose=np.eye(4, dtype=np.float64),
    )
    result = deserialize_frame(ZedFrameBuffer.template(W, H), serialize_frame(obj))
    _assert_frame_equal(obj, result)


def test_point_cloud_buffer_round_trip():
    n = H * W
    obj = PointCloudBuffer(
        frame_id=np.array([6], dtype=np.uint64),
        frame_timestamp=np.array([5.0], dtype=np.float64),
        point_cloud_positions=np.random.rand(n, 3).astype(np.float32),
        point_cloud_colors=np.zeros((n, 3), dtype=np.uint8),
        point_cloud_valid=np.array([n // 2], dtype=np.int32),
    )
    result = deserialize_frame(PointCloudBuffer.template(W, H), serialize_frame(obj))
    _assert_frame_equal(obj, result)


def test_spatial_map_buffer_round_trip():
    max_chunks, max_points = 10, 1000
    obj = SpatialMapBuffer(
        frame_id=np.array([0], dtype=np.uint64),
        frame_timestamp=np.array([6.0], dtype=np.float64),
        num_chunks=np.array([3], dtype=np.int32),
        chunks_updated=np.array([True, False, True] + [False] * (max_chunks - 3), dtype=np.bool_),
        chunk_sizes=np.array([100, 200, 300] + [0] * (max_chunks - 3), dtype=np.int32),
        point_positions=np.random.rand(max_points, 3).astype(np.float32),
        point_colors=np.zeros((max_points, 3), dtype=np.uint8),
    )
    result = deserialize_frame(SpatialMapBuffer.template(max_chunks, max_points), serialize_frame(obj))
    _assert_frame_equal(obj, result)


# ---------------------------------------------------------------------------
# Edge case: deserialized arrays are independent copies (not shared memory)
# ---------------------------------------------------------------------------


def test_deserialized_arrays_are_copies():
    obj = FpsIdl(fps=np.array([30.0], dtype=np.float64))
    data = serialize_frame(obj)
    result = deserialize_frame(FpsIdl.template(), data)
    result.fps[0] = 99.0
    # Modifying the result must not affect the original bytes
    result2 = deserialize_frame(FpsIdl.template(), data)
    assert result2.fps.item() == pytest.approx(30.0)
