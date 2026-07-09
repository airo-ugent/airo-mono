"""Data structures for frame buffers used with Zenoh IPC."""

import dataclasses
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


def serialize_frame(obj: Any) -> bytes:
    """Serialize a frame buffer dataclass to raw bytes.

    All numpy fields are concatenated in dataclass field declaration order.
    The schema (field order, shapes, dtypes) must be agreed upon by both sides
    via the corresponding ``template()`` classmethod.
    """
    parts = [getattr(obj, f.name).ravel().view(np.uint8) for f in dataclasses.fields(obj)]
    flat = np.empty(sum(p.nbytes for p in parts), dtype=np.uint8)
    np.concatenate(parts, out=flat)
    return bytes(flat)


def deserialize_frame(template: T, data: bytes) -> T:
    """Deserialize raw bytes back into a frame buffer dataclass instance.

    Args:
        template: A template instance (from ``FrameBuffer.template()``) that
            defines the expected field shapes and dtypes.
        data: Raw bytes produced by :func:`serialize_frame`.

    Returns:
        A new dataclass instance with numpy arrays filled from ``data``.
    """
    kwargs: dict = {}
    offset = 0
    for f in dataclasses.fields(template):  # type: ignore[arg-type]
        arr: np.ndarray = getattr(template, f.name)
        chunk = data[offset : offset + arr.nbytes]
        kwargs[f.name] = np.frombuffer(chunk, dtype=arr.dtype).reshape(arr.shape).copy()
        offset += arr.nbytes
    return template.__class__(**kwargs)  # type: ignore[return-value]


@dataclass
class FpsIdl:
    """Frame rate metadata published alongside camera frames."""

    fps: np.ndarray

    @staticmethod
    def template() -> Any:
        """Construct a new FpsIdl template with pre-allocated arrays."""
        return FpsIdl(fps=np.empty((1,), dtype=np.float64))


@dataclass
class ResolutionIdl:
    """Resolution metadata published alongside camera frames."""

    resolution: np.ndarray

    @staticmethod
    def template() -> Any:
        """Construct a new ResolutionIdl template with pre-allocated arrays."""
        return ResolutionIdl(resolution=np.empty((2,), dtype=np.int32))


@dataclass
class BaseFrameBuffer:
    """Base frame buffer containing timestamp and frame ID for synchronization."""

    # Frame ID for synchronization (monotonically increasing)
    frame_id: np.ndarray
    # Timestamp when the frame was captured (seconds since epoch)
    frame_timestamp: np.ndarray


@dataclass
class RGBFrameBuffer(BaseFrameBuffer):
    """Frame buffer containing RGB image data and camera intrinsics."""

    # Color image data (height x width x channels)
    rgb: np.ndarray
    # Intrinsic camera parameters (camera matrix)
    intrinsics: np.ndarray

    @staticmethod
    def template(width: int, height: int) -> Any:
        """Construct a new RGBFrameBuffer with shared memory backed arrays."""
        return RGBFrameBuffer(
            frame_id=np.empty((1,), dtype=np.uint64),
            frame_timestamp=np.empty((1,), dtype=np.float64),
            rgb=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics=np.empty((3, 3), dtype=np.float64),
        )


@dataclass
class RGBDFrameBuffer(RGBFrameBuffer):
    """Frame buffer containing RGB-D data (color + depth)."""

    # Depth image data (height x width)
    depth_image: np.ndarray
    # Depth map (height x width)
    depth: np.ndarray

    @staticmethod
    def template(width: int, height: int) -> Any:
        """Construct a new RGBDFrameBuffer with shared memory backed arrays."""
        return RGBDFrameBuffer(
            frame_id=np.empty((1,), dtype=np.uint64),
            frame_timestamp=np.empty((1,), dtype=np.float64),
            rgb=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics=np.empty((3, 3), dtype=np.float64),
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth=np.empty((height, width), dtype=np.float32),
        )


@dataclass
class RGBDFrameBufferWithPointCloud(RGBDFrameBuffer):
    """Frame buffer containing RGB-D data along with point cloud data."""

    # Point cloud positions (N x 3)
    point_cloud_positions: np.ndarray
    # Point cloud colors (N x 3)
    point_cloud_colors: np.ndarray
    # Number of valid points in the point cloud
    num_valid_points: np.ndarray

    @staticmethod
    def template(width: int, height: int) -> Any:
        """Construct a new RGBDFrameBufferWithPointCloud with shared memory backed arrays."""
        return RGBDFrameBufferWithPointCloud(
            frame_id=np.empty((1,), dtype=np.uint64),
            frame_timestamp=np.empty((1,), dtype=np.float64),
            rgb=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics=np.empty((3, 3), dtype=np.float64),
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth=np.empty((height, width), dtype=np.float32),
            point_cloud_positions=np.empty((height * width, 3), dtype=np.float32),
            point_cloud_colors=np.empty((height * width, 3), dtype=np.uint8),
            num_valid_points=np.empty((1,), dtype=np.int32),
        )


@dataclass
class StereoRGBDFrameBuffer(RGBDFrameBuffer):
    """Frame buffer containing stereo RGB-D data (left + right cameras)."""

    # Right camera RGB image
    rgb_right: np.ndarray
    # Right camera intrinsics
    intrinsics_right: np.ndarray
    # Pose of right camera in left camera frame
    pose_right_in_left: np.ndarray

    @staticmethod
    def template(width: int, height: int) -> Any:
        """Construct a new StereoRGBDFrameBuffer with shared memory backed arrays."""
        return StereoRGBDFrameBuffer(
            frame_id=np.empty((1,), dtype=np.uint64),
            frame_timestamp=np.empty((1,), dtype=np.float64),
            rgb=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics=np.empty((3, 3), dtype=np.float64),
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth=np.empty((height, width), dtype=np.float32),
            rgb_right=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics_right=np.empty((3, 3), dtype=np.float64),
            pose_right_in_left=np.empty((4, 4), dtype=np.float64),
        )


@dataclass
class StereoRGBDFrameBufferWithPointCloud(StereoRGBDFrameBuffer):
    """Frame buffer containing stereo RGB-D data along with point cloud data."""

    # Point cloud positions (N x 3)
    point_cloud_positions: np.ndarray
    # Point cloud colors (N x 3)
    point_cloud_colors: np.ndarray
    # Number of valid points in the point cloud
    num_valid_points: np.ndarray

    @staticmethod
    def template(width: int, height: int) -> Any:
        """Construct a new StereoRGBDFrameBufferWithPointCloud with shared memory backed arrays."""
        return StereoRGBDFrameBufferWithPointCloud(
            frame_id=np.empty((1,), dtype=np.uint64),
            frame_timestamp=np.empty((1,), dtype=np.float64),
            rgb=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics=np.empty((3, 3), dtype=np.float64),
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth=np.empty((height, width), dtype=np.float32),
            rgb_right=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics_right=np.empty((3, 3), dtype=np.float64),
            pose_right_in_left=np.empty((4, 4), dtype=np.float64),
            point_cloud_positions=np.empty((height * width, 3), dtype=np.float32),
            point_cloud_colors=np.empty((height * width, 3), dtype=np.uint8),
            num_valid_points=np.empty((1,), dtype=np.int32),
        )


@dataclass
class ZedFrameBuffer(StereoRGBDFrameBuffer):
    """Frame buffer containing Zed camera data including camera pose."""

    # Camera pose in world coordinates
    camera_pose: np.ndarray

    @staticmethod
    def template(width: int, height: int) -> Any:
        """Construct a new ZedFrameBuffer with shared memory backed arrays."""
        return ZedFrameBuffer(
            frame_id=np.empty((1,), dtype=np.uint64),
            frame_timestamp=np.empty((1,), dtype=np.float64),
            rgb=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics=np.empty((3, 3), dtype=np.float64),
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth=np.empty((height, width), dtype=np.float32),
            rgb_right=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics_right=np.empty((3, 3), dtype=np.float64),
            pose_right_in_left=np.empty((4, 4), dtype=np.float64),
            camera_pose=np.empty((4, 4), dtype=np.float64),
        )


@dataclass
class PointCloudBuffer:
    """Buffer containing point cloud data."""

    # Frame ID for synchronization
    frame_id: np.ndarray
    # Timestamp of the point cloud
    frame_timestamp: np.ndarray
    # Point cloud positions (height * width x 3)
    point_cloud_positions: np.ndarray
    # Point cloud colors (height * width x 3)
    point_cloud_colors: np.ndarray
    # Valid point cloud points (scalar), for sparse point clouds
    point_cloud_valid: np.ndarray

    @staticmethod
    def template(width: int, height: int) -> Any:
        """Construct a new PointCloudBuffer with shared memory backed arrays."""
        return PointCloudBuffer(
            frame_id=np.empty((1,), dtype=np.uint64),
            frame_timestamp=np.empty((1,), dtype=np.float64),
            point_cloud_positions=np.empty((height * width, 3), dtype=np.float32),
            point_cloud_colors=np.empty((height * width, 3), dtype=np.uint8),
            point_cloud_valid=np.empty((1,), dtype=np.int32),
        )


@dataclass
class SpatialMapBuffer:
    """Buffer containing spatial map data from Zed camera."""

    # Frame ID for synchronization
    frame_id: np.ndarray
    # Timestamp of the spatial map
    frame_timestamp: np.ndarray
    # Amount of chunks in the spatial map
    num_chunks: np.ndarray
    # Array indicating which chunks have been updated
    chunks_updated: np.ndarray
    # Size of each chunk (number of points)
    chunk_sizes: np.ndarray
    # Arrays of concatenated chunk data (point positions and colors)
    point_positions: np.ndarray
    point_colors: np.ndarray

    @staticmethod
    def template(max_chunks: int, max_points: int) -> Any:
        """Construct a new SpatialMapBuffer with shared memory backed arrays."""
        return SpatialMapBuffer(
            frame_id=np.empty((1,), dtype=np.uint64),
            frame_timestamp=np.empty((1,), dtype=np.float64),
            num_chunks=np.empty((1,), dtype=np.int32),
            chunks_updated=np.empty((max_chunks,), dtype=np.bool_),
            chunk_sizes=np.empty((max_chunks,), dtype=np.int32),
            point_positions=np.empty((max_points, 3), dtype=np.float32),
            point_colors=np.empty((max_points, 3), dtype=np.uint8),
        )
