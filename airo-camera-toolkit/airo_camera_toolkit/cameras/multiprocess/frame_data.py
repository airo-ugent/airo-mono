"""Data structures for frame buffers used with Zenoh IPC."""

import dataclasses
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

T = TypeVar("T")


def frame_field_specs(template: Any) -> list[tuple[str, np.dtype, tuple[int, ...], int]]:
    """Describe the wire format defined by *template*.

    Args:
        template: A template instance (from ``FrameBuffer.template()``).

    Returns:
        One ``(name, dtype, shape, nbytes)`` tuple per dataclass field, in
        declaration order.
    """
    specs = []
    for f in dataclasses.fields(template):
        arr: np.ndarray = getattr(template, f.name)
        specs.append((f.name, arr.dtype, arr.shape, arr.nbytes))
    return specs


def validate_frame(template: Any, obj: Any) -> None:
    """Check that *obj* matches the wire format defined by *template*.

    The wire format is a bare concatenation of the field bytes, so a field with
    an unexpected dtype or shape would silently shift every field after it.
    This check turns that into an error at the publishing side.

    Args:
        template: A template instance (from ``FrameBuffer.template()``).
        obj: A frame buffer dataclass instance to validate.

    Raises:
        TypeError: If *obj* is not of the same type as *template*, or if a
            field is not a numpy array.
        ValueError: If a field's dtype or shape differs from the template.
    """
    if type(obj) is not type(template):
        raise TypeError(f"Expected a {type(template).__name__} instance, but got {type(obj).__name__}.")
    for name, dtype, shape, _ in frame_field_specs(template):
        arr = getattr(obj, name)
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"Field '{name}' must be a numpy array, but is {type(arr).__name__}.")
        if arr.dtype != dtype:
            raise ValueError(f"Field '{name}' has dtype {arr.dtype}, but the wire format expects {dtype}.")
        if arr.shape != shape:
            raise ValueError(f"Field '{name}' has shape {arr.shape}, but the wire format expects {shape}.")


def serialize_frame(obj: Any, template: Any = None) -> bytes:
    """Serialize a frame buffer dataclass to raw bytes.

    All numpy fields are concatenated in dataclass field declaration order.
    The schema (field order, shapes, dtypes) must be agreed upon by both sides
    via the corresponding ``template()`` classmethod.

    Args:
        obj: A frame buffer dataclass instance.
        template: Optional template instance to validate *obj* against before
            serializing.  Passing it is strongly recommended: without it, a
            field with an unexpected dtype or shape produces bytes that the
            receiver silently misinterprets.

    Returns:
        The concatenated field bytes.
    """
    if template is not None:
        validate_frame(template, obj)
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

    Raises:
        ValueError: If ``data`` does not have exactly the length the template's
            field layout describes.  Silently accepting a mismatch would return
            a frame with garbage in (some of) its fields.
    """
    specs = frame_field_specs(template)
    expected_nbytes = sum(nbytes for _, _, _, nbytes in specs)
    if len(data) != expected_nbytes:
        raise ValueError(
            f"Cannot deserialize a {type(template).__name__}: got {len(data)} bytes, "
            f"but its field layout describes {expected_nbytes} bytes. "
            "Publisher and receiver disagree on the wire format."
        )

    kwargs: dict = {}
    offset = 0
    for name, dtype, shape, nbytes in specs:
        chunk = data[offset : offset + nbytes]
        kwargs[name] = np.frombuffer(chunk, dtype=dtype).reshape(shape).copy()
        offset += nbytes
    return template.__class__(**kwargs)


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
