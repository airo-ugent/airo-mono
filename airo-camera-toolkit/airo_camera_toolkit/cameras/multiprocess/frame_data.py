"""Data structures for frame buffers used with Zenoh.

A frame buffer is a dataclass whose fields are all numpy arrays. Its wire format is the bare
concatenation of the fields' bytes in declaration order; a :class:`FrameLayout` describes that
format (each field's dtype and shape) without allocating any arrays.
"""

import dataclasses
import json
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Generic, Self, TypeVar

import numpy as np
from numpy.typing import DTypeLike

Frame = TypeVar("Frame")

# (dtype, shape) of a single field, as declared by a frame buffer's ``_field_specs()``.
_FieldSpec = tuple[DTypeLike, tuple[int, ...]]


@dataclass(frozen=True)
class FieldLayout:
    """The dtype and shape of a single frame buffer field."""

    name: str
    dtype: np.dtype
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize


@dataclass(frozen=True)
class FrameLayout(Generic[Frame]):
    """The wire format of a frame buffer dataclass: its fields' dtypes and shapes, in declaration order."""

    frame_type: type[Frame]
    fields: tuple[FieldLayout, ...]

    @staticmethod
    def of(frame_type: type[Frame], specs: dict[str, _FieldSpec]) -> "FrameLayout[Frame]":
        """Build the layout of *frame_type* from a ``{field name: (dtype, shape)}`` dict.

        Raises:
            TypeError: If *specs* does not describe exactly the dataclass fields of *frame_type*.
        """
        names = [f.name for f in dataclasses.fields(frame_type)]  # type: ignore[arg-type]
        if set(names) != set(specs):
            raise TypeError(
                f"Layout of {frame_type.__name__} describes fields {sorted(specs)}, but the dataclass has {sorted(names)}."
            )
        return FrameLayout(
            frame_type, tuple(FieldLayout(name, np.dtype(specs[name][0]), specs[name][1]) for name in names)
        )

    @cached_property
    def _spans(self) -> tuple[tuple[FieldLayout, int, int], ...]:
        """``(field, start, end)`` byte range of every field in a serialized frame."""
        spans = []
        offset = 0
        for field in self.fields:
            spans.append((field, offset, offset + field.nbytes))
            offset += field.nbytes
        return tuple(spans)

    @property
    def nbytes(self) -> int:
        """Size of a serialized frame in bytes."""
        return self._spans[-1][2] if self._spans else 0

    def describe(self) -> list[list[Any]]:
        """A JSON-serializable ``[[name, dtype, shape], ...]`` description of the wire format."""
        return [[field.name, field.dtype.str, list(field.shape)] for field in self.fields]

    def validate(self, frame: Any) -> None:
        """Check that *frame* matches this layout.

        The wire format is a bare concatenation of the field bytes, so a field with an
        unexpected dtype or shape would silently shift every field after it. This check turns
        that into an error at the publishing side.

        Raises:
            TypeError: If *frame* is not of this layout's frame type, or if a field is not a
                numpy array.
            ValueError: If a field's dtype or shape differs from the layout.
        """
        if type(frame) is not self.frame_type:
            raise TypeError(f"Expected a {self.frame_type.__name__} instance, but got {type(frame).__name__}.")
        for field in self.fields:
            arr = getattr(frame, field.name)
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Field '{field.name}' must be a numpy array, but is {type(arr).__name__}.")
            if arr.dtype != field.dtype:
                raise ValueError(
                    f"Field '{field.name}' has dtype {arr.dtype}, but the wire format expects {field.dtype}."
                )
            if arr.shape != field.shape:
                raise ValueError(
                    f"Field '{field.name}' has shape {arr.shape}, but the wire format expects {field.shape}."
                )

    def write_into(self, frame: Frame, buf: Any) -> None:
        """Copy every field of a validated *frame* into *buf* (``nbytes`` long), field by field.

        Args:
            frame: A frame that passed :meth:`validate`.
            buf: A writable buffer supporting slice assignment of ``bytes``. A ``memoryview`` is
                filled straight from the arrays' memory, without a temporary ``bytes`` copy per field.
        """
        for field, start, end in self._spans:
            arr = getattr(frame, field.name)
            if isinstance(buf, memoryview) and arr.flags.c_contiguous:
                buf[start:end] = memoryview(arr).cast("B")
            else:
                # Zenoh's SHM buffers only accept bytes, as do non-contiguous arrays.
                buf[start:end] = arr.tobytes()

    def deserialize(self, data: bytes) -> Frame:
        """Deserialize raw bytes back into a frame buffer dataclass instance.

        The returned arrays are **read-only views** into ``data``.
        Call ``.copy()`` on a field if you need a writable array, e.g. to draw on a
        received image in place.

        Args:
            data: Raw bytes laid out as the concatenation of the fields in declaration order.

        Returns:
            A new dataclass instance whose numpy arrays are read-only views into ``data``.

        Raises:
            ValueError: If ``data`` does not have exactly :attr:`nbytes` bytes. Silently
                accepting a mismatch would return a frame with garbage in (some of) its fields.
        """
        if len(data) != self.nbytes:
            raise ValueError(
                f"Cannot deserialize a {self.frame_type.__name__}: got {len(data)} bytes, "
                f"but its field layout describes {self.nbytes} bytes. "
                "Publisher and receiver disagree on the wire format."
            )

        kwargs: dict[str, np.ndarray] = {}
        for field, start, end in self._spans:
            # np.frombuffer views into data (rather than slicing it, which copies) and
            # keeps it alive through arr.base.  The view is read-only because bytes is.
            count = (end - start) // field.dtype.itemsize
            kwargs[field.name] = np.frombuffer(data, dtype=field.dtype, count=count, offset=start).reshape(field.shape)
        return self.frame_type(**kwargs)


@dataclass(frozen=True)
class PublisherMetadata:
    """Static information a publisher announces alongside its frames.

    Besides the camera's resolution and fps, it describes the wire format of every key expression
    the publisher writes to, so a receiver configured differently from its publisher fails with a
    clear error up front instead of timing out or failing on every frame.
    """

    resolution: tuple[int, int]
    fps: float
    # Key expression -> FrameLayout.describe() of the frames published on it.
    streams: dict[str, list[list[Any]]]

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    def from_json(data: str) -> "PublisherMetadata":
        fields = json.loads(data)
        return PublisherMetadata(
            resolution=(int(fields["resolution"][0]), int(fields["resolution"][1])),
            fps=float(fields["fps"]),
            streams=fields["streams"],
        )

    def check_stream(self, key_expr: str, layout: FrameLayout[Any]) -> None:
        """Check that the publisher writes frames with *layout* to *key_expr*.

        Raises:
            ValueError: If the publisher does not publish on *key_expr*, or with a different layout.
        """
        hint = "Make sure the receiver is created with options (e.g. enable_pointcloud) the publisher supports."
        if key_expr not in self.streams:
            raise ValueError(
                f"The publisher does not publish '{key_expr}' (it publishes {sorted(self.streams)}). {hint}"
            )
        published = self.streams[key_expr]
        expected = layout.describe()
        if published != expected:
            raise ValueError(
                f"The publisher writes a different wire format to '{key_expr}' than this receiver expects.\n"
                f"  Published: {_format_description(published)}\n"
                f"  Expected:  {_format_description(expected)}\n"
                f"{hint}"
            )


def _format_description(description: list[list[Any]]) -> str:
    return ", ".join(f"{name}: {dtype}{tuple(shape)}" for name, dtype, shape in description)


@dataclass
class BaseFrameBuffer:
    """Base frame buffer containing timestamp and frame ID for synchronization."""

    # Frame ID for synchronization (monotonically increasing)
    frame_id: np.ndarray
    # Timestamp when the frame was captured (seconds since epoch)
    frame_timestamp: np.ndarray

    @staticmethod
    def _header_specs() -> dict[str, _FieldSpec]:
        return {"frame_id": (np.uint64, (1,)), "frame_timestamp": (np.float64, (1,))}


@dataclass
class RGBFrameBuffer(BaseFrameBuffer):
    """Frame buffer containing RGB image data and camera intrinsics."""

    # Color image data (height x width x channels)
    rgb: np.ndarray
    # Intrinsic camera parameters (camera matrix)
    intrinsics: np.ndarray

    @classmethod
    def _field_specs(cls, width: int, height: int) -> dict[str, _FieldSpec]:
        """The ``(dtype, shape)`` of every field; subclasses extend their parent's specs."""
        return {
            **cls._header_specs(),
            "rgb": (np.uint8, (height, width, 3)),
            "intrinsics": (np.float64, (3, 3)),
        }

    @classmethod
    def layout(cls, width: int, height: int) -> FrameLayout[Self]:
        """The wire format of this frame buffer for images of the given resolution."""
        return FrameLayout.of(cls, cls._field_specs(width, height))


@dataclass
class RGBDFrameBuffer(RGBFrameBuffer):
    """Frame buffer containing RGB-D data (color + depth)."""

    # Depth image data (height x width)
    depth_image: np.ndarray
    # Depth map (height x width)
    depth: np.ndarray

    @classmethod
    def _field_specs(cls, width: int, height: int) -> dict[str, _FieldSpec]:
        return {
            **super()._field_specs(width, height),
            "depth_image": (np.uint8, (height, width, 3)),
            "depth": (np.float32, (height, width)),
        }


@dataclass
class StereoRGBDFrameBuffer(RGBDFrameBuffer):
    """Frame buffer containing stereo RGB-D data (left + right cameras)."""

    # Right camera RGB image
    rgb_right: np.ndarray
    # Right camera intrinsics
    intrinsics_right: np.ndarray
    # Pose of right camera in left camera frame
    pose_right_in_left: np.ndarray

    @classmethod
    def _field_specs(cls, width: int, height: int) -> dict[str, _FieldSpec]:
        return {
            **super()._field_specs(width, height),
            "rgb_right": (np.uint8, (height, width, 3)),
            "intrinsics_right": (np.float64, (3, 3)),
            "pose_right_in_left": (np.float64, (4, 4)),
        }


@dataclass
class ZedFrameBuffer(StereoRGBDFrameBuffer):
    """Frame buffer containing Zed camera data including camera pose."""

    # Camera pose in world coordinates
    camera_pose: np.ndarray

    @classmethod
    def _field_specs(cls, width: int, height: int) -> dict[str, _FieldSpec]:
        return {**super()._field_specs(width, height), "camera_pose": (np.float64, (4, 4))}


@dataclass
class PointCloudFrameBuffer(BaseFrameBuffer):
    """A camera's point cloud, published next to its frames with the same ``frame_id``.

    The buffers have one point slot per pixel, since cameras can return sparse point clouds.
    """

    # Point cloud positions (N x 3); entries past num_valid_points are NaN
    point_cloud_positions: np.ndarray
    # Point cloud colors (N x 3)
    point_cloud_colors: np.ndarray
    # Number of valid points in the point cloud
    num_valid_points: np.ndarray

    @classmethod
    def layout(cls, width: int, height: int) -> FrameLayout[Self]:
        return FrameLayout.of(
            cls,
            {
                **cls._header_specs(),
                "point_cloud_positions": (np.float32, (height * width, 3)),
                "point_cloud_colors": (np.uint8, (height * width, 3)),
                "num_valid_points": (np.int32, (1,)),
            },
        )


@dataclass
class SpatialMapBuffer(BaseFrameBuffer):
    """Buffer containing spatial map data from Zed camera."""

    # Amount of chunks in the spatial map
    num_chunks: np.ndarray
    # Array indicating which chunks have been updated
    chunks_updated: np.ndarray
    # Size of each chunk (number of points)
    chunk_sizes: np.ndarray
    # Arrays of concatenated chunk data (point positions and colors)
    point_positions: np.ndarray
    point_colors: np.ndarray

    @classmethod
    def layout(cls, max_chunks: int, max_points: int) -> FrameLayout[Self]:
        return FrameLayout.of(
            cls,
            {
                **cls._header_specs(),
                "num_chunks": (np.int32, (1,)),
                "chunks_updated": (np.bool_, (max_chunks,)),
                "chunk_sizes": (np.int32, (max_chunks,)),
                "point_positions": (np.float32, (max_points, 3)),
                "point_colors": (np.uint8, (max_points, 3)),
            },
        )
