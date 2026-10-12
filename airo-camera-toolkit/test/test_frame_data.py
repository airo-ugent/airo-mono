"""Unit tests for frame_data serialization round-trips."""

import dataclasses
import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from airo_camera_toolkit.cameras.multiprocess.frame_data import (
    FrameLayout,
    PointCloudFrameBuffer,
    PublisherMetadata,
    RGBDFrameBuffer,
    RGBFrameBuffer,
    SpatialMapBuffer,
    StereoRGBDFrameBuffer,
    ZedFrameBuffer,
)

W, H = 64, 48  # small resolution for fast tests


def serialize_frame(layout: FrameLayout, obj: Any) -> bytes:
    """Serialize *obj* like ZenohWriter does without SHM: into a memoryview of a bytearray."""
    layout.validate(obj)
    buf = bytearray(layout.nbytes)
    layout.write_into(obj, memoryview(buf))
    return bytes(buf)


def _round_trip(layout: FrameLayout, obj: Any) -> Any:
    return layout.deserialize(serialize_frame(layout, obj))


def _raw_concat(obj: Any) -> bytes:
    """Concatenate the fields of *obj* without validation, to simulate a misbehaving publisher."""
    return b"".join(getattr(obj, f.name).tobytes() for f in dataclasses.fields(obj))


def _assert_frame_equal(a, b) -> None:
    for f in dataclasses.fields(a):
        np.testing.assert_array_equal(getattr(a, f.name), getattr(b, f.name), err_msg=f"field '{f.name}' mismatch")


@dataclass
class _Scalar:
    """A minimal frame buffer."""

    value: np.ndarray


_SCALAR_LAYOUT = FrameLayout.of(_Scalar, {"value": (np.float64, (1,))})


# ---------------------------------------------------------------------------
# Publisher metadata
# ---------------------------------------------------------------------------


def _metadata(**streams: FrameLayout) -> PublisherMetadata:
    return PublisherMetadata(
        resolution=(W, H),
        fps=30.0,
        streams={key_expr: layout.describe() for key_expr, layout in streams.items()},
    )


def test_publisher_metadata_json_round_trip():
    metadata = _metadata(camera=RGBDFrameBuffer.layout(W, H), camera_pointcloud=PointCloudFrameBuffer.layout(W, H))
    assert PublisherMetadata.from_json(metadata.to_json()) == metadata


def test_publisher_metadata_accepts_matching_stream():
    metadata = PublisherMetadata.from_json(_metadata(camera=RGBDFrameBuffer.layout(W, H)).to_json())
    metadata.check_stream("camera", RGBDFrameBuffer.layout(W, H))


def test_publisher_metadata_rejects_missing_stream():
    metadata = _metadata(camera=RGBDFrameBuffer.layout(W, H))
    with pytest.raises(ValueError, match="does not publish 'camera_spatial_map'"):
        metadata.check_stream("camera_spatial_map", RGBDFrameBuffer.layout(W, H))


def test_publisher_metadata_rejects_different_layout():
    """E.g. an RGBD receiver for a stereo publisher."""
    metadata = _metadata(camera=StereoRGBDFrameBuffer.layout(W, H))
    with pytest.raises(ValueError, match="different wire format"):
        metadata.check_stream("camera", RGBDFrameBuffer.layout(W, H))


def test_publisher_metadata_rejects_different_resolution():
    metadata = _metadata(camera=RGBFrameBuffer.layout(W, H))
    with pytest.raises(ValueError, match="different wire format"):
        metadata.check_stream("camera", RGBFrameBuffer.layout(W * 2, H * 2))


# ---------------------------------------------------------------------------
# Frame buffers
# ---------------------------------------------------------------------------


_LAYOUTS = [
    RGBFrameBuffer.layout(W, H),
    RGBDFrameBuffer.layout(W, H),
    StereoRGBDFrameBuffer.layout(W, H),
    ZedFrameBuffer.layout(W, H),
    PointCloudFrameBuffer.layout(W, H),
    SpatialMapBuffer.layout(10, 1000),
]


def _random_frame(layout: FrameLayout) -> Any:
    rng = np.random.default_rng(0)
    return layout.frame_type(
        **{field.name: rng.integers(0, 100, field.shape).astype(field.dtype) for field in layout.fields}
    )


@pytest.mark.parametrize("layout", _LAYOUTS, ids=lambda layout: layout.frame_type.__name__)
def test_frame_buffer_round_trip(layout: FrameLayout):
    obj = _random_frame(layout)
    _assert_frame_equal(obj, _round_trip(layout, obj))


@pytest.mark.parametrize("layout", _LAYOUTS, ids=lambda layout: layout.frame_type.__name__)
def test_bytes_buffers_get_the_same_wire_format(layout: FrameLayout):
    """Buffers that only accept bytes (like Zenoh's SHM buffers) are filled without memoryviews."""
    obj = _random_frame(layout)
    buf = bytearray(layout.nbytes)
    layout.write_into(obj, buf)
    assert bytes(buf) == serialize_frame(layout, obj)


# ---------------------------------------------------------------------------
# Deserialized arrays are read-only views into the payload, not copies
# ---------------------------------------------------------------------------


def test_deserialized_arrays_are_read_only_views():
    obj = _Scalar(value=np.array([30.0], dtype=np.float64))
    data = serialize_frame(_SCALAR_LAYOUT, obj)
    result = _SCALAR_LAYOUT.deserialize(data)

    assert not result.value.flags.writeable
    with pytest.raises(ValueError):
        result.value[0] = 99.0

    # A copy is writable and independent of the payload.
    writable = result.value.copy()
    writable[0] = 99.0
    assert _SCALAR_LAYOUT.deserialize(data).value.item() == pytest.approx(30.0)


def test_deserialized_arrays_do_not_copy_the_payload():
    """The fields must view into the payload bytes rather than copy them."""
    rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    obj = RGBFrameBuffer(
        frame_id=np.array([7], dtype=np.uint64),
        frame_timestamp=np.array([1.0], dtype=np.float64),
        rgb=rgb,
        intrinsics=np.eye(3, dtype=np.float64),
    )
    data = serialize_frame(RGBFrameBuffer.layout(W, H), obj)
    result = RGBFrameBuffer.layout(W, H).deserialize(data)

    def root_buffer(arr):
        """Follow the view chain (reshape adds one level) to the object owning the memory."""
        while isinstance(arr, np.ndarray):
            arr = arr.base
        return arr

    for field in ("frame_id", "frame_timestamp", "rgb", "intrinsics"):
        assert root_buffer(getattr(result, field)) is data, f"field '{field}' does not view into the payload"
    np.testing.assert_array_equal(result.rgb, rgb)


def test_deserialized_views_outlive_the_local_payload_reference():
    """Fields keep the payload alive, so views stay valid after it goes out of scope."""

    def make():
        obj = _Scalar(value=np.array([30.0], dtype=np.float64))
        return _round_trip(_SCALAR_LAYOUT, obj)

    result = make()
    gc.collect()
    assert result.value.item() == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Schema validation: a field whose dtype or shape drifted from the layout
# would silently shift every field after it on the wire.
# ---------------------------------------------------------------------------


def _valid_rgbd_frame():
    return RGBDFrameBuffer(
        frame_id=np.array([1], dtype=np.uint64),
        frame_timestamp=np.array([1.0], dtype=np.float64),
        rgb=np.zeros((H, W, 3), dtype=np.uint8),
        intrinsics=np.eye(3, dtype=np.float64),
        depth_image=np.zeros((H, W, 3), dtype=np.uint8),
        depth=np.zeros((H, W), dtype=np.float32),
    )


def test_validate_frame_accepts_matching_frame():
    RGBDFrameBuffer.layout(W, H).validate(_valid_rgbd_frame())


def test_validate_frame_rejects_wrong_dtype():
    obj = _valid_rgbd_frame()
    obj.depth = obj.depth.astype(np.float64)
    with pytest.raises(ValueError, match="depth"):
        RGBDFrameBuffer.layout(W, H).validate(obj)


def test_validate_frame_rejects_wrong_shape():
    obj = _valid_rgbd_frame()
    obj.rgb = np.zeros((H, W, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="rgb"):
        RGBDFrameBuffer.layout(W, H).validate(obj)


def test_validate_frame_rejects_wrong_type():
    with pytest.raises(TypeError):
        RGBDFrameBuffer.layout(W, H).validate(_Scalar(value=np.array([1.0])))


def test_deserialize_frame_rejects_wrong_length():
    """A dtype mismatch used to deserialize silently, corrupting every later field."""
    obj = _valid_rgbd_frame()
    obj.depth = obj.depth.astype(np.float64)
    data = _raw_concat(obj)  # unvalidated: simulates a publisher without validation
    with pytest.raises(ValueError, match="wire format"):
        RGBDFrameBuffer.layout(W, H).deserialize(data)


def test_layout_must_describe_exactly_the_dataclass_fields():
    with pytest.raises(TypeError, match="describes fields"):
        FrameLayout.of(_Scalar, {"value": (np.float64, (1,)), "extra": (np.int32, (1,))})
