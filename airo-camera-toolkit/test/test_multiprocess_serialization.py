import numpy as np
from airo_typing import PointCloud

"""These tests checkwhether airo_camera_toolkit.multiprocess.schema serialization/deserialization correctly preserves NumPy arrays."""
from airo_camera_toolkit.cameras.multiprocess.schema import (
    CameraSchema,
    DepthSchema,
    PointCloudSchema,
    RGBSchema,
    StereoRGBSchema,
)
from airo_camera_toolkit.interfaces import StereoRGBDCamera


def test_camera_schema_serialization_deserialization():
    schema = CameraSchema()
    resolution = (640, 480)
    buffer = schema.allocate(resolution)

    class DummyCamera:
        def __init__(self):
            self._resolution = resolution
            self._fps = 30.0
            self._intrinsics_matrix = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
            self._timestamp = 1234567890.0

        @property
        def resolution(self):
            return self._resolution

        @property
        def fps(self):
            return self._fps

        def intrinsics_matrix(self):
            return self._intrinsics_matrix

        def get_current_timestamp(self):
            return self._timestamp

    camera = DummyCamera()

    schema.serialize(camera, buffer)

    class DummyReceiver:
        pass

    receiver = DummyReceiver()
    schema.deserialize(buffer, receiver)

    assert buffer.resolution.tolist() == list(resolution)
    assert buffer.fps[0] == camera.fps
    np.testing.assert_array_almost_equal(buffer.intrinsics_matrix, camera.intrinsics_matrix())
    assert buffer.timestamp[0] != 0.0  # Timestamp should be set during serialization
    assert receiver._metadata_frame is buffer
    np.testing.assert_array_almost_equal(receiver._metadata_frame.resolution, buffer.resolution)
    np.testing.assert_array_almost_equal(receiver._metadata_frame.intrinsics_matrix, buffer.intrinsics_matrix)


def test_rgb_schema_serialization_deserialization():
    schema = RGBSchema()
    resolution = (640, 480)
    buffer = schema.allocate(resolution)

    class DummyCamera:
        def __init__(self):
            self._rgb_image = np.random.randint(0, 256, (resolution[1], resolution[0], 3), dtype=np.uint8)

        def _retrieve_rgb_image_as_int(self):
            return self._rgb_image

    camera = DummyCamera()

    schema.serialize(camera, buffer)

    class DummyReceiver:
        pass

    receiver = DummyReceiver()
    schema.deserialize(buffer, receiver)

    np.testing.assert_array_equal(buffer.rgb, camera._rgb_image)
    assert receiver._rgb_frame is buffer
    np.testing.assert_array_equal(receiver._rgb_frame.rgb, buffer.rgb)


def test_stereo_rgb_schema_serialization_deserialization():
    schema = StereoRGBSchema()
    resolution = (640, 480)
    buffer = schema.allocate(resolution)

    class DummyCamera:
        def __init__(self):
            self._rgb_left = np.random.randint(0, 256, (resolution[1], resolution[0], 3), dtype=np.uint8)
            self._rgb_right = np.random.randint(0, 256, (resolution[1], resolution[0], 3), dtype=np.uint8)
            self._intrinsics_left = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)
            self._intrinsics_right = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float64)

        def _retrieve_rgb_image_as_int(self, view):
            if view == StereoRGBDCamera.LEFT_RGB:
                return self._rgb_left
            else:
                return self._rgb_right

        def intrinsics_matrix(self, view):
            if view == "left_rgb":
                return self._intrinsics_left
            else:
                return self._intrinsics_right

        def pose_of_right_view_in_left_view(self):
            return np.eye(4, dtype=np.float64)

    camera = DummyCamera()
    schema.serialize(camera, buffer)

    class DummyReceiver:
        pass

    receiver = DummyReceiver()
    schema.deserialize(buffer, receiver)
    np.testing.assert_array_equal(buffer.rgb_left, camera._rgb_left)
    np.testing.assert_array_equal(buffer.rgb_right, camera._rgb_right)
    np.testing.assert_array_almost_equal(buffer.intrinsics_left, camera._intrinsics_left)
    np.testing.assert_array_almost_equal(buffer.intrinsics_right, camera._intrinsics_right)
    assert receiver._stereo_frame is buffer
    np.testing.assert_array_equal(receiver._stereo_frame.rgb_left, buffer.rgb_left)
    np.testing.assert_array_equal(receiver._stereo_frame.rgb_right, buffer.rgb_right)
    np.testing.assert_array_almost_equal(receiver._stereo_frame.intrinsics_left, buffer.intrinsics_left)
    np.testing.assert_array_almost_equal(receiver._stereo_frame.intrinsics_right, buffer.intrinsics_right)


def test_depth_schema_serialization_deserialization():
    schema = DepthSchema()
    resolution = (640, 480)
    buffer = schema.allocate(resolution)

    class DummyCamera:
        def __init__(self):
            self._depth_image = (
                np.random.rand(resolution[1], resolution[0]).astype(np.float32) * 10.0
            )  # Depth in meters

        def _retrieve_depth_image(self):
            return self._depth_image

        def _retrieve_depth_map(self):
            return self._depth_image.astype(np.uint16) / 10

        def _retrieve_confidence_map(self):
            return np.ones((resolution[1], resolution[0]), dtype=np.float32)

    camera = DummyCamera()

    schema.serialize(camera, buffer)

    class DummyReceiver:
        pass

    receiver = DummyReceiver()
    schema.deserialize(buffer, receiver)

    np.testing.assert_array_almost_equal(buffer.depth_image, camera._depth_image)
    assert receiver._depth_frame is buffer
    np.testing.assert_array_almost_equal(receiver._depth_frame.depth_image, buffer.depth_image)


def test_point_cloud_schema_serialization_deserialization():
    schema = PointCloudSchema()
    resolution = (640, 480)
    buffer = schema.allocate(resolution)

    class DummyCamera:
        def __init__(self):
            self._point_cloud_positions = np.random.rand(resolution[1] * resolution[0], 3).astype(
                np.float32
            )  # XYZ positions
            self._point_cloud_colors = np.random.randint(
                0, 256, (resolution[1] * resolution[0], 3), dtype=np.uint8
            )  # RGB colors

        def _retrieve_colored_point_cloud(self):
            return PointCloud(self._point_cloud_positions, self._point_cloud_colors)

    camera = DummyCamera()

    schema.serialize(camera, buffer)

    class DummyReceiver:
        pass

    receiver = DummyReceiver()
    schema.deserialize(buffer, receiver)
    np.testing.assert_array_almost_equal(buffer.point_cloud_positions, camera._point_cloud_positions)
    assert isinstance(receiver._point_cloud_frame, PointCloud)
    np.testing.assert_array_almost_equal(receiver._point_cloud_frame.points, buffer.point_cloud_positions)
    np.testing.assert_array_almost_equal(receiver._point_cloud_frame.colors, buffer.point_cloud_colors)
