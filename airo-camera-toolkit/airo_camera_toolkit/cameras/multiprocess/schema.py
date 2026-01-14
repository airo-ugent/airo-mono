"""Schemas define the serialization and deserialization logic for buffers that are shared over shared memory."""

import time
from abc import ABC, abstractmethod
from typing import Optional, Type

import numpy as np
from airo_camera_toolkit.cameras.multiprocess.buffer import (
    Buffer,
    CameraMetadataBuffer,
    DepthFrameBuffer,
    PointCloudBuffer,
    RGBFrameBuffer,
    StereoRGBFrameBuffer,
)
from airo_camera_toolkit.cameras.multiprocess.mixin import (
    CameraMixin,
    DepthMixin,
    Mixin,
    PointCloudMixin,
    RGBMixin,
    StereoRGBMixin,
)
from airo_camera_toolkit.interfaces import Camera, DepthCamera, RGBCamera, RGBDCamera, StereoRGBDCamera
from airo_typing import CameraResolutionType, PointCloud


class Schema(ABC):
    def __init__(self, topic: str, type: Type[Buffer]) -> None:
        """Initialize a new schema with a given topic name and buffer type.

        Args:
            topic: A unique name.
            type: The Buffer class type."""
        self._topic = topic
        self._buffer_type = type

        self._buffer: Optional[Buffer] = None

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def buffer(self) -> Buffer:
        """Retrieve the buffer. If it is not allocated, this raises a ValueError. Call allocate_empty first."""
        self._assert_buffer_allocated()
        assert self._buffer is not None and isinstance(self._buffer, Buffer)  # for mypy
        return self._buffer

    def _assert_buffer_allocated(self) -> None:
        if self._buffer is None:
            raise ValueError("The internal buffer has not yet been allocated by the schema.")

    @abstractmethod
    def allocate_empty(self, resolution: CameraResolutionType) -> None:
        """Allocate an empty buffer (cf. malloc in C) with the required size.

        This method must be used on the publisher and receiver side to allocate memory.

        Args:
            resolution: The camera resolution."""

    @abstractmethod
    def fill_from_camera(self, camera: Camera) -> None:
        """Fill the previously allocated buffer with relevant data (serializing it into numpy arrays) obtained via the camera object.

        This method must be used on the publisher side - as it is the one that has access to the camera.

        Args:
            camera: The camera instance."""

    @abstractmethod
    def read_into_receiver(self, frame: Buffer, receiver: Mixin) -> None:
        """Read the data from the provided frame, deserialize it, and set the necessary fields on the receiver.

        The fields that should be set are determined by the mixin. See mixin.py.

        Args:
            frame: The buffer obtained from the receiver, who read it from shared memory.
            receiver: The receiver object, which will be filled in with the new data."""

    def __repr__(self) -> str:
        return self.__class__.__name__


class CameraSchema(Schema):
    """Camera metadata. See CameraMetadataBuffer and CameraMixin."""

    def __init__(self) -> None:
        super().__init__("metadata", CameraMetadataBuffer)

    def allocate_empty(self, resolution: CameraResolutionType) -> None:
        self._buffer = CameraMetadataBuffer(
            timestamp=np.empty((1,), dtype=np.float64),
            resolution=np.empty_like(resolution, dtype=np.uint32),
            intrinsics_matrix=np.empty((3, 3), dtype=np.float32),
            fps=np.empty((1,), dtype=np.float32),
        )

    def fill_from_camera(self, camera: Camera) -> None:
        self._assert_buffer_allocated()
        assert isinstance(camera, RGBCamera)  # for mypy
        assert isinstance(self._buffer, CameraMetadataBuffer)  # for mypy
        self._buffer.timestamp[0] = time.time()
        self._buffer.resolution = np.array(camera.resolution).astype(np.uint32)
        self._buffer.intrinsics_matrix = camera.intrinsics_matrix().astype(np.float32)
        self._buffer.fps[0] = camera.fps

    def read_into_receiver(self, frame: CameraMetadataBuffer, receiver: Mixin) -> None:
        assert isinstance(receiver, CameraMixin)  # for mypy
        receiver._metadata_frame = frame


class RGBSchema(Schema):
    """RGB data. See RGBFrameBuffer and RGBMixin."""

    def __init__(self) -> None:
        super().__init__("rgb", RGBFrameBuffer)

    def allocate_empty(self, resolution: CameraResolutionType) -> None:
        width, height = resolution
        self._buffer = RGBFrameBuffer(
            rgb=np.empty((height, width, 3), dtype=np.uint8),
        )

    def fill_from_camera(self, camera: Camera) -> None:
        self._assert_buffer_allocated()
        assert isinstance(camera, RGBCamera)  # for mypy
        assert isinstance(self._buffer, RGBFrameBuffer)  # for mypy
        image = camera._retrieve_rgb_image_as_int()
        self._buffer.rgb = image

    def read_into_receiver(self, frame: RGBFrameBuffer, receiver: Mixin) -> None:
        assert isinstance(receiver, RGBMixin)  # for mypy
        receiver._rgb_frame = frame


class StereoRGBSchema(Schema):
    """Stereo RGB data. See StereoRGBFrameBuffer and StereoRGBMixin."""

    def __init__(self) -> None:
        super().__init__("stereo", StereoRGBFrameBuffer)

    def allocate_empty(self, resolution: CameraResolutionType) -> None:
        width, height = resolution

        self._buffer = StereoRGBFrameBuffer(
            rgb_left=np.empty((height, width, 3), dtype=np.uint8),
            rgb_right=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics_left=np.empty((3, 3), dtype=np.float64),
            intrinsics_right=np.empty((3, 3), dtype=np.float64),
            pose_right_in_left=np.empty((4, 4), dtype=np.float64),
        )

    def fill_from_camera(self, camera: Camera) -> None:
        self._assert_buffer_allocated()
        assert isinstance(camera, StereoRGBDCamera)  # for mypy
        assert isinstance(self._buffer, StereoRGBFrameBuffer)  # for mypy

        image_left = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.LEFT_RGB)
        image_right = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.RIGHT_RGB)
        intrinsics_left = camera.intrinsics_matrix(StereoRGBDCamera.LEFT_RGB)
        intrinsics_right = camera.intrinsics_matrix(StereoRGBDCamera.RIGHT_RGB)
        pose_right_in_left = camera.pose_of_right_view_in_left_view

        self._buffer.rgb_left = image_left
        self._buffer.rgb_right = image_right
        self._buffer.intrinsics_left = intrinsics_left
        self._buffer.intrinsics_right = intrinsics_right
        self._buffer.pose_right_in_left = pose_right_in_left

    def read_into_receiver(self, frame: StereoRGBFrameBuffer, receiver: Mixin) -> None:
        assert isinstance(receiver, StereoRGBMixin)  # for mypy
        receiver._stereo_frame = frame


class DepthSchema(Schema):
    """Depth data. See DepthFrameBuffer and DepthMixin."""

    def __init__(self) -> None:
        super().__init__("depth", DepthFrameBuffer)

    def allocate_empty(self, resolution: CameraResolutionType) -> None:
        width, height = resolution

        self._buffer = DepthFrameBuffer(
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth_map=np.empty((height, width), dtype=np.float32),
            confidence_map=np.empty((height, width), dtype=np.float32),
        )

    def fill_from_camera(self, camera: Camera) -> None:
        self._assert_buffer_allocated()
        assert isinstance(camera, DepthCamera)  # for mypy
        assert isinstance(self._buffer, DepthFrameBuffer)  # for mypy

        depth_image = camera._retrieve_depth_image()
        depth_map = camera._retrieve_depth_map()
        confidence_map = camera._retrieve_confidence_map()

        self._buffer.depth_image = depth_image
        self._buffer.depth_map = depth_map
        self._buffer.confidence_map = confidence_map

    def read_into_receiver(self, frame: DepthFrameBuffer, receiver: Mixin) -> None:
        assert isinstance(receiver, DepthMixin)  # for mypy
        receiver._depth_frame = frame


class PointCloudSchema(Schema):
    """Point cloud data. See PointCloudBuffer and PointCloudMixin."""

    def __init__(self) -> None:
        super().__init__("pcd", PointCloudBuffer)

    def allocate_empty(self, resolution: CameraResolutionType) -> None:
        width, height = resolution

        self._buffer = PointCloudBuffer(
            point_cloud_positions=np.empty((height * width, 3), dtype=np.float32),
            point_cloud_colors=np.empty((height * width, 3), dtype=np.uint8),
            point_cloud_valid=np.empty((1,), dtype=np.uint32),
        )

    def fill_from_camera(self, camera: Camera) -> None:
        self._assert_buffer_allocated()
        assert isinstance(camera, RGBDCamera)  # for mypy
        assert isinstance(self._buffer, PointCloudBuffer)  # for mypy

        point_cloud = camera._retrieve_colored_point_cloud()

        self._buffer.point_cloud_positions[: point_cloud.points.shape[0]] = point_cloud.points
        if point_cloud.colors is not None:
            self._buffer.point_cloud_colors[: point_cloud.colors.shape[0]] = point_cloud.colors
        else:
            self._buffer.point_cloud_colors[: point_cloud.points.shape[0]] = 0  # If no colors, use black.
        self._buffer.point_cloud_valid[0] = point_cloud.points.shape[0]

    def read_into_receiver(self, frame: PointCloudBuffer, receiver: Mixin) -> None:
        assert isinstance(receiver, PointCloudMixin)  # for mypy
        num_points = frame.point_cloud_valid.item()
        positions = frame.point_cloud_positions[:num_points]
        colors = frame.point_cloud_colors[:num_points]
        point_cloud = PointCloud(positions, colors)

        receiver._point_cloud_frame = point_cloud
