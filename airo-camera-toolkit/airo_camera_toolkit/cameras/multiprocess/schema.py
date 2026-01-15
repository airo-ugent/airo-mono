"""Schemas define the serialization and deserialization logic for buffers that are shared over shared memory."""

import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

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

B = TypeVar("B", bound=Buffer)
C = TypeVar("C", bound=Camera)
M = TypeVar("M", bound=Mixin)


class Schema(Generic[B, C, M], ABC):
    def __init__(self, topic: str) -> None:
        """Initialize a new schema with a given topic name and buffer type.

        Args:
            topic: A unique name."""
        self._topic = topic

    @property
    def topic(self) -> str:
        return self._topic

    @abstractmethod
    def allocate(self, resolution: CameraResolutionType) -> B:
        """Allocate an empty buffer (cf. malloc in C) with the required size.

        This method must be used on the publisher and receiver side to allocate memory.

        Args:
            resolution: The camera resolution."""

    @abstractmethod
    def serialize(self, camera: C, buffer: B) -> None:
        """Fill a previously allocated buffer with relevant data (serializing it into numpy arrays) obtained via the camera object.

        This method must be used on the publisher side - as it is the one that has access to the camera.

        Args:
            camera: The camera instance.
            buffer: The buffer instance."""

    @abstractmethod
    def deserialize(self, frame: B, receiver: M) -> None:
        """Read the data from the provided frame, deserialize it, and set the necessary fields on the receiver.

        The fields that should be set are determined by the mixin. See mixin.py.

        Args:
            frame: The buffer obtained from the receiver, who read it from shared memory.
            receiver: The receiver object, which will be filled in with the new data."""

    def __repr__(self) -> str:
        return self.__class__.__name__


class CameraSchema(Schema[CameraMetadataBuffer, RGBCamera, CameraMixin]):
    """Camera metadata. See CameraMetadataBuffer and CameraMixin."""

    def __init__(self) -> None:
        super().__init__("metadata")

    def allocate(self, resolution: CameraResolutionType) -> CameraMetadataBuffer:
        return CameraMetadataBuffer(
            timestamp=np.empty((1,), dtype=np.float64),
            resolution=np.empty_like(resolution, dtype=np.uint32),
            intrinsics_matrix=np.empty((3, 3), dtype=np.float32),
            fps=np.empty((1,), dtype=np.float32),
        )

    def serialize(self, camera: RGBCamera, buffer: CameraMetadataBuffer) -> None:
        buffer.timestamp[0] = time.time()
        buffer.resolution = np.array(camera.resolution).astype(np.uint32)
        buffer.intrinsics_matrix = camera.intrinsics_matrix().astype(np.float32)
        buffer.fps[0] = camera.fps

    def deserialize(self, frame: CameraMetadataBuffer, receiver: CameraMixin) -> None:
        receiver._metadata_frame = frame


class RGBSchema(Schema[RGBFrameBuffer, RGBCamera, RGBMixin]):
    """RGB data. See RGBFrameBuffer and RGBMixin."""

    def __init__(self) -> None:
        super().__init__("rgb")

    def allocate(self, resolution: CameraResolutionType) -> RGBFrameBuffer:
        width, height = resolution
        return RGBFrameBuffer(
            rgb=np.empty((height, width, 3), dtype=np.uint8),
        )

    def serialize(self, camera: RGBCamera, buffer: RGBFrameBuffer) -> None:
        image = camera._retrieve_rgb_image_as_int()
        buffer.rgb = image

    def deserialize(self, frame: RGBFrameBuffer, receiver: RGBMixin) -> None:
        receiver._rgb_frame = frame


class StereoRGBSchema(Schema[StereoRGBFrameBuffer, StereoRGBDCamera, StereoRGBMixin]):
    """Stereo RGB data. See StereoRGBFrameBuffer and StereoRGBMixin."""

    def __init__(self) -> None:
        super().__init__("stereo")

    def allocate(self, resolution: CameraResolutionType) -> StereoRGBFrameBuffer:
        width, height = resolution

        return StereoRGBFrameBuffer(
            rgb_left=np.empty((height, width, 3), dtype=np.uint8),
            rgb_right=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics_left=np.empty((3, 3), dtype=np.float64),
            intrinsics_right=np.empty((3, 3), dtype=np.float64),
            pose_right_in_left=np.empty((4, 4), dtype=np.float64),
        )

    def serialize(self, camera: StereoRGBDCamera, buffer: StereoRGBFrameBuffer) -> None:
        image_left = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.LEFT_RGB)
        image_right = camera._retrieve_rgb_image_as_int(StereoRGBDCamera.RIGHT_RGB)
        intrinsics_left = camera.intrinsics_matrix(StereoRGBDCamera.LEFT_RGB)
        intrinsics_right = camera.intrinsics_matrix(StereoRGBDCamera.RIGHT_RGB)
        pose_right_in_left = camera.pose_of_right_view_in_left_view

        buffer.rgb_left = image_left
        buffer.rgb_right = image_right
        buffer.intrinsics_left = intrinsics_left
        buffer.intrinsics_right = intrinsics_right
        buffer.pose_right_in_left = pose_right_in_left

    def deserialize(self, frame: StereoRGBFrameBuffer, receiver: StereoRGBMixin) -> None:
        receiver._stereo_frame = frame


class DepthSchema(Schema[DepthFrameBuffer, DepthCamera, DepthMixin]):
    """Depth data. See DepthFrameBuffer and DepthMixin."""

    def __init__(self) -> None:
        super().__init__("depth")

    def allocate(self, resolution: CameraResolutionType) -> DepthFrameBuffer:
        width, height = resolution

        return DepthFrameBuffer(
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth_map=np.empty((height, width), dtype=np.float32),
            confidence_map=np.empty((height, width), dtype=np.float32),
        )

    def serialize(self, camera: DepthCamera, buffer: DepthFrameBuffer) -> None:
        depth_image = camera._retrieve_depth_image()
        depth_map = camera._retrieve_depth_map()
        confidence_map = camera._retrieve_confidence_map()

        buffer.depth_image = depth_image
        buffer.depth_map = depth_map
        buffer.confidence_map = confidence_map

    def deserialize(self, frame: DepthFrameBuffer, receiver: DepthMixin) -> None:
        receiver._depth_frame = frame


class PointCloudSchema(Schema[PointCloudBuffer, RGBDCamera, PointCloudMixin]):
    """Point cloud data. See PointCloudBuffer and PointCloudMixin."""

    def __init__(self) -> None:
        super().__init__("pcd")

    def allocate(self, resolution: CameraResolutionType) -> PointCloudBuffer:
        width, height = resolution

        return PointCloudBuffer(
            point_cloud_positions=np.empty((height * width, 3), dtype=np.float32),
            point_cloud_colors=np.empty((height * width, 3), dtype=np.uint8),
            point_cloud_valid=np.empty((1,), dtype=np.uint32),
        )

    def serialize(self, camera: RGBDCamera, buffer: PointCloudBuffer) -> None:
        point_cloud = camera._retrieve_colored_point_cloud()

        buffer.point_cloud_positions[: point_cloud.points.shape[0]] = point_cloud.points
        if point_cloud.colors is not None:
            buffer.point_cloud_colors[: point_cloud.colors.shape[0]] = point_cloud.colors
        else:
            buffer.point_cloud_colors[: point_cloud.points.shape[0]] = 0  # If no colors, use black.
        buffer.point_cloud_valid[0] = point_cloud.points.shape[0]

    def deserialize(self, frame: PointCloudBuffer, receiver: PointCloudMixin) -> None:
        num_points = frame.point_cloud_valid.item()
        positions = frame.point_cloud_positions[:num_points]
        colors = frame.point_cloud_colors[:num_points]
        point_cloud = PointCloud(positions, colors)

        receiver._point_cloud_frame = point_cloud
