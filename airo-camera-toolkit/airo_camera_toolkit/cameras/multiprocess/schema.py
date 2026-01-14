import time
from abc import ABC, abstractmethod
from typing import Type

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
from airo_camera_toolkit.interfaces import Camera, DepthCamera, RGBCamera, StereoRGBDCamera
from airo_typing import CameraResolutionType, PointCloud


class Schema(ABC):
    def __init__(self, topic: str, type: Type[Buffer]) -> None:
        self._topic = topic
        self._buffer_type = type

        self._buffer = None

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def buffer(self) -> Buffer:
        self._assert_buffer_allocated()
        return self._buffer

    def _assert_buffer_allocated(self) -> None:
        if self._buffer is None:
            raise ValueError("The internal buffer has not yet been allocated by the schema.")

    @abstractmethod
    def allocate_empty(self, resolution: CameraResolutionType) -> None:
        pass

    @abstractmethod
    def fill_from_camera(self, camera: Camera) -> None:
        pass

    @abstractmethod
    def read_into_receiver(self, frame: Buffer, receiver: Mixin) -> None:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


class CameraSchema(Schema):
    def __init__(self):
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
        self._buffer.timestamp[0] = time.time()
        self._buffer.resolution = np.array(camera.resolution).astype(np.uint32)
        self._buffer.intrinsics_matrix = camera.intrinsics_matrix().astype(np.float32)
        self._buffer.fps[0] = camera.fps

    def read_into_receiver(self, frame: CameraMetadataBuffer, receiver: CameraMixin) -> None:
        receiver._metadata_frame = frame


class RGBSchema(Schema):
    def __init__(self):
        super().__init__("rgb", RGBFrameBuffer)

    def allocate_empty(self, resolution: CameraResolutionType) -> None:
        width, height = resolution
        self._buffer = RGBFrameBuffer(
            rgb=np.empty((height, width, 3), dtype=np.uint8),
        )

    def fill_from_camera(self, camera: RGBCamera) -> None:
        self._assert_buffer_allocated()
        image = camera._retrieve_rgb_image_as_int()
        self._buffer.rgb = image

    def read_into_receiver(self, frame: RGBFrameBuffer, receiver: RGBMixin) -> None:
        receiver._rgb_frame = frame


class StereoRGBSchema(Schema):
    def __init__(self):
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

    def fill_from_camera(self, camera: StereoRGBDCamera) -> None:
        self._assert_buffer_allocated()

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

    def read_into_receiver(self, frame: StereoRGBFrameBuffer, receiver: StereoRGBMixin) -> None:
        receiver._stereo_frame = frame


class DepthSchema(Schema):
    def __init__(self):
        super().__init__("depth", DepthFrameBuffer)

    def allocate_empty(self, resolution: CameraResolutionType) -> None:
        width, height = resolution

        self._buffer = DepthFrameBuffer(
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth_map=np.empty((height, width), dtype=np.float32),
            confidence_map=np.empty((height, width), dtype=np.float32),
        )

    def fill_from_camera(self, camera: DepthCamera) -> None:
        self._assert_buffer_allocated()

        depth_image = camera._retrieve_depth_image()
        depth_map = camera._retrieve_depth_map()
        confidence_map = camera._retrieve_confidence_map()

        self._buffer.depth_image = depth_image
        self._buffer.depth_map = depth_map
        self._buffer.confidence_map = confidence_map

    def read_into_receiver(self, frame: DepthFrameBuffer, receiver: DepthMixin) -> None:
        receiver._depth_frame = frame


class PointCloudSchema(Schema):
    def __init__(self):
        super().__init__("pcd", PointCloudBuffer)

    def allocate_empty(self, resolution: CameraResolutionType) -> None:
        width, height = resolution

        self._buffer = PointCloudBuffer(
            point_cloud_positions=np.empty((height * width, 3), dtype=np.float32),
            point_cloud_colors=np.empty((height * width, 3), dtype=np.uint8),
            point_cloud_valid=np.empty((1,), dtype=np.uint32),
        )

    def fill_from_camera(self, camera: DepthCamera) -> None:
        self._assert_buffer_allocated()

        point_cloud = camera._retrieve_colored_point_cloud()

        self._buffer.point_cloud_positions[: point_cloud.points.shape[0]] = point_cloud.points
        if point_cloud.colors is not None:
            self._buffer.point_cloud_colors[: point_cloud.colors.shape[0]] = point_cloud.colors
        else:
            self._buffer.point_cloud_colors[: point_cloud.colors.shape[0]] = 0  # If no colors, use black.
        self._buffer.point_cloud_valid[0] = point_cloud.points.shape[0]

    def read_into_receiver(self, frame: PointCloudBuffer, receiver: PointCloudMixin) -> None:
        num_points = frame.point_cloud_valid.item()
        positions = frame.point_cloud_positions[:num_points]
        colors = frame.point_cloud_colors[:num_points]
        point_cloud = PointCloud(positions, colors)

        receiver._point_cloud_frame = point_cloud
