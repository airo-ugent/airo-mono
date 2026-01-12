from abc import ABC, abstractmethod
from typing import Type

from airo_camera_toolkit.cameras.multiprocess.idl import (
    Buffer,
    CameraMetadataBuffer,
    DepthFrameBuffer,
    PointCloudBuffer,
    RGBFrameBuffer,
)
from airo_camera_toolkit.cameras.multiprocess.mixin import CameraMixin, DepthMixin, Mixin, PointCloudMixin, RGBMixin
from airo_camera_toolkit.interfaces import Camera
from airo_ipc.cyclone_shm.idl_shared_memory.base_idl import BaseIdl
from airo_typing import CameraResolutionType


class Schema(ABC):
    def __init__(self, topic: str, type: Type[Buffer]) -> None:
        self._topic = topic
        self._buffer_type = type

    @property
    def topic(self) -> str:
        return self._topic

    def allocate_empty(self, resolution: CameraResolutionType) -> BaseIdl:
        return self._buffer_type.allocate_empty(resolution)

    def allocate_from_camera(self, camera: Camera) -> BaseIdl:
        return self._buffer_type.allocate_from_camera(camera)

    @abstractmethod
    def read_into_receiver(self, frame: BaseIdl, receiver: Mixin) -> None:
        pass


class CameraSchema(Schema):
    def __init__(self):
        super().__init__("metadata", CameraMetadataBuffer)

    def read_into_receiver(self, frame: CameraMetadataBuffer, receiver: CameraMixin) -> None:
        receiver._metadata_frame = frame


class RGBSchema(Schema):
    def __init__(self):
        super().__init__("rgb", RGBFrameBuffer)

    def read_into_receiver(self, frame: RGBFrameBuffer, receiver: RGBMixin) -> None:
        receiver._rgb_frame = frame


class DepthSchema(Schema):
    def __init__(self):
        super().__init__("depth", DepthFrameBuffer)

    def read_into_receiver(self, frame: DepthFrameBuffer, receiver: DepthMixin) -> None:
        receiver._depth_frame = frame


class PointCloudSchema(Schema):
    def __init__(self):
        super().__init__("pcd", PointCloudBuffer)

    def read_into_receiver(self, frame: PointCloudBuffer, receiver: PointCloudMixin) -> None:
        receiver._point_cloud_frame = frame
