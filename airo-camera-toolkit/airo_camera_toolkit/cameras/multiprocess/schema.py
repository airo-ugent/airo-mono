from abc import ABC, abstractmethod
from typing import Type

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
from airo_camera_toolkit.interfaces import Camera
from airo_ipc.cyclone_shm.idl_shared_memory.base_idl import BaseIdl
from airo_typing import CameraResolutionType


class Schema(ABC):
    def __init__(self, topic: str, type: Type[Buffer]) -> None:
        self._topic = topic
        self._buffer_type = type

        self._buffer = None

    @property
    def topic(self) -> str:
        return self._topic

    def allocate_empty(self, resolution: CameraResolutionType) -> BaseIdl:
        self._buffer = self._buffer_type.allocate_empty(resolution)
        return self._buffer

    def fill_from_camera(self, camera: Camera) -> BaseIdl:
        self._buffer.fill_from_camera(camera)
        return self._buffer

    @abstractmethod
    def read_into_receiver(self, frame: Buffer, receiver: Mixin) -> None:
        pass

    def __repr__(self) -> str:
        return self.__class__.__name__


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


class StereoRGBSchema(Schema):
    def __init__(self):
        super().__init__("stereo", StereoRGBFrameBuffer)

    def read_into_receiver(self, frame: StereoRGBFrameBuffer, receiver: StereoRGBMixin) -> None:
        receiver._stereo_frame = frame


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
