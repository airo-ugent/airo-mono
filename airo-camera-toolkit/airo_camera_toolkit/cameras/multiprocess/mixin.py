from abc import ABC

from airo_camera_toolkit.interfaces import DepthCamera, RGBCamera, RGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    NumpyConfidenceMapType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
    PointCloud,
)


class Mixin(ABC):
    pass


class CameraMixin(Mixin):
    @property
    def resolution(self) -> CameraResolutionType:
        return self._metadata_frame.resolution

    @property
    def fps(self) -> float:
        return self._metadata_frame.fps.item()

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self._metadata_frame.intrinsics


class RGBMixin(Mixin, RGBCamera):
    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        return ImageConverter.from_numpy_int_format(self._retrieve_rgb_image_as_int()).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        return self._rgb_frame.rgb


class DepthMixin(Mixin, DepthCamera):
    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        return self._depth_frame.depth_map

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        return self._depth_frame.depth_image

    def _retrieve_confidence_map(self) -> NumpyConfidenceMapType:
        return self._depth_frame.confidence_map


class PointCloudMixin(Mixin, RGBDCamera):
    def _retrieve_colored_point_cloud(self) -> PointCloud:
        return self._point_cloud_frame.point_cloud
