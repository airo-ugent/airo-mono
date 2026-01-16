"""These mixins provide the necessary methods to retrieve data from shared memory. By subclassing from a mixin, you get implementations
belonging to `Camera`, `RGBCamera`, etcetera for free - assuming the receiver uses the necessary schemas."""

from abc import ABC

from airo_camera_toolkit.cameras.multiprocess.buffer import (
    CameraMetadataBuffer,
    DepthFrameBuffer,
    RGBFrameBuffer,
    StereoRGBFrameBuffer,
)
from airo_camera_toolkit.interfaces import Camera, DepthCamera, RGBCamera, RGBDCamera, StereoRGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    HomogeneousMatrixType,
    NumpyConfidenceMapType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
    PointCloud,
)


class Mixin(ABC):
    pass


class CameraMixin(Mixin, Camera):
    """Implements the Camera interface for SharedMemoryReceiver."""

    _metadata_frame: CameraMetadataBuffer

    @property
    def resolution(self) -> CameraResolutionType:
        width, height = self._metadata_frame.resolution
        return width, height

    @property
    def fps(self) -> float:
        return self._metadata_frame.fps.item()

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self._metadata_frame.intrinsics_matrix

    def get_current_timestamp(self) -> float:
        return self._metadata_frame.timestamp.item()


class RGBMixin(Mixin, RGBCamera):
    """Implements the RGBCamera interface for SharedMemoryReceiver."""

    _rgb_frame: RGBFrameBuffer

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        return ImageConverter.from_numpy_int_format(self._retrieve_rgb_image_as_int()).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        return self._rgb_frame.rgb


class StereoRGBMixin(Mixin, StereoRGBDCamera):
    """Implements the StereoRGBDCamera interface for SharedMemoryReceiver."""

    _stereo_frame: StereoRGBFrameBuffer

    def _retrieve_rgb_image(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyFloatImageType:
        return ImageConverter.from_numpy_int_format(self._retrieve_rgb_image_as_int(view)).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyIntImageType:
        if view == StereoRGBDCamera.LEFT_RGB:
            return self._stereo_frame.rgb_left
        else:
            return self._stereo_frame.rgb_right

    def intrinsics_matrix(self, view: str = StereoRGBDCamera.LEFT_RGB) -> CameraIntrinsicsMatrixType:
        if view == StereoRGBDCamera.LEFT_RGB:
            return self._stereo_frame.intrinsics_left
        else:
            return self._stereo_frame.intrinsics_right

    @property
    def pose_of_right_view_in_left_view(self) -> HomogeneousMatrixType:
        return self._stereo_frame.pose_right_in_left


class DepthMixin(Mixin, DepthCamera):
    """Implements the DepthCamera interface for SharedMemoryReceiver."""

    _depth_frame: DepthFrameBuffer

    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        return self._depth_frame.depth_map

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        return self._depth_frame.depth_image

    def _retrieve_confidence_map(self) -> NumpyConfidenceMapType:
        return self._depth_frame.confidence_map


class PointCloudMixin(Mixin, RGBDCamera):
    """Implements part of the RGBDCamera interface for SharedMemoryReceiver."""

    _point_cloud_frame: PointCloud

    def _retrieve_colored_point_cloud(self) -> PointCloud:
        return self._point_cloud_frame
