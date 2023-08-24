from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError:
    print("install the Realsense SDK and pyrealsense2 first")
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_camera_toolkit.utils import ImageConverter
from airo_typing import (
    CameraIntrinsicsMatrixType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
    OpenCVIntImageType,
)


class Realsense(RGBDCamera):
    """Initial wrapper around the pyrealsense2 library to use the realsense cameras.
    Written mainly for use with the D415 and D435.
    """

    # Resolutions and fps that are supported by the D415
    # To figure out which fps are supported for each resolution, see the zip in this discussion:
    # https://community.intel.com/t5/Items-with-no-label/RGB-feed-fps-for-D415-or-D435/m-p/611235
    RESOLUTION_1080 = (1920, 1080)
    RESOLUTION_720 = (1280, 720)
    RESOLUTION_480 = (640, 480)
    RESOLUTION_240 = (424, 240)
    DEPTH_RESOLUTION_1280_720 = (1280, 720)
    DEPTH_RESOLUTION_848_480 = (848, 480)
    DEPTH_RESOLUTION_640_480 = (640, 480)
    DEPTH_RESOLUTION_640_360 = (640, 360)
    DEPTH_RESOLUTION_480_270 = (480, 270)
    DEPTH_RESOLUTION_256_144 = (256, 144)

    def __init__(
        self,
        resolution: Tuple[int, int] = RESOLUTION_720,
        fps: int = 15,
        depth_resolution: Tuple[int, int] = DEPTH_RESOLUTION_640_480,
        depth_fps: int = 15,
    ) -> None:
        self.resolution = resolution
        self.fps = fps
        self.depth_enabled = depth_resolution is not None and depth_fps is not None
        self.depth_resolution = depth_resolution
        self.depth_fps = depth_fps

        self._composite_frame: Optional[rs.composite_frame] = None  # type: ignore
        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        color_sensor = device.first_color_sensor()
        depth_sensor = device.first_depth_sensor()
        self.depth_factor = depth_sensor.get_depth_scale()

        color_profile = self._enable_color_stream(config, color_sensor, self.fps, *self.resolution)
        intrinsics = color_profile.get_intrinsics()
        self._intrinsics_matrix = np.array(
            [[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]]
        )

        if self.depth_enabled:
            self._enable_depth_stream(config, depth_sensor, self.depth_fps, *self.depth_resolution)
            self._setup_depth_transforms()

        self.pipeline.start(config)

    def __enter__(self) -> RGBDCamera:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.pipeline.stop()

    def _enable_color_stream(self, config, color_sensor, fps, width, height) -> rs.stream:
        color_stream_profiles = color_sensor.get_stream_profiles()
        for color_profile in color_stream_profiles:
            color_profile = color_profile.as_video_stream_profile()
            # print(color_profile) # uncomment to print all available profiles
            if color_profile.fps() != fps:
                continue
            if color_profile.width() != width:
                continue
            if color_profile.height() != height:
                continue

            config.enable_stream(
                color_profile.stream_type(),
                color_profile.width(),
                color_profile.height(),
                format=rs.format.bgr8,
                framerate=color_profile.fps(),
            )
            return color_profile

        raise ValueError(f"No color profile found for resolution {width}x{height} at {fps} fps")

    def _enable_depth_stream(self, config, depth_sensor, fps, width, height) -> rs.stream:
        depth_stream_profiles = depth_sensor.get_stream_profiles()
        for depth_profile in depth_stream_profiles:
            depth_profile = depth_profile.as_video_stream_profile()
            # print(depth_profile) # uncomment to print all available profiles
            if str(depth_profile.stream_type()) != "stream.depth":
                continue
            if depth_profile.fps() != fps:
                continue
            if depth_profile.width() != width:
                continue
            if depth_profile.height() != height:
                continue

            config.enable_stream(
                depth_profile.stream_type(),
                depth_profile.width(),
                depth_profile.height(),
                format=rs.format.z16,
                framerate=depth_profile.fps(),
            )
            return depth_profile

        raise ValueError(f"No depth profile found for resolution {width}x{height} at {fps} fps")

    def _setup_depth_transforms(self) -> None:
        # Configure depth filters and transfrom, adapted from:
        # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
        # https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
        self.align_transform = rs.align(rs.stream.color)
        self.decimation_filter = rs.decimation_filter()
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter()

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self._intrinsics_matrix

    def _grab_images(self) -> None:
        self._composite_frame = self.pipeline.wait_for_frames()

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        image = self._retrieve_rgb_image_as_int()
        return ImageConverter.from_numpy_int_format(image).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        assert isinstance(self._composite_frame, rs.composite_frame)
        color_frame = self._composite_frame.get_color_frame()
        image: OpenCVIntImageType = np.asanyarray(color_frame.get_data())
        image = image[..., ::-1]  # convert from BGR to RGB
        return image

    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        image = self._retrieve_depth_frame().astype(np.float32)
        return image * self.depth_factor

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        image = self._retrieve_depth_frame()  # uint16

        # Clip out of range
        val_max = 3000  # recommended max accurate range = 3000 mm
        clip_mask = image > val_max
        image[clip_mask] = val_max

        # Convert to depth image
        image_uint8 = np.zeros(image.shape, dtype=np.uint8)
        image_uint8[:, :] = (val_max - image[:, :]) / (val_max / 256)
        return image_uint8

    def _retrieve_depth_frame(self) -> NumpyIntImageType:
        assert isinstance(self._composite_frame, rs.composite_frame)

        # Currently aligment
        aligned_frames = self.align_transform.process(self._composite_frame)

        frame = aligned_frames.get_depth_frame()
        frame = self.decimation_filter.process(frame)
        frame = self.depth_to_disparity.process(frame)
        frame = self.spatial_filter.process(frame)
        frame = self.temporal_filter.process(frame)
        frame = self.disparity_to_depth.process(frame)
        frame = self.hole_filling.process(frame)

        image = np.asanyarray(frame.get_data())
        return image


if __name__ == "__main__":
    import airo_camera_toolkit.cameras.test_hw as test
    import cv2

    camera = Realsense(fps=30, depth_resolution=Realsense.DEPTH_RESOLUTION_1280_720, depth_fps=30)
    print("Camera Intrinsics: \n", camera.intrinsics_matrix())

    # Perform tests
    test.manual_test_camera(camera)
    test.manual_test_rgb_camera(camera)
    test.manual_test_depth_camera(camera)
    test.profile_rgb_throughput(camera)

    # Live viewer
    cv2.namedWindow("RealSense RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RealSense Depth Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RealSense Depth Map", cv2.WINDOW_NORMAL)

    while True:
        color_image = camera.get_rgb_image()
        color_image = ImageConverter.from_numpy_format(color_image).image_in_opencv_format
        depth_image = camera._retrieve_depth_image()
        depth_map = camera._retrieve_depth_map()

        cv2.imshow("RealSense RGB", color_image)
        cv2.imshow("RealSense Depth Image", depth_image)
        cv2.imshow("RealSense Depth Map", depth_map)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
