from __future__ import annotations

from typing import Any, Tuple

import numpy as np

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError:
    print("install the Realsense SDK and pyrealsense2 first")
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_camera_toolkit.utils import ImageConverter
from airo_typing import CameraIntrinsicsMatrixType, NumpyFloatImageType, NumpyDepthMapType, NumpyIntImageType, OpenCVIntImageType


class Realsense(RGBDCamera):
    """Initial wrapper around the pyrealsense2 library to use the realsense cameras.
    Written mainly for use with the D415 and D435.

    Depth images are not yet supported.
    """

    # Resolutions and fps that are supported by the D415
    RESOLUTION_1080 = (1920, 1080)  # fps: 8
    RESOLUTION_720 = (1280, 720)  # fps: 15. 10, 6
    RESOLUTION_480 = (640, 480)  # fps: 30, 15, 6
    RESOLUTION_240 = (424, 240)  # fps: 60, 30, 15, 6
    DEPTH_RESOLUTION_1280_720 = (1280, 720)  # fps: 6
    DEPTH_RESOLUTION_848_480 = (848, 480)  # fps: 10, 8, 6
    DEPTH_RESOLUTION_640_480 = (640, 480)  # fps: 30, 15, 6
    DEPTH_RESOLUTION_640_360 = (640, 360)  # fps: 30
    DEPTH_RESOLUTION_480_270 = (480, 270)  # fps: 60, 30, 15, 6
    DEPTH_RESOLUTION_256_144 = (256, 144)  # fps: 90

    def __init__(
        self,
        resolution: Tuple[int, int] = RESOLUTION_720,
        fps: int = 15,
        depth_resolution: Tuple[int, int] = DEPTH_RESOLUTION_640_480,
        depth_fps: int = 15,
    ) -> None:
        self.width, self.height = resolution
        self.fps = fps
        self.depth_width, self.depth_height = depth_resolution
        self.depth_fps = depth_fps

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        color_sensor = device.first_color_sensor()
        depth_sensor = device.first_depth_sensor()

        # Search color profile
        color_profile_found = False
        color_stream_profiles = color_sensor.get_stream_profiles()
        for color_profile in color_stream_profiles:
            color_profile = color_profile.as_video_stream_profile()
            #print(color_profile) # uncomment to print all available profiles
            if color_profile.fps() != self.fps:
                continue
            if color_profile.width() != self.width:
                continue
            if color_profile.height() != self.height:
                continue

            intrinsics = color_profile.get_intrinsics()

            self._intrinsics_matrix = np.zeros((3, 3))
            self._intrinsics_matrix[0, 0] = intrinsics.fx
            self._intrinsics_matrix[1, 1] = intrinsics.fy
            self._intrinsics_matrix[0, 2] = intrinsics.ppx
            self._intrinsics_matrix[1, 2] = intrinsics.ppy
            self._intrinsics_matrix[2, 2] = 1

            config.enable_stream(
                color_profile.stream_type(),
                color_profile.width(),
                color_profile.height(),
                format=rs.format.bgr8,
                framerate=color_profile.fps(),
            )
            color_profile_found = True
            break

        if not color_profile_found:
            raise ValueError(f"No color profile found for resolution {self.width}x{self.height} at {self.fps} fps")

        # Search depth profile
        depth_profile_found = False
        depth_stream_profiles = depth_sensor.get_stream_profiles()
        for depth_profile in depth_stream_profiles:
            depth_profile = depth_profile.as_video_stream_profile()
            #print(depth_profile) # uncomment to print all available profiles
            if str(depth_profile.stream_type()) != "stream.depth":
                continue
            if depth_profile.fps() != self.depth_fps:
                continue
            if depth_profile.width() != self.depth_width:
                continue
            if depth_profile.height() != self.depth_height:
                continue

            intrinsics = depth_profile.get_intrinsics()

            self._depth_intrinsics_matrix = np.zeros((3, 3))
            self._depth_intrinsics_matrix[0, 0] = intrinsics.fx
            self._depth_intrinsics_matrix[1, 1] = intrinsics.fy
            self._depth_intrinsics_matrix[0, 2] = intrinsics.ppx
            self._depth_intrinsics_matrix[1, 2] = intrinsics.ppy
            self._depth_intrinsics_matrix[2, 2] = 1

            config.enable_stream(
                depth_profile.stream_type(),
                depth_profile.width(),
                depth_profile.height(),
                format=rs.format.z16,
                framerate=depth_profile.fps(),
            )
            self.depth_factor = depth_sensor.get_depth_scale()

            depth_profile_found = True
            break

        if not depth_profile_found:
            raise ValueError(f"No depth profile found for resolution {self.depth_width}x{self.depth_height} at {self.depth_fps} fps")

        # Start streaming
        self.pipeline.start(config)

    def __enter__(self) -> RGBCamera:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.pipeline.stop()

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self._intrinsics_matrix

    def get_rgb_image(self) -> NumpyFloatImageType:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        image: OpenCVIntImageType = np.asanyarray(color_frame.get_data())
        return ImageConverter.from_opencv_format(image).image_in_numpy_format

    def get_depth_map(self) -> NumpyDepthMapType:
        image: NumpyDepthMapType = self.get_depth_image()
        return image * self.depth_factor

    def get_depth_image(self) -> NumpyIntImageType:
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        image: NumpyIntImageType = np.asanyarray(depth_frame.get_data())
        return image


if __name__ == "__main__":
    import cv2

    camera = Realsense(
        fps=15,
        depth_resolution=Realsense.DEPTH_RESOLUTION_1280_720,
        depth_fps=6
    )
    print("Camera Intrinsics: \n", camera.intrinsics_matrix())

    while True:
        color_image = camera.get_rgb_image()
        color_image = ImageConverter.from_numpy_format(color_image).image_in_opencv_format
        depth_image = camera.get_depth_map()

        cv2.imshow("RealSense RGB", color_image)
        cv2.imshow("RealSense Depth", depth_image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

