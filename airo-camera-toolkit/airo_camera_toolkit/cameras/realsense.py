from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError:
    print("install the Realsense SDK and pyrealsense2 first")
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils import ImageConverter
from airo_typing import CameraIntrinsicsMatrixType, NumpyFloatImageType, OpenCVIntImageType


class Realsense(RGBCamera):
    """Initial wrapper around the pyrealsense2 library to use the realsense cameras.
    Written mainly for use with the D415 and D435.

    Depth images are not yet supported.
    """

    # Resolutions and fps that are supported by the D415
    RESOLUTION_1080 = (1920, 1080)  # fps: 8
    RESOLUTION_720 = (1280, 720)  # fps: 15. 10, 6
    RESOLUTION_480 = (640, 480)  # fps: 30, 15, 6
    RESOLUTION_240 = (424, 240)  # fps: 60, 30, 15, 6

    def __init__(
        self,
        resolution: Tuple[int, int] = RESOLUTION_720,
        fps: int = 15,
    ) -> None:
        self.width, self.height = resolution
        self.fps = fps

        # Configure depth and color streams
        self._frames: Optional[rs.composite_frame] = None  # type: ignore
        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        color_sensor = device.first_color_sensor()

        profile_found = False
        color_stream_profiles = color_sensor.get_stream_profiles()
        for profile in color_stream_profiles:
            profile = profile.as_video_stream_profile()
            # print(profile) # uncomment to print all available profiles
            if profile.fps() != self.fps:
                continue
            if profile.width() != self.width:
                continue
            if profile.height() != self.height:
                continue

            intrinsics = profile.get_intrinsics()

            self._intrinsics_matrix = np.zeros((3, 3))
            self._intrinsics_matrix[0, 0] = intrinsics.fx
            self._intrinsics_matrix[1, 1] = intrinsics.fy
            self._intrinsics_matrix[0, 2] = intrinsics.ppx
            self._intrinsics_matrix[1, 2] = intrinsics.ppy
            self._intrinsics_matrix[2, 2] = 1

            config.enable_stream(
                profile.stream_type(),
                profile.width(),
                profile.height(),
                format=rs.format.bgr8,
                framerate=profile.fps(),
            )
            profile_found = True

        if not profile_found:
            raise ValueError(f"No profile found for resolution {self.width}x{self.height} at {self.fps} fps")

        # Start streaming
        self.pipeline.start(config)

    def __enter__(self) -> RGBCamera:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.pipeline.stop()

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self._intrinsics_matrix

    def _grab_images(self) -> None:
        self._frames = self.pipeline.wait_for_frames()

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        assert isinstance(self._frames, rs.composite_frame)
        color_frame = self._frames.get_color_frame()
        image: OpenCVIntImageType = np.asanyarray(color_frame.get_data())
        return ImageConverter.from_opencv_format(image).image_in_numpy_format


if __name__ == "__main__":
    import cv2

    camera = Realsense(fps=15)
    print("Camera Intrinsics: \n", camera.intrinsics_matrix())

    while True:
        image = camera.get_rgb_image()
        print(image.shape)
        image = ImageConverter.from_numpy_format(image).image_in_opencv_format

        cv2.imshow("RealSense", image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
