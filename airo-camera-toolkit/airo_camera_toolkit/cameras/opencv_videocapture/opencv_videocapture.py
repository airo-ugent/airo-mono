from __future__ import annotations

import math
import os
from typing import Any, Optional, Tuple

import cv2
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import CameraIntrinsicsMatrixType, CameraResolutionType, NumpyFloatImageType, NumpyIntImageType


class OpenCVVideoCapture(RGBCamera):
    """Wrapper around OpenCV's VideoCapture so we can test the camera interface without external cameras."""

    # Some standard resolutions that are likely to be supported by webcams.
    # 16:9
    RESOLUTION_1080 = (1920, 1080)
    RESOLUTION_720 = (1280, 720)
    # 4:3
    RESOLUTION_768 = (1024, 768)
    RESOLUTION_480 = (640, 480)

    def __init__(
        self,
        video_capture_args: Tuple[Any] = (0,),
        intrinsics_matrix: Optional[CameraIntrinsicsMatrixType] = None,
        resolution: CameraResolutionType = RESOLUTION_480,
        fps: int = 30,
    ) -> None:
        self.video_capture = cv2.VideoCapture(*video_capture_args)

        # If passing a video file, we want to check if it exists. Then, we can throw a more meaningful
        # error if it does not.
        if len(video_capture_args) > 0 and isinstance(video_capture_args[0], str):
            if not os.path.isfile(video_capture_args[0]):
                raise FileNotFoundError(f"Could not find video file {video_capture_args[0]}")
        if not self.video_capture.isOpened():
            raise RuntimeError(f"Cannot open camera {video_capture_args[0]}. Is it connected?")

        # Note that the following will not forcibly set the resolution. If the user's webcam
        # does not support the desired resolution, OpenCV will silently select a close match.
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.video_capture.set(cv2.CAP_PROP_FPS, fps)

        self._intrinsics_matrix = intrinsics_matrix

        self._fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        self._resolution = (
            math.floor(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            math.floor(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def fps(self) -> int:
        """The frames per second of the camera."""
        return self._fps

    @property
    def resolution(self) -> CameraResolutionType:
        return self._resolution

    def __enter__(self) -> RGBCamera:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.video_capture.release()

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        """Obtain the intrinsics matrix of the camera.

        Raises:
            RuntimeError: You must explicitly pass an intrinsics object to the constructor.

        Returns:
            CameraIntrinsicsMatrixType: The intrinsics matrix.
        """
        if self._intrinsics_matrix is None:
            raise RuntimeError(
                "OpenCVVideoCapture does not have a preset intrinsics matrix. Pass it to the constructor if you know it."
            )
        return self._intrinsics_matrix

    def _grab_images(self) -> None:
        ret, image = self.video_capture.read()
        if not ret:  # When streaming a video, we will at some point reach the end.
            raise EOFError("Can't receive frame (stream end?). Exiting...")

        self._frame = image

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        return ImageConverter.from_opencv_format(self._frame).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        return ImageConverter.from_opencv_format(self._frame).image_in_numpy_int_format


if __name__ == "__main__":
    import airo_camera_toolkit.cameras.manual_test_hw as test
    import numpy as np

    camera = OpenCVVideoCapture(intrinsics_matrix=np.eye(3))

    # Perform tests
    test.manual_test_camera(camera)
    test.manual_test_rgb_camera(camera)
    test.profile_rgb_throughput(camera)

    # Live viewer
    cv2.namedWindow("OpenCV Webcam RGB", cv2.WINDOW_NORMAL)

    while True:
        color_image = camera.get_rgb_image_as_int()
        color_image = ImageConverter.from_numpy_int_format(color_image).image_in_opencv_format

        cv2.imshow("OpenCV Webcam RGB", color_image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
