from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import cv2
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import CameraIntrinsicsMatrixType, CameraResolutionType, NumpyFloatImageType, NumpyIntImageType


class OpenCVVideoCapture(RGBCamera):
    """Wrapper around OpenCV's VideoCapure so we can test the camera interface without external cameras."""

    def __init__(
        self, video_capture_args: Tuple[Any] = (0,), intrinsics_matrix: Optional[CameraIntrinsicsMatrixType] = None
    ) -> None:
        self.video_capture = cv2.VideoCapture(*video_capture_args)
        if not self.video_capture.isOpened():
            raise RuntimeError("Cannot open camera")

        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self._intrinsics_matrix = intrinsics_matrix

        self._resolution = (
            math.floor(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            math.floor(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

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

    def _grab_images(self):
        ret, image = self.video_capture.read()
        if not ret:
            raise RuntimeError("Can't receive frame (stream end?). Exiting...")

        self._frame = image

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        return ImageConverter.from_opencv_format(self._frame).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        return ImageConverter.from_opencv_format(self._frame).image_in_numpy_int_format


if __name__ == "__main__":
    camera = OpenCVVideoCapture()

    while True:
        image = camera.get_rgb_image()
        print(image.shape)
        image = ImageConverter.from_numpy_format(image).image_in_opencv_format

        cv2.imshow("VideoCapture", image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
