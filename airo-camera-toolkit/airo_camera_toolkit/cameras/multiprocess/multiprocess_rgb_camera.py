"""Publisher and receiver classes for multiprocess camera sharing."""

import multiprocessing
import time
from typing import Any

import numpy as np
from airo_camera_toolkit.cameras.multiprocess.base_publisher import BaseCameraPublisher
from airo_camera_toolkit.cameras.multiprocess.base_receiver import BaseCameraReceiver
from airo_camera_toolkit.cameras.multiprocess.frame_data import RGBFrameBuffer
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import CameraIntrinsicsMatrixType, NumpyFloatImageType, NumpyIntImageType
from loguru import logger


class MultiprocessRGBPublisher(BaseCameraPublisher):
    """Publishes RGB camera data to shared memory blocks."""

    def _get_frame_buffer_template(self, width: int, height: int) -> Any:
        """Return RGB frame buffer template."""
        return RGBFrameBuffer.template(width, height)

    def _capture_frame_data(self, frame_id: int, frame_timestamp: float) -> None:
        """Capture RGB image and intrinsics."""
        self._current_frame_id = frame_id
        self._current_frame_timestamp = frame_timestamp
        self._current_rgb_image = self._camera._retrieve_rgb_image_as_int()
        self._current_intrinsics = self._camera.intrinsics_matrix()

    def _write_frame_data(self) -> None:
        """Write RGB frame data to shared memory."""
        frame_data = RGBFrameBuffer(
            frame_id=np.array([self._current_frame_id], dtype=np.uint64),
            frame_timestamp=np.array([self._current_frame_timestamp], dtype=np.float64),
            rgb=self._current_rgb_image,
            intrinsics=self._current_intrinsics,
        )
        self._writer(frame_data)


class MultiprocessRGBReceiver(BaseCameraReceiver, RGBCamera):
    """Receives RGB camera data from shared memory."""

    def _get_frame_buffer_template(self, width: int, height: int) -> Any:
        """Return RGB frame buffer template."""
        return RGBFrameBuffer.template(width, height)

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        image = self._retrieve_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image).image_in_numpy_format
        return image

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        return self._last_frame.rgb

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self._last_frame.intrinsics


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBPublisher and MultiprocessRGBReceiver.
    You can also use the MultiprocessRGBReceiver in a different process (e.g. in a different python script)
    """
    camera_fps = 15
    namespace = "camera"

    import cv2  # type:ignore
    from airo_camera_toolkit.cameras.zed.zed import Zed

    multiprocessing.set_start_method("spawn", force=True)

    publisher = MultiprocessRGBPublisher(
        Zed,
        camera_kwargs={"resolution": Zed.InitParams.RESOLUTION_1080, "fps": camera_fps},
        shared_memory_namespace=namespace,
    )
    publisher.start()

    # The receiver behaves just like a regular RGBCamera
    receiver = MultiprocessRGBReceiver(namespace)

    cv2.namedWindow(namespace, cv2.WINDOW_NORMAL)

    time_current = None
    time_previous = None

    while True:
        time_previous = time_current
        time_current = time.time()

        image_rgb = receiver.get_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        cv2.imshow(namespace, image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

        if time_previous is not None:
            fps = 1 / (time_current - time_previous)

            fps_str = f"{fps:.2f}".rjust(6, " ")
            camera_fps_str = f"{camera_fps:.2f}".rjust(6, " ")
            if fps < 0.9 * camera_fps:
                logger.warning(f"FPS: {fps_str} / {camera_fps_str} (too slow)")
            else:
                logger.debug(f"FPS: {fps_str} / {camera_fps_str}")

    publisher.stop()
    publisher.join()
    cv2.destroyAllWindows()
