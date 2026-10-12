"""Publisher and receiver classes for multiprocess RGBD camera sharing."""

import multiprocessing
import time
from typing import Any

from airo_camera_toolkit.cameras.multiprocess.base_publisher import BaseCameraPublisher
from airo_camera_toolkit.cameras.multiprocess.frame_data import FrameLayout, RGBDFrameBuffer
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import MultiprocessRGBReceiver
from airo_camera_toolkit.cameras.multiprocess.pointcloud_stream import (
    PointCloudPublisherMixin,
    PointCloudReceiverMixin,
)
from airo_camera_toolkit.cameras.multiprocess.zenoh_reader import DEFAULT_FIRST_MESSAGE_TIMEOUT
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import NumpyDepthMapType, NumpyIntImageType
from loguru import logger


class MultiprocessRGBDPublisher(PointCloudPublisherMixin, BaseCameraPublisher):
    """Publishes RGBD camera data over Zenoh, and the point cloud on its own key expression if enabled."""

    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        shared_memory_namespace: str = "camera",
        enable_pointcloud: bool = True,
    ):
        self.enable_pointcloud = enable_pointcloud
        super().__init__(camera_cls, camera_kwargs, shared_memory_namespace)

    def _get_frame_layout(self, width: int, height: int) -> FrameLayout[Any]:
        """Return RGBD frame buffer layout."""
        return RGBDFrameBuffer.layout(width, height)

    def _capture_frame(self, frame_id: int, frame_timestamp: float) -> Any:
        """Capture the RGB-D data."""
        return RGBDFrameBuffer(
            **self._header(frame_id, frame_timestamp),
            rgb=self._camera.retrieve_rgb_image_as_int(),
            intrinsics=self._intrinsics,
            depth=self._camera.retrieve_depth_map(),
            depth_image=self._camera.retrieve_depth_image(),
        )


class MultiprocessRGBDReceiver(PointCloudReceiverMixin, MultiprocessRGBReceiver, RGBDCamera):
    """Receives RGBD camera data over Zenoh, and the matching point clouds if enabled."""

    def __init__(
        self,
        shared_memory_namespace: str,
        enable_pointcloud: bool = True,
        timeout: float | None = DEFAULT_FIRST_MESSAGE_TIMEOUT,
    ) -> None:
        self.enable_pointcloud = enable_pointcloud
        super().__init__(shared_memory_namespace, timeout=timeout)

    def _get_frame_layout(self, width: int, height: int) -> FrameLayout[Any]:
        """Return RGBD frame buffer layout."""
        return RGBDFrameBuffer.layout(width, height)

    def retrieve_depth_map(self) -> NumpyDepthMapType:
        return self._last_frame.depth

    def retrieve_depth_image(self) -> NumpyIntImageType:
        return self._last_frame.depth_image


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBDPublisher and MultiprocessRGBDReceiver.
    You can also use the MultiprocessRGBDReceiver in a different process (e.g. in a different python script)
    """
    camera_fps = 15

    import cv2
    from airo_camera_toolkit.cameras.zed.zed import Zed

    multiprocessing.set_start_method("spawn", force=True)

    publisher = MultiprocessRGBDPublisher(
        Zed,
        camera_kwargs={
            "resolution": Zed.InitParams.RESOLUTION_1080,
            "fps": camera_fps,
            "depth_mode": Zed.InitParams.NEURAL_DEPTH_MODE,
        },
    )

    publisher.start()
    receiver = MultiprocessRGBDReceiver("camera")
    try:
        cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
        cv2.namedWindow("DEPTH", cv2.WINDOW_NORMAL)

        time_current = None
        time_previous = None

        while True:
            time_previous = time_current
            time_current = time.time()

            receiver.grab_images()
            pcd = receiver.retrieve_colored_point_cloud()
            depth_image = receiver.retrieve_depth_map()

            image_rgb = receiver.retrieve_rgb_image_as_int()
            image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
            cv2.imshow("RGB", image)
            cv2.imshow("DEPTH", depth_image)
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
    finally:
        receiver.stop()
        publisher.stop()
        publisher.join()
        cv2.destroyAllWindows()
