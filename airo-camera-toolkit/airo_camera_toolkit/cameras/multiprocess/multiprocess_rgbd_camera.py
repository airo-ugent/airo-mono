"""Publisher and receiver classes for multiprocess camera sharing."""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import (
    MultiprocessRGBPublisher,
    MultiprocessRGBReceiver,
    ResolutionIdl,
)
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_ipc.cyclone_shm.idl_shared_memory.base_idl import BaseIdl  # type: ignore
from airo_ipc.cyclone_shm.patterns.sm_reader import SMReader  # type: ignore
from airo_ipc.cyclone_shm.patterns.sm_writer import SMWriter  # type: ignore
from airo_typing import CameraResolutionType, NumpyDepthMapType, NumpyIntImageType, PointCloud
from loguru import logger


@dataclass
class RGBDFrameBuffer(BaseIdl):  # type: ignore
    """This struct, sent over shared memory, contains a timestamp, an RGB image, the camera intrinsics, a depth image, a depth map, and a point cloud."""

    # Timestamp of the frame (seconds)
    timestamp: np.ndarray
    # Color image data (height x width x channels)
    rgb: np.ndarray
    # Intrinsic camera parameters (camera matrix)
    intrinsics: np.ndarray
    # Depth image data (height x width)
    depth_image: np.ndarray
    # Depth map (height x width)
    depth: np.ndarray
    # Point cloud (colors, positions x height * width x 3)
    point_cloud: np.ndarray
    # Valid point cloud points (scalar), for sparse point clouds
    point_cloud_valid: np.ndarray

    @staticmethod
    def template(width: int, height: int) -> Any:
        return RGBDFrameBuffer(
            timestamp=np.empty((1,), dtype=np.float64),
            rgb=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics=np.empty((3, 3), dtype=np.float64),
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth=np.empty((height, width), dtype=np.float32),
            point_cloud=np.empty((2, height * width, 3), dtype=np.float32),
            point_cloud_valid=np.empty((1,), dtype=np.uint32),
        )


class MultiprocessRGBDPublisher(MultiprocessRGBPublisher):
    """Publishes the data of a camera that implements the RGBDCamera interface to shared memory blocks."""

    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        shared_memory_namespace: str = "camera",
    ):
        super().__init__(camera_cls, camera_kwargs, shared_memory_namespace)

    def _setup(self) -> None:
        super()._setup()

        # Some cameras, such as the Realsense D435i, can return a sparse point cloud. This is not supported by the
        # current implementation of the RGBDFrameBuffer. Therefore, we make sure that we always retrieve a point
        # for every pixel in the RBG image.
        self._pcd_buf = np.zeros((2, self._camera.resolution[0] * self._camera.resolution[1], 3), dtype=np.float32)

    def _setup_sm_writer(self) -> None:
        # Create the shared memory writer
        self._writer = SMWriter(
            domain_participant=self._dp,
            topic_name=self._shared_memory_namespace,
            idl_dataclass=RGBDFrameBuffer.template(self._camera.resolution[0], self._camera.resolution[1]),
        )

    def stop(self) -> None:
        self.shutdown_event.set()

    def run(self) -> None:
        logger.info(f"{self.__class__.__name__} process started.")
        self._setup()
        assert isinstance(self._camera, RGBDCamera)  # For mypy
        logger.info(f'{self.__class__.__name__} starting to publish to "{self._shared_memory_namespace}".')

        while not self.shutdown_event.is_set():
            self._resolution_writer(ResolutionIdl(width=self._camera.resolution[0], height=self._camera.resolution[1]))

            image = self._camera.get_rgb_image_as_int()
            depth_map = self._camera._retrieve_depth_map()
            depth_image = self._camera._retrieve_depth_image()
            point_cloud = self._camera._retrieve_colored_point_cloud()

            # Some camera's, such as the Realsense D435i, return a sparse point cloud. This is not supported by the
            # current implementation of the RGBDFrameBuffer. Therefore, we make sure that we always retrieve a point
            # for every pixel in the RBG image.
            self._pcd_buf.fill(np.nan)
            self._pcd_buf[0, : point_cloud.points.shape[0]] = point_cloud.points
            if point_cloud.colors is not None:
                self._pcd_buf[1, : point_cloud.colors.shape[0]] = (
                    point_cloud.colors / 255.0
                )  # Colors are in [0, 255], but buffer is float.
            else:
                self._pcd_buf[1, : point_cloud.points.shape[0]] = 0.0  # If no colors, use black.

            self._writer(
                RGBDFrameBuffer(
                    timestamp=np.array([time.time()], dtype=np.float64),
                    rgb=image,
                    intrinsics=self._camera.intrinsics_matrix(),
                    depth=depth_map,
                    depth_image=depth_image,
                    point_cloud=self._pcd_buf,
                    point_cloud_valid=np.array([point_cloud.points.shape[0]], dtype=np.uint32),
                )
            )


class MultiprocessRGBDReceiver(MultiprocessRGBReceiver, RGBDCamera):
    """Implements the RGBD camera interface for a camera that is running in a different process, to be used with the Publisher class."""

    def __init__(self, shared_memory_namespace: str) -> None:
        super().__init__(shared_memory_namespace)

    def _setup_sm_reader(self, resolution: CameraResolutionType) -> None:
        # Create the shared memory reader
        self._reader = SMReader(
            domain_participant=self._dp,
            topic_name=self._shared_memory_namespace,
            idl_dataclass=RGBDFrameBuffer.template(self.resolution[0], self.resolution[1]),
        )

        # Initialize a first frame.
        self._last_frame = RGBDFrameBuffer.template(self.resolution[0], self.resolution[1])

    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        return self._last_frame.depth

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        return self._last_frame.depth_image

    def _retrieve_colored_point_cloud(self) -> PointCloud:
        num_points = self._last_frame.point_cloud_valid.item()
        positions = self._last_frame.point_cloud[0, :num_points]
        colors = self._last_frame.point_cloud[1, :num_points]
        colors = (colors * 255.0).astype(np.uint8)
        point_cloud = PointCloud(positions, colors)
        return point_cloud


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBDPublisher and MultiprocessRGBDReceiver.
    You can also use the MultiprocessRGBDReceiver in a different process (e.g. in a different python script)
    """
    camera_fps = 15

    import cv2
    from airo_camera_toolkit.cameras.zed.zed import Zed

    publisher = MultiprocessRGBDPublisher(
        Zed,
        camera_kwargs={"resolution": Zed.RESOLUTION_1080, "fps": camera_fps, "depth_mode": Zed.NEURAL_DEPTH_MODE},
    )

    publisher.start()
    receiver = MultiprocessRGBDReceiver("camera")

    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DEPTH", cv2.WINDOW_NORMAL)

    time_current = None
    time_previous = None

    while True:
        time_previous = time_current
        time_current = time.time()

        pcd = receiver.get_colored_point_cloud()
        depth_image = receiver.get_depth_image()

        image_rgb = receiver.get_rgb_image_as_int()
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

    publisher.stop()
    publisher.join()
    cv2.destroyAllWindows()
