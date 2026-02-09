"""Publisher and receiver classes for multiprocess RGBD camera sharing."""

import multiprocessing
import time
from typing import Any

import numpy as np
from airo_camera_toolkit.cameras.multiprocess.base_publisher import BaseCameraPublisher
from airo_camera_toolkit.cameras.multiprocess.frame_data import RGBDFrameBuffer, RGBDFrameBufferWithPointCloud
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import MultiprocessRGBReceiver
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import NumpyDepthMapType, NumpyIntImageType, PointCloud
from loguru import logger


class MultiprocessRGBDPublisher(BaseCameraPublisher):
    """Publishes RGBD camera data to shared memory blocks."""

    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        shared_memory_namespace: str = "camera",
        enable_pointcloud: bool = True,
    ):
        self.enable_pointcloud = enable_pointcloud
        super().__init__(camera_cls, camera_kwargs, shared_memory_namespace)

    def _get_frame_buffer_template(self, width: int, height: int) -> Any:
        """Return RGBD frame buffer template."""
        if self.enable_pointcloud:
            return RGBDFrameBufferWithPointCloud.template(width, height)
        else:
            return RGBDFrameBuffer.template(width, height)

    def _setup(self) -> None:
        """Set up camera and prepare point cloud buffers if needed."""
        super()._setup()

        if self.enable_pointcloud:
            # Some cameras can return sparse point clouds. We ensure we always have
            # a buffer for every pixel in the RGB image.
            self._pcd_pos_buf = np.zeros(
                (self._camera.resolution[0] * self._camera.resolution[1], 3),
                dtype=np.float32,
            )
            self._pcd_col_buf = np.zeros(
                (self._camera.resolution[0] * self._camera.resolution[1], 3),
                dtype=np.uint8,
            )

    def _retrieve_frame_data(self, frame_id: int, frame_timestamp: float) -> None:
        """Retrieve RGB-D data and optionally point cloud."""
        self._current_frame_id = frame_id
        self._current_frame_timestamp = frame_timestamp
        self._current_rgb_image = self._camera._retrieve_rgb_image_as_int()
        self._current_depth_map = self._camera._retrieve_depth_map()
        self._current_depth_image = self._camera._retrieve_depth_image()
        self._current_intrinsics = self._camera.intrinsics_matrix()

        if self.enable_pointcloud:
            point_cloud = self._camera._retrieve_colored_point_cloud()

            # Handle sparse point clouds by filling buffer with NaN
            self._pcd_pos_buf.fill(np.nan)
            self._pcd_pos_buf[: point_cloud.points.shape[0]] = point_cloud.points

            if point_cloud.colors is not None:
                self._pcd_col_buf[: point_cloud.colors.shape[0]] = point_cloud.colors
            else:
                self._pcd_col_buf[: point_cloud.points.shape[0]] = 0  # Use black if no colors

            self._current_pcd_num_points = point_cloud.points.shape[0]

    def _write_frame_data(self) -> None:
        """Write RGBD frame data and optionally point cloud to shared memory."""
        if self.enable_pointcloud:
            self._writer(
                RGBDFrameBufferWithPointCloud(
                    frame_id=np.array([self._current_frame_id], dtype=np.uint64),
                    frame_timestamp=np.array([self._current_frame_timestamp], dtype=np.float64),
                    rgb=self._current_rgb_image,
                    intrinsics=self._current_intrinsics,
                    depth=self._current_depth_map,
                    depth_image=self._current_depth_image,
                    point_cloud_positions=self._pcd_pos_buf,
                    point_cloud_colors=self._pcd_col_buf,
                    num_valid_points=np.array([self._current_pcd_num_points], dtype=np.int32),
                )
            )
        else:
            # Write main RGBD frame
            self._writer(
                RGBDFrameBuffer(
                    frame_id=np.array([self._current_frame_id], dtype=np.uint64),
                    frame_timestamp=np.array([self._current_frame_timestamp], dtype=np.float64),
                    rgb=self._current_rgb_image,
                    intrinsics=self._current_intrinsics,
                    depth=self._current_depth_map,
                    depth_image=self._current_depth_image,
                )
            )


class MultiprocessRGBDReceiver(MultiprocessRGBReceiver, RGBDCamera):
    """Receives RGBD camera data from shared memory."""

    def __init__(self, shared_memory_namespace: str, enable_pointcloud: bool = True) -> None:
        self.enable_pointcloud = enable_pointcloud
        super().__init__(shared_memory_namespace)

    def _get_frame_buffer_template(self, width: int, height: int) -> Any:
        """Return RGBD frame buffer template."""
        if self.enable_pointcloud:
            return RGBDFrameBufferWithPointCloud.template(width, height)
        else:
            return RGBDFrameBuffer.template(width, height)

    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        return self._last_frame.depth

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        return self._last_frame.depth_image

    def _retrieve_colored_point_cloud(self) -> PointCloud:
        if not self.enable_pointcloud:
            raise RuntimeError("Cannot retrieve point cloud when point cloud is not enabled.")
        num_points = self._last_frame.num_valid_points.item()
        positions = self._last_frame.point_cloud_positions[:num_points]
        colors = self._last_frame.point_cloud_colors[:num_points]
        point_cloud = PointCloud(positions, colors)
        return point_cloud


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

    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DEPTH", cv2.WINDOW_NORMAL)

    time_current = None
    time_previous = None

    while True:
        time_previous = time_current
        time_current = time.time()

        pcd = receiver.get_colored_point_cloud()
        depth_image = receiver._retrieve_depth_map()

        image_rgb = receiver._retrieve_rgb_image_as_int()
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
