"""Publishing a camera's point cloud on its own key expression, next to its frames.

The RGBD, stereo RGBD and Zed publishers publish the point cloud as a :class:`PointCloudFrameBuffer`
on ``<namespace>_pointcloud`` rather than inside their frames, so receivers that do not need it
(``enable_pointcloud=False``) never receive those bytes. Receivers that do need it match each point
cloud to its frame by ``frame_id``.
"""

from typing import Any, Callable

import numpy as np
from airo_camera_toolkit.cameras.multiprocess.base_publisher import BaseCameraPublisher
from airo_camera_toolkit.cameras.multiprocess.base_receiver import BaseCameraReceiver
from airo_camera_toolkit.cameras.multiprocess.frame_data import PointCloudFrameBuffer
from airo_camera_toolkit.cameras.multiprocess.zenoh_reader import DEFAULT_FIRST_MESSAGE_TIMEOUT, ZenohReader
from airo_typing import CameraResolutionType, PointCloud


def pointcloud_key_expr(namespace: str) -> str:
    """The key expression the point cloud of the publisher on *namespace* is published on."""
    return f"{namespace}_pointcloud"


def allocate_pointcloud_buffers(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Allocate position/color buffers sized for one point per pixel.

    Args:
        width: Image width.
        height: Image height.

    Returns:
        A ``(positions, colors)`` tuple of zero-initialized ``(width * height, 3)`` arrays,
        with dtypes ``float32`` and ``uint8`` respectively.
    """
    num_points = width * height
    positions = np.zeros((num_points, 3), dtype=np.float32)
    colors = np.zeros((num_points, 3), dtype=np.uint8)
    return positions, colors


def fill_pointcloud_buffers(positions_buf: np.ndarray, colors_buf: np.ndarray, point_cloud: PointCloud) -> int:
    """Stage a (possibly sparse) point cloud into fixed-size buffers, in place.

    Positions beyond the valid point count are filled with NaN, so a receiver that doesn't check
    the returned count still sees invalid data rather than a previous, larger point cloud's
    leftovers. Missing colors are filled with black.

    Args:
        positions_buf: Pre-allocated ``(N, 3)`` float32 buffer to fill in place.
        colors_buf: Pre-allocated ``(N, 3)`` uint8 buffer to fill in place.
        point_cloud: The point cloud to stage.

    Returns:
        The number of valid points written (``point_cloud.points.shape[0]``).
    """
    num_points = point_cloud.points.shape[0]
    positions_buf[:num_points] = point_cloud.points
    positions_buf[num_points:] = np.nan
    if point_cloud.colors is not None:
        colors_buf[:num_points] = point_cloud.colors
    else:
        colors_buf[:num_points] = 0  # Use black if no colors
    return num_points


def read_matching_point_cloud(
    frame_reader: ZenohReader[Any],
    frame: Any,
    frame_count: int,
    pointcloud_reader: ZenohReader[PointCloudFrameBuffer],
    wait: Callable[[ZenohReader[Any], int], None],
) -> tuple[Any, int, PointCloudFrameBuffer]:
    """Find the point cloud with the same ``frame_id`` as *frame*, moving on to a newer frame if needed.

    Both readers only hold their latest message. A point cloud is published just before its frame,
    so the one matching *frame* has normally arrived already, but it may also still be in flight, or
    already replaced by the next frame's point cloud. In that last case the frame moves on as well.

    Args:
        frame_reader: Reader of the frames.
        frame: The frame read last from *frame_reader*.
        frame_count: The message count of *frame*.
        pointcloud_reader: Reader of the point clouds.
        wait: ``wait(reader, after)`` blocks until *reader* has a message newer than message count
            *after*, raising a ``TimeoutError`` if it does not arrive in time.

    Returns:
        The ``(frame, frame_count, point_cloud)`` with matching frame ids.
    """
    point_cloud, point_cloud_count = pointcloud_reader.read()
    while (point_cloud_id := int(point_cloud.frame_id.item())) != (frame_id := int(frame.frame_id.item())):
        if point_cloud_id < frame_id:
            wait(pointcloud_reader, point_cloud_count)
        else:
            # Also when the publisher restarted: its new frame ids start from 0 again.
            wait(frame_reader, frame_count)
            frame, frame_count = frame_reader.read()
        point_cloud, point_cloud_count = pointcloud_reader.read()
    return frame, frame_count, point_cloud


class PointCloudPublisherMixin(BaseCameraPublisher):
    """Publishes the camera's point cloud on its own key expression when ``enable_pointcloud`` is set."""

    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        shared_memory_namespace: str = "camera",
        enable_pointcloud: bool = True,
    ):
        self.enable_pointcloud = enable_pointcloud
        super().__init__(camera_cls, camera_kwargs, shared_memory_namespace)

    def _setup(self) -> None:
        super()._setup()

        if self.enable_pointcloud:
            # Some cameras return sparse point clouds, so there is a buffer slot for every pixel.
            self._pcd_pos_buf, self._pcd_col_buf = allocate_pointcloud_buffers(*self._camera.resolution)
            self._pointcloud_writer = self._declare_writer(
                pointcloud_key_expr(self._shared_memory_namespace),
                PointCloudFrameBuffer.layout(*self._camera.resolution),
            )

    def _publish_side_streams(self, frame_id: int, frame_timestamp: float) -> None:
        super()._publish_side_streams(frame_id, frame_timestamp)

        # Retrieving and serializing a point cloud is expensive, so skip it while nobody receives it.
        if self.enable_pointcloud and self._pointcloud_writer.has_subscribers:
            point_cloud = self._camera.retrieve_colored_point_cloud()
            num_valid_points = fill_pointcloud_buffers(self._pcd_pos_buf, self._pcd_col_buf, point_cloud)
            self._pointcloud_writer(
                PointCloudFrameBuffer(
                    **self._header(frame_id, frame_timestamp),
                    point_cloud_positions=self._pcd_pos_buf,
                    point_cloud_colors=self._pcd_col_buf,
                    num_valid_points=np.array([num_valid_points], dtype=np.int32),
                )
            )


class PointCloudReceiverMixin(BaseCameraReceiver):
    """Receives the point cloud matching each frame when ``enable_pointcloud`` is set."""

    def __init__(
        self,
        shared_memory_namespace: str,
        enable_pointcloud: bool = True,
        timeout: float | None = DEFAULT_FIRST_MESSAGE_TIMEOUT,
    ) -> None:
        self.enable_pointcloud = enable_pointcloud
        self._pointcloud_reader: ZenohReader[PointCloudFrameBuffer] | None = None
        # Connects, so the attributes above must be set first.
        super().__init__(shared_memory_namespace, timeout=timeout)

    def _setup_frame_reader(self, resolution: CameraResolutionType) -> None:
        super()._setup_frame_reader(resolution)

        if self.enable_pointcloud:
            self._pointcloud_reader = self._declare_reader(
                pointcloud_key_expr(self._shared_memory_namespace), PointCloudFrameBuffer.layout(*resolution)
            )

    def grab_images(self) -> None:
        """Read the latest frame and, if enabled, the point cloud with the same ``frame_id``."""
        super().grab_images()

        # The frame reader is set whenever the point cloud reader is, and grab_images() raised above if not.
        if self._pointcloud_reader is not None and self._reader is not None:
            self._last_frame, self._consumed_count, self._last_point_cloud = read_matching_point_cloud(
                self._reader, self._last_frame, self._consumed_count, self._pointcloud_reader, self._wait_for_message
            )

    def retrieve_colored_point_cloud(self) -> PointCloud:
        if not self.enable_pointcloud:
            raise RuntimeError("Cannot retrieve point cloud when point cloud is not enabled.")
        num_points = self._last_point_cloud.num_valid_points.item()
        positions = self._last_point_cloud.point_cloud_positions[:num_points]
        colors = self._last_point_cloud.point_cloud_colors[:num_points]
        return PointCloud(positions, colors)
