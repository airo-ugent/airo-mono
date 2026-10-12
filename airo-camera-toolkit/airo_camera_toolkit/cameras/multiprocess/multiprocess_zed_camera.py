"""Publisher and receiver classes for multiprocess Zed camera sharing."""

import multiprocessing
import time
from typing import Any

import loguru
import numpy as np
from airo_camera_toolkit.cameras.multiprocess.frame_data import SpatialMapBuffer, StereoRGBDFrameBuffer, ZedFrameBuffer
from airo_camera_toolkit.cameras.multiprocess.multiprocess_stereo_rgbd_camera import (
    MultiprocessStereoRGBDPublisher,
    MultiprocessStereoRGBDReceiver,
)
from airo_camera_toolkit.cameras.multiprocess.zenoh_reader import DEFAULT_FIRST_MESSAGE_TIMEOUT, ZenohReader
from airo_camera_toolkit.cameras.zed.zed import Zed, ZedSpatialMap
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_typing import CameraResolutionType, PointCloud

logger = loguru.logger


class MultiprocessZedPublisher(MultiprocessStereoRGBDPublisher):
    """Publishes Zed camera data including positional tracking and spatial mapping over Zenoh.

    Publishes the frames of :class:`MultiprocessStereoRGBDPublisher` with the camera pose added,
    and the spatial map (if enabled) on the ``<namespace>_spatial_map`` key expression.
    """

    _frame_type: type[StereoRGBDFrameBuffer] = ZedFrameBuffer

    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        shared_memory_namespace: str = "camera",
        enable_pointcloud: bool = True,
        max_spatial_map_chunks: int = 10000,
        max_spatial_map_points: int = 1000000,
        map_refresh_interval: int = 5,  # Map is refreshed every N frames
    ):
        self.enable_positional_tracking = camera_kwargs.get("camera_tracking_params") is not None
        self.enable_spatial_mapping = camera_kwargs.get("camera_mapping_params") is not None
        self.max_spatial_map_chunks = max_spatial_map_chunks
        self.max_spatial_map_points = max_spatial_map_points
        self.map_refresh_interval = map_refresh_interval
        super().__init__(camera_cls, camera_kwargs, shared_memory_namespace, enable_pointcloud=enable_pointcloud)

    def _setup(self) -> None:
        """Set up the inherited stereo RGB-D state, then the spatial map writer."""
        super()._setup()

        if self.enable_spatial_mapping:
            self._spatial_map_chunks_updated = np.zeros(self.max_spatial_map_chunks, dtype=np.bool_)
            self._spatial_map_chunk_sizes = np.zeros(self.max_spatial_map_chunks, dtype=np.int32)
            self._spatial_map_point_positions = np.zeros((self.max_spatial_map_points, 3), dtype=np.float32)
            self._spatial_map_point_colors = np.zeros((self.max_spatial_map_points, 3), dtype=np.uint8)
            self._spatial_map_writer = self._declare_writer(
                f"{self._shared_memory_namespace}_spatial_map",
                SpatialMapBuffer.layout(self.max_spatial_map_chunks, self.max_spatial_map_points),
            )

    def _capture_extra_fields(self) -> dict[str, np.ndarray]:
        """Capture the camera pose (identity if positional tracking is disabled)."""
        camera_pose = (
            self._camera.retrieve_camera_pose() if self.enable_positional_tracking else np.eye(4, dtype=np.float64)
        )
        return {"camera_pose": camera_pose}

    def _publish_side_streams(self, frame_id: int, frame_timestamp: float) -> None:
        """Publish the point cloud (if enabled) and, every few frames, the spatial map (if enabled)."""
        super()._publish_side_streams(frame_id, frame_timestamp)

        if (
            self.enable_spatial_mapping
            and frame_id % self.map_refresh_interval == 0
            and self._spatial_map_writer.has_subscribers
        ):
            assert isinstance(self._camera, Zed)
            self._camera.request_spatial_map_update()
            spatial_map = self._camera.retrieve_spatial_map()
            self._write_spatial_map(frame_id, frame_timestamp, spatial_map)

    def _write_spatial_map(self, frame_id: int, frame_timestamp: float, spatial_map: ZedSpatialMap) -> None:
        """Stage *spatial_map* into the spatial map buffers and publish it."""
        num_chunks = spatial_map.num_chunks
        chunks_updated = spatial_map.chunks_updated
        chunk_sizes = spatial_map.chunk_sizes
        point_positions = spatial_map.full_pointcloud.points
        point_colors = spatial_map.full_pointcloud.colors

        # For simplicity, just throw an error for now if the map exceeds the preallocated buffers.
        # Could later be improved to only send a partial map.
        if num_chunks > self.max_spatial_map_chunks:
            raise RuntimeError(
                f"Spatial map has {num_chunks} chunks, exceeding the maximum of {self.max_spatial_map_chunks}."
            )
        if spatial_map.size > self.max_spatial_map_points:
            raise RuntimeError(
                f"Spatial map has {spatial_map.size} points, exceeding the maximum of {self.max_spatial_map_points}."
            )

        self._spatial_map_chunks_updated[:num_chunks] = chunks_updated
        self._spatial_map_chunk_sizes[:num_chunks] = chunk_sizes
        self._spatial_map_point_positions[: spatial_map.size, :] = point_positions
        if point_colors is not None:
            self._spatial_map_point_colors[: spatial_map.size, :] = point_colors
        else:
            self._spatial_map_point_colors[: spatial_map.size, :] = 0

        self._spatial_map_writer(
            SpatialMapBuffer(
                **self._header(frame_id, frame_timestamp),
                num_chunks=np.array([num_chunks], dtype=np.int32),
                chunks_updated=self._spatial_map_chunks_updated,
                chunk_sizes=self._spatial_map_chunk_sizes,
                point_positions=self._spatial_map_point_positions,
                point_colors=self._spatial_map_point_colors,
            )
        )

    def _publisher_config(self) -> dict[str, Any]:
        return {**super()._publisher_config(), "positional_tracking": self.enable_positional_tracking}


class MultiprocessZedReceiver(MultiprocessStereoRGBDReceiver):
    """Receives Zed camera data over Zenoh."""

    _frame_type: type[StereoRGBDFrameBuffer] = ZedFrameBuffer

    def __init__(
        self,
        shared_memory_namespace: str,
        enable_pointcloud: bool = True,
        enable_positional_tracking: bool = False,
        enable_spatial_mapping: bool = False,
        max_spatial_map_chunks: int = 10000,
        max_spatial_map_points: int = 1000000,
        timeout: float | None = DEFAULT_FIRST_MESSAGE_TIMEOUT,
    ) -> None:
        self.enable_positional_tracking = enable_positional_tracking
        self.enable_spatial_mapping = enable_spatial_mapping
        self.max_spatial_map_chunks = max_spatial_map_chunks
        self.max_spatial_map_points = max_spatial_map_points
        self._reader_spatial_map: ZenohReader[SpatialMapBuffer] | None = None

        super().__init__(shared_memory_namespace, enable_pointcloud=enable_pointcloud, timeout=timeout)

    def _setup_frame_reader(self, resolution: CameraResolutionType) -> None:
        # The camera pose is part of every frame, so a mismatch would not show in the wire format:
        # without tracking, the publisher fills in an identity pose.
        if self.enable_positional_tracking and not self._metadata.config.get("positional_tracking", False):
            raise ValueError(
                "enable_positional_tracking is set, but the publisher was not created with camera_tracking_params."
            )

        super()._setup_frame_reader(resolution)

        if self.enable_spatial_mapping:
            self._reader_spatial_map = self._declare_reader(
                f"{self._shared_memory_namespace}_spatial_map",
                SpatialMapBuffer.layout(self.max_spatial_map_chunks, self.max_spatial_map_points),
            )

    def grab_images(self) -> None:
        """Read the latest frame, and the latest spatial map if enabled.

        The spatial map is published on a separate key expression and only refreshed every few
        frames, so it generally belongs to an earlier ``frame_id`` than the frame.
        """
        super().grab_images()
        if self._reader_spatial_map is not None:
            self._last_spatial_map_frame = self._reader_spatial_map()

    def stop(self) -> None:
        """Undeclare the spatial map reader, then the base reader and session."""
        if self._reader_spatial_map is not None:
            self._reader_spatial_map.stop()
            self._reader_spatial_map = None
        super().stop()

    def retrieve_camera_pose(self) -> np.ndarray:
        """Returns the 4x4 global pose matrix of the camera if tracking is enabled."""
        if not self.enable_positional_tracking:
            raise RuntimeError("Cannot retrieve camera pose when positional tracking is not enabled.")
        return self._last_frame.camera_pose

    def retrieve_spatial_map(self) -> ZedSpatialMap:
        """
        Reconstructs the spatial map from the frame captured by the last grab_images().

        Returns:
            list[tuple[PointCloud, bool]]: A list of tuples, each containing a PointCloud object
                                           and a boolean indicating whether the chunk has been updated.
        """
        if not self.enable_spatial_mapping:
            raise RuntimeError("Cannot retrieve spatial map when it is not enabled.")
        if self._reader_spatial_map is None:
            raise RuntimeError("This receiver was stopped; call reconnect() to use it again.")

        # Get the last spatial map frame
        buf = self._last_spatial_map_frame

        # Get total chunks
        num_chunks = int(buf.num_chunks.item())

        # If no chunks, return empty spatial map
        if num_chunks == 0:
            return ZedSpatialMap([], [])

        current_offset = 0

        chunks = []
        chunks_updated = []

        # Iterate through the chunks described in the spatial map frame
        for i in range(num_chunks):
            chunk_size = int(buf.chunk_sizes[i])
            has_been_updated = bool(buf.chunks_updated[i])

            # Slice the flattened arrays based on the current chunk size
            # We use .copy() to decouple the PointCloud data from the received payload
            positions = buf.point_positions[current_offset : current_offset + chunk_size].copy()
            colors = buf.point_colors[current_offset : current_offset + chunk_size].copy()

            # Reconstruct the ZedSpatialMap
            chunks.append(PointCloud(positions, colors))
            chunks_updated.append(has_been_updated)

            # Increment offset for the next chunk
            current_offset += chunk_size

        return ZedSpatialMap(chunks, chunks_updated)


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBDPublisher and MultiprocessRGBDReceiver.
    You can also use the MultiprocessRGBDReceiver in a different process (e.g. in a different python script)
    """
    import cv2
    import rerun as rr

    def camera_to_rerun(points_cam: np.ndarray) -> np.ndarray:
        """
        Convert points from camera frame (X right, Y down, Z forward)
        to Rerun frame (X right, Y forward, Z up).
        """
        assert points_cam.shape[1] == 3

        points_rerun = np.empty_like(points_cam)

        points_rerun[:, 0] = points_cam[:, 0]  # X -> X
        points_rerun[:, 1] = points_cam[:, 2]  # Z -> Y
        points_rerun[:, 2] = -points_cam[:, 1]  # -Y -> Z

        return points_rerun

    multiprocessing.set_start_method("spawn", force=True)

    resolution = Zed.InitParams.RESOLUTION_720
    camera_fps = 15

    camera_tracking_params = Zed.TrackingParams()
    camera_mapping_params = Zed.MappingParams()
    publisher = MultiprocessZedPublisher(
        Zed,
        camera_kwargs={
            "resolution": resolution,
            "fps": camera_fps,
            "depth_mode": Zed.InitParams.NEURAL_DEPTH_MODE,
            "camera_tracking_params": camera_tracking_params,
            "camera_mapping_params": camera_mapping_params,
        },
    )

    publisher.start()

    receiver = MultiprocessZedReceiver("camera", enable_positional_tracking=True, enable_spatial_mapping=True)
    try:
        # while not receiver.is_ready():
        #     logger.warning("Waiting for receiver to be ready...")
        #     time.sleep(1.0)

        receiver.grab_images()

        with np.printoptions(precision=3, suppress=True):
            print("Intrinsics left:\n", receiver.intrinsics_matrix())
            print(
                "Intrinsics right:\n",
                receiver.intrinsics_matrix(view=StereoRGBDCamera.RIGHT_RGB),
            )
            print("Pose right in left:\n", receiver.pose_of_right_view_in_left_view)

        cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("RGB Image Right", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Confidence Map", cv2.WINDOW_NORMAL)

        rr.init(f"{MultiprocessZedReceiver.__name__}")
        rr.spawn(memory_limit="2GB")

        # 1. Setup World Coordinate System (Z-Up)
        rr.log("World", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        rr.log(
            "World/Camera/Pinhole",
            rr.Pinhole(
                resolution=[1280, 720],  # Arbitrary resolution for visualization
                focal_length=700,  # Arbitrary FOV for visualization
            ),
            static=True,
        )

        log_point_cloud = False
        log_spatial_map = True

        time_current = None
        time_previous = None

        while True:
            time_previous = time_current
            time_current = time.time()

            # Retrieve images and data from the last received frame
            receiver.grab_images()
            image = receiver.retrieve_rgb_image_as_int()
            image_right = receiver.retrieve_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB)
            depth_map = receiver.retrieve_depth_map()
            depth_image = receiver.retrieve_depth_image()
            # confidence_map = receiver.retrieve_confidence_map()
            point_cloud = receiver.retrieve_colored_point_cloud()

            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_right_bgr = cv2.cvtColor(image_right, cv2.COLOR_RGB2BGR)

            spatial_map = receiver.retrieve_spatial_map()
            pose_matrix = receiver.retrieve_camera_pose()

            # Visualize images using OpenCV
            cv2.imshow("RGB Image", image_bgr)
            cv2.imshow("RGB Image Right", image_right_bgr)
            cv2.imshow("Depth Map", depth_map)
            cv2.imshow("Depth Image", depth_image)
            # cv2.imshow("Confidence Map", confidence_map)

            # If enabled, log point cloud to rerun
            if log_point_cloud:
                # Received arrays are read-only views, so replace NaNs in a copy. Colors are uint8 and cannot be NaN.
                positions = np.nan_to_num(point_cloud.points)
                rr.log(
                    "World/point_cloud",
                    rr.Points3D(
                        positions=camera_to_rerun(positions),
                        colors=point_cloud.colors,
                    ),
                )

            # If enabled, log spatial map to rerun
            if log_spatial_map:
                full_pointcloud = spatial_map.full_pointcloud
                rr.log(
                    "World/spatial_map",
                    rr.Points3D(
                        positions=camera_to_rerun(full_pointcloud.points),
                        colors=full_pointcloud.colors,
                    ),
                )

            # Visualize camera pose in Rerun

            # Extract Translation (first 3 rows, 4th column)
            translation = pose_matrix[:3, 3]

            # Extract Rotation Matrix (3x3 top-left)
            rotation_mat = pose_matrix[:3, :3]

            R_x_90 = np.array(
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                ],
                dtype=float,
            )

            rotation_mat = rotation_mat @ R_x_90

            # Log the transform.
            # This moves "World/Camera" (and its child "Pinhole") to the new location.
            rr.log("World/Camera", rr.Transform3D(translation=translation, mat3x3=rotation_mat))

            key = cv2.waitKey(10)
            if key == ord("q"):
                break
            elif key == ord("l"):
                log_point_cloud = not log_point_cloud
    finally:
        receiver.stop()
        publisher.stop()
        publisher.join()
