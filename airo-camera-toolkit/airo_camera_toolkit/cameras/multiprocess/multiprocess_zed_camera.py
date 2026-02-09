"""Publisher and receiver classes for multiprocess Zed camera sharing."""

import multiprocessing
import time
from typing import Any, Optional

import loguru
import numpy as np
from airo_camera_toolkit.cameras.multiprocess.base_publisher import BaseCameraPublisher
from airo_camera_toolkit.cameras.multiprocess.frame_data import PointCloudBuffer, SpatialMapBuffer, ZedFrameBuffer
from airo_camera_toolkit.cameras.multiprocess.multiprocess_stereo_rgbd_camera import MultiprocessStereoRGBDReceiver
from airo_camera_toolkit.cameras.zed.zed import Zed, ZedSpatialMap
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_ipc.cyclone_shm.patterns.sm_reader import SMReader
from airo_ipc.cyclone_shm.patterns.sm_writer import SMWriter
from airo_typing import CameraResolutionType, HomogeneousMatrixType, NumpyDepthMapType, NumpyIntImageType, PointCloud

logger = loguru.logger


class MultiprocessZedPublisher(BaseCameraPublisher):
    """Publishes Zed camera data including positional tracking and spatial mapping to shared memory."""

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
        self.enable_pointcloud = enable_pointcloud
        self.enable_positional_tracking = camera_kwargs.get("camera_tracking_params") is not None
        self.enable_spatial_mapping = camera_kwargs.get("camera_mapping_params") is not None
        self.max_spatial_map_chunks = max_spatial_map_chunks
        self.max_spatial_map_points = max_spatial_map_points
        self.map_refresh_interval = map_refresh_interval
        super().__init__(camera_cls, camera_kwargs, shared_memory_namespace)

    def _get_frame_buffer_template(self, width: int, height: int) -> Any:
        """Return Zed frame buffer template."""
        return ZedFrameBuffer.template(width, height)

    def _setup(self) -> None:
        """Set up camera and prepare buffers for point clouds and spatial mapping."""
        super()._setup()

        # Cache static camera parameters
        assert isinstance(self._camera, StereoRGBDCamera)
        self._pose_right_in_left = self._camera.pose_of_right_view_in_left_view
        self._intrinsics_left = self._camera.intrinsics_matrix(view=StereoRGBDCamera.LEFT_RGB)
        self._intrinsics_right = self._camera.intrinsics_matrix(view=StereoRGBDCamera.RIGHT_RGB)

        if self.enable_pointcloud:
            # Prepare buffers for point cloud data
            self._pcd_pos_buf = np.zeros(
                (self._camera.resolution[0] * self._camera.resolution[1], 3),
                dtype=np.float32,
            )
            self._pcd_col_buf = np.zeros(
                (self._camera.resolution[0] * self._camera.resolution[1], 3),
                dtype=np.uint8,
            )
            self._pcd_writer = SMWriter(
                domain_participant=self._dp,
                topic_name=f"{self._shared_memory_namespace}_pcd",
                idl_dataclass=PointCloudBuffer.template(self._camera.resolution[0], self._camera.resolution[1]),
            )

        if self.enable_spatial_mapping:
            # Initialize buffers for spatial map data
            self._spatial_map_chunks_updated = np.zeros(self.max_spatial_map_chunks, dtype=np.bool_)
            self._spatial_map_chunk_sizes = np.zeros(self.max_spatial_map_chunks, dtype=np.int32)
            self._spatial_map_point_positions = np.zeros((self.max_spatial_map_points, 3), dtype=np.float32)
            self._spatial_map_point_colors = np.zeros((self.max_spatial_map_points, 3), dtype=np.uint8)
            self._spatial_map_writer = SMWriter(
                domain_participant=self._dp,
                topic_name=f"{self._shared_memory_namespace}_spatial_map",
                idl_dataclass=SpatialMapBuffer.template(self.max_spatial_map_chunks, self.max_spatial_map_points),
            )

    def _retrieve_frame_data(self, frame_id: int, frame_timestamp: float) -> None:
        """Retrieve Zed stereo RGB-D data, pose, point cloud, and optionally spatial map."""
        self._current_frame_id = frame_id
        self._current_frame_timestamp = frame_timestamp

        # Capture left and right images
        self._current_rgb_left = self._camera.get_rgb_image_as_int()
        self._current_rgb_right = self._camera._retrieve_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB)

        # Capture depth data
        self._current_depth_map = self._camera.get_depth_map()
        self._current_depth_image = self._camera.get_depth_image()

        # Capture camera pose if tracking is enabled
        if self.enable_positional_tracking:
            self._current_camera_pose = self._camera._retrieve_camera_pose()
        else:
            self._current_camera_pose = np.eye(4, dtype=np.float64)

        # Capture point cloud if enabled
        if self.enable_pointcloud:
            point_cloud = self._camera._retrieve_colored_point_cloud()

            # Handle sparse point clouds
            self._pcd_pos_buf.fill(np.nan)
            self._pcd_pos_buf[: point_cloud.points.shape[0]] = point_cloud.points

            if point_cloud.colors is not None:
                self._pcd_col_buf[: point_cloud.colors.shape[0]] = point_cloud.colors
            else:
                self._pcd_col_buf[: point_cloud.points.shape[0]] = 0  # Use black if no colors

            self._current_pcd_num_points = point_cloud.points.shape[0]

        # Capture spatial map if enabled and on refresh interval
        self._current_spatial_map: Optional[ZedSpatialMap] = None
        if self.enable_spatial_mapping and frame_id % self.map_refresh_interval == 0:
            assert isinstance(self._camera, Zed)
            self._camera._request_spatial_map_update()
            self._current_spatial_map = self._camera._retrieve_spatial_map()

    def _write_frame_data(self) -> None:
        """Write Zed frame data, point cloud, and spatial map to shared memory."""
        # Write main Zed frame
        frame_data = ZedFrameBuffer(
            frame_id=np.array([self._current_frame_id], dtype=np.uint64),
            frame_timestamp=np.array([self._current_frame_timestamp], dtype=np.float64),
            rgb=self._current_rgb_left,
            rgb_right=self._current_rgb_right,
            intrinsics=self._intrinsics_left,
            intrinsics_right=self._intrinsics_right,
            pose_right_in_left=self._pose_right_in_left,
            depth=self._current_depth_map,
            depth_image=self._current_depth_image,
            camera_pose=self._current_camera_pose,
        )
        self._writer(frame_data)

        # Write point cloud if enabled
        if self.enable_pointcloud:
            pcd_data = PointCloudBuffer(
                frame_id=np.array([self._current_frame_id], dtype=np.uint64),
                frame_timestamp=np.array([self._current_frame_timestamp], dtype=np.float64),
                point_cloud_positions=self._pcd_pos_buf,
                point_cloud_colors=self._pcd_col_buf,
                point_cloud_valid=np.array([self._current_pcd_num_points], dtype=np.int32),
            )
            self._pcd_writer(pcd_data)

        # Write spatial map if available
        if self.enable_spatial_mapping and self._current_spatial_map is not None:
            self._write_spatial_map()

    def _write_spatial_map(self) -> None:
        """Write spatial map data to shared memory."""
        if self._current_spatial_map is None:
            raise AssertionError("Spatial map data is not available for writing.")

        num_chunks = self._current_spatial_map.num_chunks
        chunks_updated = self._current_spatial_map.chunks_updated
        chunk_sizes = self._current_spatial_map.chunk_sizes
        point_positions = self._current_spatial_map.full_pointcloud.points
        point_colors = self._current_spatial_map.full_pointcloud.colors
        # Check if number of chunks exceeds maximum allowed
        # For simplicity, just throw error for now. Could later be improved to only send partial map.
        if num_chunks > self.max_spatial_map_chunks:
            raise RuntimeError(
                f"Spatial map has {num_chunks} chunks, exceeding the maximum of {self.max_spatial_map_chunks}."
            )

        # Check if number of points exceeds maximum allowed.
        # For simplicity, just throw error for now. Could later be improved to only send partial map.
        if self._current_spatial_map.size > self.max_spatial_map_points:
            raise RuntimeError(
                f"Spatial map has {self._current_spatial_map.size} points, exceeding the maximum of {self.max_spatial_map_points}."
            )

        # Put data in buffers
        self._spatial_map_chunks_updated[:num_chunks] = np.array(chunks_updated)
        self._spatial_map_chunk_sizes[:num_chunks] = np.array(chunk_sizes)
        self._spatial_map_point_positions[: self._current_spatial_map.size, :] = np.array(point_positions)
        self._spatial_map_point_colors[: self._current_spatial_map.size, :] = np.array(point_colors)

        # Write spatial map buffer
        spatial_map_data = SpatialMapBuffer(
            frame_id=np.array([self._current_frame_id], dtype=np.uint64),
            frame_timestamp=np.array([self._current_frame_timestamp], dtype=np.float64),
            num_chunks=np.array([num_chunks], dtype=np.int32),
            chunks_updated=self._spatial_map_chunks_updated,
            chunk_sizes=self._spatial_map_chunk_sizes,
            point_positions=self._spatial_map_point_positions,
            point_colors=self._spatial_map_point_colors,
        )
        self._spatial_map_writer(spatial_map_data)


class MultiprocessZedReceiver(MultiprocessStereoRGBDReceiver, StereoRGBDCamera):
    """Receives Zed camera data from shared memory."""

    def __init__(
        self,
        shared_memory_namespace: str,
        enable_pointcloud: bool = True,
        enable_positional_tracking: bool = False,
        enable_spatial_mapping: bool = False,
        max_spatial_map_chunks: int = 10000,
        max_spatial_map_points: int = 1000000,
    ) -> None:
        self.enable_pointcloud = enable_pointcloud
        self.enable_positional_tracking = enable_positional_tracking
        self.enable_spatial_mapping = enable_spatial_mapping
        self.max_spatial_map_chunks = max_spatial_map_chunks
        self.max_spatial_map_points = max_spatial_map_points

        super().__init__(shared_memory_namespace)

    def _setup_frame_reader(self, resolution: CameraResolutionType) -> None:
        super()._setup_frame_reader(resolution)

        if self.enable_pointcloud:
            self._reader_pcd = SMReader(
                domain_participant=self._dp,
                topic_name=f"{self._shared_memory_namespace}_pcd",
                idl_dataclass=PointCloudBuffer.template(resolution[0], resolution[1]),
            )
            self._last_pcd_frame = PointCloudBuffer.template(resolution[0], resolution[1])

        if self.enable_spatial_mapping:
            self._reader_spatial_map = SMReader(
                domain_participant=self._dp,
                topic_name=f"{self._shared_memory_namespace}_spatial_map",
                idl_dataclass=SpatialMapBuffer.template(self.max_spatial_map_chunks, self.max_spatial_map_points),
            )
            self._last_spatial_map_frame = SpatialMapBuffer.template(
                self.max_spatial_map_chunks, self.max_spatial_map_points
            )

    def _get_frame_buffer_template(self, width: int, height: int) -> Any:
        """Return Zed frame buffer template."""
        return ZedFrameBuffer.template(width, height)

    def _retrieve_rgb_image_as_int(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyIntImageType:
        """Retrieve RGB image as integer array."""
        if view == StereoRGBDCamera.LEFT_RGB:
            return self._last_frame.rgb
        else:
            return self._last_frame.rgb_right

    @property
    def pose_of_right_view_in_left_view(self) -> HomogeneousMatrixType:
        """Get the pose of the right camera in left camera frame."""
        return self._last_frame.pose_right_in_left

    def intrinsics_matrix(self, view: str = StereoRGBDCamera.LEFT_RGB) -> np.ndarray:
        """Get camera intrinsics matrix."""
        if view == StereoRGBDCamera.LEFT_RGB:
            return self._last_frame.intrinsics
        else:
            return self._last_frame.intrinsics_right

    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        """Retrieve depth map from frame buffer."""
        return self._last_frame.depth

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        """Retrieve depth image from frame buffer."""
        return self._last_frame.depth_image

    def _retrieve_colored_point_cloud(self) -> PointCloud:
        """Retrieve colored point cloud."""
        if not self.enable_pointcloud:
            raise RuntimeError("Cannot retrieve point cloud when point cloud is not enabled.")
        self._last_pcd_frame = self._reader_pcd()
        num_points = self._last_pcd_frame.point_cloud_valid.item()
        positions = self._last_pcd_frame.point_cloud_positions[:num_points]
        colors = self._last_pcd_frame.point_cloud_colors[:num_points]
        return PointCloud(positions, colors)

    def _retrieve_camera_pose(self) -> np.ndarray:
        """Returns the 4x4 global pose matrix of the camera if tracking is enabled."""
        if not self.enable_positional_tracking:
            raise RuntimeError("Cannot retrieve camera pose when positional tracking is not enabled.")
        return self._last_frame.camera_pose

    def _retrieve_spatial_map(self) -> ZedSpatialMap:
        """
        Reconstructs the spatial map from the shared memory buffer.

        Returns:
            list[tuple[PointCloud, bool]]: A list of tuples, each containing a PointCloud object
                                           and a boolean indicating whether the chunk has been updated.
        """
        if not self.enable_spatial_mapping:
            raise RuntimeError("Cannot retrieve spatial map when it is not enabled.")

        self._last_spatial_map_frame = self._reader_spatial_map()

        # Get the last spatial map frame from shared memory
        buf = self._last_spatial_map_frame

        # Get total chunks
        num_chunks = int(buf.num_chunks.item())

        # If no chunks, return empty spatial map
        if num_chunks == 0:
            return ZedSpatialMap([], [])

        current_offset = 0

        chunks = []
        chunks_updated = []

        # Iterate through the chunks described in the shared memory
        for i in range(num_chunks):
            chunk_size = int(buf.chunk_sizes[i])
            has_been_updated = bool(buf.chunks_updated[i])

            # Slice the flattened arrays based on the current chunk size
            # We use .copy() to decouple the PointCloud data from the shared memory buffer
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

    # while not receiver.is_ready():
    #     logger.warning("Waiting for receiver to be ready...")
    #     time.sleep(1.0)

    receiver._grab_images()

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

        # Retrieve images and data from shared memory
        image = receiver.get_rgb_image_as_int()
        image_right = receiver._retrieve_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB)
        depth_map = receiver._retrieve_depth_map()
        depth_image = receiver._retrieve_depth_image()
        # confidence_map = receiver._retrieve_confidence_map()
        point_cloud = receiver._retrieve_colored_point_cloud()

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_right_bgr = cv2.cvtColor(image_right, cv2.COLOR_RGB2BGR)

        spatial_map = receiver._retrieve_spatial_map()
        pose_matrix = receiver._retrieve_camera_pose()

        # Visualize images using OpenCV
        cv2.imshow("RGB Image", image_bgr)
        cv2.imshow("RGB Image Right", image_right_bgr)
        cv2.imshow("Depth Map", depth_map)
        cv2.imshow("Depth Image", depth_image)
        # cv2.imshow("Confidence Map", confidence_map)

        # If enabled, log point cloud to rerun
        if log_point_cloud:
            point_cloud.points[np.isnan(point_cloud.points)] = 0
            if point_cloud.colors is not None:
                point_cloud.colors[np.isnan(point_cloud.colors)] = 0
            rr.log(
                "World/point_cloud",
                rr.Points3D(
                    positions=camera_to_rerun(point_cloud.points),
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

    publisher.stop()
    publisher.join()
