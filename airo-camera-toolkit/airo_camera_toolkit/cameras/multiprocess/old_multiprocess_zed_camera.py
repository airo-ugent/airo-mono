"""code for sharing the data of the ZED camera between processes using shared memory"""

import multiprocessing
import time
from dataclasses import dataclass
from typing import Any

import loguru
import numpy as np
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import ResolutionIdl
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgbd_camera import PointCloudBuffer
from airo_camera_toolkit.cameras.multiprocess.multiprocess_stereo_rgbd_camera import (
    MultiprocessStereoRGBDPublisher,
    MultiprocessStereoRGBDReceiver,
    StereoRGBDFrameBuffer,
)
from airo_camera_toolkit.cameras.zed.zed import Zed, ZedSpatialMap
from airo_ipc.cyclone_shm.idl_shared_memory.base_idl import BaseIdl  # type: ignore
from airo_ipc.cyclone_shm.patterns.sm_reader import SMReader  # type: ignore
from airo_ipc.cyclone_shm.patterns.sm_writer import SMWriter  # type: ignore

logger = loguru.logger
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_typing import CameraResolutionType, HomogeneousMatrixType, NumpyDepthMapType, NumpyIntImageType, PointCloud


@dataclass
class ZedFrameBuffer(StereoRGBDFrameBuffer):
    """This struct, sent over shared memory, contains a timestamp, two RGB images,
    the camera intrinsics, a depth image, a depth map and a global pose.
    It also contains the relative pose of the right camera in the left camera frame."""

    # Camera pose
    camera_pose: np.ndarray

    @staticmethod
    def template(width: int, height: int) -> Any:
        return ZedFrameBuffer(
            timestamp=np.empty((1,), dtype=np.float64),
            rgb_left=np.empty((height, width, 3), dtype=np.uint8),
            rgb_right=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics_left=np.empty((3, 3), dtype=np.float64),
            intrinsics_right=np.empty((3, 3), dtype=np.float64),
            pose_right_in_left=np.empty((4, 4), dtype=np.float64),
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth=np.empty((height, width), dtype=np.float32),
            camera_pose=np.empty((4, 4), dtype=np.float64),
        )


@dataclass
class SpatialMapBuffer(BaseIdl):  # type: ignore
    """This struct, sent over shared memory, contains the spatial map data of the Zed camera."""

    # Timestamp of the spatial map
    timestamp: np.ndarray
    # Amount of chunks in the spatial map
    num_chunks: np.ndarray
    # Array indicating which chunks have been updated
    chunks_updated: np.ndarray
    # Size of each chunk (number of points)
    chunk_sizes: np.ndarray
    # Arrays of concatenated chunk data (point positions and colors)
    point_positions: np.ndarray
    point_colors: np.ndarray

    @staticmethod
    def template(max_chunks: int, max_points: int) -> Any:
        return SpatialMapBuffer(
            timestamp=np.empty((1,), dtype=np.float64),
            num_chunks=np.empty((1,), dtype=np.uint32),
            chunks_updated=np.empty((max_chunks,), dtype=np.uint8),
            chunk_sizes=np.empty((max_chunks,), dtype=np.uint32),
            point_positions=np.empty((max_points, 3), dtype=np.float32),
            point_colors=np.empty((max_points, 3), dtype=np.uint8),
        )


@dataclass
class DynamicCameraData:
    """
    A data class representing dynamic camera data captured from a ZED camera.

    Attributes:
        image_left (NumpyIntImageType): Left stereo image from the ZED camera.
        image_right (NumpyIntImageType): Right stereo image from the ZED camera.
        depth_map (NumpyDepthMapType): Raw depth map data from the ZED sensor.
        depth_image (NumpyIntImageType): Depth information represented as an integer image.
        camera_pose (HomogeneousMatrixType): 4x4 homogeneous transformation matrix representing
            the camera's position and orientation in 3D space.
        point_cloud_valid (np.ndarray): Boolean mask indicating valid points in the point cloud.
        spatial_map (ZedSpatialMap): Spatial mapping data from the ZED camera's SLAM system.
    """

    image_left: NumpyIntImageType
    image_right: NumpyIntImageType
    depth_map: NumpyDepthMapType
    depth_image: NumpyIntImageType
    camera_pose: HomogeneousMatrixType
    point_cloud_valid: np.ndarray
    spatial_map: ZedSpatialMap


class MultiprocessZedPublisher(MultiprocessStereoRGBDPublisher):
    """Publishes the data of a Zed camera that implements the StereoRGBDCamera interface to shared memory blocks."""

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
        super().__init__(camera_cls, camera_kwargs, shared_memory_namespace, enable_pointcloud)

    def _setup(self) -> None:
        super()._setup()

        if self.enable_spatial_mapping:
            # Initialize buffers for spatial map data
            self._spatial_map_chunks_updated = np.zeros(self.max_spatial_map_chunks, dtype=np.bool_)
            self._spatial_map_chunk_sizes = np.zeros(self.max_spatial_map_chunks, dtype=np.uint32)
            self._spatial_map_point_positions = np.zeros((self.max_spatial_map_points, 3), dtype=np.float32)
            self._spatial_map_point_colors = np.zeros((self.max_spatial_map_points, 3), dtype=np.uint8)

    def _setup_sm_writer(self) -> None:
        # Create the shared memory writer for the frame information
        self._writer = SMWriter(
            domain_participant=self._dp,
            topic_name=self._shared_memory_namespace,
            idl_dataclass=ZedFrameBuffer.template(self._camera.resolution[0], self._camera.resolution[1]),
        )
        # Create the shared memory writer for the pointcloud
        if self.enable_pointcloud:
            self._pcd_writer = SMWriter(
                domain_participant=self._dp,
                topic_name=f"{self._shared_memory_namespace}_pcd",
                idl_dataclass=PointCloudBuffer.template(self._camera.resolution[0], self._camera.resolution[1]),
            )
        # Create the shared memory writer for the spatial map
        if self.enable_spatial_mapping:
            self._spatial_map_writer = SMWriter(
                domain_participant=self._dp,
                topic_name=f"{self._shared_memory_namespace}_spatial_map",
                idl_dataclass=SpatialMapBuffer.template(self.max_spatial_map_chunks, self.max_spatial_map_points),
            )

    def stop(self) -> None:
        self.shutdown_event.set()

    def _retrieve_dynamic_camera_data(self, frame_count: int) -> "DynamicCameraData":
        # Get information for the ZEDFrameBuffer
        image_left = self._camera.get_rgb_image_as_int()
        image_right = self._camera._retrieve_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB)
        depth_map = self._camera.get_depth_map()
        depth_image = self._camera.get_depth_image()
        if self.enable_positional_tracking:
            camera_pose = self._camera._retrieve_camera_pose()
        else:
            camera_pose = np.empty((4, 4), dtype=np.float64)

        # If point cloud is enabled, retrieve it for the PointCloudBuffer
        if self.enable_pointcloud:
            point_cloud = self._camera._retrieve_colored_point_cloud()

            # Some camera's, such as the Realsense D435i, return a sparse point cloud. This is not supported by the
            # current implementation of the RGBDFrameBuffer. Therefore, we make sure that we always retrieve a point
            # for every pixel in the RBG image.
            self._pcd_pos_buf.fill(np.nan)
            self._pcd_pos_buf[: point_cloud.points.shape[0]] = point_cloud.points
            if point_cloud.colors is not None:
                self._pcd_col_buf[: point_cloud.colors.shape[0]] = point_cloud.colors
            else:
                self._pcd_col_buf[: point_cloud.points.shape[0]] = 0  # If no colors, use black.
            point_cloud_valid = np.array([point_cloud.points.shape[0]], dtype=np.uint32)

        # If spatial mapping is enabled, request an update and try to retrieve the spatial map for the SpatialMapBuffer
        if self.enable_spatial_mapping:
            self._camera._request_spatial_map_update()  # Request an update each frame; if one is pending, nothing happens (see ZED-sdk docs)

            # Only update the spatial map every N frames to reduce load and suppress warning messages
            # If the spatial map is not updated, it gets the value None and is not sent to shared memory.
            spatial_map = None
            if frame_count % self.map_refresh_interval == 0:
                spatial_map = self._camera._retrieve_spatial_map()

        return DynamicCameraData(
            image_left,
            image_right,
            depth_map,
            depth_image,
            camera_pose,
            point_cloud_valid,
            spatial_map,  # type: ignore
        )

    def _write_camera_data_to_sm(
        self,
        timestamp: float,
        intrinsics_left: np.ndarray,
        intrinsics_right: np.ndarray,
        pose_right_in_left: np.ndarray,
        dyn_camera_data: "DynamicCameraData",
    ) -> None:
        # Write the ZedFrameBuffer to shared memory
        self._writer(
            ZedFrameBuffer(
                timestamp=np.array([timestamp], dtype=np.float64),
                rgb_left=dyn_camera_data.image_left,
                rgb_right=dyn_camera_data.image_right,
                intrinsics_left=intrinsics_left,
                intrinsics_right=intrinsics_right,
                pose_right_in_left=pose_right_in_left,
                depth=dyn_camera_data.depth_map,
                depth_image=dyn_camera_data.depth_image,
                camera_pose=dyn_camera_data.camera_pose,
            )
        )

        # If enabled, write the PointCloudBuffer to shared memory
        if self.enable_pointcloud:
            self._pcd_writer(
                PointCloudBuffer(
                    timestamp=np.array([timestamp], dtype=np.float64),
                    point_cloud_positions=self._pcd_pos_buf,
                    point_cloud_colors=self._pcd_col_buf,
                    point_cloud_valid=dyn_camera_data.point_cloud_valid,
                )
            )

        # If spatial mapping is enabled, the spatial map has been updated, and the spatial map is non-empty, write the SpatialMapBuffer to shared memory
        # Can be empty in the beginning when the map is still being built.
        if self.enable_spatial_mapping and dyn_camera_data.spatial_map and dyn_camera_data.spatial_map.size > 0:
            # Prepare spatial map data for shared memory
            num_chunks = dyn_camera_data.spatial_map.num_chunks
            chunks_updated = dyn_camera_data.spatial_map.chunks_updated
            chunk_sizes = dyn_camera_data.spatial_map.chunk_sizes
            point_positions = dyn_camera_data.spatial_map.full_pointcloud.points
            point_colors = dyn_camera_data.spatial_map.full_pointcloud.colors

            # Check if number of chunks exceeds maximum allowed
            # For simplicity, just throw error for now. Could later be improved to only send partial map.
            if num_chunks > self.max_spatial_map_chunks:
                raise RuntimeError(
                    f"Spatial map has {num_chunks} chunks, exceeding the maximum of {self.max_spatial_map_chunks}."
                )

            # Check if number of points exceeds maximum allowed.
            # For simplicity, just throw error for now. Could later be improved to only send partial map.
            if dyn_camera_data.spatial_map.size > self.max_spatial_map_points:
                raise RuntimeError(
                    f"Spatial map has {dyn_camera_data.spatial_map.size} points, exceeding the maximum of {self.max_spatial_map_points}."
                )

            # Put data in buffers
            self._spatial_map_chunks_updated[:num_chunks] = np.array(chunks_updated)
            self._spatial_map_chunk_sizes[:num_chunks] = np.array(chunk_sizes)
            self._spatial_map_point_positions[: dyn_camera_data.spatial_map.size, :] = np.array(point_positions)
            self._spatial_map_point_colors[: dyn_camera_data.spatial_map.size, :] = np.array(point_colors)

            self._spatial_map_writer(
                SpatialMapBuffer(
                    timestamp=np.array([timestamp], dtype=np.float64),
                    num_chunks=np.array([num_chunks], dtype=np.uint32),
                    chunks_updated=self._spatial_map_chunks_updated,
                    chunk_sizes=self._spatial_map_chunk_sizes,
                    point_positions=self._spatial_map_point_positions,
                    point_colors=self._spatial_map_point_colors,
                )
            )

    def run(self) -> None:
        logger.info(f"{self.__class__.__name__} process started.")
        self._setup()
        assert isinstance(self._camera, Zed)  # For mypy
        logger.info(f'{self.__class__.__name__} starting to publish to "{self._shared_memory_namespace}".')

        # Retrieve static camera data
        pose_right_in_left = self._camera.pose_of_right_view_in_left_view
        intrinsics_left = self._camera.intrinsics_matrix(view=StereoRGBDCamera.LEFT_RGB)
        intrinsics_right = self._camera.intrinsics_matrix(view=StereoRGBDCamera.RIGHT_RGB)

        frame_count = 0

        while not self.shutdown_event.is_set():
            # Write resolution info using the DDS resolution writer
            self._resolution_writer(ResolutionIdl(width=self._camera.resolution[0], height=self._camera.resolution[1]))

            # Get current timestamp
            timestamp = time.time()

            # Retrieve dynamic camera data
            dyn_camera_data = self._retrieve_dynamic_camera_data(frame_count)

            # Write camera data to shared memory
            self._write_camera_data_to_sm(
                timestamp,
                intrinsics_left,
                intrinsics_right,
                pose_right_in_left,
                dyn_camera_data,
            )

            frame_count += 1


class MultiprocessZedReceiver(MultiprocessStereoRGBDReceiver, StereoRGBDCamera):
    def __init__(
        self,
        shared_memory_namespace: str,
        enable_pointcloud: bool = True,
        enable_positional_tracking: bool = False,
        enable_spatial_mapping: bool = False,
        max_spatial_map_chunks: int = 10000,
        max_spatial_map_points: int = 1000000,
    ) -> None:
        self.enable_positional_tracking = enable_positional_tracking
        self.enable_spatial_mapping = enable_spatial_mapping

        # Mapping limits needed to initialize the shared memory template reader
        self.max_spatial_map_chunks = max_spatial_map_chunks
        self.max_spatial_map_points = max_spatial_map_points

        super().__init__(shared_memory_namespace, enable_pointcloud)

    def _setup_sm_reader(self, resolution: CameraResolutionType) -> None:
        # 1. Main Frame Reader
        # We switch to ZedFrameBuffer to include camera_pose
        self._reader = SMReader(
            domain_participant=self._dp,
            topic_name=self._shared_memory_namespace,
            idl_dataclass=ZedFrameBuffer.template(
                self.resolution[0],
                self.resolution[1],
            ),
        )

        # 2. Point Cloud Reader
        if self.enable_pointcloud:
            self._reader_pcd = SMReader(
                domain_participant=self._dp,
                topic_name=f"{self._shared_memory_namespace}_pcd",
                idl_dataclass=PointCloudBuffer.template(self.resolution[0], self.resolution[1]),
            )

        # 3. Spatial Map Reader
        if self.enable_spatial_mapping:
            self._reader_spatial_map = SMReader(
                domain_participant=self._dp,
                topic_name=f"{self._shared_memory_namespace}_spatial_map",
                idl_dataclass=SpatialMapBuffer.template(self.max_spatial_map_chunks, self.max_spatial_map_points),
            )

        # Initialize first frames to prevent NoneType errors before first read
        self._last_frame = ZedFrameBuffer.template(
            self.resolution[0],
            self.resolution[1],
        )

        if self.enable_pointcloud:
            self._last_pcd_frame = PointCloudBuffer.template(self.resolution[0], self.resolution[1])

        if self.enable_spatial_mapping:
            self._last_spatial_map_frame = SpatialMapBuffer.template(
                self.max_spatial_map_chunks, self.max_spatial_map_points
            )

    def _retrieve_camera_pose(self) -> np.ndarray:
        """
        Returns the 4x4 global pose matrix of the camera if tracking is enabled.
        """
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

    def _grab_images(self) -> None:
        # Retrieve standard frame (RGB, Depth, Pose)
        super()._grab_images()

        # Retrieve Point Cloud if enabled
        if self.enable_pointcloud:
            self._last_pcd_frame = self._reader_pcd()

        # Retrieve Spatial Map if enabled
        if self.enable_spatial_mapping:
            self._last_spatial_map_frame = self._reader_spatial_map()


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
        depth_map = receiver.get_depth_map()
        depth_image = receiver.get_depth_image()
        # confidence_map = receiver._retrieve_confidence_map()
        point_cloud = receiver.get_colored_point_cloud()

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
