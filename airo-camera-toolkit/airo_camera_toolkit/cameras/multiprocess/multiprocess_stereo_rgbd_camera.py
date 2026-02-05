"""Publisher and receiver classes for multiprocess stereo RGBD camera sharing."""

import multiprocessing
import time
from typing import Any

import loguru
import numpy as np
from airo_camera_toolkit.cameras.multiprocess.base_publisher import BaseCameraPublisher
from airo_camera_toolkit.cameras.multiprocess.base_receiver import BaseCameraReceiver
from airo_camera_toolkit.cameras.multiprocess.frame_data import PointCloudBuffer, StereoRGBDFrameBuffer
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_ipc.cyclone_shm.patterns.sm_reader import SMReader
from airo_ipc.cyclone_shm.patterns.sm_writer import SMWriter
from airo_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    HomogeneousMatrixType,
    NumpyFloatImageType,
    NumpyIntImageType,
    PointCloud,
)

logger = loguru.logger


class MultiprocessStereoRGBDPublisher(BaseCameraPublisher):
    """Publishes stereo RGBD camera data to shared memory blocks."""

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
        """Return stereo RGBD frame buffer template."""
        return StereoRGBDFrameBuffer.template(width, height)

    def _setup(self) -> None:
        """Set up camera and prepare point cloud buffers and static camera parameters."""
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

    def _setup_additional_writers(self) -> None:
        """Set up point cloud writer if enabled."""
        if self.enable_pointcloud:
            self._pcd_writer = SMWriter(
                domain_participant=self._dp,
                topic_name=f"{self._shared_memory_namespace}_pcd",
                idl_dataclass=PointCloudBuffer.template(self._camera.resolution[0], self._camera.resolution[1]),
            )

    def _capture_frame_data(self, frame_id: int, frame_timestamp: float) -> None:
        """Capture stereo RGB-D data and optionally point cloud."""
        self._current_frame_id = frame_id
        self._current_frame_timestamp = frame_timestamp

        # Capture left and right images
        self._current_rgb_left = self._camera.get_rgb_image_as_int()
        self._current_rgb_right = self._camera._retrieve_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB)

        # Capture depth data
        self._current_depth_map = self._camera.get_depth_map()
        self._current_depth_image = self._camera.get_depth_image()

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

    def _write_frame_data(self) -> None:
        """Write stereo RGBD frame data and optionally point cloud to shared memory."""
        # Write main stereo RGBD frame
        frame_data = StereoRGBDFrameBuffer(
            frame_id=np.array([self._current_frame_id], dtype=np.uint64),
            frame_timestamp=np.array([self._current_frame_timestamp], dtype=np.float64),
            rgb=self._current_rgb_left,
            rgb_right=self._current_rgb_right,
            intrinsics=self._intrinsics_left,
            intrinsics_right=self._intrinsics_right,
            pose_right_in_left=self._pose_right_in_left,
            depth=self._current_depth_map,
            depth_image=self._current_depth_image,
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


class MultiprocessStereoRGBDReceiver(BaseCameraReceiver, StereoRGBDCamera):
    """Receives stereo RGBD camera data from shared memory."""

    def __init__(self, shared_memory_namespace: str, enable_pointcloud: bool = True) -> None:
        self.enable_pointcloud = enable_pointcloud
        super().__init__(shared_memory_namespace)

    def _get_frame_buffer_template(self, width: int, height: int) -> Any:
        """Return stereo RGBD frame buffer template."""
        return StereoRGBDFrameBuffer.template(width, height)

    def _setup_additional_readers(self, resolution: CameraResolutionType) -> None:
        """Set up point cloud reader if enabled."""
        if self.enable_pointcloud:
            self._reader_pcd = SMReader(
                domain_participant=self._dp,
                topic_name=f"{self._shared_memory_namespace}_pcd",
                idl_dataclass=PointCloudBuffer.template(resolution[0], resolution[1]),
            )
            # Initialize an empty point cloud frame
            self._last_pcd_frame = PointCloudBuffer.template(resolution[0], resolution[1])

    def _grab_additional_data(self) -> None:
        """Read point cloud if enabled."""
        if self.enable_pointcloud:
            self._last_pcd_frame = self._reader_pcd()

    def _retrieve_rgb_image(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyFloatImageType:
        return ImageConverter.from_numpy_int_format(self._retrieve_rgb_image_as_int(view=view)).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyIntImageType:
        if view == StereoRGBDCamera.LEFT_RGB:
            return self._last_frame.rgb
        else:
            return self._last_frame.rgb_right

    @property
    def pose_of_right_view_in_left_view(self) -> HomogeneousMatrixType:
        return self._last_frame.pose_right_in_left

    def intrinsics_matrix(self, view: str = StereoRGBDCamera.LEFT_RGB) -> CameraIntrinsicsMatrixType:
        if view == StereoRGBDCamera.LEFT_RGB:
            return self._last_frame.intrinsics
        else:
            return self._last_frame.intrinsics_right

    def _retrieve_depth_map(self) -> NumpyIntImageType:
        """Retrieve depth map from frame buffer."""
        return self._last_frame.depth

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        """Retrieve depth image from frame buffer."""
        return self._last_frame.depth_image

    def _retrieve_colored_point_cloud(self) -> PointCloud:
        if not self.enable_pointcloud:
            raise RuntimeError("Cannot retrieve point cloud when point cloud is not enabled.")
        num_points = self._last_pcd_frame.point_cloud_valid.item()
        positions = self._last_pcd_frame.point_cloud_positions[:num_points]
        colors = self._last_pcd_frame.point_cloud_colors[:num_points]
        point_cloud = PointCloud(positions, colors)
        return point_cloud


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBDPublisher and MultiprocessRGBDReceiver.
    You can also use the MultiprocessRGBDReceiver in a different process (e.g. in a different python script)
    """
    import cv2
    from airo_camera_toolkit.cameras.zed.zed import Zed

    multiprocessing.set_start_method("spawn", force=True)

    resolution = Zed.InitParams.RESOLUTION_720
    camera_fps = 15

    publisher = MultiprocessStereoRGBDPublisher(
        Zed,
        camera_kwargs={
            "resolution": resolution,
            "fps": camera_fps,
            "depth_mode": Zed.InitParams.NEURAL_DEPTH_MODE,
        },
    )

    publisher.start()
    receiver = MultiprocessStereoRGBDReceiver("camera")

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

    import rerun as rr

    rr.init(f"{MultiprocessStereoRGBDReceiver.__name__} - Point cloud", spawn=True)

    log_point_cloud = False

    time_current = None
    time_previous = None

    while True:
        time_previous = time_current
        time_current = time.time()

        image = receiver.get_rgb_image_as_int()
        image_right = receiver._retrieve_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB)
        depth_map = receiver.get_depth_map()
        depth_image = receiver.get_depth_image()
        # confidence_map = receiver._retrieve_confidence_map()
        point_cloud = receiver.get_colored_point_cloud()

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_right_bgr = cv2.cvtColor(image_right, cv2.COLOR_RGB2BGR)

        cv2.imshow("RGB Image", image_bgr)
        cv2.imshow("RGB Image Right", image_right_bgr)
        cv2.imshow("Depth Map", depth_map)
        cv2.imshow("Depth Image", depth_image)
        # cv2.imshow("Confidence Map", confidence_map)

        if log_point_cloud:
            point_cloud.points[np.isnan(point_cloud.points)] = 0
            if point_cloud.colors is not None:
                point_cloud.colors[np.isnan(point_cloud.colors)] = 0
            rr.log(
                "point_cloud",
                rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors),
            )

        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord("l"):
            log_point_cloud = not log_point_cloud

    publisher.stop()
    publisher.join()
