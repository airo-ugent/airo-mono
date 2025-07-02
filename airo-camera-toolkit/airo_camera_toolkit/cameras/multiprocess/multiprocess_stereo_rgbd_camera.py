"""code for sharing the data of a camera that implements the RGBDCamera interface between processes using shared memory"""

import time
from dataclasses import dataclass
from typing import Any

import loguru
import numpy as np
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import ResolutionIdl
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgbd_camera import (
    MultiprocessRGBDPublisher,
    MultiprocessRGBDReceiver,
)
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_ipc.cyclone_shm.idl_shared_memory.base_idl import BaseIdl  # type: ignore
from airo_ipc.cyclone_shm.patterns.sm_reader import SMReader  # type: ignore
from airo_ipc.cyclone_shm.patterns.sm_writer import SMWriter  # type: ignore

logger = loguru.logger
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    HomogeneousMatrixType,
    NumpyFloatImageType,
    NumpyIntImageType,
)


@dataclass
class StereoRGBDFrameBuffer(BaseIdl):  # type: ignore
    """This struct, sent over shared memory, contains a timestamp, two RGB images, the camera intrinsics, a depth image, a depth map, and a point cloud.
    It also contains the pose of the right camera in the left camera frame."""

    # Timestamp of the frame
    timestamp: np.ndarray
    # Color image data (height x width x channels)
    rgb_left: np.ndarray
    rgb_right: np.ndarray
    # Intrinsic camera parameters (camera matrix)
    intrinsics_left: np.ndarray
    intrinsics_right: np.ndarray
    # Extrinsic camera parameters (camera matrix)
    pose_right_in_left: np.ndarray
    # Depth image data (height x width)
    depth_image: np.ndarray
    # Depth map (height x width)
    depth: np.ndarray
    # Point cloud (colors, positions x height * width x 3)
    point_cloud: np.ndarray

    @staticmethod
    def template(width: int, height: int) -> Any:
        return StereoRGBDFrameBuffer(
            timestamp=np.empty((1,), dtype=np.float64),
            rgb_left=np.empty((height, width, 3), dtype=np.uint8),
            rgb_right=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics_left=np.empty((3, 3), dtype=np.float64),
            intrinsics_right=np.empty((3, 3), dtype=np.float64),
            pose_right_in_left=np.empty((4, 4), dtype=np.float64),
            depth_image=np.empty((height, width, 3), dtype=np.uint8),
            depth=np.empty((height, width), dtype=np.float32),
            point_cloud=np.empty((2, height * width, 3), dtype=np.float32),
        )


class MultiprocessStereoRGBDPublisher(MultiprocessRGBDPublisher):
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

        # Some camera's, such as the Realsense D435i, return a sparse point cloud. This is not supported by the
        # current implementation of the RGBDFrameBuffer. Therefore, we make sure that we always retrieve a point
        # for every pixel in the RBG image.
        self._pcd_buf = np.zeros((2, self._camera.resolution[0] * self._camera.resolution[1], 3), dtype=np.float32)

    def _setup_sm_writer(self) -> None:
        # Create the shared memory writer
        self._writer = SMWriter(
            domain_participant=self._dp,
            topic_name=self._shared_memory_namespace,
            idl_dataclass=StereoRGBDFrameBuffer.template(self._camera.resolution[0], self._camera.resolution[1]),
            nr_of_buffers=4,
        )

    def stop(self) -> None:
        self.shutdown_event.set()

    def run(self) -> None:
        logger.info(f"{self.__class__.__name__} process started.")
        self._setup()
        assert isinstance(self._camera, StereoRGBDCamera)  # For mypy
        logger.info(f'{self.__class__.__name__} starting to publish to "{self._shared_memory_namespace}".')

        pose_right_in_left = self._camera.pose_of_right_view_in_left_view
        intrinsics_left = self._camera.intrinsics_matrix(view=StereoRGBDCamera.LEFT_RGB)
        intrinsics_right = self._camera.intrinsics_matrix(view=StereoRGBDCamera.RIGHT_RGB)

        while not self.shutdown_event.is_set():
            self._resolution_writer(ResolutionIdl(width=self._camera.resolution[0], height=self._camera.resolution[1]))

            image_left = self._camera.get_rgb_image_as_int()
            image_right = self._camera._retrieve_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB)
            depth_map = self._camera.get_depth_map()
            depth_image = self._camera.get_depth_image()
            point_cloud = self._camera.get_colored_point_cloud()

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
                StereoRGBDFrameBuffer(
                    timestamp=np.array([time.time()], dtype=np.float64),
                    rgb_left=image_left,
                    rgb_right=image_right,
                    intrinsics_left=intrinsics_left,
                    intrinsics_right=intrinsics_right,
                    pose_right_in_left=pose_right_in_left,
                    depth=depth_map,
                    depth_image=depth_image,
                    point_cloud=self._pcd_buf,
                )
            )


class MultiprocessStereoRGBDReceiver(MultiprocessRGBDReceiver, StereoRGBDCamera):
    def __init__(self, shared_memory_namespace: str) -> None:
        super().__init__(shared_memory_namespace)

    def _setup_sm_reader(self, resolution: CameraResolutionType) -> None:
        # Create the shared memory reader
        self._reader = SMReader(
            domain_participant=self._dp,
            topic_name=self._shared_memory_namespace,
            idl_dataclass=StereoRGBDFrameBuffer.template(self.resolution[0], self.resolution[1]),
            nr_of_buffers=4,
        )

        # Initialize a first frame.
        self._last_frame = StereoRGBDFrameBuffer.template(self.resolution[0], self.resolution[1])

    def _retrieve_rgb_image(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyFloatImageType:
        return ImageConverter.from_numpy_int_format(self._retrieve_rgb_image_as_int(view=view)).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyIntImageType:
        if view == StereoRGBDCamera.LEFT_RGB:
            return self._last_frame.rgb_left
        else:
            return self._last_frame.rgb_right

    @property
    def pose_of_right_view_in_left_view(self) -> HomogeneousMatrixType:
        return self._last_frame.pose_right_in_left

    def intrinsics_matrix(self, view: str = StereoRGBDCamera.LEFT_RGB) -> CameraIntrinsicsMatrixType:
        if view == StereoRGBDCamera.LEFT_RGB:
            return self._last_frame.intrinsics_left
        else:
            return self._last_frame.intrinsics_right


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBDPublisher and MultiprocessRGBDReceiver.
    You can also use the MultiprocessRGBDReceiver in a different process (e.g. in a different python script)
    """
    import cv2
    from airo_camera_toolkit.cameras.zed.zed import Zed

    resolution = Zed.RESOLUTION_720
    camera_fps = 15

    publisher = MultiprocessStereoRGBDPublisher(
        Zed,
        camera_kwargs={
            "resolution": resolution,
            "fps": camera_fps,
            "depth_mode": Zed.NEURAL_DEPTH_MODE,
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
        print("Intrinsics right:\n", receiver.intrinsics_matrix(view=StereoRGBDCamera.RIGHT_RGB))
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
            rr.log("point_cloud", rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors))

        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord("l"):
            log_point_cloud = not log_point_cloud

    publisher.stop()
    publisher.join()
