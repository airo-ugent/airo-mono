"""Publisher and receiver classes for multiprocess stereo RGBD camera sharing."""

import multiprocessing
import time
from typing import Any

import loguru
import numpy as np
from airo_camera_toolkit.cameras.multiprocess.base_publisher import BaseCameraPublisher
from airo_camera_toolkit.cameras.multiprocess.base_receiver import BaseCameraReceiver
from airo_camera_toolkit.cameras.multiprocess.frame_data import FrameLayout, StereoRGBDFrameBuffer
from airo_camera_toolkit.cameras.multiprocess.pointcloud_stream import (
    PointCloudPublisherMixin,
    PointCloudReceiverMixin,
)
from airo_camera_toolkit.cameras.multiprocess.zenoh_reader import DEFAULT_FIRST_MESSAGE_TIMEOUT
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import (
    CameraIntrinsicsMatrixType,
    HomogeneousMatrixType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
)

logger = loguru.logger


class MultiprocessStereoRGBDPublisher(PointCloudPublisherMixin, BaseCameraPublisher):
    """Publishes stereo RGBD camera data over Zenoh, and the point cloud on its own key expression if enabled.

    Subclasses can publish extra per-frame fields by setting ``_frame_type`` to a frame buffer
    that adds those fields, and returning them from ``_capture_extra_fields()``.
    """

    _frame_type: type[StereoRGBDFrameBuffer] = StereoRGBDFrameBuffer

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
        """Return stereo RGBD frame buffer layout."""
        return self._frame_type.layout(width, height)

    def _setup(self) -> None:
        """Set up camera, point cloud publishing and static camera parameters."""
        super()._setup()

        # Cache static camera parameters
        assert isinstance(self._camera, StereoRGBDCamera)
        self._pose_right_in_left = self._camera.pose_of_right_view_in_left_view
        self._intrinsics_left = self._camera.intrinsics_matrix(view=StereoRGBDCamera.LEFT_RGB)
        self._intrinsics_right = self._camera.intrinsics_matrix(view=StereoRGBDCamera.RIGHT_RGB)

    def _capture_extra_fields(self) -> dict[str, np.ndarray]:
        """Capture the fields a subclass's ``_frame_type`` adds to :class:`StereoRGBDFrameBuffer`."""
        return {}

    def _capture_frame(self, frame_id: int, frame_timestamp: float) -> Any:
        """Capture the stereo RGB-D data."""
        return self._frame_type(
            **self._header(frame_id, frame_timestamp),
            rgb=self._camera.retrieve_rgb_image_as_int(),
            rgb_right=self._camera.retrieve_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB),
            intrinsics=self._intrinsics_left,
            intrinsics_right=self._intrinsics_right,
            pose_right_in_left=self._pose_right_in_left,
            depth=self._camera.retrieve_depth_map(),
            depth_image=self._camera.retrieve_depth_image(),
            **self._capture_extra_fields(),
        )


class MultiprocessStereoRGBDReceiver(PointCloudReceiverMixin, BaseCameraReceiver, StereoRGBDCamera):
    """Receives stereo RGBD camera data over Zenoh, and the matching point clouds if enabled."""

    # Must match the publisher's frame type; see MultiprocessStereoRGBDPublisher.
    _frame_type: type[StereoRGBDFrameBuffer] = StereoRGBDFrameBuffer

    def __init__(
        self,
        shared_memory_namespace: str,
        enable_pointcloud: bool = True,
        timeout: float | None = DEFAULT_FIRST_MESSAGE_TIMEOUT,
    ) -> None:
        self.enable_pointcloud = enable_pointcloud
        super().__init__(shared_memory_namespace, timeout=timeout)

    def _get_frame_layout(self, width: int, height: int) -> FrameLayout[Any]:
        """Return stereo RGBD frame buffer layout."""
        return self._frame_type.layout(width, height)

    def retrieve_rgb_image(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyFloatImageType:
        return ImageConverter.from_numpy_int_format(self.retrieve_rgb_image_as_int(view=view)).image_in_numpy_format

    def retrieve_rgb_image_as_int(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyIntImageType:
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

    def retrieve_depth_map(self) -> NumpyDepthMapType:
        """Retrieve depth map from frame buffer."""
        return self._last_frame.depth

    def retrieve_depth_image(self) -> NumpyIntImageType:
        """Retrieve depth image from frame buffer."""
        return self._last_frame.depth_image


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

        import rerun as rr

        rr.init(f"{MultiprocessStereoRGBDReceiver.__name__} - Point cloud", spawn=True)

        log_point_cloud = False

        time_current = None
        time_previous = None

        while True:
            time_previous = time_current
            time_current = time.time()

            receiver.grab_images()
            image = receiver.retrieve_rgb_image_as_int()
            image_right = receiver.retrieve_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB)
            depth_map = receiver.retrieve_depth_map()
            depth_image = receiver.retrieve_depth_image()
            # confidence_map = receiver.retrieve_confidence_map()
            point_cloud = receiver.retrieve_colored_point_cloud()

            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_right_bgr = cv2.cvtColor(image_right, cv2.COLOR_RGB2BGR)

            cv2.imshow("RGB Image", image_bgr)
            cv2.imshow("RGB Image Right", image_right_bgr)
            cv2.imshow("Depth Map", depth_map)
            cv2.imshow("Depth Image", depth_image)
            # cv2.imshow("Confidence Map", confidence_map)

            if log_point_cloud:
                # Received arrays are read-only views, so replace NaNs in a copy. Colors are uint8 and cannot be NaN.
                positions = np.nan_to_num(point_cloud.points)
                rr.log(
                    "point_cloud",
                    rr.Points3D(positions=positions, colors=point_cloud.colors),
                )

            key = cv2.waitKey(10)
            if key == ord("q"):
                break
            elif key == ord("l"):
                log_point_cloud = not log_point_cloud
    finally:
        receiver.stop()
        publisher.stop()
        publisher.join()
