from airo_camera_toolkit.cameras.multiprocess.mixin import CameraMixin, DepthMixin, PointCloudMixin, StereoRGBMixin
from airo_camera_toolkit.cameras.multiprocess.publisher import CameraPublisher
from airo_camera_toolkit.cameras.multiprocess.receiver import SharedMemoryReceiver
from airo_camera_toolkit.cameras.multiprocess.schema import (
    CameraSchema,
    DepthSchema,
    PointCloudSchema,
    StereoRGBSchema,
)
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_typing import CameraResolutionType


class MultiprocessStereoRGBDCameraPublisher(CameraPublisher):
    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        shared_memory_namespace: str = "camera",
        enable_pointcloud: bool = True,
    ) -> None:
        schemas = [CameraSchema(), StereoRGBSchema(), DepthSchema()]
        if enable_pointcloud:
            schemas.append(PointCloudSchema())
        super().__init__(camera_cls, camera_kwargs, schemas, shared_memory_namespace)


# Inheritance order matters! The first class encountered determines which method is used, is it if defined in >1 Mixin.
# StereoRGBMixin MUST be before CameraMixin and SharedMemoryReceiver for intrinsics_matrix()!
class MultiprocessStereoRGBDReceiver(StereoRGBMixin, CameraMixin, DepthMixin, PointCloudMixin, SharedMemoryReceiver):
    """Multiprocess implementation of StereoRGBDCamera. To be used with MultiprocessStereoRGBDCameraPublisher."""

    def __init__(
        self,
        namespace: str,
        resolution: CameraResolutionType,
        enable_pointcloud: bool = True,
    ):
        schemas = [CameraSchema(), StereoRGBSchema(), DepthSchema()]
        if enable_pointcloud:
            schemas.append(PointCloudSchema())
        SharedMemoryReceiver.__init__(self, resolution, schemas, namespace)


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBDPublisher and MultiprocessRGBDReceiver.
    You can also use the MultiprocessRGBDReceiver in a different process (e.g. in a different python script)
    """
    CAMERA_FPS = 30
    NAMESPACE = "camera"

    import multiprocessing
    import time

    import cv2
    from airo_camera_toolkit.cameras.multiprocess.publisher import CameraPublisher
    from airo_camera_toolkit.cameras.zed.zed import Zed
    from airo_camera_toolkit.utils.image_converter import ImageConverter
    from loguru import logger

    multiprocessing.set_start_method("spawn", force=True)

    publisher = MultiprocessStereoRGBDCameraPublisher(
        Zed,
        camera_kwargs={
            "resolution": Zed.InitParams.RESOLUTION_720,
            "fps": CAMERA_FPS,
            "depth_mode": Zed.InitParams.NEURAL_DEPTH_MODE,
        },
        shared_memory_namespace=NAMESPACE,
    )
    publisher.start()

    time.sleep(5)

    # The receiver behaves just like a regular RGBCamera
    receiver = MultiprocessStereoRGBDReceiver(NAMESPACE, resolution=Zed.InitParams.RESOLUTION_720)

    cv2.namedWindow("RGB_LEFT", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RGB_RIGHT", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DEPTH", cv2.WINDOW_NORMAL)

    time_current = None
    time_previous = None

    while True:
        time_previous = time_current
        time_current = time.time()

        image_rgb = receiver.get_rgb_image_as_int()
        image_rgb_right = receiver.get_rgb_image_as_int(StereoRGBDCamera.RIGHT_RGB)
        depth_image = receiver.get_depth_image()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        image_right = ImageConverter.from_numpy_int_format(image_rgb_right).image_in_opencv_format
        depth_image = ImageConverter.from_numpy_int_format(depth_image).image_in_opencv_format
        cv2.imshow("RGB_LEFT", image)
        cv2.imshow("RGB_RIGHT", image_right)
        cv2.imshow("DEPTH", depth_image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

        if time_previous is not None:
            fps = 1 / (time_current - time_previous)

            fps_str = f"{fps:.2f}".rjust(6, " ")
            camera_fps_str = f"{CAMERA_FPS:.2f}".rjust(6, " ")
            if fps < 0.9 * CAMERA_FPS:
                logger.warning(f"FPS: {fps_str} / {camera_fps_str} (too slow)")
            else:
                logger.debug(f"FPS: {fps_str} / {camera_fps_str}")

    publisher.stop()
    publisher.join()
    cv2.destroyAllWindows()
