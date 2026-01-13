import numpy as np
from airo_camera_toolkit.cameras.multiprocess.mixin import CameraMixin, DepthMixin, PointCloudMixin, RGBMixin
from airo_camera_toolkit.cameras.multiprocess.publisher import CameraPublisher
from airo_camera_toolkit.cameras.multiprocess.receiver import SharedMemoryReceiver
from airo_camera_toolkit.cameras.multiprocess.schema import CameraSchema, DepthSchema, PointCloudSchema, RGBSchema
from airo_typing import CameraResolutionType


class MultiprocessRGBDCameraPublisher(CameraPublisher):
    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        shared_memory_namespace: str = "camera",
        enable_pointcloud: bool = True,
    ) -> None:
        schemas = [CameraSchema(), RGBSchema(), DepthSchema()]
        if enable_pointcloud:
            schemas.append(PointCloudSchema())
        super().__init__(camera_cls, camera_kwargs, schemas, shared_memory_namespace)


class MultiprocessRGBDReceiver(SharedMemoryReceiver, CameraMixin, RGBMixin, DepthMixin, PointCloudMixin):
    def __init__(
        self,
        namespace: str,
        resolution: CameraResolutionType,
        enable_pointcloud: bool = True,
    ):
        schemas = [CameraSchema(), RGBSchema(), DepthSchema()]
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
    import rerun as rr
    from airo_camera_toolkit.cameras.multiprocess.publisher import CameraPublisher
    from airo_camera_toolkit.cameras.zed.zed import Zed
    from airo_camera_toolkit.utils.image_converter import ImageConverter
    from loguru import logger

    multiprocessing.set_start_method("spawn", force=True)

    publisher = MultiprocessRGBDCameraPublisher(
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
    receiver = MultiprocessRGBDReceiver(NAMESPACE, resolution=Zed.InitParams.RESOLUTION_720)

    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("DEPTH", cv2.WINDOW_NORMAL)

    time_current = None
    time_previous = None

    log_point_cloud = False
    if log_point_cloud:
        rr.init("multiprocess_rgbd_camera", spawn=True)

    while True:
        time_previous = time_current
        time_current = time.time()

        image_rgb = receiver.get_rgb_image_as_int()
        depth_image = receiver._retrieve_depth_image()
        point_cloud = receiver._retrieve_colored_point_cloud()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        depth_image = ImageConverter.from_numpy_int_format(depth_image).image_in_opencv_format
        cv2.imshow("RGB", image)
        cv2.imshow("DEPTH", depth_image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

        if log_point_cloud:
            point_cloud.points[np.isnan(point_cloud.points)] = 0
            if point_cloud.colors is not None:
                point_cloud.colors[np.isnan(point_cloud.colors)] = 0
            rr.log(
                "point_cloud",
                rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors),
            )

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
