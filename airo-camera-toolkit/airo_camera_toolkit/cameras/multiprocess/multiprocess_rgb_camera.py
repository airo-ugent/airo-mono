from typing import List

from airo_camera_toolkit.cameras.multiprocess.mixin import CameraMixin, RGBMixin
from airo_camera_toolkit.cameras.multiprocess.publisher import CameraPublisher
from airo_camera_toolkit.cameras.multiprocess.receiver import SharedMemoryReceiver
from airo_camera_toolkit.cameras.multiprocess.schema import CameraSchema, RGBSchema, Schema
from airo_typing import CameraResolutionType


class MultiprocessRGBCameraPublisher(CameraPublisher):
    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        shared_memory_namespace: str = "camera",
    ) -> None:
        schemas: List[Schema] = [CameraSchema(), RGBSchema()]
        super().__init__(camera_cls, camera_kwargs, schemas, shared_memory_namespace)


# Inheritance order matters! The first class encountered determines which method is used, is it if defined in >1 Mixin.
# CameraMixin must be before SharedMemoryReceiver for intrinsics_matrix()!
class MultiprocessRGBReceiver(CameraMixin, RGBMixin, SharedMemoryReceiver):
    """Multiprocess implementation of RGBCamera. To be used with MultiprocessRGBCameraPublisher."""

    def __init__(self, namespace: str, resolution: CameraResolutionType):
        SharedMemoryReceiver.__init__(
            self,
            resolution,
            schemas=[CameraSchema(), RGBSchema()],
            shared_memory_namespace=namespace,
        )


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBPublisher and MultiprocessRGBReceiver.
    You can also use the MultiprocessRGBReceiver in a different process (e.g. in a different python script)
    """
    CAMERA_FPS = 30
    NAMESPACE = "camera"

    import multiprocessing
    import time

    import cv2
    import numpy as np
    from airo_camera_toolkit.cameras.multiprocess.publisher import CameraPublisher
    from airo_camera_toolkit.cameras.zed.zed import Zed
    from airo_camera_toolkit.utils.image_converter import ImageConverter
    from loguru import logger

    multiprocessing.set_start_method("spawn", force=True)

    publisher = MultiprocessRGBCameraPublisher(
        Zed,
        camera_kwargs={
            "resolution": Zed.InitParams.RESOLUTION_720,
            "fps": CAMERA_FPS,
        },
        shared_memory_namespace=NAMESPACE,
    )
    publisher.start()

    time.sleep(5)

    # The receiver behaves just like a regular RGBCamera
    receiver = MultiprocessRGBReceiver(NAMESPACE, resolution=Zed.InitParams.RESOLUTION_720)

    cv2.namedWindow(NAMESPACE, cv2.WINDOW_NORMAL)

    with np.printoptions(precision=3, suppress=True):
        print("Intrinsics matrix:")
        print(receiver.intrinsics_matrix())

    time_current = None
    time_previous = None

    while True:
        time_previous = time_current
        time_current = time.time()

        image_rgb = receiver.get_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format

        cv2.imshow(NAMESPACE, image)
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
