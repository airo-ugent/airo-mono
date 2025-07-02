import multiprocessing
import time
from multiprocessing.context import SpawnProcess
from typing import Optional

import loguru
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import MultiprocessRGBReceiver
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgbd_camera import MultiprocessRGBDReceiver
from airo_camera_toolkit.image_transforms.image_transform import ImageTransform
from airo_camera_toolkit.utils.image_converter import ImageConverter

logger = loguru.logger


class MultiprocessRGBRerunLogger(SpawnProcess):
    def __init__(
        self,
        shared_memory_namespace: str,
        rerun_application_id: str = "rerun",
        image_transform: Optional[ImageTransform] = None,
        entity_path: Optional[str] = None,
    ):
        super().__init__(daemon=True)
        self._shared_memory_namespace = shared_memory_namespace
        self.shutdown_event = multiprocessing.Event()
        self._rerun_application_id = rerun_application_id
        self._image_transform = image_transform

        # If the entity path is not given, we use the `_shared_memory_namespace` value as entity path (maintaining backwards compatibility).
        self._entity_path = entity_path if entity_path is not None else shared_memory_namespace

    def _log_rgb_image(self) -> None:
        import rerun as rr

        image = self._receiver.get_rgb_image()
        # This randomly fails, just don't log an image if it does
        try:
            image_bgr = ImageConverter.from_numpy_format(image).image_in_opencv_format
        except AssertionError as e:
            print(e)
            return
        image_rgb = image_bgr[:, :, ::-1]
        if self._image_transform is not None:
            image_rgb = self._image_transform.transform_image(image_rgb)

        rr.log(self._entity_path, rr.Image(image_rgb).compress(jpeg_quality=90))

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        import rerun as rr

        rr.init(self._rerun_application_id)
        rr.connect_grpc()

        self._receiver = MultiprocessRGBReceiver(self._shared_memory_namespace)

        while not self.shutdown_event.is_set():
            self._log_rgb_image()

    def stop(self) -> None:
        self.shutdown_event.set()


class MultiprocessRGBDRerunLogger(MultiprocessRGBRerunLogger):
    def __init__(
        self,
        shared_memory_namespace: str,
        rerun_application_id: str = "rerun",
        image_transform: Optional[ImageTransform] = None,
        entity_path: Optional[str] = None,
        entity_path_depth: Optional[str] = None,
    ):
        super().__init__(
            shared_memory_namespace,
            rerun_application_id,
            image_transform,
            entity_path,
        )

        self._entity_path_depth = entity_path_depth if entity_path_depth is not None else f"{self._entity_path}_depth"

    def _log_depth_image(self) -> None:
        import rerun as rr

        assert isinstance(self._receiver, MultiprocessRGBDReceiver)

        depth_image = self._receiver.get_depth_image()
        if self._image_transform is not None:
            depth_image = self._image_transform.transform_image(depth_image)
        rr.log(self._entity_path_depth, rr.Image(depth_image).compress(jpeg_quality=90))

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        import rerun

        rerun.init(self._rerun_application_id)
        rerun.connect_grpc()

        self._receiver = MultiprocessRGBDReceiver(self._shared_memory_namespace)

        while not self.shutdown_event.is_set():
            self._log_rgb_image()
            self._log_depth_image()

    def stop(self) -> None:
        self.shutdown_event.set()


if __name__ == "__main__":
    rerun_logger = MultiprocessRGBDRerunLogger("camera")
    rerun_logger.start()
    time.sleep(10)
    rerun_logger.stop()
