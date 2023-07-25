import multiprocessing
import time
from multiprocessing import Process

import loguru
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import MultiprocessRGBReceiver
from airo_camera_toolkit.image_transforms.image_transform import ImageTransform
from airo_camera_toolkit.utils import ImageConverter

logger = loguru.logger


class MultiprocessRGBRerunLogger(Process):
    def __init__(
        self,
        shared_memory_namespace: str,
        rerun_application_id: str = "rerun",
        image_transform: ImageTransform = None,
    ):
        super().__init__(daemon=True)
        self._shared_memory_namespace = shared_memory_namespace
        self.shutdown_event = multiprocessing.Event()
        self._rerun_application_id = rerun_application_id

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        import rerun

        rerun.init(self._rerun_application_id)
        rerun.connect()

        receiver = MultiprocessRGBReceiver(self._shared_memory_namespace)

        while not self.shutdown_event.is_set():
            image = receiver.get_rgb_image()
            image_bgr = ImageConverter.from_numpy_format(image).image_in_opencv_format  #
            image_rgb = image_bgr[:, :, ::-1]
            rerun.log_image(self._shared_memory_namespace, image_rgb, jpeg_quality=90)

    def stop(self) -> None:
        self.shutdown_event.set()


if __name__ == "__main__":
    logger = MultiprocessRGBRerunLogger("camera")
    logger.start()
    time.sleep(10)
    logger.stop()
