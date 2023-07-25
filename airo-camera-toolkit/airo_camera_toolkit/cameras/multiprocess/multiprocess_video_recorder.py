import multiprocessing
import os
import time
from multiprocessing import Process

import loguru
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import MultiprocessRGBReceiver
from airo_camera_toolkit.image_transforms.image_transform import ImageTransform
from airo_camera_toolkit.utils import ImageConverter

logger = loguru.logger
# import ffmpegcv
import datetime


class MultiprocessVideoRecorder(Process):
    def __init__(
        self,
        shared_memory_namespace: str,
        video_path: str = None,
        image_transform: ImageTransform = None,
        fps: int = 30,
    ):
        super().__init__(daemon=True)
        self._shared_memory_namespace = shared_memory_namespace
        self.shutdown_event = multiprocessing.Event()
        self._fps = fps

        if video_path is None:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%H:%M:%S:%f")[:-3]
            video_name = f"{timestamp}.mp4"
            video_path = os.path.join(output_dir, video_name)

        self._video_path = video_path

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        import ffmpegcv

        receiver = MultiprocessRGBReceiver(self._shared_memory_namespace)

        print(self._video_path)
        print(self._fps)

        video_writer = ffmpegcv.VideoWriter(self._video_path, "hevc", self._fps)

        while not self.shutdown_event.is_set():
            image_float = receiver.get_rgb_image()
            image = ImageConverter.from_numpy_format(image_float).image_in_opencv_format
            video_writer.write(image)

        video_writer.release()

    def stop(self) -> None:
        self.shutdown_event.set()


if __name__ == "__main__":
    """Records 10 seconds of video. Assumes there's being published to the "camera" namespace."""
    recorder = MultiprocessVideoRecorder("camera")
    recorder.start()
    time.sleep(10)
    recorder.stop()
