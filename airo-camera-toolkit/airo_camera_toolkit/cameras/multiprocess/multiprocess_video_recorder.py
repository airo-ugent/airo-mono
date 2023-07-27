import datetime
import multiprocessing
import os
import time
from collections import deque
from multiprocessing import Process
from typing import Optional

import loguru
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import MultiprocessRGBReceiver
from airo_camera_toolkit.image_transforms.image_transform import ImageTransform
from airo_camera_toolkit.utils import ImageConverter

logger = loguru.logger


class FPSMonitor:
    def __init__(self, target_fps: float, queue_size: int = 100, tolerance: float = 0.05, name: str = "FPSMonitor"):
        self.target_fps = target_fps
        self.tolerance = tolerance
        self.name = name
        self._durations: deque[float] = deque(maxlen=queue_size)
        self._last_time: Optional[float] = None

    def get_fps(self) -> float:
        if len(self._durations) == 0:
            return 0.0

        average_duration = sum(self._durations) / len(self._durations)
        return 1 / average_duration

    def check_fps(self) -> None:
        fps = self.get_fps()
        fps_relative_error = abs(fps - self.target_fps) / self.target_fps

        if fps_relative_error > self.tolerance:
            logger.warning(
                f"{self.name} FPS is {fps:.2f} but should be {self.target_fps:.2f} (Error: {fps_relative_error:.3f}))"
            )

    def tick(self) -> None:
        current_time = time.time()
        if self._last_time is None:
            self._last_time = current_time
            return

        duration = current_time - self._last_time
        self._durations.append(duration)
        self._last_time = current_time

        if len(self._durations) == self._durations.maxlen:
            self.check_fps()


class MultiprocessVideoRecorder(Process):
    def __init__(
        self,
        shared_memory_namespace: str,
        video_path: Optional[str] = None,
        image_transform: Optional[ImageTransform] = None,
    ):
        super().__init__(daemon=True)
        self._shared_memory_namespace = shared_memory_namespace
        self._image_transform = image_transform
        self.shutdown_event = multiprocessing.Event()

        if video_path is None:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%H:%M:%S:%f")[:-3]
            video_name = f"{timestamp}.mp4"
            video_path = os.path.join(output_dir, video_name)
            video_path = os.path.abspath(video_path)

        self._video_path = video_path

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        import ffmpegcv  # type: ignore

        receiver = MultiprocessRGBReceiver(self._shared_memory_namespace)
        fps = receiver.fps_shm_array[0]
        video_writer = ffmpegcv.VideoWriter(self._video_path, "hevc", fps)

        fps_monitor = FPSMonitor(fps, name=f"{self._shared_memory_namespace} video recorder")

        while not self.shutdown_event.is_set():
            image_float = receiver.get_rgb_image()
            image = ImageConverter.from_numpy_format(image_float).image_in_opencv_format
            if self._image_transform is not None:
                image = self._image_transform.transform_image(image)

            video_writer.write(image)
            fps_monitor.tick()

        video_writer.release()

    def stop(self) -> None:
        self.shutdown_event.set()


if __name__ == "__main__":
    """Records 10 seconds of video. Assumes there's being published to the "camera" namespace."""
    recorder = MultiprocessVideoRecorder("zed_top")
    recorder.start()
    time.sleep(10)
    recorder.stop()
