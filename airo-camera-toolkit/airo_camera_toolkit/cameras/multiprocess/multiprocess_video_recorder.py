import datetime
import multiprocessing
import os
import time
from multiprocessing import Process
from typing import Optional

from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import MultiprocessRGBReceiver
from airo_camera_toolkit.image_transforms.image_transform import ImageTransform
from airo_camera_toolkit.utils.image_converter import ImageConverter
from loguru import logger


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
        self.recording_started_event = multiprocessing.Event()
        self.recording_finished_event = multiprocessing.Event()
        self.shutdown_event = multiprocessing.Event()

        if video_path is None:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%H:%M:%S:%f")[:-3]
            video_name = f"{timestamp}.mp4"
            video_path = os.path.join(output_dir, video_name)
            video_path = os.path.abspath(video_path)

        self._video_path = video_path

    def start(self) -> None:
        super().start()
        # Block until the recording has started
        self.recording_started_event.wait()

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        import ffmpegcv  # type: ignore

        receiver = MultiprocessRGBReceiver(self._shared_memory_namespace)
        fps = receiver.fps_shm_array[0]
        height, width, _ = receiver.rgb_shm_array.shape
        video_writer = ffmpegcv.VideoWriter(self._video_path, "hevc", fps, (width, height))

        logger.info(f"Recording video to {self._video_path}")

        while not self.shutdown_event.is_set():
            image_rgb = receiver.get_rgb_image_as_int()
            image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
            # Known bug: vertical images still give horizontal videos
            if self._image_transform is not None:
                image = self._image_transform.transform_image(image)
            video_writer.write(image)
            self.recording_started_event.set()

        video_writer.release()

        logger.info(f"Video saved to {self._video_path}")

        self.recording_finished_event.set()

    def stop(self) -> None:
        self.shutdown_event.set()
        self.recording_finished_event.wait()


if __name__ == "__main__":
    """Records 10 seconds of video. Assumes there's being published to the "camera" namespace."""
    recorder = MultiprocessVideoRecorder("camera")
    recorder.start()
    time.sleep(10)
    recorder.stop()
