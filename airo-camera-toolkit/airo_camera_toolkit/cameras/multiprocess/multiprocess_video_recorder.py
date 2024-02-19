import datetime
import multiprocessing
import os
import time
from multiprocessing import Process
from typing import Optional

import cv2
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import MultiprocessRGBReceiver
from airo_camera_toolkit.image_transforms.image_transform import ImageTransform
from loguru import logger


class MultiprocessVideoRecorder(Process):
    def __init__(
        self,
        shared_memory_namespace: str,
        video_path: Optional[str] = None,
        image_transform: Optional[ImageTransform] = None,
        log_fps: bool = False,
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
            # - instead of : because : in file name can give problems
            timestamp = datetime.datetime.now().strftime("%H-%M-%S-%f")[:-3]
            video_name = f"{timestamp}.mp4"
            video_path = os.path.join(output_dir, video_name)
            video_path = os.path.abspath(video_path)

        self._video_path = video_path
        self.log_fps = log_fps

    def start(self) -> None:
        super().start()
        # Block until the recording has started
        self.recording_started_event.wait()

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        import ffmpegcv  # type: ignore

        receiver = MultiprocessRGBReceiver(self._shared_memory_namespace)
        camera_fps = receiver.fps_shm_array[0]
        camera_period = 1 / camera_fps

        height, width, _ = receiver.rgb_shm_array.shape
        video_writer = ffmpegcv.VideoWriter(self._video_path, "hevc", camera_fps, (width, height))

        logger.info(f"Recording video to {self._video_path}")

        timestamp_prev_frame = None

        while not self.shutdown_event.is_set():
            image_rgb = receiver.get_rgb_image_as_int()
            timestamp_current_frame = receiver.get_current_timestamp()

            if timestamp_prev_frame is not None:
                # This method of detecting dropped frames is better than simply checking FPS
                timestamp_difference = timestamp_current_frame - timestamp_prev_frame
                if timestamp_difference >= 1.2 * camera_period:
                    logger.warning(
                        f"Timestamp difference with previous frame: {timestamp_difference:.3f} s (Frame dropped?)"
                    )
                else:
                    logger.debug(f"Timestamp difference with previous frame: {timestamp_difference:.3f} s")

            timestamp_prev_frame = timestamp_current_frame

            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # # Known bug: vertical images still give horizontal videos
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
    recorder = MultiprocessVideoRecorder("camera", log_fps=True)
    recorder.start()
    time.sleep(10)
    recorder.stop()
