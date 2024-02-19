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
        fill_missing_frames: bool = True,
    ):
        super().__init__(daemon=True)
        self._shared_memory_namespace = shared_memory_namespace
        self._image_transform = image_transform
        self.recording_started_event = multiprocessing.Event()
        self.recording_finished_event = multiprocessing.Event()
        self.shutdown_event = multiprocessing.Event()
        self.fill_missing_frames = fill_missing_frames

        if video_path is None:
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            # - instead of : because : in file name can give problems
            timestamp = datetime.datetime.now().strftime("%H-%M-%S-%f")[:-3]
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
        camera_fps = receiver.fps_shm_array[0]
        camera_period = 1 / camera_fps

        height, width, _ = receiver.rgb_shm_array.shape
        video_writer = ffmpegcv.VideoWriter(self._video_path, "hevc", camera_fps, (width, height))

        logger.info(f"Recording video to {self._video_path}")

        timestamp_prev_frame = None
        image_previous = receiver.get_rgb_image_as_int()
        timestamp_prev_frame = receiver.get_current_timestamp()
        video_writer.write(cv2.cvtColor(image_previous, cv2.COLOR_RGB2BGR))
        self.recording_started_event.set()
        n_consecutive_frames_dropped = 0

        while not self.shutdown_event.is_set():
            timestamp_receiver = receiver.get_current_timestamp()

            if timestamp_receiver <= timestamp_prev_frame:
                continue

            # New frame arrived
            image_rgb_new = receiver._retrieve_rgb_image_as_int()

            timestamp_difference = timestamp_receiver - timestamp_prev_frame
            missed_frames = int(timestamp_difference / camera_period) - 1

            if missed_frames > 0:
                logger.warning(f"Missed {missed_frames} frames (fill_missing_frames = {self.fill_missing_frames}).")

                if self.fill_missing_frames:
                    image_fill = cv2.cvtColor(image_previous, cv2.COLOR_RGB2BGR)
                    for _ in range(missed_frames):
                        video_writer.write(image_fill)
                        n_consecutive_frames_dropped += 1

            timestamp_prev_frame = timestamp_receiver
            image_previous = image_rgb_new

            image = cv2.cvtColor(image_rgb_new, cv2.COLOR_RGB2BGR)

            # Known bug: vertical images still give horizontal videos
            if self._image_transform is not None:
                image = self._image_transform.transform_image(image)

            video_writer.write(image)

        video_writer.release()
        logger.info(f"Video saved to {self._video_path}")
        self.recording_finished_event.set()

    def stop(self) -> None:
        self.shutdown_event.set()
        logger.info("Set video recording shutdown event.")
        self.recording_finished_event.wait()


if __name__ == "__main__":
    """Records 10 seconds of video. Assumes there's being published to the "camera" namespace."""
    recorder = MultiprocessVideoRecorder("camera")
    recorder.start()
    time.sleep(10)
    recorder.stop()
