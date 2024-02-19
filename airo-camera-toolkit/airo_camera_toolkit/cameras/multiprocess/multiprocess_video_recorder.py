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

            # 1. wait until new timestamp is available, but if it takes > 2*camera_period, log a warning
            if timestamp_receiver == timestamp_prev_frame:
                # if receiver timestamp is not updated, check if it's been too long since last frame
                timestamp_current = time.time()
                timestamp_difference = timestamp_current - timestamp_prev_frame

                if timestamp_difference >= 2.0 * camera_period:
                    n_consecutive_frames_dropped += 1
                    logger.warning(
                        f"No frame received within {2.0 * camera_period:.3f} s, repeating previous frame in video (n = {n_consecutive_frames_dropped})."
                    )
                    image_rgb = image_previous
                    timestamp_prev_frame += camera_period  # pretend that the frame was received
                else:
                    continue  # wait a bit longer before taking action
            else:
                # new frame arrived
                if n_consecutive_frames_dropped > 0:
                    logger.info(f"New frame received after missing {n_consecutive_frames_dropped} frames.")
                n_consecutive_frames_dropped = 0
                image_rgb = receiver._retrieve_rgb_image_as_int()
                timestamp_prev_frame = timestamp_receiver

            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

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
