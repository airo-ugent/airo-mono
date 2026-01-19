"""Base classes for multiprocess camera publishers and receivers."""

import multiprocessing
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from airo_camera_toolkit.cameras.multiprocess.frame_data import FpsIdl, ResolutionIdl
from airo_camera_toolkit.interfaces import RGBCamera
from airo_ipc.cyclone_shm.patterns.sm_writer import SMWriter  # type: ignore
from cyclonedds.domain import DomainParticipant
from loguru import logger


class BaseCameraPublisher(multiprocessing.context.Process, ABC):
    """Base class for camera publishers that write frame data to shared memory.

    Subclasses should implement:
    - _get_frame_buffer_template(): Return the appropriate frame buffer template
    - _setup_additional_writers(): Set up any additional shared memory writers
    - _capture_frame_data(): Capture all data for a single frame
    - _write_frame_data(): Write captured data to shared memory
    """

    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        shared_memory_namespace: str = "camera",
    ):
        """Initialize the camera publisher.

        Args:
            camera_cls: The camera class to instantiate (e.g., Zed, RealSense)
            camera_kwargs: Keyword arguments to pass to the camera constructor
            shared_memory_namespace: Prefix for shared memory blocks
        """
        super().__init__()

        self._camera_cls = camera_cls
        self._camera_kwargs = camera_kwargs
        self._shared_memory_namespace = shared_memory_namespace

        self.shutdown_event = multiprocessing.Event()
        self._frame_id = 0

    def _setup(self) -> None:
        """Initialize the camera and shared memory infrastructure.

        Note: Camera must be instantiated in the publisher process to retrieve images.
        """
        # Initialize DDS domain participant
        self._dp = DomainParticipant()
        self._resolution_writer = SMWriter(
            self._dp, f"{self._shared_memory_namespace}_resolution", ResolutionIdl.template()
        )
        self._fps_writer = SMWriter(self._dp, f"{self._shared_memory_namespace}_fps", FpsIdl.template())

        # Instantiate the camera
        logger.info(f"Instantiating a {self._camera_cls.__name__} camera.")
        self._camera = self._camera_cls(**self._camera_kwargs)

        if not isinstance(self._camera, RGBCamera):
            raise TypeError(f"camera_cls must be a subclass of RGBCamera, but is {self._camera_cls.__name__}")

        logger.info(f"Successfully instantiated a {self._camera_cls.__name__} camera.")

        # Set up shared memory writers
        self._setup_frame_writer()
        self._setup_additional_writers()

    def _setup_frame_writer(self) -> None:
        """Set up the main frame data writer."""
        frame_buffer_template = self._get_frame_buffer_template(self._camera.resolution[0], self._camera.resolution[1])

        self._writer = SMWriter(
            domain_participant=self._dp,
            topic_name=self._shared_memory_namespace,
            idl_dataclass=frame_buffer_template,
        )

    def _setup_additional_writers(self) -> None:
        """Set up additional shared memory writers (e.g., for optional data).

        Override in subclasses if needed.
        """

    def _publish_metadata(self) -> None:
        """Publish camera metadata (resolution and FPS)."""
        self._resolution_writer(
            ResolutionIdl(
                resolution=np.array([self._camera.resolution[0], self._camera.resolution[1]], dtype=np.int32),
            )
        )
        self._fps_writer(FpsIdl(fps=np.array([self._camera.fps], dtype=np.float64)))

    def _next_frame_id(self) -> int:
        """Get the next frame ID and increment the counter."""
        frame_id = self._frame_id
        self._frame_id += 1
        return frame_id

    def stop(self) -> None:
        """Signal the publisher to stop."""
        self.shutdown_event.set()

    def run(self) -> None:
        """Main loop of the publisher process."""
        logger.info(f"{self.__class__.__name__} process started.")
        self._setup()
        logger.info(f'{self.__class__.__name__} starting to publish to "{self._shared_memory_namespace}".')

        try:
            while not self.shutdown_event.is_set():
                self._publish_metadata()

                # Capture frame with timestamp
                self._camera._grab_images()
                frame_timestamp = time.time()
                frame_id = self._next_frame_id()

                # Capture and write frame data
                self._capture_frame_data(frame_id, frame_timestamp)
                self._write_frame_data()

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
            raise
        finally:
            logger.info(f"{self.__class__.__name__} process terminated.")

    @abstractmethod
    def _get_frame_buffer_template(self, width: int, height: int) -> Any:
        """Return the frame buffer template for this camera type.

        Args:
            width: Image width
            height: Image height

        Returns:
            Frame buffer template instance
        """

    @abstractmethod
    def _capture_frame_data(self, frame_id: int, frame_timestamp: float) -> None:
        """Capture all data for the current frame.

        This method should retrieve all necessary data from the camera and store it
        in instance variables for later writing.

        Args:
            frame_id: Monotonically increasing frame identifier
            frame_timestamp: Timestamp when the frame was captured
        """

    @abstractmethod
    def _write_frame_data(self) -> None:
        """Write the captured frame data to shared memory.

        This method should construct the appropriate frame buffer from previously
        captured data and write it using self._writer.
        """
