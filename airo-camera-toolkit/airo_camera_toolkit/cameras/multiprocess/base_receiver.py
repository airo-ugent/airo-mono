"""Base class for multiprocess camera receivers."""

import time
from abc import ABC, abstractmethod
from typing import Any

from airo_camera_toolkit.cameras.multiprocess.frame_data import FpsIdl, ResolutionIdl
from airo_camera_toolkit.interfaces import RGBCamera
from airo_ipc.cyclone_shm.patterns.sm_reader import SMReader
from airo_typing import CameraResolutionType
from cyclonedds.domain import DomainParticipant
from loguru import logger


class BaseCameraReceiver(RGBCamera, ABC):
    """Base class for camera receivers that read frame data from shared memory.

    Subclasses should implement:
    - _get_frame_buffer_template(): Return the appropriate frame buffer template
    - _setup_additional_readers(): Set up any additional shared memory readers
    - _grab_additional_data(): Read additional data (e.g., point clouds, depth maps)
    """

    def __init__(self, shared_memory_namespace: str, block_until_new_frame: bool = True) -> None:
        """Initialize the camera receiver.

        Args:
            shared_memory_namespace: Prefix for shared memory blocks to read from
            block_until_new_frame: Whether to block until a new frame is available
        """
        super().__init__()

        self._shared_memory_namespace = shared_memory_namespace
        self._block_until_new_frame = block_until_new_frame

        # Initialize the DDS domain participant
        self._dp = DomainParticipant()

        # Read static camera information
        self._resolution = self._read_resolution(shared_memory_namespace)
        self._fps = self._read_fps(shared_memory_namespace)

        # Set up shared memory readers
        self._setup_frame_reader(self._resolution)
        self._setup_additional_readers(self._resolution)

        # Grab first frame
        self._grab_images()

    def _setup_frame_reader(self, resolution: CameraResolutionType) -> None:
        """Set up the main frame data reader."""
        frame_buffer_template = self._get_frame_buffer_template(resolution[0], resolution[1])

        self._reader = SMReader(
            domain_participant=self._dp,
            topic_name=self._shared_memory_namespace,
            idl_dataclass=frame_buffer_template,
        )

        # Initialize an empty frame
        self._last_frame = frame_buffer_template

        if self._block_until_new_frame:
            # If blocking is enabled, the frame_buffer_template must have a timestamp
            if not hasattr(frame_buffer_template, "frame_timestamp"):
                raise ValueError(
                    "Blocking until new frame is enabled, but frame buffer template has no 'frame_timestamp'"
                )

    def _setup_additional_readers(self, resolution: CameraResolutionType) -> None:
        """Set up additional shared memory readers (e.g., for optional data).

        Override in subclasses if needed.
        """

    def _read_fps(self, shared_memory_namespace: str) -> int:
        """Read the camera FPS from shared memory."""
        logger.info(f"Reading FPS from {shared_memory_namespace}_fps")
        fps_reader = SMReader(self._dp, f"{shared_memory_namespace}_fps", FpsIdl.template())
        fps_data: FpsIdl = fps_reader()
        fps = int(fps_data.fps.item())
        logger.info(f"Camera FPS: {fps}")
        return fps

    def _read_resolution(self, shared_memory_namespace: str) -> CameraResolutionType:
        """Read the camera resolution from shared memory."""
        logger.info(f"Reading resolution from {shared_memory_namespace}_resolution")
        resolution_reader = SMReader(self._dp, f"{shared_memory_namespace}_resolution", ResolutionIdl.template())
        resolution_data: ResolutionIdl = resolution_reader()
        resolution = (int(resolution_data.resolution[0]), int(resolution_data.resolution[1]))
        logger.info(f"Camera resolution: {resolution}")
        return resolution

    @property
    def fps(self) -> int:
        """The frames per second of the camera."""
        return self._fps

    @property
    def resolution(self) -> CameraResolutionType:
        """The resolution of the camera."""
        return self._resolution

    def get_current_timestamp(self) -> float:
        """Get the timestamp of the current frame."""
        return self._last_frame.frame_timestamp.item()

    def get_current_frame_id(self) -> int:
        """Get the frame ID of the current frame."""
        return self._last_frame.frame_id.item()

    def _grab_images(self) -> None:
        """Read the latest frame from shared memory."""
        previous_timestamp = self._last_frame.frame_timestamp.item()

        # Block until we get a frame with a newer timestamp
        if self._block_until_new_frame:
            while True:
                self._last_frame = self._reader()
                current_timestamp = self._last_frame.frame_timestamp.item()
                if current_timestamp > previous_timestamp:
                    break
                time.sleep(0.001)  # Sleep briefly to avoid busy waiting
        else:
            self._last_frame = self._reader()

        self._grab_additional_data()

    def _grab_additional_data(self) -> None:
        """Read additional data (e.g., point clouds, depth maps).

        Override in subclasses if needed.
        """

    @abstractmethod
    def _get_frame_buffer_template(self, width: int, height: int) -> Any:
        """Return the frame buffer template for this camera type.

        Args:
            width: Image width
            height: Image height

        Returns:
            Frame buffer template instance
        """
