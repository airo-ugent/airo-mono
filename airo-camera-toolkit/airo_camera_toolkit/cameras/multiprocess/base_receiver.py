"""Base class for multiprocess camera receivers."""

import time
from abc import ABC, abstractmethod
from typing import Any

import zenoh
from airo_camera_toolkit.cameras.multiprocess.base_publisher import _make_zenoh_config
from airo_camera_toolkit.cameras.multiprocess.frame_data import FpsIdl, ResolutionIdl
from airo_camera_toolkit.cameras.multiprocess.zenoh_reader import ZenohReader
from airo_camera_toolkit.interfaces import RGBCamera
from airo_typing import CameraResolutionType
from loguru import logger


class BaseCameraReceiver(RGBCamera, ABC):
    """Base class for camera receivers that read frame data from shared memory.

    Subclasses should implement:
    - _get_frame_buffer_template(): Return the appropriate frame buffer template
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

        # Open a Zenoh session for receiving
        self._session = zenoh.open(_make_zenoh_config())

        # Read static camera information
        self._resolution = self._read_resolution(shared_memory_namespace)
        self._fps = self._read_fps(shared_memory_namespace)

        # Set up shared memory readers
        self._setup_frame_reader(self._resolution)

        # Grab first frame
        self.grab_images()

    def _setup_frame_reader(self, resolution: CameraResolutionType) -> None:
        """Set up the main frame data reader."""
        frame_buffer_template = self._get_frame_buffer_template(resolution[0], resolution[1])

        self._reader = ZenohReader(
            session=self._session,
            key_expr=self._shared_memory_namespace,
            template=frame_buffer_template,
        )

        # Initialize an empty frame
        self._last_frame = frame_buffer_template

        if self._block_until_new_frame:
            # If blocking is enabled, the frame_buffer_template must have a timestamp
            if not hasattr(frame_buffer_template, "frame_timestamp"):
                raise ValueError(
                    "Blocking until new frame is enabled, but frame buffer template has no 'frame_timestamp'"
                )

    def _read_fps(self, shared_memory_namespace: str) -> int:
        """Read the camera FPS from shared memory."""
        logger.info(f"Reading FPS from {shared_memory_namespace}_fps")
        fps_reader = ZenohReader(self._session, f"{shared_memory_namespace}_fps", FpsIdl.template())
        fps_data = fps_reader()
        fps_reader.stop()
        assert isinstance(fps_data, FpsIdl)  # for mypy
        fps = int(fps_data.fps.item())
        logger.info(f"Camera FPS: {fps}")
        return fps

    def _read_resolution(self, shared_memory_namespace: str) -> CameraResolutionType:
        """Read the camera resolution from shared memory."""
        logger.info(f"Reading resolution from {shared_memory_namespace}_resolution")
        resolution_reader = ZenohReader(
            self._session, f"{shared_memory_namespace}_resolution", ResolutionIdl.template()
        )
        resolution_data = resolution_reader()
        resolution_reader.stop()
        assert isinstance(resolution_data, ResolutionIdl)  # for mypy
        resolution = (
            int(resolution_data.resolution[0]),
            int(resolution_data.resolution[1]),
        )
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

    def grab_images(self) -> None:
        """Read the latest frame from shared memory."""
        # Block until a new message arrives (compare frame counter, not timestamp).
        # This avoids the expensive deserialization on every poll iteration.
        if self._block_until_new_frame:
            previous_count = self._reader.frame_count
            while self._reader.frame_count == previous_count:
                time.sleep(0.001)
        self._last_frame = self._reader()

    @abstractmethod
    def _get_frame_buffer_template(self, width: int, height: int) -> Any:
        """Return the frame buffer template for this camera type.

        Args:
            width: Image width
            height: Image height

        Returns:
            Frame buffer template instance
        """
