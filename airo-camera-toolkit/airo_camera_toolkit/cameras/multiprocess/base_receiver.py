"""Base class for multiprocess camera receivers."""

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import zenoh
from airo_camera_toolkit.cameras.multiprocess.base_publisher import _make_zenoh_config
from airo_camera_toolkit.cameras.multiprocess.frame_data import FpsIdl, ResolutionIdl
from airo_camera_toolkit.cameras.multiprocess.zenoh_reader import DEFAULT_FIRST_MESSAGE_TIMEOUT, ZenohReader
from airo_camera_toolkit.interfaces import RGBCamera
from airo_typing import CameraResolutionType
from loguru import logger


class BaseCameraReceiver(RGBCamera, ABC):
    """Base class for camera receivers that read frame data from shared memory.

    The arrays returned by the ``retrieve_*`` methods are read-only views into
    the last received frame, so they are not copied out of the payload.  Call
    ``.copy()`` on one if you need to modify it in place.  Views from a previous
    ``grab_images()`` stay valid, since each frame keeps its own payload alive.

    Subclasses should implement:
    - _get_frame_buffer_template(): Return the appropriate frame buffer template
    """

    def __init__(
        self,
        shared_memory_namespace: str,
        block_until_new_frame: bool = True,
        timeout: Optional[float] = DEFAULT_FIRST_MESSAGE_TIMEOUT,
    ) -> None:
        """Initialize the camera receiver.

        Args:
            shared_memory_namespace: Prefix for shared memory blocks to read from
            block_until_new_frame: Whether to block until a new frame is available
            timeout: Maximum seconds to wait for the publisher to send its first
                message, for each of the key expressions this receiver subscribes
                to.  ``None`` waits indefinitely.

        Raises:
            TimeoutError: If the publisher does not publish within *timeout*.
        """
        super().__init__()

        self._shared_memory_namespace = shared_memory_namespace
        self._block_until_new_frame = block_until_new_frame
        self._timeout = timeout
        self._reader: Optional[ZenohReader] = None
        self._stopped = False

        self._connect()

    def _connect(self) -> None:
        """Open a Zenoh session, read the static camera information and subscribe.

        Raises:
            TimeoutError: If the publisher does not publish within the
                receiver's timeout.
        """
        # Open a Zenoh session for receiving
        self._session = zenoh.open(_make_zenoh_config())
        self._stopped = False

        try:
            # Read static camera information
            self._resolution = self._read_resolution(self._shared_memory_namespace)
            self._fps = self._read_fps(self._shared_memory_namespace)

            # Set up shared memory readers
            self._setup_frame_reader(self._resolution)

            # Grab first frame
            self.grab_images()
        except BaseException:
            # A half-constructed receiver is never handed to the caller, so
            # nobody can stop() it.  Release the session here: a leaked session
            # keeps scouting and stays a peer of every publisher started later
            # in this process, which prevents those publishers from reaching
            # new receivers.
            self.stop()
            raise

    def reconnect(self) -> None:
        """Re-establish the connection to the publisher.

        Closes the Zenoh session, opens a new one and reads the publisher's
        resolution and fps again.  Use it when :meth:`grab_images` times out
        because the publisher was restarted: a new session re-runs peer
        discovery, drops the shared memory mappings of the previous publisher,
        and picks up a resolution that changed across the restart (frames from a
        publisher at another resolution do not match the previous frame buffer
        template and are rejected).

        Arrays retrieved before this call keep pointing at their own payload and
        stay valid.

        Raises:
            TimeoutError: If the publisher does not publish within the
                receiver's timeout.
        """
        self.stop()
        self._connect()

    def _setup_frame_reader(self, resolution: CameraResolutionType) -> None:
        """Set up the main frame data reader."""
        frame_buffer_template = self._get_frame_buffer_template(resolution[0], resolution[1])

        self._reader = ZenohReader(
            session=self._session,
            key_expr=self._shared_memory_namespace,
            template=frame_buffer_template,
            timeout=self._timeout,
        )

        # Initialize an empty frame
        self._last_frame = frame_buffer_template
        # Message count of the frame in self._last_frame; 0 means "nothing read yet".
        self._consumed_count = 0

        if self._block_until_new_frame:
            # If blocking is enabled, the frame_buffer_template must have a timestamp
            if not hasattr(frame_buffer_template, "frame_timestamp"):
                raise ValueError(
                    "Blocking until new frame is enabled, but frame buffer template has no 'frame_timestamp'"
                )

    def _read_fps(self, shared_memory_namespace: str) -> int:
        """Read the camera FPS from shared memory."""
        logger.info(f"Reading FPS from {shared_memory_namespace}_fps")
        fps_reader = ZenohReader(
            self._session, f"{shared_memory_namespace}_fps", FpsIdl.template(), timeout=self._timeout
        )
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
            self._session, f"{shared_memory_namespace}_resolution", ResolutionIdl.template(), timeout=self._timeout
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
        """Read the latest frame from shared memory.

        When ``block_until_new_frame`` is set, this blocks until a message newer
        than the one currently held is available.  The comparison is against the
        message count of the frame we last returned (not the count at entry), so
        a frame that arrived while the caller was busy is returned immediately
        instead of waiting for the next one.

        Raises:
            RuntimeError: If the receiver has been stopped.
            TimeoutError: If no newer frame arrives within the receiver's
                timeout, which is what a publisher that stopped, restarted or
                became unreachable looks like from here.  Call
                :meth:`reconnect` to re-establish the connection.
        """
        reader = self._reader
        if reader is None:
            raise RuntimeError("This receiver was stopped; call reconnect() to use it again.")

        if self._block_until_new_frame:
            t0 = time.time()
            # Compare message counters rather than timestamps: this avoids the
            # expensive deserialization on every poll iteration.
            while reader.frame_count == self._consumed_count:
                if self._timeout is not None and time.time() - t0 >= self._timeout:
                    raise TimeoutError(
                        f"No new frame on '{self._shared_memory_namespace}' within {self._timeout}s. "
                        "Is the publisher still running? Call reconnect() to re-establish the connection."
                    )
                time.sleep(0.001)
        self._last_frame, self._consumed_count = reader.read()

    def stop(self) -> None:
        """Undeclare the readers and close the Zenoh session.

        Safe to call more than once.  Receivers hold a Zenoh session and a
        background subscriber thread for the lifetime of the process otherwise.
        """
        if self._stopped:
            return
        self._stopped = True
        # The frame reader does not exist yet when the connection failed before
        # it was set up.
        if self._reader is not None:
            self._reader.stop()
            self._reader = None
        self._session.close()

    def __enter__(self) -> "BaseCameraReceiver":
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    @abstractmethod
    def _get_frame_buffer_template(self, width: int, height: int) -> Any:
        """Return the frame buffer template for this camera type.

        Args:
            width: Image width
            height: Image height

        Returns:
            Frame buffer template instance
        """
