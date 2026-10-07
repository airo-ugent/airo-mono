"""Base classes for multiprocess camera publishers and receivers."""

import json
import multiprocessing
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import zenoh
from airo_camera_toolkit.cameras.multiprocess.frame_data import FpsIdl, ResolutionIdl
from airo_camera_toolkit.cameras.multiprocess.zenoh_writer import ZenohWriter
from airo_camera_toolkit.interfaces import RGBCamera
from loguru import logger

# Environment variable naming the Zenoh router to connect to, e.g.
# "tcp/192.168.0.10:7447".  Publishers and receivers are confined to the local
# host unless this is set.
ZENOH_ROUTER_ENV_VAR = "AIRO_ZENOH_ROUTER"

# Environment variable to enable the Zenoh shared memory transport, e.g. "1" or
# "true".  SHM buffers are mlock()ed, which requires the process's
# RLIMIT_MEMLOCK (`ulimit -l`) to be large enough for the frame pool, so it is
# opt-in: it defaults to disabled, and a warning is logged when that default
# applies, since SHM gives a real latency benefit that is easy to miss.
ZENOH_SHM_ENV_VAR = "AIRO_ZENOH_SHM"

_LOCALHOST = "127.0.0.1"
_SHM_ENABLED_VALUES = {"1", "true", "yes", "on"}


def _shm_enabled(explicit: Optional[bool] = None) -> bool:
    """Resolve whether Zenoh shared memory transport should be used.

    Args:
        explicit: Force the result if not ``None``. Otherwise this is read
            from the ``AIRO_ZENOH_SHM`` environment variable, which defaults
            to disabled; a warning is logged in that default case.

    Returns:
        Whether SHM should be used.
    """
    if explicit is not None:
        return explicit
    if ZENOH_SHM_ENV_VAR in os.environ:
        return os.environ[ZENOH_SHM_ENV_VAR].strip().lower() in _SHM_ENABLED_VALUES
    logger.warning(
        f"Zenoh shared memory transport is disabled by default (set {ZENOH_SHM_ENV_VAR}=1 to enable it for "
        "lower latency). Frames will be copied instead. Enabling shared memory transport requires the process's 'ulimit -l' (max locked memory) "
        "to accommodate the frame pool, which is a few MB per camera stream and not always available. See multiprocess/README.md for more details."
    )
    return False


def _make_zenoh_config(shm: Optional[bool] = None, router_endpoint: Optional[str] = None) -> zenoh.Config:
    """Return the Zenoh configuration used by the multiprocess publishers and receivers.

    By default, the session is confined to the local host. You can opt in to cross-host
    through a Zenoh router by passing ``router_endpoint`` or setting the
    ``AIRO_ZENOH_ROUTER`` environment variable.

    In the local-host case, shared memory transport can be used to reduce latency further.
    This option is not available in the cross-host case.

    Args:
        shm: Whether to enable the Zenoh shared memory transport. Defaults to
            the value of ``AIRO_ZENOH_SHM`` (disabled unless set to "1",
            "true", "yes" or "on"; a warning is logged when the default
            applies). Shared memory buffers are mlock()ed, so enabling this
            requires the process's ``ulimit -l`` to accommodate the frame
            pool.
        router_endpoint: Zenoh endpoint of a router to connect to, e.g.
            ``"tcp/192.168.0.10:7447"``.  Defaults to the value of
            ``AIRO_ZENOH_ROUTER``; when that is unset too, the session is
            restricted to the local host.

    Returns:
        The Zenoh configuration.
    """
    shm = _shm_enabled(shm)

    if router_endpoint is None:
        router_endpoint = os.environ.get(ZENOH_ROUTER_ENV_VAR) or None

    conf = zenoh.Config()
    conf.insert_json5("transport/shared_memory/enabled", json.dumps(shm))

    if router_endpoint is None:
        conf.insert_json5("scouting/multicast/interface", json.dumps(_LOCALHOST))
        conf.insert_json5("listen/endpoints", json.dumps([f"tcp/{_LOCALHOST}:0"]))
    else:
        logger.info(f"Using Zenoh router at {router_endpoint}; frames may travel over the network.")
        conf.insert_json5("scouting/multicast/enabled", "false")
        conf.insert_json5("connect/endpoints", json.dumps([router_endpoint]))

    return conf


class BaseCameraPublisher(multiprocessing.context.Process, ABC):
    """Base class for camera publishers that write frame data to shared memory.

    Subclasses should implement:
    - _get_frame_buffer_template(): Return the appropriate frame buffer template
    - _retrieve_frame_data(): Retrieve all data for a single frame
    - _write_frame_data(): Write retrieved data to shared memory
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
        """Initialize the camera and Zenoh publishing infrastructure.

        Note: Camera must be instantiated in the publisher process to retrieve images.
        """
        self._shm_enabled = _shm_enabled()
        self._session = zenoh.open(_make_zenoh_config(shm=self._shm_enabled))
        self._resolution_writer = ZenohWriter(
            self._session,
            f"{self._shared_memory_namespace}_resolution",
            ResolutionIdl.template(),
            shm=self._shm_enabled,
        )
        self._fps_writer = ZenohWriter(
            self._session, f"{self._shared_memory_namespace}_fps", FpsIdl.template(), shm=self._shm_enabled
        )

        # Instantiate the camera
        logger.info(f"Instantiating a {self._camera_cls.__name__} camera.")
        self._camera = self._camera_cls(**self._camera_kwargs)

        if not isinstance(self._camera, RGBCamera):
            raise TypeError(f"camera_cls must be a subclass of RGBCamera, but is {self._camera_cls.__name__}")

        logger.info(f"Successfully instantiated a {self._camera_cls.__name__} camera.")

        # Set up frame writer
        self._setup_frame_writer()

    def _setup_frame_writer(self) -> None:
        """Set up the main frame data writer."""
        frame_buffer_template = self._get_frame_buffer_template(self._camera.resolution[0], self._camera.resolution[1])

        self._writer = ZenohWriter(
            session=self._session,
            key_expr=self._shared_memory_namespace,
            template=frame_buffer_template,
            shm=self._shm_enabled,
        )

    def _publish_metadata(self) -> None:
        """Publish camera metadata (resolution and FPS)."""
        self._resolution_writer(
            ResolutionIdl(
                resolution=np.array(
                    [self._camera.resolution[0], self._camera.resolution[1]],
                    dtype=np.int32,
                ),
            )
        )
        self._fps_writer(FpsIdl(fps=np.array([self._camera.fps], dtype=np.float64)))

    def _stop_writers(self) -> None:
        """Stop all ZenohWriters before closing the session.

        Subclasses that create additional writers should override this,
        stop their own writers, and call ``super()._stop_writers()``.
        """
        if hasattr(self, "_writer"):
            self._writer.stop()
        if hasattr(self, "_resolution_writer"):
            self._resolution_writer.stop()
        if hasattr(self, "_fps_writer"):
            self._fps_writer.stop()

    def _next_frame_id(self) -> int:
        """Get the next frame ID and increment the counter."""
        frame_id = self._frame_id
        self._frame_id += 1
        return frame_id

    @staticmethod
    def _header(frame_id: int, frame_timestamp: float) -> dict:
        """Build the ``frame_id``/``frame_timestamp`` keyword arguments shared by every frame buffer.

        Args:
            frame_id: Monotonically increasing frame identifier.
            frame_timestamp: Timestamp when the frame was captured.

        Returns:
            A dict suitable for ``**``-passing into a frame buffer dataclass constructor.
        """
        return {
            "frame_id": np.array([frame_id], dtype=np.uint64),
            "frame_timestamp": np.array([frame_timestamp], dtype=np.float64),
        }

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
                self._camera.grab_images()
                frame_timestamp = time.time()
                frame_id = self._next_frame_id()

                # Capture and write the frame
                frame = self._capture_frame(frame_id, frame_timestamp)
                self._writer(frame)

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
            raise
        finally:
            self._stop_writers()
            self._session.close()
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
    def _capture_frame(self, frame_id: int, frame_timestamp: float) -> Any:
        """Capture the current camera frame and return the frame buffer to publish.

        **Important**: This should retrieve data using methods starting with `retrieve_`, not
        `get_`, since `grab_images()` has already been called by `run()` and calling a `get_`
        method would trigger an extra, unwanted frame capture.

        Implementations may also publish extra data on their own writers (e.g. a point cloud or
        spatial map on a side key expression) before returning.

        Args:
            frame_id: Monotonically increasing frame identifier
            frame_timestamp: Timestamp when the frame was captured

        Returns:
            The frame buffer dataclass instance to write via ``self._writer``.
        """
