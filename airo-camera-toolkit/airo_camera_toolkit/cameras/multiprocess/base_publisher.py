"""Base classes for multiprocess camera publishers and receivers."""

import json
import multiprocessing
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import zenoh
from airo_camera_toolkit.cameras.multiprocess.frame_data import Frame, FrameLayout, PublisherMetadata
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


def _shm_enabled(explicit: bool | None = None) -> bool:
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


def _make_zenoh_config(shm: bool | None = None, router_endpoint: str | None = None) -> zenoh.Config:
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
    """Base class for camera publishers that publish frame data over Zenoh.

    The publisher runs in its own process. Frames are published with a ``ZenohWriter``, using
    shared memory transport if enabled (see ``AIRO_ZENOH_SHM``).

    Subclasses should implement:
    - _get_frame_layout(): Return the layout of the frame buffer this camera type publishes
    - _capture_frame(): Retrieve all data for a single frame and return it as a frame buffer,
      which ``run()`` then publishes

    Writers for extra key expressions should be created with ``_declare_writer()``, so that their
    layout is announced to receivers and they are stopped on shutdown.
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
            shared_memory_namespace: Zenoh key expression the frames are published on. Metadata and
                extra data use this name with a suffix (``_resolution``, ``_fps``, ...)
        """
        super().__init__()

        self._camera_cls = camera_cls
        self._camera_kwargs = camera_kwargs
        self._shared_memory_namespace = shared_memory_namespace

        self.shutdown_event = multiprocessing.Event()
        self._frame_id = 0
        # Filled in by _declare_writer(), in the publisher process.
        self._writers: dict[str, ZenohWriter[Any]] = {}
        self._stream_layouts: dict[str, FrameLayout[Any]] = {}

    def _setup(self) -> None:
        """Initialize the camera and Zenoh publishing infrastructure.

        Note: Camera must be instantiated in the publisher process to retrieve images.
        """
        self._shm_enabled = _shm_enabled()
        self._session = zenoh.open(_make_zenoh_config(shm=self._shm_enabled))
        self._metadata_publisher = self._session.declare_publisher(f"{self._shared_memory_namespace}_metadata")

        # Instantiate the camera
        logger.info(f"Instantiating a {self._camera_cls.__name__} camera.")
        self._camera = self._camera_cls(**self._camera_kwargs)

        if not isinstance(self._camera, RGBCamera):
            raise TypeError(f"camera_cls must be a subclass of RGBCamera, but is {self._camera_cls.__name__}")

        logger.info(f"Successfully instantiated a {self._camera_cls.__name__} camera.")
        # Intrinsics are static, and retrieving them can be costly (e.g. several SDK calls for a Zed).
        self._intrinsics = self._camera.intrinsics_matrix()

        # Set up frame writer
        self._setup_frame_writer()

    def _setup_frame_writer(self) -> None:
        """Set up the main frame data writer."""
        self._writer = self._declare_writer(
            self._shared_memory_namespace,
            self._get_frame_layout(self._camera.resolution[0], self._camera.resolution[1]),
        )

    def _declare_writer(self, key_expr: str, layout: FrameLayout[Frame]) -> ZenohWriter[Frame]:
        """Create a writer on *key_expr*, announce its layout in the metadata and stop it on shutdown."""
        writer = ZenohWriter(self._session, key_expr, layout, shm=self._shm_enabled)
        self._writers[key_expr] = writer
        self._stream_layouts[key_expr] = layout
        return writer

    def _publish_side_streams(self, frame_id: int, frame_timestamp: float) -> None:
        """Publish data that goes on its own key expression, such as a point cloud, for the current frame.

        Runs right after :meth:`_capture_frame`, before the frame itself is written. Subclasses that
        publish side streams should override this and call ``super()._publish_side_streams()``.
        Side streams let receivers that do not need the data skip receiving it.
        """

    def _publisher_config(self) -> dict[str, Any]:
        """Publisher options that receivers must agree with but that do not show in the wire format.

        Subclasses with such options should override this and extend ``super()._publisher_config()``.
        """
        return {}

    def _publish_metadata(self) -> None:
        """Publish the camera metadata, so receivers that start late can still read it."""
        self._metadata_publisher.put(self._metadata_json)

    def _stop_writers(self) -> None:
        """Stop all writers and the metadata publisher before closing the session."""
        for writer in self._writers.values():
            writer.stop()
        if hasattr(self, "_metadata_publisher"):
            self._metadata_publisher.undeclare()

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

        try:
            self._setup()
            self._metadata_json = PublisherMetadata(
                resolution=(self._camera.resolution[0], self._camera.resolution[1]),
                fps=float(self._camera.fps),
                streams={key_expr: layout.describe() for key_expr, layout in self._stream_layouts.items()},
                config=self._publisher_config(),
            ).to_json()
            logger.info(f'{self.__class__.__name__} starting to publish to "{self._shared_memory_namespace}".')

            while not self.shutdown_event.is_set():
                self._publish_metadata()

                # Capture frame with timestamp
                self._camera.grab_images()
                frame_timestamp = time.time()
                frame_id = self._next_frame_id()

                # Capture the frame, publish its side streams and then the frame itself, so a
                # receiver that has the frame normally already has its side streams too.
                frame = self._capture_frame(frame_id, frame_timestamp)
                self._publish_side_streams(frame_id, frame_timestamp)
                self._writer(frame)

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
            raise
        finally:
            self._stop_writers()
            # The session does not exist yet if _setup() failed before opening it.
            if hasattr(self, "_session"):
                self._session.close()
            logger.info(f"{self.__class__.__name__} process terminated.")

    @abstractmethod
    def _get_frame_layout(self, width: int, height: int) -> FrameLayout[Any]:
        """Return the frame buffer layout for this camera type.

        Args:
            width: Image width
            height: Image height

        Returns:
            The frame buffer layout
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
