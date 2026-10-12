"""Base classes for multiprocess camera publishers and receivers."""

import itertools
import json
import multiprocessing
import os
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
import zenoh
from airo_camera_toolkit.cameras.multiprocess.frame_data import Frame, FrameLayout, PublisherMetadata, RGBFrameBuffer
from airo_camera_toolkit.cameras.multiprocess.zenoh_writer import ZenohWriter
from airo_camera_toolkit.interfaces import RGBCamera
from loguru import logger

# Environment variable naming the Zenoh router to connect to, e.g.
# "tcp/192.168.0.10:7447".  Publishers and receivers are confined to the local
# host unless this is set.
ZENOH_ROUTER_ENV_VAR = "AIRO_ZENOH_ROUTER"

# Environment variable to enable the Zenoh shared memory transport, e.g. "1" or
# "true". Opt-in, since it needs a large enough `ulimit -l` (see the README).
ZENOH_SHM_ENV_VAR = "AIRO_ZENOH_SHM"

_LOCALHOST = "127.0.0.1"
_SHM_ENABLED_VALUES = {"1", "true", "yes", "on"}


def _shm_enabled() -> bool:
    """Whether ``AIRO_ZENOH_SHM`` enables the Zenoh shared memory transport (logs a warning if unset)."""
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
            "true", "yes" or "on").
        router_endpoint: Zenoh endpoint of a router to connect to, e.g.
            ``"tcp/192.168.0.10:7447"``.  Defaults to the value of
            ``AIRO_ZENOH_ROUTER``; when that is unset too, the session is
            restricted to the local host.

    Returns:
        The Zenoh configuration.
    """
    if shm is None:
        shm = _shm_enabled()

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

    Subclasses should set ``_frame_type`` to the frame buffer they publish, and implement
    ``_capture_frame()``, which retrieves all data for a single frame and returns it as a frame
    buffer of that type for ``run()`` to publish.

    Writers for extra key expressions should be created with ``_declare_writer()``, so that their
    layout is announced to receivers and they are stopped on shutdown.
    """

    _frame_type: ClassVar[type[RGBFrameBuffer]]

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
                extra data use this name with a suffix (``_metadata``, ``_pointcloud``, ...)
        """
        super().__init__()

        self._camera_cls = camera_cls
        self._camera_kwargs = camera_kwargs
        self._shared_memory_namespace = shared_memory_namespace

        self.shutdown_event = multiprocessing.Event()
        # Filled in by _declare_writer(), in the publisher process.
        self._writers: dict[str, ZenohWriter[Any]] = {}

    def _setup(self) -> None:
        """Instantiate the camera and declare the frame writer.

        Note: Camera must be instantiated in the publisher process to retrieve images.
        """
        # Instantiate the camera
        logger.info(f"Instantiating a {self._camera_cls.__name__} camera.")
        self._camera = self._camera_cls(**self._camera_kwargs)

        if not isinstance(self._camera, RGBCamera):
            raise TypeError(f"camera_cls must be a subclass of RGBCamera, but is {self._camera_cls.__name__}")

        logger.info(f"Successfully instantiated a {self._camera_cls.__name__} camera.")
        # Intrinsics are static, and retrieving them can be costly (e.g. several SDK calls for a Zed).
        self._intrinsics = self._camera.intrinsics_matrix()

        self._writer = self._declare_writer(
            self._shared_memory_namespace, self._frame_type.layout(*self._camera.resolution)
        )

    def _declare_writer(self, key_expr: str, layout: FrameLayout[Frame]) -> ZenohWriter[Frame]:
        """Create a writer on *key_expr*, announce its layout in the metadata and stop it on shutdown."""
        writer = ZenohWriter(self._session, key_expr, layout, shm=self._shm_enabled)
        self._writers[key_expr] = writer
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
        self._shm_enabled = _shm_enabled()
        self._session = zenoh.open(_make_zenoh_config(shm=self._shm_enabled))
        metadata_publisher = self._session.declare_publisher(f"{self._shared_memory_namespace}_metadata")

        try:
            self._setup()
            metadata_json = PublisherMetadata(
                resolution=(self._camera.resolution[0], self._camera.resolution[1]),
                fps=float(self._camera.fps),
                streams={key_expr: writer.layout.describe() for key_expr, writer in self._writers.items()},
                config=self._publisher_config(),
            ).to_json()
            logger.info(f'{self.__class__.__name__} starting to publish to "{self._shared_memory_namespace}".')

            for frame_id in itertools.count():
                if self.shutdown_event.is_set():
                    break
                # Republished with every frame, so receivers that start late can still read it.
                metadata_publisher.put(metadata_json)

                # Capture frame with timestamp
                self._camera.grab_images()
                frame_timestamp = time.time()

                # Capture the frame, publish its side streams and then the frame itself, so a
                # receiver that has the frame normally already has its side streams too.
                frame = self._capture_frame(frame_id, frame_timestamp)
                self._publish_side_streams(frame_id, frame_timestamp)
                self._writer(frame)

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
            raise
        finally:
            for writer in self._writers.values():
                writer.stop()
            metadata_publisher.undeclare()
            self._session.close()
            logger.info(f"{self.__class__.__name__} process terminated.")

    @abstractmethod
    def _capture_frame(self, frame_id: int, frame_timestamp: float) -> Any:
        """Capture the current camera frame and return the frame buffer to publish.

        **Important**: This should retrieve data using methods starting with `retrieve_`, not
        `get_`, since `grab_images()` has already been called by `run()` and calling a `get_`
        method would trigger an extra, unwanted frame capture.

        Data on other key expressions, such as a point cloud, is published by
        :meth:`_publish_side_streams` instead.

        Args:
            frame_id: Monotonically increasing frame identifier
            frame_timestamp: Timestamp when the frame was captured

        Returns:
            The frame buffer dataclass instance to write via ``self._writer``.
        """
