from airo_camera_toolkit.cameras.multiprocess.schema import Schema
from airo_camera_toolkit.interfaces import Camera
from airo_ipc.cyclone_shm.patterns.sm_reader import SMReader  # type: ignore
from airo_typing import CameraIntrinsicsMatrixType, CameraResolutionType
from cyclonedds.domain import DomainParticipant
from loguru import logger


class SharedMemoryReceiver(Camera):
    """A SharedMemoryReceiver is a subscriber for one or more topics communicated over shared memory, published by a CameraPublisher (see publisher.py)."""

    def __init__(
        self,
        camera_resolution: CameraResolutionType,
        schemas: list[Schema] = [],
        shared_memory_namespace: str = "camera",
    ):
        self._camera_resolution = camera_resolution
        self._schemas = schemas
        self._shared_memory_namespace = shared_memory_namespace

        logger.info(f"Initializing SharedMemoryReceiver with schemas: {schemas}")

        self._dp = DomainParticipant()

        self._readers = dict()
        for s in self._schemas:
            self._readers[s] = SMReader(
                self._dp, f"{self._shared_memory_namespace}_{s.topic}", s.allocate(self._camera_resolution)
            )

        # Initial grab
        self._grab_images()

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        # Needed to instantiate. Otherwise this class is abstract.
        raise NotImplementedError("The intrinsics_matrix() method is supposed to be implemented via Mixins.")

    def _grab_images(self) -> None:
        for schema in self._schemas:
            # Read serialized buffer from shared memory.
            buffer_data = self._readers[schema]()
            # Deserialize buffer and read into field defined by mixins.
            schema.deserialize(buffer_data, self)
