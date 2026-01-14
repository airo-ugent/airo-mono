from airo_camera_toolkit.cameras.multiprocess.schema import Schema
from airo_camera_toolkit.interfaces import Camera
from airo_ipc.cyclone_shm.patterns.sm_reader import SMReader
from airo_typing import CameraIntrinsicsMatrixType, CameraResolutionType
from cyclonedds.domain import DomainParticipant
from loguru import logger


class SharedMemoryReceiver(Camera):
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
            s.allocate_empty(self._camera_resolution)
            self._readers[s] = SMReader(self._dp, f"{self._shared_memory_namespace}_{s.topic}", s.buffer)

        # Initial grab
        self._grab_images()

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        # Needed to instantiate. Otherwise this class is abstract.
        raise NotImplementedError("The intrinsics_matrix() method is supposed to be implemented via Mixins.")

    def _grab_images(self) -> None:
        for s in self._schemas:
            s.read_into_receiver(self._readers[s](), self)
