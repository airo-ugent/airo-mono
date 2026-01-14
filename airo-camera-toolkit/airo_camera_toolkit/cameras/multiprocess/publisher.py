import multiprocessing
from multiprocessing.context import Process

from airo_camera_toolkit.cameras.multiprocess.schema import Schema
from airo_camera_toolkit.interfaces import Camera
from airo_ipc.cyclone_shm.patterns.sm_writer import SMWriter
from cyclonedds.domain import DomainParticipant
from loguru import logger


class CameraPublisher(Process):
    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        schemas: list[Schema] = [],
        shared_memory_namespace: str = "camera",
    ) -> None:
        super().__init__()

        self._camera_cls = camera_cls
        self._camera_kwargs = camera_kwargs
        self._schemas = schemas
        self._shared_memory_namespace = shared_memory_namespace

        self.shutdown_event = multiprocessing.Event()

    def _setup(self) -> None:
        """Note: to be able to retrieve camera image from the Publisher process, the camera must be instantiated in the
        Publisher process. For this reason, we do not instantiate the camera in __init__ but, here instead."""

        # Instantiate the camera.
        logger.info(f"Instantiating a {self._camera_cls.__name__} camera.")
        self._camera: Camera = self._camera_cls(**self._camera_kwargs)
        logger.info(f"Successfully instantiated a {self._camera_cls.__name__} camera.")

        logger.info(f"Initializing CameraPublisher with schemas: {self._schemas}")

        # Initialize the DDS domain participant.
        self._dp = DomainParticipant()

        self._writers = dict()
        for s in self._schemas:
            s.allocate_empty(self._camera.resolution)
            self._writers[s] = SMWriter(self._dp, f"{self._shared_memory_namespace}_{s.topic}", s.buffer)

    def stop(self) -> None:
        self.shutdown_event.set()

    def run(self) -> None:
        logger.info(f"{self.__class__.__name__} process started.")

        self._setup()

        logger.info(f'{self.__class__.__name__} starting to publish to "{self._shared_memory_namespace}".')

        while not self.shutdown_event.is_set():
            self._camera._grab_images()

            for schema in self._schemas:
                schema.fill_from_camera(self._camera)
                self._writers[schema](schema.buffer)
