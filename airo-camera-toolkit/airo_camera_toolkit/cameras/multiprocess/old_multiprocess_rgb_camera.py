"""Publisher and receiver classes for multiprocess camera sharing."""

import multiprocessing
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_ipc.cyclone_shm.idl_shared_memory.base_idl import BaseIdl
from airo_ipc.cyclone_shm.patterns.sm_reader import SMReader
from airo_ipc.cyclone_shm.patterns.sm_writer import SMWriter
from airo_typing import CameraIntrinsicsMatrixType, CameraResolutionType, NumpyFloatImageType, NumpyIntImageType
from cyclonedds.domain import DomainParticipant
from cyclonedds.util import duration
from loguru import logger


@dataclass
class FrameMetadata(BaseIdl):
    """This struct, published over shared memory, contains metadata for frames: FPS and image resolution.

    It has no timestamp, because this is timeless data: we assume that it remains the same."""

    # FPS (scalar float)
    fps: np.ndarray
    # Resolution (width and height, (2,) uint32 array)
    resolution: np.ndarray

    @staticmethod
    def template() -> Any:
        """Construct a new FrameMetadata with shared memory backed arrays."""
        return FrameMetadata(
            fps=np.empty((1,), dtype=np.float32),
            resolution=np.empty((2,), dtype=np.uint32),
        )


@dataclass
class RGBFrameBuffer(BaseIdl):
    """This struct, sent over shared memory, contains a timestamp, an RGB image, and the camera intrinsics."""

    # Timestamp of the frame (seconds)
    timestamp: np.ndarray
    # Color image data (height x width x channels)
    rgb: np.ndarray
    # Intrinsic camera parameters (camera matrix)
    intrinsics: np.ndarray

    @staticmethod
    def template(width: int, height: int) -> Any:
        """Construct a new RGBFrameBuffer with shared memory backed arrays initialized with the given width and height."""
        return RGBFrameBuffer(
            timestamp=np.empty((1,), dtype=np.float64),
            rgb=np.empty((height, width, 3), dtype=np.uint8),
            intrinsics=np.empty((3, 3), dtype=np.float64),
        )


class MultiprocessRGBPublisher(multiprocessing.context.Process):
    """Publishes the data of a camera that implements the RGBCamera interface to shared memory blocks."""

    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        shared_memory_namespace: str = "camera",
    ):
        """Instantiates the publisher. Note that the publisher (and its process) will not start until start() is called.

        Args:
            camera_cls (type): The class, e.g., Zed2i, that this publisher will instantiate.
            camera_kwargs (dict, optional): The kwargs that will be passed to the camera_cls constructor.
            shared_memory_namespace (str, optional): The string that will be used to prefix the shared memory blocks that this class will create.
        """
        super().__init__()

        self._camera_cls = camera_cls
        self._camera_kwargs = camera_kwargs
        self._shared_memory_namespace = shared_memory_namespace

        self.shutdown_event = multiprocessing.Event()

    def _setup(self) -> None:
        """Note: to be able to retrieve camera image from the Publisher process, the camera must be instantiated in the
        Publisher process. For this reason, we do not instantiate the camera in __init__ but, here instead."""

        # Initialize the DDS domain participant.
        self._dp = DomainParticipant()
        self._metadata_writer = SMWriter(
            self._dp,
            f"{self._shared_memory_namespace}_metadata",
            FrameMetadata.template(),
        )

        # Instantiate the camera.
        logger.info(f"Instantiating a {self._camera_cls.__name__} camera.")
        self._camera = self._camera_cls(**self._camera_kwargs)
        if not isinstance(self._camera, RGBCamera):  # Check whether user passed a valid camera class
            raise TypeError(f"camera_cls must be a subclass of RGBCamera, but is {self._camera_cls.__name__}")
        logger.info(f"Successfully instantiated a {self._camera_cls.__name__} camera.")

        self._setup_framebuffer_writer()  # Overwritten in base classes.

    def _setup_framebuffer_writer(self) -> None:
        # Create the shared memory writer for the frame buffer.
        self._writer = SMWriter(
            domain_participant=self._dp,
            topic_name=self._shared_memory_namespace,
            idl_dataclass=RGBFrameBuffer.template(self._camera.resolution[0], self._camera.resolution[1]),
        )

    def stop(self) -> None:
        self.shutdown_event.set()

    def run(self) -> None:
        """Main loop of the process, runs until the process is terminated."""

        logger.info(f"{self.__class__.__name__} process started.")
        self._setup()
        assert isinstance(self._camera, RGBCamera)  # Just to make mypy happy, already checked in _setup()
        logger.info(f'{self.__class__.__name__} starting to publish to "{self._shared_memory_namespace}".')

        try:
            while not self.shutdown_event.is_set():
                self._metadata_writer(
                    FrameMetadata(
                        fps=np.array([self._camera.fps], dtype=np.float32),
                        resolution=self._camera.resolution,
                    )
                )

                self._camera._grab_images()
                timestamp = time.time()
                image = self._camera._retrieve_rgb_image_as_int()

                frame_data = RGBFrameBuffer(
                    timestamp=np.array([timestamp], dtype=np.float64),
                    rgb=image,
                    intrinsics=self._camera.intrinsics_matrix(),
                )

                self._writer(frame_data)
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
        finally:
            logger.info(f"{self.__class__.__name__} process terminated.")


class MultiprocessRGBReceiver(RGBCamera):
    """Implements the RGB camera interface for a camera that is running in a different process, to be used with the Publisher class."""

    def __init__(
        self,
        shared_memory_namespace: str,
    ) -> None:
        super().__init__()

        self._shared_memory_namespace = shared_memory_namespace

        # Initialize the DDS domain participant.
        self._dp = DomainParticipant()
        metadata = self._read_metadata(self._shared_memory_namespace)
        self._fps, self._resolution = metadata.fps, metadata.resolution

        # Overwritten in base class.
        self._setup_framebuffer_reader(self._resolution)

    def _setup_framebuffer_reader(self, resolution: CameraResolutionType) -> None:
        # Create the shared memory reader.
        self._reader = SMReader(
            domain_participant=self._dp,
            topic_name=self._shared_memory_namespace,
            idl_dataclass=RGBFrameBuffer.template(resolution[0], resolution[1]),
        )

        # Initialize an empty frame. Avoids allocating every tick.
        self._last_frame = RGBFrameBuffer.template(resolution[0], resolution[1])

        metadata = self._read_metadata(self._shared_memory_namespace)
        self._fps, self._resolution = metadata.fps, metadata.resolution

    def _read_metadata(self, shared_memory_namespace: str) -> FrameMetadata:
        logger.info(f"Reading metadata from {shared_memory_namespace}_metadata")
        metadata_reader = SMReader(self._dp, f"{shared_memory_namespace}_metadata", FrameMetadata.template())
        try:
            metadata = metadata_reader.reader.read_one(timeout=duration(seconds=10))
            logger.info(f"Camera metadata: {metadata}")
        except StopIteration:
            raise TimeoutError("Timeout while waiting for the metadata message.")
        return metadata

    @property
    def fps(self) -> int:
        """The frames per second of the camera."""
        return self._fps

    @property
    def resolution(self) -> CameraResolutionType:
        return self._resolution

    def get_current_timestamp(self) -> float:
        return self._last_frame.timestamp.item()

    def _grab_images(self) -> None:
        self._last_frame = self._reader()

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        image = self._retrieve_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image).image_in_numpy_format
        return image

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        return self._last_frame.rgb

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self._last_frame.intrinsics


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBPublisher and MultiprocessRGBReceiver.
    You can also use the MultiprocessRGBReceiver in a different process (e.g. in a different python script)
    """
    camera_fps = 15
    namespace = "camera"

    from airo_camera_toolkit.cameras.zed.zed import Zed

    multiprocessing.set_start_method("spawn", force=True)

    publisher = MultiprocessRGBPublisher(
        Zed,
        camera_kwargs={"resolution": Zed.InitParams.RESOLUTION_1080, "fps": camera_fps},
        shared_memory_namespace=namespace,
    )
    publisher.start()

    # The receiver behaves just like a regular RGBCamera
    receiver = MultiprocessRGBReceiver(namespace)

    cv2.namedWindow(namespace, cv2.WINDOW_NORMAL)

    time_current = None
    time_previous = None

    while True:
        time_previous = time_current
        time_current = time.time()

        image_rgb = receiver.get_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        cv2.imshow(namespace, image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

        if time_previous is not None:
            fps = 1 / (time_current - time_previous)

            fps_str = f"{fps:.2f}".rjust(6, " ")
            camera_fps_str = f"{camera_fps:.2f}".rjust(6, " ")
            if fps < 0.9 * camera_fps:
                logger.warning(f"FPS: {fps_str} / {camera_fps_str} (too slow)")
            else:
                logger.debug(f"FPS: {fps_str} / {camera_fps_str}")

    publisher.stop()
    publisher.join()
    cv2.destroyAllWindows()
