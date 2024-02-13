"""code for sharing the data of a camera that implements the RGBDCamera interface between processes using shared memory"""

import time
from multiprocessing import resource_tracker, shared_memory
from typing import Optional

import cv2
import loguru
import numpy as np
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import (
    MultiprocessRGBPublisher,
    MultiprocessRGBReceiver,
    shared_memory_block_like,
)

logger = loguru.logger
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_typing import NumpyDepthMapType, NumpyFloatImageType, NumpyIntImageType

_DEPTH_SHM_NAME = "depth"
_DEPTH_SHAPE_SHM_NAME = "depth_shape"
_DEPTH_IMAGE_SHM_NAME = "depth_image"
_DEPTH_IMAGE_SHAPE_SHM_NAME = "depth_image_shape"
_CONFIDENCE_MAP_SHM_NAME = "confidence_map"
_CONFIDENCE_MAP_SHAPE_SHM_NAME = "confidence_map_shape"


class MultiprocessRGBDPublisher(MultiprocessRGBPublisher):
    """publishes the data of a camera that implements the RGBDCamera interface to shared memory blocks.
    Shared memory blocks can then be accessed in other processes using their names,
    cf. https://docs.python.org/3/library/multiprocessing.shared_memory.html#module-multiprocessing.shared_memory

    The Receiver class is a convenient way of doing so and is the intended way of using this class.
    """

    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        shared_memory_namespace: str = "camera",
    ):
        super().__init__(camera_cls, camera_kwargs, shared_memory_namespace)

        self.depth_shm: Optional[shared_memory.SharedMemory] = None
        self.depth_shape_shm: Optional[shared_memory.SharedMemory] = None
        self.depth_image_shm: Optional[shared_memory.SharedMemory] = None
        self.depth_image_shape_shm: Optional[shared_memory.SharedMemory] = None
        self.confidence_map_shm: Optional[shared_memory.SharedMemory] = None
        self.confidence_map_shape_shm: Optional[shared_memory.SharedMemory] = None

    def _setup(self) -> None:
        """in-process creation of camera object and shared memory blocks"""
        super()._setup()
        assert isinstance(self._camera, RGBDCamera)

        depth_name = f"{self._shared_memory_namespace}_{_DEPTH_SHM_NAME}"
        depth_shape_name = f"{self._shared_memory_namespace}_{_DEPTH_SHAPE_SHM_NAME}"
        depth_image_name = f"{self._shared_memory_namespace}_{_DEPTH_IMAGE_SHM_NAME}"
        depth_image_shape_name = f"{self._shared_memory_namespace}_{_DEPTH_IMAGE_SHAPE_SHM_NAME}"
        confidence_map_name = f"{self._shared_memory_namespace}_{_CONFIDENCE_MAP_SHM_NAME}"
        confidence_map_shape_name = f"{self._shared_memory_namespace}_{_CONFIDENCE_MAP_SHAPE_SHM_NAME}"

        depth_map = self._camera.get_depth_map()
        depth_map_shape = np.array([depth_map.shape])
        depth_image = self._camera._retrieve_depth_image()
        depth_image_shape = np.array([depth_image.shape])
        confidence_map = self._camera._retrieve_confidence_map()  # TODO this is not an interface function yet.
        confidence_map_shape = np.array([confidence_map.shape])

        logger.info("Creating depth shared memory blocks.")
        self.depth_shm, self.depth_shm_array = shared_memory_block_like(depth_map, depth_name)
        self.depth_shape_shm, self.depth_shape_shm_array = shared_memory_block_like(depth_map_shape, depth_shape_name)
        self.depth_image_shm, self.depth_image_shm_array = shared_memory_block_like(depth_image, depth_image_name)
        (
            self.depth_image_shape_shm,
            self.depth_image_shape_shm_array,
        ) = shared_memory_block_like(depth_image_shape, depth_image_shape_name)
        self.confidence_map_shm, self.confidence_map_shm_array = shared_memory_block_like(
            confidence_map, confidence_map_name
        )
        self.confidence_map_shape_shm, self.confidence_map_shape_shm_array = shared_memory_block_like(
            confidence_map_shape, confidence_map_shape_name
        )
        logger.info("Created depth shared memory blocks.")

    def stop(self) -> None:
        self.shutdown_event.set()

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""

        logger.info(f"{self.__class__.__name__} process started.")
        self._setup()
        assert isinstance(self._camera, RGBDCamera)  # For mypy
        logger.info(f'{self.__class__.__name__} starting to publish to "{self._shared_memory_namespace}".')

        try:
            while not self.shutdown_event.is_set():
                image = self._camera.get_rgb_image_as_int()
                depth_map = self._camera._retrieve_depth_map()
                depth_image = self._camera._retrieve_depth_image()
                confidence_map = self._camera._retrieve_confidence_map()
                self.rgb_shm_array[:] = image[:]
                self.depth_shm_array[:] = depth_map[:]
                self.depth_image_shm_array[:] = depth_image[:]
                self.confidence_map_shm_array[:] = confidence_map[:]
                self.timestamp_shm_array[:] = np.array([time.time()])[:]
                self.running_event.set()
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
        finally:
            self.unlink_shared_memory()
            logger.info(f"{self.__class__.__name__} process terminated.")

    def unlink_shared_memory(self) -> None:
        """unlink the shared memory blocks so that they are deleted when the process is terminated"""
        super().unlink_shared_memory()
        print(f"Unlinking depth shared memory blocks of {self.__class__.__name__}")

        if self.depth_shm is not None:
            self.depth_shm.close()
            self.depth_shm.unlink()
            self.depth_shm = None

        if self.depth_shape_shm is not None:
            self.depth_shape_shm.close()
            self.depth_shape_shm.unlink()
            self.depth_shape_shm = None

        if self.depth_image_shm is not None:
            self.depth_image_shm.close()
            self.depth_image_shm.unlink()
            self.depth_image_shm = None

        if self.depth_image_shape_shm is not None:
            self.depth_image_shape_shm.close()
            self.depth_image_shape_shm.unlink()
            self.depth_image_shape_shm = None

        if self.confidence_map_shm is not None:
            self.confidence_map_shm.close()
            self.confidence_map_shm.unlink()
            self.confidence_map_shm = None

        if self.confidence_map_shape_shm is not None:
            self.confidence_map_shape_shm.close()
            self.confidence_map_shape_shm.unlink()
            self.confidence_map_shape_shm = None

    def __del__(self) -> None:
        self.unlink_shared_memory()


class MultiprocessRGBDReceiver(MultiprocessRGBReceiver, RGBDCamera):
    """Implements the RGBD camera interface for a camera that is running in a different process and shares its data using shared memory blocks.
    To be used with the Publisher class.
    """

    def __init__(self, shared_memory_namespace: str) -> None:
        super().__init__(shared_memory_namespace)

        depth_name = f"{self._shared_memory_namespace}_{_DEPTH_SHM_NAME}"
        depth_shape_name = f"{self._shared_memory_namespace}_{_DEPTH_SHAPE_SHM_NAME}"
        depth_image_name = f"{self._shared_memory_namespace}_{_DEPTH_IMAGE_SHM_NAME}"
        depth_image_shape_name = f"{self._shared_memory_namespace}_{_DEPTH_IMAGE_SHAPE_SHM_NAME}"
        confidence_map_name = f"{self._shared_memory_namespace}_{_CONFIDENCE_MAP_SHM_NAME}"
        confidence_map_shape_name = f"{self._shared_memory_namespace}_{_CONFIDENCE_MAP_SHAPE_SHM_NAME}"

        self.depth_shm = shared_memory.SharedMemory(name=depth_name)
        self.depth_shape_shm = shared_memory.SharedMemory(name=depth_shape_name)
        self.depth_image_shm = shared_memory.SharedMemory(name=depth_image_name)
        self.depth_image_shape_shm = shared_memory.SharedMemory(name=depth_image_shape_name)
        self.confidence_map_shm = shared_memory.SharedMemory(name=confidence_map_name)
        self.confidence_map_shape_shm = shared_memory.SharedMemory(name=confidence_map_shape_name)

        # Same comment as in base class:
        resource_tracker.unregister(self.depth_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.depth_shape_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.depth_image_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.depth_image_shape_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.confidence_map_shm._name, "shared_memory")  # type: ignore[attr-defined)
        resource_tracker.unregister(self.confidence_map_shape_shm._name, "shared_memory")  # type: ignore[attr-defined)

        self.depth_shape_shm_array: np.ndarray = np.ndarray((2,), dtype=np.int64, buffer=self.depth_shape_shm.buf)
        self.depth_image_shape_shm_array: np.ndarray = np.ndarray(
            (3,), dtype=np.int64, buffer=self.depth_image_shape_shm.buf
        )
        self.confidence_map_shape_shm_array: np.ndarray = np.ndarray(
            (2,), dtype=np.int64, buffer=self.confidence_map_shape_shm.buf
        )

        depth_shape = tuple(self.depth_shape_shm_array[:])
        depth_image_shape = tuple(self.depth_image_shape_shm_array[:])
        confidence_map_shape = tuple(self.confidence_map_shape_shm_array[:])

        self.depth_shm_array: np.ndarray = np.ndarray(depth_shape, dtype=np.float32, buffer=self.depth_shm.buf)
        self.depth_image_shm_array: np.ndarray = np.ndarray(
            depth_image_shape, dtype=np.uint8, buffer=self.depth_image_shm.buf
        )
        self.confidence_map_shm_array: np.ndarray = np.ndarray(
            confidence_map_shape, dtype=np.float32, buffer=self.confidence_map_shm.buf
        )

    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        return self.depth_shm_array

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        return self.depth_image_shm_array

    def _retrieve_confidence_map(self) -> NumpyFloatImageType:
        return self.confidence_map_shm_array

    def _close_shared_memory(self) -> None:
        """Closing shared memory signal that"""
        super()._close_shared_memory()
        print(f"Closing depth shared memory blocks of {self.__class__.__name__}")

        if self.depth_shm is not None:
            self.depth_shm.close()
            self.depth_shm = None

        if self.depth_shape_shm is not None:
            self.depth_shape_shm.close()
            self.depth_shape_shm = None

        if self.depth_image_shm is not None:
            self.depth_image_shm.close()
            self.depth_image_shm = None

        if self.depth_image_shape_shm is not None:
            self.depth_image_shape_shm.close()
            self.depth_image_shape_shm = None

        if self.confidence_map_shm is not None:
            self.confidence_map_shm.close()
            self.confidence_map_shm = None

        if self.confidence_map_shape_shm is not None:
            self.confidence_map_shape_shm.close()
            self.confidence_map_shape_shm = None

    def __del__(self) -> None:
        super().__del__()
        self._close_shared_memory()


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBDPublisher and MultiprocessRGBDReceiver.
    You can also use the MultiprocessRGBDReceiver in a different process (e.g. in a different python script)
    """

    import multiprocessing

    from airo_camera_toolkit.cameras.zed.zed2i import Zed2i

    multiprocessing.set_start_method("spawn")

    resolution = Zed2i.RESOLUTION_720

    publisher = MultiprocessRGBDPublisher(
        Zed2i,
        camera_kwargs={
            "resolution": resolution,
            "fps": 30,
            "depth_mode": Zed2i.NEURAL_DEPTH_MODE,
        },
    )

    publisher.start()
    receiver = MultiprocessRGBDReceiver("camera")

    cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Confidence Map", cv2.WINDOW_NORMAL)

    while True:
        logger.info("Getting image")
        depth_map = receiver.get_depth_map()
        depth_image = receiver._retrieve_depth_image()
        confidence_map = receiver._retrieve_confidence_map()
        # point_cloud = receiver._retrieve_colored_point_cloud()

        cv2.imshow("Depth Map", depth_map)
        cv2.imshow("Depth Image", depth_image)
        cv2.imshow("Confidence Map", confidence_map)

        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    receiver._close_shared_memory()
    publisher.stop()
    publisher.join()
