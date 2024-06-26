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
from airo_typing import NumpyDepthMapType, NumpyIntImageType

_DEPTH_SHM_NAME = "depth"
_DEPTH__SHAPE_SHM_NAME = "depth_shape"
_DEPTH_IMAGE_SHM_NAME = "depth_image"
_DEPTH_IMAGE_SHAPE_SHM_NAME = "depth_image_shape"


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
        self.depth_image_shm: Optional[shared_memory.SharedMemory] = None

    def _setup(self) -> None:
        """in-process creation of camera object and shared memory blocks"""
        super()._setup()
        assert isinstance(self._camera, RGBDCamera)

        depth_name = f"{self._shared_memory_namespace}_{_DEPTH_SHM_NAME}"
        depth_shape_name = f"{self._shared_memory_namespace}_{_DEPTH__SHAPE_SHM_NAME}"
        depth_image_name = f"{self._shared_memory_namespace}_{_DEPTH_IMAGE_SHM_NAME}"
        depth_image_shape_name = f"{self._shared_memory_namespace}_{_DEPTH_IMAGE_SHAPE_SHM_NAME}"

        depth_map = self._camera.get_depth_map()
        depth_map_shape = np.array([depth_map.shape])
        depth_image = self._camera.get_depth_image()
        depth_image_shape = np.array([depth_image.shape])

        self.depth_shm, self.depth_shm_array = shared_memory_block_like(depth_map, depth_name)
        self.depth_shape_shm, self.depth_shape_shm_array = shared_memory_block_like(depth_map_shape, depth_shape_name)
        self.depth_image_shm, self.depth_image_shm_array = shared_memory_block_like(depth_image, depth_image_name)
        (
            self.depth_image_shape_shm,
            self.depth_image_shape_shm_array,
        ) = shared_memory_block_like(depth_image_shape, depth_image_shape_name)

    def stop(self) -> None:
        self.shutdown_event.set()

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        self._setup()
        assert isinstance(self._camera, RGBDCamera)  # For mypy

        while not self.shutdown_event.is_set():
            image = self._camera.get_rgb_image_as_int()
            depth_map = self._camera._retrieve_depth_map()
            depth_image = self._camera._retrieve_depth_image()
            self.rgb_shm_array[:] = image[:]
            self.depth_shm_array[:] = depth_map[:]
            self.depth_image_shm_array[:] = depth_image[:]
            self.timestamp_shm_array[:] = np.array([time.time()])[:]

        self.unlink_shared_memory()

    def unlink_shared_memory(self) -> None:
        """unlink the shared memory blocks so that they are deleted when the process is terminated"""
        super().unlink_shared_memory()

        assert isinstance(self.depth_shm, shared_memory.SharedMemory)
        assert isinstance(self.depth_image_shm, shared_memory.SharedMemory)

        self.depth_shm.close()
        self.depth_image_shm.close()
        self.depth_shm.unlink()
        self.depth_image_shm.unlink()


class MultiprocessRGBDReceiver(MultiprocessRGBReceiver, RGBDCamera):
    """Implements the RGBD camera interface for a camera that is running in a different process and shares its data using shared memory blocks.
    To be used with the Publisher class.
    """

    def __init__(self, shared_memory_namespace: str) -> None:
        super().__init__(shared_memory_namespace)

        depth_name = f"{self._shared_memory_namespace}_{_DEPTH_SHM_NAME}"
        depth_shape_name = f"{self._shared_memory_namespace}_{_DEPTH__SHAPE_SHM_NAME}"
        depth_image_name = f"{self._shared_memory_namespace}_{_DEPTH_IMAGE_SHM_NAME}"
        depth_image_shape_name = f"{self._shared_memory_namespace}_{_DEPTH_IMAGE_SHAPE_SHM_NAME}"

        # Attach to existing shared memory blocks. Retry a few times to give the publisher time to start up (opening
        # connection to a camera can take a while).
        is_shm_found = False
        for i in range(10):
            try:
                self.depth_shm = shared_memory.SharedMemory(name=depth_name)
                self.depth_shape_shm = shared_memory.SharedMemory(name=depth_shape_name)
                self.depth_image_shm = shared_memory.SharedMemory(name=depth_image_name)
                self.depth_image_shape_shm = shared_memory.SharedMemory(name=depth_image_shape_name)
                is_shm_found = True
                break
            except FileNotFoundError:
                logger.debug(
                    f'SharedMemory namespace "{self._shared_memory_namespace}" not found yet, retrying in .5 seconds.'
                )
                time.sleep(0.5)

        if not is_shm_found:
            raise FileNotFoundError("Shared memory not found.")

        # Same comment as in base class:
        resource_tracker.unregister(self.depth_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.depth_shape_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.depth_image_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.depth_image_shape_shm._name, "shared_memory")  # type: ignore[attr-defined]

        self.depth_shape_shm_array: np.ndarray = np.ndarray((2,), dtype=np.int64, buffer=self.depth_shape_shm.buf)
        self.depth_image_shape_shm_array: np.ndarray = np.ndarray(
            (3,), dtype=np.int64, buffer=self.depth_image_shape_shm.buf
        )

        depth_shape = tuple(self.depth_shape_shm_array[:])
        depth_image_shape = tuple(self.depth_image_shape_shm_array[:])

        self.depth_shm_array: np.ndarray = np.ndarray(depth_shape, dtype=np.float32, buffer=self.depth_shm.buf)
        self.depth_image_shm_array: np.ndarray = np.ndarray(
            depth_image_shape, dtype=np.uint8, buffer=self.depth_image_shm.buf
        )

        self.previous_depth_map_timestamp = time.time()
        self.previous_depth_image_timestamp = time.time()

    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        return self.depth_shm_array

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        return self.depth_image_shm_array

    def _close_shared_memory(self) -> None:
        """Closing shared memory signal that"""
        super()._close_shared_memory()

        self.depth_shm.close()
        self.depth_image_shm.close()

    def __del__(self) -> None:
        self._close_shared_memory()


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBDPublisher and MultiprocessRGBDReceiver.
    You can also use the MultiprocessRGBDReceiver in a different process (e.g. in a different python script)
    """

    from airo_camera_toolkit.cameras.zed.zed2i import Zed2i

    resolution = Zed2i.RESOLUTION_720

    p = MultiprocessRGBDPublisher(
        Zed2i,
        camera_kwargs={
            "resolution": resolution,
            "fps": 30,
            "depth_mode": Zed2i.NEURAL_DEPTH_MODE,
        },
    )
    p.start()
    receiver = MultiprocessRGBDReceiver("camera")

    while True:
        logger.info("Getting image")
        depth_map = receiver.get_depth_map()
        depth_image = receiver.get_depth_image()
        cv2.imshow("Depth Map", depth_map)
        cv2.imshow("Depth Image", depth_image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
