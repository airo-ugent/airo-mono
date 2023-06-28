"""code for sharing the data of a camera that implements the RGBDCamera interface between processes using shared memory"""
import time
from multiprocessing import resource_tracker, shared_memory
from typing import Optional

import cv2
import loguru
import numpy as np
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import (
    MultiProcessRerunRGBLogger,
    MultiProcessRGBPublisher,
    MultiProcessRGBReceiver,
)
from airo_camera_toolkit.utils import ImageConverter

logger = loguru.logger
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_typing import NumpyDepthMapType, NumpyIntImageType

_DEPTH_SHM_NAME = "depth"
_DEPTH_IMAGE_SHM_NAME = "depth_image"


class MultiProcessRGBDPublisher(MultiProcessRGBPublisher):
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
        # create shared memory blocks here so that we can use the actual image sizes and don't need to pass resolutions in the constructor
        dummy_depth_map = self._camera.get_depth_map()
        dummpy_depth_image = self._camera.get_depth_image()

        self.depth_shm = shared_memory.SharedMemory(
            create=True,
            size=dummy_depth_map.nbytes,
            name=f"{self._shared_memory_namespace}_{_DEPTH_SHM_NAME}",
        )
        self.depth_shm_array: np.ndarray = np.ndarray(
            dummy_depth_map.shape,
            dtype=dummy_depth_map.dtype,
            buffer=self.depth_shm.buf,
        )

        self.depth_image_shm = shared_memory.SharedMemory(
            create=True,
            size=dummpy_depth_image.nbytes,
            name=f"{self._shared_memory_namespace}_{_DEPTH_IMAGE_SHM_NAME}",
        )

        self.depth_image_shm_array: np.ndarray = np.ndarray(
            dummpy_depth_image.shape,
            dtype=dummpy_depth_image.dtype,
            buffer=self.depth_image_shm.buf,
        )

    def stop_publishing(self) -> None:
        self.shutdown_event.set()

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        self._setup()
        assert isinstance(self._camera, RGBDCamera)

        # only write intrinsics once, these do not change.
        self.intrinsics_shm_array[:] = self._camera.intrinsics_matrix()[:]
        while not self.shutdown_event.is_set():
            # TODO: use Lock to make sure that the data is not overwritten while it is being read and avoid tearing
            img = self._camera.get_rgb_image()
            depth = self._camera.get_depth_map()
            depth_image = self._camera.get_depth_image()
            self.rgb_shm_array[:] = img[:]
            self.depth_shm_array[:] = depth[:]
            self.depth_image_shm_array[:] = depth_image[:]
            self.rgb_timestamp_shm.buf[:] = np.array([time.time()]).tobytes()

        self.unlink_shared_memory()

    def unlink_shared_memory(self) -> None:
        """unlink the shared memory blocks so that they are deleted when the process is terminated"""
        super().unlink_shared_memory()
        assert self.depth_shm is not None
        assert self.depth_image_shm is not None
        self.depth_shm.close()
        self.depth_image_shm.close()
        self.depth_shm.unlink()
        self.depth_image_shm.unlink()


class MultiProcessRGBDReceiver(MultiProcessRGBReceiver, RGBDCamera):
    """Implements the RGBD camera interface for a camera that is running in a different process and shares its data using shared memory blocks.
    To be used with the Publisher class.
    """

    def __init__(
        self,
        shared_memory_namespace: str,
        camera_resolution_width: int,
        camera_resolution_height: int,
    ) -> None:
        super().__init__(shared_memory_namespace, camera_resolution_width, camera_resolution_height)

        is_shm_created = False
        for _ in range(3):
            # if the sender process was just started, the shared memory blocks might not be created yet
            # so retry a few times to acces them
            try:
                self.depth_shm = shared_memory.SharedMemory(name=f"{shared_memory_namespace}_{_DEPTH_SHM_NAME}")
                self.depth_image_shm = shared_memory.SharedMemory(
                    name=f"{shared_memory_namespace}_{_DEPTH_IMAGE_SHM_NAME}"
                )

                is_shm_created = True
                break
            except FileNotFoundError as e:
                print(e)
                print("SHM not found, waiting 2 second")
                time.sleep(2)
        if not is_shm_created:
            raise FileNotFoundError("Shared memory not found")

        # Same comment as in base class:
        resource_tracker.unregister(self.depth_shm._name, "shared_memory")
        resource_tracker.unregister(self.depth_image_shm._name, "shared_memory")

        # create numpy arrays that are backed by the shared memory blocks
        self.depth_shm_array: np.ndarray = np.ndarray(
            (camera_resolution_height, camera_resolution_width),
            dtype=np.float32,
            buffer=self.depth_shm.buf,
        )

        self.depth_image_shm_array: np.ndarray = np.ndarray(
            (camera_resolution_height, camera_resolution_width, 3),
            dtype=np.uint8,
            buffer=self.depth_image_shm.buf,
        )

    def get_depth_map(self) -> NumpyDepthMapType:
        return self.depth_shm_array

    def get_depth_image(self) -> NumpyIntImageType:
        return self.depth_image_shm_array

    def _close_shared_memory(self) -> None:
        """Closing shared memory signal that"""
        self.depth_shm.close()
        self.depth_image_shm.close()

        # Same comment as in base class:
        # resource_tracker.unregister(self.depth_shm._name, "shared_memory")
        # resource_tracker.unregister(self.depth_image_shm._name, "shared_memory")

    def stop_receiving(self) -> None:
        super().stop_receiving()
        self._close_shared_memory()


class MultiProcessRerunRGBDLogger(MultiProcessRerunRGBLogger):
    def __init__(
        self,
        shared_memory_namespace: str,
        camera_resolution_width: int,
        camera_resolution_height: int,
        rotation_degrees_clockwise: int = 0,
    ):
        super().__init__(
            shared_memory_namespace,
            camera_resolution_width,
            camera_resolution_height,
            rotation_degrees_clockwise,
        )

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        import rerun

        rerun.connect()
        self.multiProcessRGBDReceiver = MultiProcessRGBDReceiver(
            self._shared_memory_namespace,
            self._camera_resolution_width,
            self._camera_resolution_height,
        )

        previous_timestamp = time.time()

        while not self.shutdown_event.is_set():
            timestamp = self.multiProcessRGBDReceiver.get_rgb_image_timestamp()
            if timestamp <= previous_timestamp:
                time.sleep(0.001)  # Check every millisecond
                continue

            image = self.multiProcessRGBDReceiver.get_rgb_image()
            depth_image = self.multiProcessRGBDReceiver.get_depth_image()

            # Float to int conversion for faster logging
            image_bgr = ImageConverter.from_numpy_format(image).image_in_opencv_format
            image = image_bgr[:, :, ::-1]

            if self._numpy_rot90_k != 0:
                image = np.rot90(image, self._numpy_rot90_k)
                depth_image = np.rot90(depth_image, self._numpy_rot90_k)

            rerun.log_image(self._shared_memory_namespace, image)

            rerun.log_image(f"{self._shared_memory_namespace}_depth", depth_image)
            previous_timestamp = timestamp

        self.multiProcessRGBDReceiver.stop_receiving()


if __name__ == "__main__":
    """example of how to use the MultiProcessRGBDPublisher and MultiProcessRGBDReceiver.
    You can also use the MultiProcessRGBDReceiver in a different process (e.g. in a different python script)
    """

    from airo_camera_toolkit.cameras.zed2i import Zed2i

    resolution_identifier = Zed2i.RESOLUTION_720
    resolution = Zed2i.resolution_sizes[resolution_identifier]

    p = MultiProcessRGBDPublisher(
        Zed2i,
        camera_kwargs={
            "resolution": resolution_identifier,
            "fps": 30,
            "depth_mode": Zed2i.NEURAL_DEPTH_MODE,
        },
    )
    p.start()
    receiver = MultiProcessRGBDReceiver("camera", *resolution)

    while True:
        logger.info("Getting image")
        depth_map = receiver.get_depth_map()
        depth_image = receiver.get_depth_image()
        cv2.imshow("Dpeth Map", depth_map)
        cv2.imshow("Depth Image", depth_image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
