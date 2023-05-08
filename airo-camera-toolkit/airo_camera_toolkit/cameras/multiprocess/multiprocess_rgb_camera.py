"""code for sharing the data of a camera that implements the RGBDCamera interface between processes using shared memory"""
import multiprocessing
import time
from multiprocessing import Process, shared_memory

import loguru
import numpy as np

logger = loguru.logger
from airo_camera_toolkit.interfaces import RGBCamera
from airo_typing import CameraIntrinsicsMatrixType, NumpyFloatImageType

_RGB_SHM_NAME = "rgb"
_RGB_TIMESTAMP_SHM_NAME = "rgb_timestamp"
_INTRINSICS_SHM_NAME = "intrinsics"


class MultiProcessRGBPublisher(Process):
    """publishes the data of a camera that implements the RGBCamera interface to shared memory blocks.
    Shared memory blocks can then be accessed in other processes using their names,
    cf. https://docs.python.org/3/library/multiprocessing.shared_memory.html#module-multiprocessing.shared_memory

    The Receiver class is a convenient way of doing so and is the intended way of using this class.
    """

    def __init__(
        self,
        camera_cls: type,
        camera_kwargs={},
        shared_memory_namespace: str = "camera",
    ):
        super().__init__(daemon=True)
        self._shared_memory_namespace = shared_memory_namespace
        self._camera_cls = camera_cls
        self._camera_kwargs = camera_kwargs
        self._camera = None
        self.shutdown_event = multiprocessing.Event()

    def _setup(self):
        """in-process creation of camera object and shared memory blocks"""

        # have to create camera here to make sure the shared memory blocks in the camera are created in the same process
        self._camera = self._camera_cls(**self._camera_kwargs)

        # create shared memory blocks here so that we can use the actual image sizes and don't need to pass resolutions in the constructor
        dummy_rgb_image = self._camera.get_rgb_image()
        dummy_intrinsics = self._camera.intrinsics_matrix()

        self.rgb_shm = shared_memory.SharedMemory(
            create=True,
            size=dummy_rgb_image.nbytes,
            name=f"{self._shared_memory_namespace}_{_RGB_SHM_NAME}",
        )
        self.rgb_shm_array = np.ndarray(dummy_rgb_image.shape, dtype=dummy_rgb_image.dtype, buffer=self.rgb_shm.buf)

        self.rgb_timestamp_shm = shared_memory.SharedMemory(
            create=True,
            size=8,
            name=f"{self._shared_memory_namespace}_{_RGB_TIMESTAMP_SHM_NAME}",
        )

        self.intrinsics_shm = shared_memory.SharedMemory(
            create=True,
            size=dummy_intrinsics.nbytes,
            name=f"{self._shared_memory_namespace}_{_INTRINSICS_SHM_NAME}",
        )
        self.intrinsics_shm_array = np.ndarray(
            dummy_intrinsics.shape,
            dtype=dummy_intrinsics.dtype,
            buffer=self.intrinsics_shm.buf,
        )

    def stop_publishing(self):
        self.shutdown_event.set()

    def run(self):
        """main loop of the process, runs until the process is terminated"""
        self._setup()
        assert isinstance(self._camera, RGBCamera)

        # only write intrinsics once, these do not change.
        self.intrinsics_shm_array[:] = self._camera.intrinsics_matrix()[:]

        time.time()

        while not self.shutdown_event.is_set():
            # TODO: use Lock to make sure that the data is not overwritten while it is being read and avoid tearing
            img = self._camera.get_rgb_image()
            self.rgb_shm_array[:] = img[:]
            self.rgb_timestamp_shm.buf[:] = np.array([time.time()]).tobytes()

            # print(1.0 / (time.time() - prev_time))
            time.time()

        self.unlink_shared_memory()

    def unlink_shared_memory(self):
        """unlink the shared memory blocks so that they are deleted when the process is terminated"""
        print("Unlinking shared memory")
        self.rgb_shm.close()
        self.intrinsics_shm.close()
        self.rgb_shm.unlink()
        self.intrinsics_shm.unlink()
        self.depth_image_shm.unlink()


class MultiProcessRGBReceiver(RGBCamera):
    """Implements the RGBD camera interface for a camera that is running in a different process and shares its data using shared memory blocks.
    To be used with the Publisher class.
    """

    def __init__(
        self,
        shared_memory_namespace: str,
        camera_resolution_width: int,
        camera_resolution_height: int,
    ) -> None:
        super().__init__()

        is_shm_created = False
        for _ in range(10):
            # if the sender process was just started, the shared memory blocks might not be created yet
            # so retry a few times to access them
            try:
                self.rgb_shm = shared_memory.SharedMemory(name=f"{shared_memory_namespace}_{_RGB_SHM_NAME}")
                self.rgb_timestamp_shm = shared_memory.SharedMemory(
                    name=f"{shared_memory_namespace}_{_RGB_TIMESTAMP_SHM_NAME}"
                )
                self.intrinsics_shm = shared_memory.SharedMemory(
                    name=f"{shared_memory_namespace}_{_INTRINSICS_SHM_NAME}"
                )
                is_shm_created = True
                break
            except FileNotFoundError as e:
                print(e)
                print("SharedMemory not found yet, waiting 2 second and retrying.")
                time.sleep(2)
        if not is_shm_created:
            raise FileNotFoundError("Shared memory not found.")

        # create numpy arrays that are backed by the shared memory blocks
        self.rgb_shm_array = np.ndarray(
            (camera_resolution_height, camera_resolution_width, 3),
            dtype=np.float32,
            buffer=self.rgb_shm.buf,
        )

        self.intrinsics_shm_array = np.ndarray((3, 3), dtype=np.float32, buffer=self.intrinsics_shm.buf)

    def get_rgb_image_timestamp(self) -> float:
        return np.frombuffer(self.rgb_timestamp_shm.buf, dtype=np.float64)[0]

    # TODO: use locks to make sure that the data is not overwritten while it is being read and avoid tearing
    # https://docs.python.org/3/library/multiprocessing.html#synchronization-between-processes
    def get_rgb_image(self) -> NumpyFloatImageType:
        return self.rgb_shm_array

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self.intrinsics_shm_array

    def _close_shared_memory(self):
        """Closing shared memory signal that"""
        self.rgb_shm.close()
        self.rgb_timestamp_shm.close()
        self.intrinsics_shm.close()

    def stop_receiving(self):
        self._close_shared_memory()

    def __del__(self):
        self._close_shared_memory()


class MultiProcessRGBLogger(Process):
    def __init__(
        self,
        shared_memory_namespace: str,
        camera_resolution_width: int,
        camera_resolution_height: int,
    ):
        super().__init__(daemon=True)
        self._shared_memory_namespace = shared_memory_namespace
        self._camera_resolution_width = camera_resolution_width
        self._camera_resolution_height = camera_resolution_height
        self.shutdown_event = multiprocessing.Event()

    def run(self):
        """main loop of the process, runs until the process is terminated"""
        import rerun

        rerun.connect()
        self.multiProcessRGBReceiver = MultiProcessRGBReceiver(
            self._shared_memory_namespace,
            self._camera_resolution_width,
            self._camera_resolution_height,
        )

        previous_timestamp = time.time()

        while not self.shutdown_event.is_set():
            timestamp = self.multiProcessRGBReceiver.get_rgb_image_timestamp()
            if timestamp > previous_timestamp:
                image = self.multiProcessRGBReceiver.get_rgb_image()
                rerun.log_image(self._shared_memory_namespace, image)
                previous_timestamp = timestamp
            time.sleep(0.001)  # Check every millisecond

        self.multiProcessRGBReceiver.stop_receiving()

    def stop_logging(self):
        self.shutdown_event.set()


if __name__ == "__main__":
    """example of how to use the MultiProcessRGBDPublisher and MultiProcessRGBDReceiver.
    You can also use the MultiProcessRGBDReceiver in a different process (e.g. in a different python script)
    """

    import cv2
    from airo_camera_toolkit.cameras.zed2i import Zed2i

    p = MultiProcessRGBPublisher(
        Zed2i,
        camera_kwargs={
            "resolution": Zed2i.RESOLUTION_1080,
            "fps": 30,
            "depth_mode": Zed2i.NONE_DEPTH_MODE,
        },
    )
    p.start()
    receiver = MultiProcessRGBReceiver("camera", 1920, 1080)

    while True:
        logger.info("Getting image")
        img = receiver.get_rgb_image()
        cv2.imshow("test", img)
        cv2.waitKey(10)
