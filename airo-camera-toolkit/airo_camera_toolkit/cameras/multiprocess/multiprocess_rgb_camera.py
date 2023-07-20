"""code for sharing the data of a camera that implements the RGBDCamera interface between processes using shared memory"""
import multiprocessing
import time
from multiprocessing import Process, resource_tracker, shared_memory
from typing import Optional

import loguru
import numpy as np
from airo_camera_toolkit.utils import ImageConverter

logger = loguru.logger
import cv2
from airo_camera_toolkit.interfaces import RGBCamera
from airo_typing import CameraIntrinsicsMatrixType, NumpyFloatImageType

_RGB_SHM_NAME = "rgb"
_RGB_TIMESTAMP_SHM_NAME = "rgb_timestamp"
_INTRINSICS_SHM_NAME = "intrinsics"


class MultiprocessRGBPublisher(Process):
    """publishes the data of a camera that implements the RGBCamera interface to shared memory blocks.
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
        super().__init__(daemon=True)
        self._shared_memory_namespace = shared_memory_namespace
        self._camera_cls = camera_cls
        self._camera_kwargs = camera_kwargs
        self._camera = None
        self.shutdown_event = multiprocessing.Event()

        self.rgb_shm: Optional[shared_memory.SharedMemory] = None
        self.intrinsics_shm: Optional[shared_memory.SharedMemory] = None

    def _setup(self) -> None:
        """in-process creation of camera object and shared memory blocks"""
        # have to create camera here to make sure the shared memory blocks in the camera are created in the same process

        # Opening cameras fails often so retry a few times
        # for _ in range(3):
        #     try:
        self._camera = self._camera_cls(**self._camera_kwargs)
        assert isinstance(self._camera, RGBCamera)
        # except Exception as e:
        #     print(f"Camera {self._shared_memory_namespace} could not be instantiated, waiting 10 seconds and retrying, expection was:")
        #     print(e)
        #     time.sleep(10)

        # create shared memory blocks here so that we can use the actual image sizes and don't need to pass resolutions in the constructor
        dummy_rgb_image = self._camera.get_rgb_image()
        dummy_intrinsics = self._camera.intrinsics_matrix()

        self.rgb_shm = shared_memory.SharedMemory(
            create=True,
            size=dummy_rgb_image.nbytes,
            name=f"{self._shared_memory_namespace}_{_RGB_SHM_NAME}",
        )
        self.rgb_shm_array: np.ndarray = np.ndarray(
            dummy_rgb_image.shape, dtype=dummy_rgb_image.dtype, buffer=self.rgb_shm.buf
        )

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

        self.intrinsics_shm_array: np.ndarray = np.ndarray(
            dummy_intrinsics.shape,
            dtype=dummy_intrinsics.dtype,
            buffer=self.intrinsics_shm.buf,
        )

    def stop_publishing(self) -> None:
        self.shutdown_event.set()

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        self._setup()
        assert isinstance(self._camera, RGBCamera)

        # only write intrinsics once, these do not change.
        self.intrinsics_shm_array[:] = self._camera.intrinsics_matrix()[:]

        while not self.shutdown_event.is_set():
            # TODO: use Lock to make sure that the data is not overwritten while it is being read and avoid tearing
            img = self._camera.get_rgb_image()
            self.rgb_shm_array[:] = img[:]
            self.rgb_timestamp_shm.buf[:] = np.array([time.time()]).tobytes()

        self.unlink_shared_memory()

    def unlink_shared_memory(self) -> None:
        """unlink the shared memory blocks so that they are deleted when the process is terminated"""
        assert self.rgb_shm is not None
        assert self.intrinsics_shm is not None

        self.rgb_shm.close()
        self.intrinsics_shm.close()
        self.rgb_shm.unlink()
        self.intrinsics_shm.unlink()
        # self.depth_image_shm.unlink()


class MultiprocessRGBReceiver(RGBCamera):
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
        for _ in range(20):
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

        resource_tracker.unregister(self.rgb_shm._name, "shared_memory")
        resource_tracker.unregister(self.intrinsics_shm._name, "shared_memory")
        resource_tracker.unregister(self.rgb_timestamp_shm._name, "shared_memory")

        # create numpy arrays that are backed by the shared memory blocks
        self.rgb_shm_array: np.ndarray = np.ndarray(
            (camera_resolution_height, camera_resolution_width, 3),
            dtype=np.float32,
            buffer=self.rgb_shm.buf,
        )

        self.intrinsics_shm_array: np.ndarray = np.ndarray((3, 3), dtype=np.float64, buffer=self.intrinsics_shm.buf)

    def get_rgb_image_timestamp(self) -> float:
        return np.frombuffer(self.rgb_timestamp_shm.buf, dtype=np.float64)[0]

    # TODO: use locks to make sure that the data is not overwritten while it is being read and avoid tearing
    # https://docs.python.org/3/library/multiprocessing.html#synchronization-between-processes
    def get_rgb_image(self) -> NumpyFloatImageType:
        return self.rgb_shm_array

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self.intrinsics_shm_array

    def _close_shared_memory(self) -> None:
        """Closing shared memory, to signal that they are no longer need by this object."""
        self.rgb_shm.close()
        self.rgb_timestamp_shm.close()
        self.intrinsics_shm.close()

        # Normally, we wouldn't have to do this unregistering. However, without it, the resource tracker incorrectly
        # destroys access to the shared memory blocks when the process is terminated. This is a known 3 year old bug
        # that hasn't been resolved yet: https://bugs.python.org/issue39959
        # Concretely, the problem was that once any MultiprocessRGBReceiver object was destroyed, all further access to
        # the shared memory blocks would fail with a FileNotFoundError.
        # resource_tracker.unregister(self.rgb_shm._name, "shared_memory")
        # resource_tracker.unregister(self.intrinsics_shm._name, "shared_memory")
        # resource_tracker.unregister(self.rgb_timestamp_shm._name, "shared_memory")

    def stop_receiving(self) -> None:
        self._close_shared_memory()

    def __del__(self) -> None:
        self._close_shared_memory()


class MultiprocessRGBRerunLogger(Process):
    def __init__(
        self,
        shared_memory_namespace: str,
        camera_resolution_width: int,
        camera_resolution_height: int,
        rotation_degrees_clockwise: int = 0,
        save_images_to_disk: bool = False,
    ):
        super().__init__(daemon=True)
        self._shared_memory_namespace = shared_memory_namespace
        self._camera_resolution_width = camera_resolution_width
        self._camera_resolution_height = camera_resolution_height
        self.save_images_to_disk = save_images_to_disk
        self.shutdown_event = multiprocessing.Event()

        if rotation_degrees_clockwise is None:
            rotation_degrees_clockwise = 0

        remainder = rotation_degrees_clockwise % 90
        if remainder != 0:
            raise ValueError("rotation_degrees must be a multiple of 90")
        # Divide by 90, and negate because numpy rotates counter-clockwise
        self._numpy_rot90_k = -rotation_degrees_clockwise // 90

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""
        import rerun

        print("logger authkey:")
        print(multiprocessing.current_process().authkey)

        print("Connecting to rerun")
        # rerun.init("Realsense test")
        rerun.init("rerun")
        rerun.connect()
        # print(rerun.get_global_data_recording())

        self.multiprocessRGBReceiver = MultiprocessRGBReceiver(
            self._shared_memory_namespace,
            self._camera_resolution_width,
            self._camera_resolution_height,
        )

        previous_timestamp = time.time()

        while not self.shutdown_event.is_set():
            timestamp = self.multiprocessRGBReceiver.get_rgb_image_timestamp()
            if timestamp <= previous_timestamp:
                time.sleep(0.001)  # Check every millisecond
                continue

            image = self.multiprocessRGBReceiver.get_rgb_image()
            if self._numpy_rot90_k != 0:
                image = np.rot90(image, self._numpy_rot90_k)

            # Float to int conversion for faster logging
            try:
                image_bgr = ImageConverter.from_numpy_format(image).image_in_opencv_format
            except Exception as e:
                print(e)
                continue

            print("Logging image")
            image_rgb = image_bgr[:, :, ::-1]
            rerun.log_image(self._shared_memory_namespace, image_rgb)
            print("Logging succeeded")

            if self.save_images_to_disk:
                # write image to disk:
                image_opencv = ImageConverter.from_numpy_format(image).image_in_opencv_format
                # format the timestamp
                timestamp_str = str(timestamp).replace(".", "_")
                cv2.imwrite(f"{self._shared_memory_namespace}_{timestamp_str}.png", image_opencv)

            previous_timestamp = timestamp

        self.multiprocessRGBReceiver.stop_receiving()

    def stop_logging(self) -> None:
        self.shutdown_event.set()


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBDPublisher and MultiprocessRGBDReceiver.
    You can also use the MultiprocessRGBDReceiver in a different process (e.g. in a different python script)
    """
    from airo_camera_toolkit.cameras.zed2i import Zed2i

    resolution_identifier = Zed2i.RESOLUTION_1080
    resolution = Zed2i.resolution_sizes[resolution_identifier]

    p = MultiprocessRGBPublisher(
        Zed2i,
        camera_kwargs={
            "resolution": resolution_identifier,
            "fps": 30,
            "depth_mode": Zed2i.NONE_DEPTH_MODE,
        },
    )
    p.start()
    receiver = MultiprocessRGBReceiver("camera", *resolution)

    while True:
        logger.info("Getting image")
        image = receiver.get_rgb_image()
        cv2.imshow("RGB Image", image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
