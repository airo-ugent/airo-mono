"""Publisher and receiver classes for multiprocess camera sharing."""

import multiprocessing
import time
from multiprocessing import resource_tracker, shared_memory
from typing import Optional, Tuple

import cv2
import numpy as np
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import CameraIntrinsicsMatrixType, CameraResolutionType, NumpyFloatImageType, NumpyIntImageType
from loguru import logger

_RGB_SHM_NAME = "rgb"
_RGB_SHAPE_SHM_NAME = "rgb_shape"
_TIMESTAMP_SHM_NAME = "timestamp"
_INTRINSICS_SHM_NAME = "intrinsics"
_FPS_SHM_NAME = "fps"


def shared_memory_block_like(array: np.ndarray, name: str) -> Tuple[shared_memory.SharedMemory, np.ndarray]:
    """Creates a shared memory block with the same shape and dtype as the given array. Additionally, the shared memory
    is initialized with the values of the given array (this convenient for data that won't change).

    Args:
        array: The array that will be used to determine the size of the shared memory block, in accordance with its shape and dtype.
        name: The name of the shared memory block.

    Returns:
        The created shared memory block and a new array that is backed by the shared memory block.
    """
    try:
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes, name=name)
    except FileExistsError:
        logger.warning(f"Shared memory block with name {name} already exists, reusing it.")
        # If we close() and unlink() here, receivers that are already accessing it freeze.
        # So we try to reuse the existing shared memory block.
        # However, if it is too small, the array creation will throw an error.
        shm = shared_memory.SharedMemory(create=False, size=array.nbytes, name=name)
    shm_array: np.ndarray = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shm_array[:] = array[:]
    return shm, shm_array


# TODO: figure out why inheriting from SpawnProcess does not solve the issue.
class MultiprocessRGBPublisher(multiprocessing.context.SpawnProcess):
    """Publishes the data of a camera that implements the RGBCamera interface to shared memory blocks.
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
        """Instantiates the publisher. Note that the publisher (and its process) will not start until start() is called.

        Args:
            camera_cls (type): The class e.g. Zed2i that this publisher will instantiate.
            camera_kwargs (dict, optional): The kwargs that will be passed to the camera_cls constructor.
            shared_memory_namespace (str, optional): The string that will be used to prefix the shared memory blocks that this class will create.
        """

        # context = multiprocessing.get_context("spawn")  # Default "fork" leads to CUDA issues.
        # Why it is a good idea in general to spawn: https://pythonspeed.com/articles/python-multiprocessing/

        super().__init__(daemon=True)
        self._shared_memory_namespace = shared_memory_namespace
        self._camera_cls = camera_cls
        self._camera_kwargs = camera_kwargs
        self._camera = None
        self.running_event = multiprocessing.Event()
        self.shutdown_event = multiprocessing.Event()

        # Declare these here so mypy doesn't complain.
        self.rgb_shm: Optional[shared_memory.SharedMemory] = None
        self.rgb_shape_shm: Optional[shared_memory.SharedMemory] = None
        self.timestamp_shm: Optional[shared_memory.SharedMemory] = None
        self.intrinsics_shm: Optional[shared_memory.SharedMemory] = None
        self.fps_shm: Optional[shared_memory.SharedMemory] = None

    def start(self) -> None:
        """Starts the process. The process will not start until this method is called."""
        super().start()
        self.running_event.wait()  # Block until the publisher has started

    def _setup(self) -> None:
        """Note: to be able to retrieve camera image from the Publisher process, the camera must be instantiated in the
        Publisher process. For this reason, we do not instantiate the camera in __init__ but, here instead.

        We also create the shared memory blocks here, so that their lifetime is bound to the lifetime of the Publisher
        process. Usually shared memory may outlive its creator process, but as the Publisher is the only process that
        writes to the shared memory blocks, we want to make sure that they are deleted when the Publisher process is
        terminated. This also frees up the names of the shared memory blocks so that they can be reused.


        Five SharedMemory blocks are created, each block is prefixed with the namespace of the publisher. Three of these
        are only written once, the other two are written continuously.

        Constant blocks:
        * intrinsics: the intrinsics matrix of the camera
        * rgb_shape: the shape that rgb image array should be
        * fps: the fps of the camera

        Blocks that are written continuously:
        * rgb: the most recently retrieved image
        * timestamp: the timestamp of that image


        To simplify access, we create numpy arrays that are backed by the shared memory blocks for the rgb image and
        the intrinsics matrix.
        """

        # Instantiating a camera.
        logger.info(f"Instantiating a {self._camera_cls.__name__} camera.")
        self._camera = self._camera_cls(**self._camera_kwargs)
        assert isinstance(self._camera, RGBCamera)  # Check whether user passed a valid camera class
        logger.info(f"Successfully instantiated a {self._camera_cls.__name__} camera.")

        rgb_name = f"{self._shared_memory_namespace}_{_RGB_SHM_NAME}"
        rgb_shape_name = f"{self._shared_memory_namespace}_{_RGB_SHAPE_SHM_NAME}"
        timestamp_name = f"{self._shared_memory_namespace}_{_TIMESTAMP_SHM_NAME}"
        intrinsics_name = f"{self._shared_memory_namespace}_{_INTRINSICS_SHM_NAME}"
        fps_name = f"{self._shared_memory_namespace}_{_FPS_SHM_NAME}"

        # Get the example arrays (this is the easiest way to initialize the shared memory blocks with the correct size).
        rgb = self._camera.get_rgb_image_as_int()  # We pass uint8 images as they consume 4x less memory
        rgb_shape = np.array(rgb.shape)
        logger.info(f"Successfully retrieved an image of shape {rgb.shape} from the camera.")

        timestamp = np.array([time.time()])
        intrinsics = self._camera.intrinsics_matrix()
        fps = np.array([self._camera.fps], dtype=np.float64)

        # Create the shared memory blocks and numpy arrays that are backed by them.
        logger.info("Creating RGB shared memory blocks.")
        self.rgb_shm, self.rgb_shm_array = shared_memory_block_like(rgb, rgb_name)
        self.rgb_shape_shm, self.rgb_shape_shm_array = shared_memory_block_like(rgb_shape, rgb_shape_name)
        self.timestamp_shm, self.timestamp_shm_array = shared_memory_block_like(timestamp, timestamp_name)
        self.intrinsics_shm, self.intrinsics_shm_array = shared_memory_block_like(intrinsics, intrinsics_name)
        self.fps_shm, self.fps_shm_array = shared_memory_block_like(fps, fps_name)
        logger.info("Created RGB shared memory blocks.")

    def stop(self) -> None:
        self.shutdown_event.set()

    def run(self) -> None:
        """Main loop of the process, runs until the process is terminated.

        Each iteration a new image is retrieved from the camera and copied to the shared memory block.

        Note that we update timestamp after image data has been copied. This ensure that if the receiver sees a new
        timestamp, it will also see the new image data. Theoretically it is possble that the recevier reads new image
        data, but the timestamp is still old. I'm not sure whether this is a problem in practice.

        # TODO: invesitgate whether a Lock is required when copying the image data to the shared memory block. Also
        whether it is possible to do this without having to spawn all processes from a single Python script (e.g. to
        pass the Lock object).
        """

        logger.info(f"{self.__class__.__name__} process started.")
        self._setup()
        assert isinstance(self._camera, RGBCamera)  # Just to make mypy happy, already checked in _setup()
        logger.info(f'{self.__class__.__name__} starting to publish to "{self._shared_memory_namespace}".')

        try:
            while not self.shutdown_event.is_set():
                image = self._camera.get_rgb_image_as_int()
                self.rgb_shm_array[:] = image[:]
                self.timestamp_shm_array[:] = np.array([time.time()])[:]
                self.running_event.set()
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
        finally:
            self.unlink_shared_memory()
            logger.info(f"{self.__class__.__name__} process terminated.")
        self.unlink_shared_memory()

    def unlink_shared_memory(self) -> None:
        """Cleanup of the SharedMemory as recommended by the docs:
        https://docs.python.org/3/library/multiprocessing.shared_memory.html

        For debugging, use:
        watch -n 0.1 ls /dev/shm/

        However, I'm not sure how essential this actually is.
        """
        print(f"Unlinking RGB shared memory blocks of {self.__class__.__name__}.")

        if self.rgb_shm is not None:
            self.rgb_shm.close()
            self.rgb_shm.unlink()
            self.rgb_shm = None

        if self.rgb_shape_shm is not None:
            self.rgb_shape_shm.close()
            self.rgb_shape_shm.unlink()
            self.rgb_shape_shm = None

        if self.timestamp_shm is not None:
            self.timestamp_shm.close()
            self.timestamp_shm.unlink()
            self.timestamp_shm = None

        if self.intrinsics_shm is not None:
            self.intrinsics_shm.close()
            self.intrinsics_shm.unlink()
            self.intrinsics_shm = None

        if self.fps_shm is not None:
            self.fps_shm.close()
            self.fps_shm.unlink()
            self.fps_shm = None

    def __del__(self) -> None:
        self.unlink_shared_memory()


class MultiprocessRGBReceiver(RGBCamera):
    """Implements the RGBD camera interface for a camera that is running in a different process and shares its data using shared memory blocks.
    To be used with the Publisher class.
    """

    def __init__(
        self,
        shared_memory_namespace: str,
    ) -> None:
        super().__init__()

        self._shared_memory_namespace = shared_memory_namespace
        rgb_name = f"{self._shared_memory_namespace}_{_RGB_SHM_NAME}"
        rgb_shape_name = f"{self._shared_memory_namespace}_{_RGB_SHAPE_SHM_NAME}"
        timestamp_name = f"{self._shared_memory_namespace}_{_TIMESTAMP_SHM_NAME}"
        intrinsics_name = f"{self._shared_memory_namespace}_{_INTRINSICS_SHM_NAME}"
        fps_name = f"{self._shared_memory_namespace}_{_FPS_SHM_NAME}"

        # Attach to existing shared memory blocks. Retry a few times to give the publisher time to start up (opening
        # connection to a camera can take a while).

        self.rgb_shm = shared_memory.SharedMemory(name=rgb_name)
        self.rgb_shape_shm = shared_memory.SharedMemory(name=rgb_shape_name)
        self.timestamp_shm = shared_memory.SharedMemory(name=timestamp_name)
        self.intrinsics_shm = shared_memory.SharedMemory(name=intrinsics_name)
        self.fps_shm = shared_memory.SharedMemory(name=fps_name)

        logger.info(f'SharedMemory namespace "{self._shared_memory_namespace}" found.')

        # Normally, we wouldn't have to do this unregistering. However, without it, the resource tracker incorrectly
        # destroys access to the shared memory blocks when the process is terminated. This is a known 3 year old bug
        # that hasn't been resolved yet: https://bugs.python.org/issue39959
        # Concretely, the problem was that once any MultiprocessRGBReceiver object was destroyed, all further access to
        # the shared memory blocks would fail with a FileNotFoundError.
        # We also ignore mypy telling us to use .name instead of ._name, because the latter is used in the registration.
        resource_tracker.unregister(self.rgb_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.rgb_shape_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.intrinsics_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.timestamp_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.fps_shm._name, "shared_memory")  # type: ignore[attr-defined]

        # Timestamp and intrinsics are the same shape for all images, so I decided that we could hardcode their shape.
        # However, images come in many shapes, which I also decided to pass via shared memory. (Previously, I required
        # the image resolution to be passed to this class' constructor, but that was inconvenient to keep in sync
        # between publisher and receiver scripts.)
        # Create numpy arrays that are backed by the shared memory blocks
        self.rgb_shape_shm_array: np.ndarray = np.ndarray((3,), dtype=np.int64, buffer=self.rgb_shape_shm.buf)
        self.intrinsics_shm_array: np.ndarray = np.ndarray((3, 3), dtype=np.float64, buffer=self.intrinsics_shm.buf)
        self.timestamp_shm_array: np.ndarray = np.ndarray((1,), dtype=np.float64, buffer=self.timestamp_shm.buf)
        self.fps_shm_array: np.ndarray = np.ndarray((1,), dtype=np.float64, buffer=self.fps_shm.buf)

        # The shape of the image is not known in advance, so we need to retrieve it from the shared memory block.
        rgb_shape = tuple(self.rgb_shape_shm_array[:])
        self.rgb_shm_array: np.ndarray = np.ndarray(rgb_shape, dtype=np.uint8, buffer=self.rgb_shm.buf)

        self.previous_timestamp = time.time()

    def get_current_timestamp(self) -> float:
        """Timestamp of the image that is currently in the shared memory block.

        Warning: our current implementation, in theory the image and the timestamp could be out of sync when reading.
        Having atomic read/writes of both the image and timestap (a la ROS) would solve this.
        """
        return self.timestamp_shm_array[0]

    @property
    def resolution(self) -> CameraResolutionType:
        """The resolution of the camera, in pixels."""
        shape_array = [int(x) for x in self.rgb_shape_shm_array[:2]]
        return (shape_array[1], shape_array[0])

    def _grab_images(self) -> None:
        while not self.get_current_timestamp() > self.previous_timestamp:
            time.sleep(0.001)
        self.previous_timestamp = self.get_current_timestamp()

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        image = self._retrieve_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image).image_in_numpy_format
        return image

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        return self.rgb_shm_array

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self.intrinsics_shm_array

    def _close_shared_memory(self) -> None:
        """Signal that the shared memory blocks are no longer needed from this process."""
        print(f"Closing RGB shared memory blocks of {self.__class__.__name__}.")

        if self.rgb_shm is not None:
            self.rgb_shm.close()
            self.rgb_shm = None

        if self.rgb_shape_shm is not None:
            self.rgb_shape_shm.close()
            self.rgb_shape_shm = None

        if self.timestamp_shm is not None:
            self.timestamp_shm.close()
            self.timestamp_shm = None

        if self.intrinsics_shm is not None:
            self.intrinsics_shm.close()
            self.intrinsics_shm = None

        if self.fps_shm is not None:
            self.fps_shm.close()
            self.fps_shm = None

    def __del__(self) -> None:
        self._close_shared_memory()


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBPublisher and MultiprocessRGBReceiver.
    You can also use the MultiprocessRGBReceiver in a different process (e.g. in a different python script)
    """
    from airo_camera_toolkit.cameras.zed.zed2i import Zed2i

    multiprocessing.set_start_method("spawn")

    namespace = "camera"

    # Creating and starting the publisher
    publisher = MultiprocessRGBPublisher(
        Zed2i,
        camera_kwargs={
            "resolution": Zed2i.RESOLUTION_1080,
            "fps": 30,
            "depth_mode": Zed2i.NONE_DEPTH_MODE,
        },
        shared_memory_namespace=namespace,
    )
    publisher.start()

    # The receiver behaves just like a regular RGBCamera
    receiver = MultiprocessRGBReceiver(namespace)

    cv2.namedWindow(namespace, cv2.WINDOW_NORMAL)
    while True:
        image_rgb = receiver.get_rgb_image_as_int()
        image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
        cv2.imshow(namespace, image)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    receiver._close_shared_memory()
    publisher.stop()
    publisher.join()
