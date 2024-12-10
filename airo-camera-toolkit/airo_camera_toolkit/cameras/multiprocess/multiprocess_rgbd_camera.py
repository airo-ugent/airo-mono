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
from airo_typing import NumpyDepthMapType, NumpyFloatImageType, NumpyIntImageType, PointCloud

_DEPTH_SHM_NAME = "depth"
_DEPTH_SHAPE_SHM_NAME = "depth_shape"
_DEPTH_IMAGE_SHM_NAME = "depth_image"
_DEPTH_IMAGE_SHAPE_SHM_NAME = "depth_image_shape"
_CONFIDENCE_MAP_SHM_NAME = "confidence_map"
_CONFIDENCE_MAP_SHAPE_SHM_NAME = "confidence_map_shape"
_POINT_CLOUD_POSITIONS_SHM_NAME = "point_cloud_positions"
_POINT_CLOUD_POSITIONS_SHAPE_SHM_NAME = "point_cloud_positions_shape"
_POINT_CLOUD_COLORS_SHM_NAME = "point_cloud_colors"
_POINT_CLOUD_COLORS_SHAPE_SHM_NAME = "point_cloud_colors_shape"


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
        log_debug: bool = False,
    ):
        super().__init__(camera_cls, camera_kwargs, shared_memory_namespace, log_debug)

        self.depth_shm: Optional[shared_memory.SharedMemory] = None
        self.depth_shape_shm: Optional[shared_memory.SharedMemory] = None
        self.depth_image_shm: Optional[shared_memory.SharedMemory] = None
        self.depth_image_shape_shm: Optional[shared_memory.SharedMemory] = None
        self.confidence_map_shm: Optional[shared_memory.SharedMemory] = None
        self.confidence_map_shape_shm: Optional[shared_memory.SharedMemory] = None
        self.point_cloud_positions_shm: Optional[shared_memory.SharedMemory] = None
        self.point_cloud_positions_shape_shm: Optional[shared_memory.SharedMemory] = None
        self.point_cloud_colors_shm: Optional[shared_memory.SharedMemory] = None
        self.point_cloud_colors_shape_shm: Optional[shared_memory.SharedMemory] = None

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
        point_cloud_positions_name = f"{self._shared_memory_namespace}_{_POINT_CLOUD_POSITIONS_SHM_NAME}"
        point_cloud_positions_shape_name = f"{self._shared_memory_namespace}_{_POINT_CLOUD_POSITIONS_SHAPE_SHM_NAME}"
        point_cloud_colors_name = f"{self._shared_memory_namespace}_{_POINT_CLOUD_COLORS_SHM_NAME}"
        point_cloud_colors_shape_name = f"{self._shared_memory_namespace}_{_POINT_CLOUD_COLORS_SHAPE_SHM_NAME}"

        depth_map = self._camera.get_depth_map()
        depth_map_shape = np.array([depth_map.shape])
        depth_image = self._camera._retrieve_depth_image()
        depth_image_shape = np.array([depth_image.shape])
        confidence_map = self._camera._retrieve_confidence_map()  # TODO this is not an interface function yet.
        confidence_map_shape = np.array([confidence_map.shape])
        colored_point_cloud = self._camera._retrieve_colored_point_cloud()
        point_cloud_positions = colored_point_cloud.points
        point_cloud_colors = colored_point_cloud.colors
        point_cloud_positions_shape = np.array([colored_point_cloud.points.shape])
        point_cloud_colors_shape = np.array([colored_point_cloud.colors.shape])

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
        self.point_cloud_positions_shm, self.point_cloud_positions_shm_array = shared_memory_block_like(
            point_cloud_positions, point_cloud_positions_name
        )
        self.point_cloud_positions_shape_shm, self.point_cloud_positions_shape_shm_array = shared_memory_block_like(
            point_cloud_positions_shape, point_cloud_positions_shape_name
        )
        self.point_cloud_colors_shm, self.point_cloud_colors_shm_array = shared_memory_block_like(
            point_cloud_colors, point_cloud_colors_name
        )
        self.point_cloud_colors_shape_shm, self.point_cloud_colors_shape_shm_array = shared_memory_block_like(
            point_cloud_colors_shape, point_cloud_colors_shape_name
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
                point_cloud = self._camera._retrieve_colored_point_cloud()

                while self.read_lock_shm_array[0] > 0 and self.write_lock_shm_array[0]:
                    time.sleep(0.00001)

                self.write_lock_shm_array[0] = True
                self.rgb_shm_array[:] = image[:]
                self.depth_shm_array[:] = depth_map[:]
                self.depth_image_shm_array[:] = depth_image[:]
                self.confidence_map_shm_array[:] = confidence_map[:]
                self.point_cloud_positions_shm_array[:] = point_cloud.points[:]
                self.point_cloud_colors_shm_array[:] = point_cloud.colors[:]
                self.timestamp_shm_array[0] = time.time()
                self.write_lock_shm_array[0] = False
                self.running_event.set()

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}")
        finally:
            self.unlink_shared_memory()
            logger.info(f"{self.__class__.__name__} process terminated.")

    def unlink_shared_memory(self) -> None:  # noqa C901
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

        if self.point_cloud_positions_shm is not None:
            self.point_cloud_positions_shm.close()
            self.point_cloud_positions_shm.unlink()
            self.point_cloud_positions_shm = None

        if self.point_cloud_positions_shape_shm is not None:
            self.point_cloud_positions_shape_shm.close()
            self.point_cloud_positions_shape_shm.unlink()
            self.point_cloud_positions_shape_shm = None

        if self.point_cloud_colors_shm is not None:
            self.point_cloud_colors_shm.close()
            self.point_cloud_colors_shm.unlink()
            self.point_cloud_colors_shm = None

        if self.point_cloud_colors_shape_shm is not None:
            self.point_cloud_colors_shape_shm.close()
            self.point_cloud_colors_shape_shm.unlink()
            self.point_cloud_colors_shape_shm = None

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
        point_cloud_positions_name = f"{self._shared_memory_namespace}_{_POINT_CLOUD_POSITIONS_SHM_NAME}"
        point_cloud_positions_shape_name = f"{self._shared_memory_namespace}_{_POINT_CLOUD_POSITIONS_SHAPE_SHM_NAME}"
        point_cloud_colors_name = f"{self._shared_memory_namespace}_{_POINT_CLOUD_COLORS_SHM_NAME}"
        point_cloud_colors_shape_name = f"{self._shared_memory_namespace}_{_POINT_CLOUD_COLORS_SHAPE_SHM_NAME}"

        self.depth_shm = shared_memory.SharedMemory(name=depth_name)
        self.depth_shape_shm = shared_memory.SharedMemory(name=depth_shape_name)
        self.depth_image_shm = shared_memory.SharedMemory(name=depth_image_name)
        self.depth_image_shape_shm = shared_memory.SharedMemory(name=depth_image_shape_name)
        self.confidence_map_shm = shared_memory.SharedMemory(name=confidence_map_name)
        self.confidence_map_shape_shm = shared_memory.SharedMemory(name=confidence_map_shape_name)
        self.point_cloud_positions_shm = shared_memory.SharedMemory(name=point_cloud_positions_name)
        self.point_cloud_positions_shape_shm = shared_memory.SharedMemory(name=point_cloud_positions_shape_name)
        self.point_cloud_colors_shm = shared_memory.SharedMemory(name=point_cloud_colors_name)
        self.point_cloud_colors_shape_shm = shared_memory.SharedMemory(name=point_cloud_colors_shape_name)

        # Same comment as in base class:
        resource_tracker.unregister(self.depth_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.depth_shape_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.depth_image_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.depth_image_shape_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.confidence_map_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.confidence_map_shape_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.point_cloud_positions_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.point_cloud_positions_shape_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.point_cloud_colors_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.point_cloud_colors_shape_shm._name, "shared_memory")  # type: ignore[attr-defined]

        self.depth_shape_shm_array: np.ndarray = np.ndarray((2,), dtype=np.int64, buffer=self.depth_shape_shm.buf)
        self.depth_image_shape_shm_array: np.ndarray = np.ndarray(
            (3,), dtype=np.int64, buffer=self.depth_image_shape_shm.buf
        )
        self.confidence_map_shape_shm_array: np.ndarray = np.ndarray(
            (2,), dtype=np.int64, buffer=self.confidence_map_shape_shm.buf
        )

        self.point_cloud_positions_shape_shm_array: np.ndarray = np.ndarray(
            (2,), dtype=np.int64, buffer=self.point_cloud_positions_shape_shm.buf
        )
        self.point_cloud_colors_shape_shm_array: np.ndarray = np.ndarray(
            (2,), dtype=np.int64, buffer=self.point_cloud_colors_shape_shm.buf
        )

        depth_shape = tuple(self.depth_shape_shm_array[:])
        depth_image_shape = tuple(self.depth_image_shape_shm_array[:])
        confidence_map_shape = tuple(self.confidence_map_shape_shm_array[:])
        point_cloud_positions_shape = tuple(self.point_cloud_positions_shape_shm_array[:])
        point_cloud_colors_shape = tuple(self.point_cloud_colors_shape_shm_array[:])

        self.depth_shm_array: np.ndarray = np.ndarray(depth_shape, dtype=np.float32, buffer=self.depth_shm.buf)
        self.depth_image_shm_array: np.ndarray = np.ndarray(
            depth_image_shape, dtype=np.uint8, buffer=self.depth_image_shm.buf
        )
        self.confidence_map_shm_array: np.ndarray = np.ndarray(
            confidence_map_shape, dtype=np.float32, buffer=self.confidence_map_shm.buf
        )

        self.point_cloud_positions_shm_array: np.ndarray = np.ndarray(
            point_cloud_positions_shape,
            dtype=np.float32,
            buffer=self.point_cloud_positions_shm.buf,
        )

        self.point_cloud_colors_shm_array: np.ndarray = np.ndarray(
            point_cloud_colors_shape, dtype=np.uint8, buffer=self.point_cloud_colors_shm.buf
        )

        self.depth_buffer_array: np.ndarray = np.ndarray(depth_shape, dtype=np.float32)
        self.depth_image_buffer_array: np.ndarray = np.ndarray(depth_image_shape, dtype=np.uint8)
        self.confidence_map_buffer_array: np.ndarray = np.ndarray(confidence_map_shape, dtype=np.float32)
        self.point_cloud_positions_buffer_array: np.ndarray = np.ndarray(point_cloud_positions_shape, dtype=np.float32)
        self.point_cloud_colors_buffer_array: np.ndarray = np.ndarray(point_cloud_colors_shape, dtype=np.uint8)

    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        while self.write_lock_shm_array[0]:
            time.sleep(0.00001)
        self.read_lock_shm_array[0] += 1
        self.depth_buffer_array[:] = self.depth_shm_array[:]
        self.read_lock_shm_array[0] -= 1
        return self.depth_buffer_array

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        while self.write_lock_shm_array[0]:
            time.sleep(0.00001)
        self.read_lock_shm_array[0] += 1
        self.depth_image_buffer_array[:] = self.depth_image_shm_array[:]
        self.read_lock_shm_array[0] -= 1
        return self.depth_image_buffer_array

    def _retrieve_confidence_map(self) -> NumpyFloatImageType:
        while self.write_lock_shm_array[0]:
            time.sleep(0.00001)
        self.read_lock_shm_array[0] += 1
        self.confidence_map_buffer_array[:] = self.confidence_map_shm_array[:]
        self.read_lock_shm_array[0] -= 1
        return self.confidence_map_buffer_array

    def _retrieve_colored_point_cloud(self) -> PointCloud:
        while self.write_lock_shm_array[0]:
            time.sleep(0.00001)
        self.read_lock_shm_array[0] += 1
        self.point_cloud_positions_buffer_array[:] = self.point_cloud_positions_shm_array[:]
        self.point_cloud_colors_buffer_array[:] = self.point_cloud_colors_shm_array[:]
        self.read_lock_shm_array[0] -= 1
        point_cloud = PointCloud(self.point_cloud_positions_buffer_array, self.point_cloud_colors_buffer_array)
        return point_cloud

    def _close_shared_memory(self) -> None:  # noqa C901
        """Closing shared memory signal that"""
        super()._close_shared_memory()
        print(f"Closing depth shared memory blocks of {self.__class__.__name__}")

        if self.depth_shm is not None:
            self.depth_shm.close()
            self.depth_shm = None  # type: ignore

        if self.depth_shape_shm is not None:
            self.depth_shape_shm.close()
            self.depth_shape_shm = None  # type: ignore

        if self.depth_image_shm is not None:
            self.depth_image_shm.close()
            self.depth_image_shm = None  # type: ignore

        if self.depth_image_shape_shm is not None:
            self.depth_image_shape_shm.close()
            self.depth_image_shape_shm = None  # type: ignore

        if self.confidence_map_shm is not None:
            self.confidence_map_shm.close()
            self.confidence_map_shm = None  # type: ignore

        if self.confidence_map_shape_shm is not None:
            self.confidence_map_shape_shm.close()
            self.confidence_map_shape_shm = None  # type: ignore

        if self.point_cloud_positions_shm is not None:
            self.point_cloud_positions_shm.close()
            self.point_cloud_positions_shm = None  # type: ignore

        if self.point_cloud_positions_shape_shm is not None:
            self.point_cloud_positions_shape_shm.close()
            self.point_cloud_positions_shape_shm = None  # type: ignore

        if self.point_cloud_colors_shm is not None:
            self.point_cloud_colors_shm.close()
            self.point_cloud_colors_shm = None  # type: ignore

        if self.point_cloud_colors_shape_shm is not None:
            self.point_cloud_colors_shape_shm.close()
            self.point_cloud_colors_shape_shm = None  # type: ignore

    def __del__(self) -> None:
        super().__del__()
        self._close_shared_memory()


if __name__ == "__main__":
    """example of how to use the MultiprocessRGBDPublisher and MultiprocessRGBDReceiver.
    You can also use the MultiprocessRGBDReceiver in a different process (e.g. in a different python script)
    """

    import multiprocessing

    from airo_camera_toolkit.cameras.zed.zed import Zed

    multiprocessing.set_start_method("spawn")

    resolution = Zed.RESOLUTION_2K
    camera_fps = 15

    publisher = MultiprocessRGBDPublisher(
        Zed,
        camera_kwargs={
            "resolution": resolution,
            "fps": camera_fps,
            "depth_mode": Zed.NEURAL_DEPTH_MODE,
        },
    )

    publisher.start()
    receiver = MultiprocessRGBDReceiver("camera")

    cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Confidence Map", cv2.WINDOW_NORMAL)

    time_current = None
    time_previous = None

    while True:
        time_previous = time_current
        time_current = time.time()

        depth_map = receiver.get_depth_map()
        depth_image = receiver._retrieve_depth_image()
        confidence_map = receiver._retrieve_confidence_map()
        point_cloud = receiver._retrieve_colored_point_cloud()

        cv2.imshow("Depth Map", depth_map)
        cv2.imshow("Depth Image", depth_image)
        cv2.imshow("Confidence Map", confidence_map)

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

    receiver._close_shared_memory()
    publisher.stop()
    publisher.join()
