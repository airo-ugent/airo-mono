"""code for sharing the data of a camera that implements the RGBDCamera interface between processes using shared memory"""

import time
from multiprocessing import resource_tracker, shared_memory
from typing import Optional

import cv2
import loguru
import numpy as np
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgb_camera import shared_memory_block_like
from airo_camera_toolkit.cameras.multiprocess.multiprocess_rgbd_camera import (
    MultiprocessRGBDPublisher,
    MultiprocessRGBDReceiver,
)
from airo_camera_toolkit.utils.image_converter import ImageConverter
from typing_extensions import deprecated

logger = loguru.logger
from airo_camera_toolkit.interfaces import RGBDCamera, StereoRGBDCamera
from airo_typing import CameraIntrinsicsMatrixType, HomogeneousMatrixType, NumpyFloatImageType, NumpyIntImageType

_RGB_RIGHT_SHM_NAME = "rgb_right"
_RGB_RIGHT_SHAPE_SHM_NAME = "rgb_right_shape"
_POSE_RIGHT_IN_LEFT_SHM_NAME = "pose_right_in_left"
_INTRINSICS_RIGHT_SHM_NAME = "intrinsics_right"


@deprecated(
    "This class uses the old shared memory implementation and will not work currently. It will be updated in the future."
)
class MultiprocessStereoRGBDPublisher(MultiprocessRGBDPublisher):
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

        self.rgb_right_shm: Optional[shared_memory.SharedMemory] = None
        self.rgb_right_shape_shm: Optional[shared_memory.SharedMemory] = None
        self.pose_right_in_left_shm: Optional[shared_memory.SharedMemory] = None
        self.intrinsics_right_shm: Optional[shared_memory.SharedMemory] = None

    def _setup(self) -> None:
        """in-process creation of camera object and shared memory blocks"""
        super()._setup()
        assert isinstance(self._camera, RGBDCamera)

        rgb_right_name = f"{self._shared_memory_namespace}_{_RGB_RIGHT_SHM_NAME}"
        rgb_right_shape_name = f"{self._shared_memory_namespace}_{_RGB_RIGHT_SHAPE_SHM_NAME}"
        pose_right_in_left_name = f"{self._shared_memory_namespace}_{_POSE_RIGHT_IN_LEFT_SHM_NAME}"
        intrinsics_right_name = f"{self._shared_memory_namespace}_{_INTRINSICS_RIGHT_SHM_NAME}"

        rgb_right = self._camera.get_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB)
        rgb_right_shape = np.array([rgb_right.shape])
        pose_right_in_left = self._camera.pose_of_right_view_in_left_view
        intrinsics_right = self._camera.intrinsics_matrix(view=StereoRGBDCamera.RIGHT_RGB)

        logger.info("Creating stereo shared memory blocks.")

        self.rgb_right_shm, self.rgb_right_shm_array = shared_memory_block_like(rgb_right, rgb_right_name)
        self.rgb_right_shape_shm, self.rgb_right_shape_shm_array = shared_memory_block_like(
            rgb_right_shape, rgb_right_shape_name
        )
        self.pose_right_in_left_shm, self.pose_right_in_left_shm_array = shared_memory_block_like(
            pose_right_in_left, pose_right_in_left_name
        )

        self.intrinsics_right_shm, self.intrinsics_right_shm_array = shared_memory_block_like(
            intrinsics_right, intrinsics_right_name
        )

    def stop(self) -> None:
        self.shutdown_event.set()

    def run(self) -> None:
        """main loop of the process, runs until the process is terminated"""

        logger.info(f"{self.__class__.__name__} process started.")
        self._setup()
        assert isinstance(self._camera, StereoRGBDCamera)  # For mypy
        logger.info(f'{self.__class__.__name__} starting to publish to "{self._shared_memory_namespace}".')

        timestamp_prev_publish = None

        try:
            while not self.shutdown_event.is_set():
                time_retreive_start = time.time()
                self._camera._grab_images()
                time_grab_end = time.time()
                image = self._camera._retrieve_rgb_image_as_int()
                image_right = self._camera._retrieve_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB)
                depth_map = self._camera._retrieve_depth_map()
                depth_image = self._camera._retrieve_depth_image()
                confidence_map = self._camera._retrieve_confidence_map()
                point_cloud = self._camera._retrieve_colored_point_cloud()
                time_retreive_end = time.time()

                time_lock_start = time.time()
                while self.read_lock_shm_array[0] > 0 and self.write_lock_shm_array[0]:
                    time.sleep(0.00001)
                time_lock_end = time.time()

                time_shm_write_start = time.time()
                self.write_lock_shm_array[0] = True
                self.rgb_shm_array[:] = image[:]
                self.rgb_right_shm_array[:] = image_right[:]
                self.depth_shm_array[:] = depth_map[:]
                self.depth_image_shm_array[:] = depth_image[:]
                self.confidence_map_shm_array[:] = confidence_map[:]
                self.point_cloud_positions_shm_array[:] = point_cloud.points[:]
                self.point_cloud_colors_shm_array[:] = point_cloud.colors[:]

                timestamp_publish = time.time()
                self.timestamp_shm_array[0] = timestamp_publish

                if timestamp_prev_publish is not None and self.log_debug:
                    publish_period = timestamp_publish - timestamp_prev_publish
                    if publish_period > 1.1 * self.camera_period:
                        logger.warning(
                            f"Time since previous publish: {publish_period:.3f} s. Publisher or camera running slow."
                        )
                    else:
                        logger.debug(f"Time since previous publish: {publish_period:.3f} s")
                timestamp_prev_publish = timestamp_publish

                self.write_lock_shm_array[0] = False
                time_shm_write_end = time.time()

                if self.log_debug:
                    logger.debug(
                        f"Retrieval time: {time_retreive_end - time_retreive_start:.3f} s, (grab time: {time_grab_end - time_retreive_start:.3f} s),"
                        f"Lock time: {time_lock_end - time_lock_start:.3f} s, "
                        f"SHM write time: {time_shm_write_end - time_shm_write_start:.3f} s"
                    )

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

        if self.rgb_right_shm is not None:
            self.rgb_right_shm.close()
            self.rgb_right_shm.unlink()
            self.rgb_right_shm = None

        if self.rgb_right_shape_shm is not None:
            self.rgb_right_shape_shm.close()
            self.rgb_right_shape_shm.unlink()
            self.rgb_right_shape_shm = None

        if self.pose_right_in_left_shm is not None:
            self.pose_right_in_left_shm.close()
            self.pose_right_in_left_shm.unlink()
            self.pose_right_in_left_shm = None

        if self.intrinsics_right_shm is not None:
            self.intrinsics_right_shm.close()
            self.intrinsics_right_shm.unlink()
            self.intrinsics_right_shm = None

    def __del__(self) -> None:
        self.unlink_shared_memory()


class MultiprocessStereoRGBDReceiver(MultiprocessRGBDReceiver, StereoRGBDCamera):
    def __init__(self, shared_memory_namespace: str) -> None:
        super().__init__(shared_memory_namespace)

        rgb_right_name = f"{self._shared_memory_namespace}_{_RGB_RIGHT_SHM_NAME}"
        rgb_right_shape_name = f"{self._shared_memory_namespace}_{_RGB_RIGHT_SHAPE_SHM_NAME}"
        pose_right_in_left_name = f"{self._shared_memory_namespace}_{_POSE_RIGHT_IN_LEFT_SHM_NAME}"
        intrinsics_right_name = f"{self._shared_memory_namespace}_{_INTRINSICS_RIGHT_SHM_NAME}"

        self.rgb_right_shm = shared_memory.SharedMemory(name=rgb_right_name)
        self.rgb_right_shape_shm = shared_memory.SharedMemory(name=rgb_right_shape_name)
        self.pose_right_in_left_shm = shared_memory.SharedMemory(name=pose_right_in_left_name)
        self.intrinsics_right_shm = shared_memory.SharedMemory(name=intrinsics_right_name)

        logger.info("Found stereo shared memory blocks.")

        resource_tracker.unregister(self.rgb_right_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.rgb_right_shape_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.pose_right_in_left_shm._name, "shared_memory")  # type: ignore[attr-defined]
        resource_tracker.unregister(self.intrinsics_right_shm._name, "shared_memory")  # type: ignore[attr-defined]

        self.rgb_right_shape_shm_array: np.ndarray = np.ndarray(
            (3,), dtype=np.int64, buffer=self.rgb_right_shape_shm.buf
        )
        self.pose_right_in_left_shm_array: np.ndarray = np.ndarray(
            (4, 4), dtype=np.float64, buffer=self.pose_right_in_left_shm.buf
        )
        self.intrinsics_right_shm_array: np.ndarray = np.ndarray(
            (3, 3), dtype=np.float64, buffer=self.intrinsics_right_shm.buf
        )

        rgb_right_shape = tuple(self.rgb_right_shape_shm_array[:])

        self.rgb_right_shm_array: np.ndarray = np.ndarray(
            rgb_right_shape, dtype=np.uint8, buffer=self.rgb_right_shm.buf
        )

        self.rgb_right_buffer_array: np.ndarray = np.ndarray(rgb_right_shape, dtype=np.uint8)

    def _retrieve_rgb_image(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyFloatImageType:
        image = self._retrieve_rgb_image_as_int(view)
        image = ImageConverter.from_numpy_int_format(image).image_in_numpy_format
        return image

    def _retrieve_rgb_image_as_int(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyIntImageType:
        while self.write_lock_shm_array[0]:
            time.sleep(0.00001)

        if view == StereoRGBDCamera.LEFT_RGB:
            self.read_lock_shm_array[0] += 1
            self.rgb_buffer_array[:] = self.rgb_shm_array[:]
            self.read_lock_shm_array[0] -= 1
            return self.rgb_buffer_array
        elif view == StereoRGBDCamera.RIGHT_RGB:
            self.read_lock_shm_array[0] += 1
            self.rgb_right_buffer_array[:] = self.rgb_right_shm_array[:]
            self.read_lock_shm_array[0] -= 1
            return self.rgb_right_buffer_array
        else:
            raise ValueError(f"Unknown view: {view}")

    def intrinsics_matrix(self, view: str = StereoRGBDCamera.LEFT_RGB) -> CameraIntrinsicsMatrixType:
        if view == StereoRGBDCamera.LEFT_RGB:
            return self.intrinsics_shm_array
        elif view == StereoRGBDCamera.RIGHT_RGB:
            return self.intrinsics_right_shm_array
        else:
            raise ValueError(f"Unknown view: {view}")

    @property
    def pose_of_right_view_in_left_view(self) -> HomogeneousMatrixType:
        return self.pose_right_in_left_shm_array

    def _close_shared_memory(self) -> None:
        """Closing shared memory signal that"""
        super()._close_shared_memory()
        print(f"Closing depth shared memory blocks of {self.__class__.__name__}")

        if self.rgb_right_shm is not None:
            self.rgb_right_shm.close()
            self.rgb_right_shm = None  # type: ignore

        if self.rgb_right_shape_shm is not None:
            self.rgb_right_shape_shm.close()
            self.rgb_right_shape_shm = None  # type: ignore

        if self.pose_right_in_left_shm is not None:
            self.pose_right_in_left_shm.close()
            self.pose_right_in_left_shm = None  # type: ignore

        if self.intrinsics_right_shm is not None:
            self.intrinsics_right_shm.close()
            self.intrinsics_right_shm = None  # type: ignore

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

    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    publisher = MultiprocessStereoRGBDPublisher(
        Zed,
        camera_kwargs={
            "resolution": resolution,
            "fps": camera_fps,
            "depth_mode": Zed.NEURAL_DEPTH_MODE,
        },
        log_debug=True,
    )

    publisher.start()
    receiver = MultiprocessStereoRGBDReceiver("camera")

    with np.printoptions(precision=3, suppress=True):
        print("Intrinsics left:\n", receiver.intrinsics_matrix())
        print("Intrinsics right:\n", receiver.intrinsics_matrix(view=StereoRGBDCamera.RIGHT_RGB))
        print("Pose right in left:\n", receiver.pose_of_right_view_in_left_view)

    cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RGB Image Right", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Confidence Map", cv2.WINDOW_NORMAL)

    import rerun as rr

    rr.init(f"{MultiprocessStereoRGBDReceiver.__name__} - Point cloud", spawn=True)

    log_point_cloud = False

    time_current = None
    time_previous = None

    while True:
        time_previous = time_current
        time_current = time.time()

        image = receiver.get_rgb_image_as_int()
        image_right = receiver._retrieve_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB)
        depth_map = receiver._retrieve_depth_map()
        depth_image = receiver._retrieve_depth_image()
        confidence_map = receiver._retrieve_confidence_map()
        point_cloud = receiver._retrieve_colored_point_cloud()

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_right_bgr = cv2.cvtColor(image_right, cv2.COLOR_RGB2BGR)

        cv2.imshow("RGB Image", image_bgr)
        cv2.imshow("RGB Image Right", image_right_bgr)
        cv2.imshow("Depth Map", depth_map)
        cv2.imshow("Depth Image", depth_image)
        cv2.imshow("Confidence Map", confidence_map)

        if log_point_cloud:
            rr.log("point_cloud", rr.Points3D(positions=point_cloud.points, colors=point_cloud.colors))

        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord("l"):
            log_point_cloud = not log_point_cloud

    receiver._close_shared_memory()
    publisher.stop()
    publisher.join()
