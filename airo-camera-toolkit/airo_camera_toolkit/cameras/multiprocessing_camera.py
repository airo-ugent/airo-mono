from multiprocessing import Process, Queue, shared_memory

import numpy as np
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_typing import CameraIntrinsicsMatrixType, NumpyDepthMapType, NumpyFloatImageType, NumpyIntImageType


class MultiprocessingRGBDCameraPublisher(Process):
    def __init__(self, camera_cls, camera_kwargs, intrinsics_queue) -> None:
        self._camera_cls = camera_cls
        self._camera_kwargs = camera_kwargs
        self._camera = None

        self._rgb_shared_memory = shared_memory.SharedMemory(create=True, size=(5000, 5000, 3))
        super().__init__()
        self._update_shared_memory()

    def run(self) -> None:
        # create camera here to make sure the shared memory blocks in the camera are created in the same process
        self._camera = self._camera_cls(**self._camera_kwargs)
        while True:
            self._update_shared_memory()
            print("Updated SHM")

    def _update_shared_memory(self) -> None:
        if self._camera is None:
            print("camera not initialized")
            return
        self._rgb_queue.put(np.zeros((1000, 1000, 3)))
        s = self._camera.intrinsics_matrix()
        self._intrinsic_queue.put(s)
        # self._rgb_queue.put(self._camera.get_rgb_image())
        # self._depth_map_queue.put(self._camera.get_depth_map())


class MultiprocessingRGBDCamera(RGBDCamera):
    def __init__(self, intrinsics_queue: Queue, rgb_queue: Queue, depth_map_queue: Queue) -> None:
        super().__init__()
        self._intrinsics_queue = intrinsics_queue
        self._rgb_queue = rgb_queue
        self._depth_map_queue = depth_map_queue

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self._intrinsics_queue[0]

    def get_rgb_image(self) -> NumpyFloatImageType:
        return self._rgb_queue.get()

    def get_depth_map(self) -> NumpyDepthMapType:
        return self._depth_map_queue.get()

    def get_depth_image(self) -> NumpyIntImageType:
        # basic implementation, don't want to use additional queue for the depth image
        depth_map = self.get_depth_map()
        normalized_depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        return (normalized_depth_map * 255).astype("uint8")


if __name__ == "__main__":
    from airo_camera_toolkit.cameras.zed2i import Zed2i

    # intrinsics_queue = Queue()
    # rgb_queue = Queue()
    # depth_map_queue = Queue()
    # publisher = MultiprocessingRGBDCameraPublisher(Zed2i, {"depth_mode": Zed2i.PERFORMANCE_DEPTH_MODE}, intrinsics_queue, rgb_queue, depth_map_queue)
    # publisher.start()
    # camera = MultiprocessingRGBDCamera(intrinsics_queue, rgb_queue, depth_map_queue)
    # while True:
    #     print(camera.get_rgb_image().shape)

    camera = Zed2i(depth_mode=Zed2i.NONE_DEPTH_MODE)
    while True:
        print(camera.get_rgb_image().shape)
