from typing import Final

try:
    import pyzed.sl as sl
except ImportError:
    raise ImportError(
        "You should install the ZED SDK and pip install the python bindings in your environment first, see the installation README."
    )

from airo_camera_toolkit.cameras.multiprocess.buffer import CameraPoseBuffer, SpatialMapBuffer
from airo_camera_toolkit.cameras.multiprocess.mixin import (
    CameraMixin,
    DepthMixin,
    Mixin,
    PointCloudMixin,
    StereoRGBMixin,
)
from airo_camera_toolkit.cameras.multiprocess.publisher import CameraPublisher
from airo_camera_toolkit.cameras.multiprocess.receiver import SharedMemoryReceiver
from airo_camera_toolkit.cameras.multiprocess.schema import (
    CameraSchema,
    DepthSchema,
    PointCloudSchema,
    Schema,
    StereoRGBSchema,
)
from airo_camera_toolkit.cameras.zed.zed import Zed, ZedSpatialMap
from airo_camera_toolkit.interfaces import Camera, StereoRGBDCamera
from airo_typing import CameraResolutionType, HomogeneousMatrixType, PointCloud

# Schemas and mixins are defined in this file, unlike more generic implementations, because they require Zed imports.


class CameraPoseMixin(Mixin):
    _camera_pose_world_frame: HomogeneousMatrixType
    _camera_pose_camera_frame: HomogeneousMatrixType

    def _retrieve_camera_pose(  # type:ignore[no-any-unimported]
        self, coordinate_frame: sl.REFERENCE_FRAME = Zed.TrackingParams.REFERENCE_FRAME_WORLD
    ) -> HomogeneousMatrixType:
        if coordinate_frame == Zed.TrackingParams.REFERENCE_FRAME_WORLD:
            return self._camera_pose_world_frame
        else:
            return self._camera_pose_camera_frame


class CameraPoseSchema(Schema):
    def __init__(self) -> None:
        super().__init__("pose", CameraPoseBuffer)

    def allocate_empty(self, resolution: CameraResolutionType) -> None:
        self._buffer = CameraPoseBuffer(
            camera_pose_world_frame=np.empty((4, 4), dtype=np.float32),
            camera_pose_camera_frame=np.empty((4, 4), dtype=np.float32),
        )

    def fill_from_camera(self, camera: Camera) -> None:
        self._assert_buffer_allocated()
        assert isinstance(camera, Zed)  # for mypy
        assert isinstance(self._buffer, CameraPoseBuffer)  # for mypy

        self._buffer.camera_pose_world_frame = camera._retrieve_camera_pose(Zed.TrackingParams.REFERENCE_FRAME_WORLD)
        self._buffer.camera_pose_camera_frame = camera._retrieve_camera_pose(Zed.TrackingParams.REFERENCE_FRAME_CAMERA)

    def read_into_receiver(self, frame: CameraPoseBuffer, receiver: Mixin) -> None:
        assert isinstance(receiver, CameraPoseMixin)  # for mypy
        receiver._camera_pose_world_frame = frame.camera_pose_world_frame
        receiver._camera_pose_camera_frame = frame.camera_pose_camera_frame


class SpatialMapMixin(Mixin):
    _spatial_map: ZedSpatialMap

    def _retrieve_spatial_map(self) -> ZedSpatialMap:
        return self._spatial_map


class SpatialMapSchema(Schema):
    def __init__(self, max_chunks: int, max_points: int, refresh_interval: int):
        super().__init__("spatial_map", SpatialMapBuffer)

        self._max_chunks = max_chunks
        self._max_points = max_points
        self._refresh_interval = refresh_interval

        self._frame_counter = -1

    def allocate_empty(self, resolution: CameraResolutionType) -> None:
        self._buffer = SpatialMapBuffer(
            num_chunks=np.empty((1,), dtype=np.uint32),
            chunks_updated=np.empty((self._max_chunks), dtype=np.uint8),
            chunk_sizes=np.empty((self._max_chunks), dtype=np.uint32),
            point_positions=np.empty((self._max_points, 3), dtype=np.float32),
            point_colors=np.empty((self._max_points, 3), dtype=np.uint8),
        )

    def fill_from_camera(self, camera: Camera) -> None:
        self._assert_buffer_allocated()
        assert isinstance(camera, Zed)  # for mypy
        assert isinstance(self._buffer, SpatialMapBuffer)  # for mypy

        self._frame_counter += 1

        if self._frame_counter == 0:
            # First frame.
            # Write to buffers an empty spatial map.
            self._buffer.num_chunks[0] = 0
            # Other data left uninitialized!

        # Request an update each frame; if one is pending, nothing happens (see ZED-sdk docs).
        camera._request_spatial_map_update()

        # Only update the spatial map every N frames to reduce load and suppress warning messages.
        if self._frame_counter % self._refresh_interval != 0:
            # Don't update the buffer, but retain the previous value.
            return

        # If we reach this point, we're going to retrieve the spatial map.

        self._frame_counter = 1  # Reset to 1 to avoid overflows on long-running processes.

        spatial_map = camera._retrieve_spatial_map()

        # Only continue if the spatial map is not empty.
        if spatial_map.size <= 0:
            return

        num_chunks = spatial_map.num_chunks
        chunk_sizes = spatial_map.chunk_sizes
        chunks_updated = spatial_map.chunks_updated
        point_positions = spatial_map.full_pointcloud.points
        point_colors = spatial_map.full_pointcloud.colors

        # Check if number of chunks exceeds maximum allowed
        # For simplicity, just throw error for now. Could later be improved to only send partial map.
        if num_chunks > self._max_chunks:
            raise RuntimeError(f"Spatial map has {num_chunks} chunks, exceeding the maximum of {self._max_chunks}.")

        # Check if number of points exceeds maximum allowed.
        # For simplicity, just throw error for now. Could later be improved to only send partial map.
        if spatial_map.size > self._max_points:
            raise RuntimeError(
                f"Spatial map has {spatial_map.size} points, exceeding the maximum of {self._max_points}."
            )

        # Write to buffers.
        self._buffer.num_chunks[0] = num_chunks
        self._buffer.chunks_updated[:num_chunks] = np.array(chunks_updated)
        self._buffer.chunk_sizes[:num_chunks] = np.array(chunk_sizes)
        self._buffer.point_positions[: spatial_map.size, :] = np.array(point_positions)
        self._buffer.point_colors[: spatial_map.size, :] = np.array(point_colors)

    def read_into_receiver(self, frame: SpatialMapBuffer, receiver: Mixin) -> None:
        # Reconstruct the ZedSpatialMap from shared memory.
        assert isinstance(receiver, SpatialMapMixin)  # for mypy

        num_chunks = int(frame.num_chunks.item())

        if num_chunks == 0:
            receiver._spatial_map = ZedSpatialMap([], [])

        current_offset = 0

        chunks = []
        chunks_updated = []

        # Iterate through the chunks described in the shared memory
        for i in range(num_chunks):
            chunk_size = int(frame.chunk_sizes[i])
            has_been_updated = bool(frame.chunks_updated[i])

            # Slice the flattened arrays based on the current chunk size
            # We use .copy() to decouple the PointCloud data from the shared memory buffer
            positions = frame.point_positions[current_offset : current_offset + chunk_size].copy()
            colors = frame.point_colors[current_offset : current_offset + chunk_size].copy()

            # Reconstruct the ZedSpatialMap
            chunks.append(PointCloud(positions, colors))
            chunks_updated.append(has_been_updated)

            # Increment offset for the next chunk
            current_offset += chunk_size

        receiver._spatial_map = ZedSpatialMap(chunks, chunks_updated)


_DEFAULT_MAX_SPATIAL_MAP_CHUNKS: Final[int] = 10000
_DEFAULT_MAX_SPATIAL_MAP_POINTS: Final[int] = 1000000


class MultiprocessZedPublisher(CameraPublisher):
    def __init__(
        self,
        camera_cls: type,
        camera_kwargs: dict = {},
        shared_memory_namespace: str = "camera",
        enable_pointcloud: bool = True,
        max_spatial_map_chunks: int = _DEFAULT_MAX_SPATIAL_MAP_CHUNKS,
        max_spatial_map_points: int = _DEFAULT_MAX_SPATIAL_MAP_POINTS,
        spatial_map_refresh_interval: int = 5,
    ) -> None:
        schemas = [CameraSchema(), StereoRGBSchema(), DepthSchema()]
        if enable_pointcloud:
            schemas.append(PointCloudSchema())

        enable_positional_tracking = camera_kwargs.get("camera_tracking_params") is not None
        if enable_positional_tracking:
            schemas.append(CameraPoseSchema())

        enable_spatial_mapping = camera_kwargs.get("camera_mapping_params") is not None
        if enable_spatial_mapping:
            schemas.append(
                SpatialMapSchema(max_spatial_map_chunks, max_spatial_map_points, spatial_map_refresh_interval)
            )

        super().__init__(camera_cls, camera_kwargs, schemas, shared_memory_namespace)


# Inheritance order matters! The first class encountered determines which method is used, is it if defined in >1 Mixin.
# StereoRGBMixin MUST be before CameraMixin and SharedMemoryReceiver for intrinsics_matrix()!
class MultiprocessZedReceiver(
    StereoRGBMixin, CameraMixin, DepthMixin, PointCloudMixin, CameraPoseMixin, SpatialMapMixin, SharedMemoryReceiver
):
    """Multiprocess implementation of Zed (which is itself an extension of StereoRGBDCamera). To be used with MultiprocessZedPublisher."""

    def __init__(
        self,
        namespace: str,
        resolution: CameraResolutionType,
        enable_pointcloud: bool = True,
        enable_positional_tracking: bool = False,
        enable_spatial_mapping: bool = False,
        max_spatial_map_chunks: int = _DEFAULT_MAX_SPATIAL_MAP_CHUNKS,
        max_spatial_map_points: int = _DEFAULT_MAX_SPATIAL_MAP_POINTS,
    ):
        schemas = [CameraSchema(), StereoRGBSchema(), DepthSchema()]
        if enable_pointcloud:
            schemas.append(PointCloudSchema())
        if enable_positional_tracking:
            schemas.append(CameraPoseSchema())
        if enable_spatial_mapping:
            # -1: The receiver does not use the refresh interval.
            schemas.append(SpatialMapSchema(max_spatial_map_chunks, max_spatial_map_points, -1))
        SharedMemoryReceiver.__init__(self, resolution, schemas, namespace)


if __name__ == "__main__":  # noqa: C901
    import multiprocessing
    import time
    from collections import deque

    import cv2
    import numpy as np
    import rerun as rr

    def camera_to_rerun(points_cam: np.ndarray) -> np.ndarray:
        """
        Convert points from camera frame (X right, Y down, Z forward)
        to Rerun frame (X right, Y forward, Z up).
        """
        points_rerun = np.empty_like(points_cam)
        points_rerun[:, 0] = points_cam[:, 0]
        points_rerun[:, 1] = points_cam[:, 2]
        points_rerun[:, 2] = -points_cam[:, 1]
        return points_rerun

    multiprocessing.set_start_method("spawn", force=True)

    resolution = Zed.InitParams.RESOLUTION_720
    camera_fps = 15

    camera_tracking_params = Zed.TrackingParams()
    camera_mapping_params = Zed.MappingParams()

    publisher = MultiprocessZedPublisher(
        Zed,
        camera_kwargs={
            "resolution": resolution,
            "fps": camera_fps,
            "depth_mode": Zed.InitParams.NEURAL_DEPTH_MODE,
            "camera_tracking_params": camera_tracking_params,
            "camera_mapping_params": camera_mapping_params,
        },
    )
    publisher.start()

    receiver = MultiprocessZedReceiver(
        "camera",
        resolution,
        enable_positional_tracking=True,
        enable_spatial_mapping=True,
    )

    # Prime shared memory
    receiver._grab_images()

    cv2.namedWindow("RGB Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RGB Image Right", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)

    rr.init(MultiprocessZedReceiver.__name__)
    rr.spawn(memory_limit="2GB")

    rr.log("World", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    rr.log(
        "World/Camera/Pinhole",
        rr.Pinhole(
            resolution=[1280, 720],
            focal_length=700,
        ),
        static=True,
    )

    log_point_cloud = False
    log_spatial_map = True

    # -----------------------------
    # PROFILING STATE
    # -----------------------------
    WINDOW = 200

    timings: dict[str, deque] = {
        "grab": deque(maxlen=WINDOW),
        "rgb": deque(maxlen=WINDOW),
        "depth": deque(maxlen=WINDOW),
        "pointcloud": deque(maxlen=WINDOW),
        "spatial_map": deque(maxlen=WINDOW),
        "rerun": deque(maxlen=WINDOW),
        "total": deque(maxlen=WINDOW),
    }

    fps_window: deque = deque(maxlen=50)

    frame_idx = 0
    previous_frame_time = time.time()
    t_last_frame = time.perf_counter()

    # -----------------------------
    # MAIN LOOP
    # -----------------------------
    while True:
        t_frame_start = time.perf_counter()

        # ---- Grab from shared memory
        t0 = time.perf_counter()
        receiver._grab_images()
        t1 = time.perf_counter()

        frame_time = receiver.get_current_timestamp()
        if frame_time <= previous_frame_time:
            continue
        previous_frame_time = frame_time

        # Track FPS
        t_now = time.perf_counter()
        fps_window.append(1.0 / (t_now - t_last_frame))
        t_last_frame = t_now

        if len(fps_window) == fps_window.maxlen:
            print(f"Camera FPS: {np.mean(fps_window):.2f}")

        # ---- RGB
        image = receiver.get_rgb_image_as_int()
        image_right = receiver._retrieve_rgb_image_as_int(view=StereoRGBDCamera.RIGHT_RGB)
        t2 = time.perf_counter()

        # ---- Depth
        depth_map = receiver.get_depth_map()
        depth_image = receiver.get_depth_image()
        t3 = time.perf_counter()

        # ---- Point cloud
        point_cloud = receiver.get_colored_point_cloud()
        t4 = time.perf_counter()

        # ---- Spatial map
        spatial_map = receiver._retrieve_spatial_map()
        t5 = time.perf_counter()

        # ---- Rerun logging
        t_rr0 = time.perf_counter()

        if log_point_cloud:
            pc = point_cloud
            pc.points[np.isnan(pc.points)] = 0
            if pc.colors is not None:
                pc.colors[np.isnan(pc.colors)] = 0

            rr.log(
                "World/point_cloud",
                rr.Points3D(
                    positions=camera_to_rerun(pc.points),
                    colors=pc.colors,
                ),
            )

        if log_spatial_map:
            full_pc = spatial_map.full_pointcloud
            rr.log(
                "World/spatial_map",
                rr.Points3D(
                    positions=camera_to_rerun(full_pc.points),
                    colors=full_pc.colors,
                ),
            )

        pose_matrix = receiver._retrieve_camera_pose()
        translation = pose_matrix[:3, 3]
        rotation_mat = pose_matrix[:3, :3]

        R_x_90 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
        rotation_mat = rotation_mat @ R_x_90

        rr.log(
            "World/Camera",
            rr.Transform3D(
                translation=translation,
                mat3x3=rotation_mat,
            ),
        )

        t_rr1 = time.perf_counter()

        # ---- OpenCV visualization
        cv2.imshow("RGB Image", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imshow("RGB Image Right", cv2.cvtColor(image_right, cv2.COLOR_RGB2BGR))
        cv2.imshow("Depth Image", depth_image)

        # ---- Record timings (ms)
        timings["grab"].append((t1 - t0) * 1e3)
        timings["rgb"].append((t2 - t1) * 1e3)
        timings["depth"].append((t3 - t2) * 1e3)
        timings["pointcloud"].append((t4 - t3) * 1e3)
        timings["spatial_map"].append((t5 - t4) * 1e3)
        timings["rerun"].append((t_rr1 - t_rr0) * 1e3)
        timings["total"].append((time.perf_counter() - t_frame_start) * 1e3)

        # ---- Print stats
        frame_idx += 1
        if frame_idx % WINDOW == 0:
            print("\n=== Throughput breakdown (ms) ===")
            for k, v in timings.items():
                print(f"{k:12s}: {np.mean(v):6.2f}")
            fps = 1000.0 / np.mean(timings["total"])
            print(f"{'FPS':12s}: {fps:6.2f}")

        # ---- Input
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord("l"):
            log_point_cloud = not log_point_cloud

    publisher.stop()
    publisher.join()
