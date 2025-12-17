from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional

from loguru import logger

try:
    import pyzed.sl as sl
except ImportError:
    raise ImportError(
        "You should install the ZED SDK and pip install the python bindings in your environment first, see the installation README."
    )

# check SDK version
_sdk_version = sl.Camera().get_sdk_version()
if _sdk_version.split(".")[0] != "5":
    raise ImportError("You should install version 5.X of the SDK!")

import time

import cv2
import numpy as np
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    HomogeneousMatrixType,
    NumpyConfidenceMapType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
    OpenCVIntImageType,
    PointCloud,
)


class Zed(StereoRGBDCamera):
    """
    Wrapper around the ZED SDK
    https://www.stereolabs.com/zed-2i/

    It is important to note that the ZED cameras are factory calibrated and hence provide undistorted images
    and corresponding intrinsics matrices.

    Also note that all depth values are relative to the left camera.

    Next to basic rgb frames and depth maps, The ZED-sdk provides other frameworks like spatial mapping or positional tracking:
    - Spatial mapping (also called 3D reconstruction) is the ability to create a 3D mesh of the environment.
      This mesh can be a pointcloud or a triangular mesh. Here we assume that a pointcloud is used.
      More information can be found here: https://www.stereolabs.com/docs/spatial-mapping
    - Positional tracking is the ability to track the pose of the camera in space over time.
      More information can be found here: https://www.stereolabs.com/docs/positional-tracking
    """

    @dataclass
    class InitParams:
        """
        InitParams is a dataclass that encapsulates initialization parameters for configuring a ZED camera.
        It also provides class-level aliases for commonly used depth modes and resolutions,
        as well as dictionaries for conversion between ZED resolution types and native airo-mono resolution types.

        Instance Attributes:
            resolution (CameraResolutionType): Selected camera resolution. Defaults to RESOLUTION_2K.
            fps (int): Frames per second. Defaults to 15.
            depth_mode (sl.DEPTH_MODE): Selected depth mode. Defaults to NONE_DEPTH_MODE.
            serial_number (Optional[str]): Camera serial number for device selection.
            svo_filepath (Optional[str]): Path to SVO file for playback.

        Methods:
            build_zed_init_params(self) -> sl.InitParameters:
                Builds and returns an sl.InitParameters object configured with the current instance attributes.
                Sets camera resolution, FPS, depth mode, coordinate units, and depth range.
                If serial_number is provided, sets it in the parameters.
                If svo_filepath is provided, configures input type for SVO playback.
        """

        # ---------------- Constants & aliases for camera initialization parameters -----------------

        # for more info on the different depth modes, see:
        # https://www.stereolabs.com/docs/depth-sensing/depth-modes
        # keep in mind though that the depth map is calculated during the `grab`operation, so the depth mode also influences the
        # fps of the rgb images, which is why the default depth mode is None

        NEURAL_DEPTH_MODE: ClassVar[sl.DEPTH_MODE] = sl.DEPTH_MODE.NEURAL
        NEURAL_LIGHT_DEPTH_MODE: ClassVar[sl.DEPTH_MODE] = sl.DEPTH_MODE.NEURAL_LIGHT
        NEURAL_PLUS_DEPTH_MODE: ClassVar[sl.DEPTH_MODE] = sl.DEPTH_MODE.NEURAL_PLUS

        # no depth mode, higher troughput of the RGB images as the GPU has to do less work
        # can also turn depth off in the runtime params, which is recommended as it allows for switching at runtime.
        NONE_DEPTH_MODE: ClassVar[sl.DEPTH_MODE] = sl.DEPTH_MODE.NONE
        DEPTH_MODES: ClassVar[tuple[sl.DEPTH_MODE, ...]] = (
            NEURAL_DEPTH_MODE,
            NONE_DEPTH_MODE,
            NEURAL_LIGHT_DEPTH_MODE,
            NEURAL_PLUS_DEPTH_MODE,
        )

        # for info on image resolution, pixel sizes, fov..., see:
        # https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view-
        # make sure to check the combination of frame rates and resolution is available.
        RESOLUTION_2K: ClassVar[CameraResolutionType] = (2208, 1242)
        RESOLUTION_1080: ClassVar[CameraResolutionType] = (1920, 1080)
        RESOLUTION_720: ClassVar[CameraResolutionType] = (1280, 720)
        RESOLUTION_VGA: ClassVar[CameraResolutionType] = (672, 376)

        resolution_to_identifier_dict: ClassVar[dict[tuple[int, int], sl.RESOLUTION]] = {
            RESOLUTION_2K: sl.RESOLUTION.HD2K,
            RESOLUTION_1080: sl.RESOLUTION.HD1080,
            RESOLUTION_720: sl.RESOLUTION.HD720,
            RESOLUTION_VGA: sl.RESOLUTION.VGA,
        }

        # ---------------- Instance attributes and methods ----------------

        resolution: CameraResolutionType
        fps: int
        depth_mode: sl.DEPTH_MODE
        serial_number: Optional[str]
        svo_filepath: Optional[str]

        def build_zed_init_params(self) -> sl.InitParameters:  # type: ignore
            init_params = sl.InitParameters()

            if self.serial_number:
                init_params.set_from_serial_number(self.serial_number)

            if self.svo_filepath:
                input_type = sl.InputType()
                input_type.set_from_svo_file(self.svo_filepath)
                init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)

            # set the init parameters
            init_params.camera_resolution = self.resolution_to_identifier_dict[self.resolution]
            init_params.camera_fps = self.fps
            # https://www.stereolabs.com/docs/depth-sensing/depth-settings/
            init_params.depth_mode = self.depth_mode
            init_params.coordinate_units = sl.UNIT.METER
            # objects closerby will have artifacts so they are filtered out (querying them will give a - Infinty)
            init_params.depth_minimum_distance = 0.3
            init_params.depth_maximum_distance = 10.0  # filter out far away objects

            return init_params

    @dataclass
    class MappingParams:
        """
        MappingParams is a dataclass that encapsulates spatial mapping parameters for configuring a ZED camera.
        It also provides class-level aliases for commonly used mapping resolutions and ranges.

        Instance Attributes:
            mapping_resolution (sl.SPATIAL_MAPPING_RESOLUTION): Selected mapping resolution. Defaults to MAPPING_RESOLUTION_MEDIUM.
            mapping_range (sl.SPATIAL_MAPPING_RANGE): Selected mapping range. Defaults to MAPPING_RANGE_SHORT.
            max_memory_usage (int): The maximum CPU memory (in MB) allocated for the meshing process. Defaults to 2048.
            chunk_only (bool): Whether to only use chunks for mapping. Defaults to False.

        Methods:
            build_zed_mapping_params(self) -> sl.SpatialMappingParameters:
                Builds and returns an sl.SpatialMappingParameters object configured with the current instance attributes.
                Sets mapping type, resolution, range, and chunk-only option.
        """

        # ---------------- Constants & aliases for camera spatial mapping parameters -----------------

        # for more info on the different mapping resolutions and ranges, see:
        # https://www.stereolabs.com/docs/spatial-mapping/using-mapping
        # note that mapping on the HIGH resolution setting is resource-intensive, and slows down spatial map updates.
        MAPPING_RESOLUTION_LOW: ClassVar[sl.MAPPING_RESOLUTION] = sl.MAPPING_RESOLUTION.LOW  # resolution of 2cm
        MAPPING_RESOLUTION_MEDIUM: ClassVar[sl.MAPPING_RESOLUTION] = sl.MAPPING_RESOLUTION.MEDIUM  # resolution of 5cm
        MAPPING_RESOLUTION_HIGH: ClassVar[sl.MAPPING_RESOLUTION] = sl.MAPPING_RESOLUTION.HIGH  # resolution of 8cm

        MAPPING_RANGE_SHORT: ClassVar[sl.MAPPING_RANGE] = sl.MAPPING_RANGE.SHORT  # integrates depth up to 3.5m
        MAPPING_RANGE_MEDIUM: ClassVar[sl.MAPPING_RANGE] = sl.MAPPING_RANGE.MEDIUM  # integrates depth up to 5m
        MAPPING_RANGE_FAR: ClassVar[sl.MAPPING_RANGE] = sl.MAPPING_RANGE.LONG  # integrates depth up to 10m

        # ---------------- Instance attributes and methods ----------------

        mapping_resolution: sl.SPATIAL_MAPPING_RESOLUTION = MAPPING_RESOLUTION_MEDIUM
        mapping_range: sl.SPATIAL_MAPPING_RANGE = MAPPING_RANGE_SHORT
        max_memory_usage: int = 2048
        chunk_only: bool = False

        def build_zed_mapping_params(self) -> sl.SpatialMappingParameters:  # type: ignore
            mapping_params = sl.SpatialMappingParameters(map_type=sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD)

            # set the mapping parameters
            mapping_params.set_resolution(self.mapping_resolution)
            mapping_params.set_range(self.mapping_range)
            mapping_params.max_memory_usage = self.max_memory_usage
            mapping_params.use_chunk_only = self.chunk_only

            return mapping_params

    @dataclass
    class TrackingParams:
        """
        TrackingParams is a dataclass that encapsulates positional tracking parameters for configuring a ZED camera.
        It also provides class-level aliases for commonly used reference frames.

        Instance Attributes:
            initial_world_transform (HomogeneousMatrixType): Position of the camera in the world frame at the start of tracking.
                                                             Defaults to identity matrix.
            enable_pose_smoothing (bool): Whether to enable smooth pose correction for small drift correction. Defaults to False.
            set_floor_as_origin (bool): Initializes the tracking to be aligned with the floor plane to better position the camera in space.
                                        Defaults to False.
            align_with_gravity (bool): Whether to override 2 of the 3 rotations from initial_world_transform using the IMU gravity.
                                       Defaults to False.
            static_camera (bool): Whether the camera is static. Defaults to False.
            enable_2d_ground_mode (bool): Whether to constrain the positional tracking to a 2D plane. Defaults to False.

        Methods:
            build_zed_tracking_params(self) -> sl.PositionalTrackingParameters:
                Builds and returns an sl.PositionalTrackingParameters object configured with the current instance attributes.
                Sets gravity as origin and static camera options.
        """

        # ---------------- Constants & aliases for camera positional tracking parameters -----------------

        REFERENCE_FRAME_WORLD: ClassVar[sl.REFERENCE_FRAME] = sl.REFERENCE_FRAME.WORLD
        REFERENCE_FRAME_CAMERA: ClassVar[sl.REFERENCE_FRAME] = sl.REFERENCE_FRAME.CAMERA

        # ---------------- Instance attributes and methods ----------------

        initial_world_transform: HomogeneousMatrixType = np.eye(4)
        enable_pose_smoothing: bool = False
        set_floor_as_origin: bool = False
        align_with_gravity: bool = False
        static_camera: bool = False
        enable_2d_ground_mode: bool = False

        def build_zed_tracking_params(self) -> sl.PositionalTrackingParameters:  # type: ignore
            tracking_params = sl.PositionalTrackingParameters()

            # set the tracking parameters
            zed_transform = sl.Transform()
            zed_transform.m[:, :] = self.initial_world_transform.astype(np.float32)

            tracking_params.set_initial_world_transform(zed_transform)
            tracking_params.enable_pose_smoothing = self.enable_pose_smoothing
            tracking_params.set_floor_as_origin = self.set_floor_as_origin
            tracking_params.set_gravity_as_origin = self.align_with_gravity
            # If the camera is static, this setting provides better performances and makes boxes stick to the ground.
            tracking_params.set_as_static = self.static_camera
            tracking_params.enable_2d_ground_mode = self.enable_2d_ground_mode

            return tracking_params

    @dataclass
    class RuntimeParams:
        """
        RuntimeParams is a dataclass that encapsulates runtime parameters for configuring a ZED camera.

        Instance Attributes:
            enable_fill_mode (bool): Whether to enable fill mode for depth computation. Defaults to False
            texture_confidence_threshold (int): Threshold to reject depth values based on their texture confidence, defaults to 100.
                                                It can be seen as a probability of error, scaled to 100.
            confidence_threshold (int): Threshold to reject depth values based on their confidence, defaults to 100.
                                        It can be seen as a probability of error, scaled to 100.
            depth_enabled (bool): Whether to compute the depth map. Defaults to True.

        Methods:
            build_zed_runtime_params(self) -> sl.RuntimeParameters:
                Builds and returns an sl.RuntimeParameters object configured with the current instance attributes.
                Sets fill mode, texture confidence threshold, confidence threshold, and depth enable options.
        """

        # ---------------- Instance attributes and methods ----------------

        enable_fill_mode: bool = False
        texture_confidence_threshold: int = 100
        confidence_threshold: int = 100
        depth_enabled: bool = True

        def build_zed_runtime_params(self) -> sl.RuntimeParameters:  # type: ignore
            runtime_params = sl.RuntimeParameters()

            # set runtime parameters
            # Enabling fill mode changed for SDK 4.0: https://www.stereolabs.com/developers/release/4.0/migration-guide/
            runtime_params.enable_fill_mode = self.enable_fill_mode  # standard > fill for accuracy. See docs.
            runtime_params.texture_confidence_threshold = self.texture_confidence_threshold
            runtime_params.confidence_threshold = self.confidence_threshold
            runtime_params.enable_depth = self.depth_enabled

            return runtime_params

    def __init__(
        self,
        resolution: CameraResolutionType = InitParams.RESOLUTION_2K,
        fps: int = 15,
        depth_mode: sl.DEPTH_MODE = InitParams.NONE_DEPTH_MODE,
        serial_number: Optional[str] = None,
        svo_filepath: Optional[str] = None,
        camera_runtime_params: Optional[sl.RuntimeParameters] = None,
        camera_tracking_params: Optional[sl.PositionalTrackingParameters] = None,
        camera_mapping_params: Optional[sl.SpatialMappingParameters] = None,
    ) -> None:

        # Create init parameters
        self.camera_init_params = self.InitParams(
            resolution=resolution,
            fps=fps,
            depth_mode=depth_mode,
            serial_number=serial_number,
            svo_filepath=svo_filepath,
        )
        self._zed_init_params = self.camera_init_params.build_zed_init_params()

        # Create runtime parameters
        self.camera_runtime_params = (
            camera_runtime_params if camera_runtime_params is not None else self.RuntimeParams()
        )
        self._zed_runtime_params = self.camera_runtime_params.build_zed_runtime_params()

        # Create tracking parameters
        self.camera_tracking_params = camera_tracking_params  # None if positional tracking not enabled
        if self.camera_tracking_params:
            self._zed_tracking_params = self.camera_tracking_params.build_zed_tracking_params()

        # Create mapping parameters
        self.camera_mapping_params = camera_mapping_params  # None if spatial mapping not enabled
        if self.camera_mapping_params:
            self._zed_mapping_params = self.camera_mapping_params.build_zed_mapping_params()

        # create camera object
        self.camera = sl.Camera()

        # if camera is open, close it first and re-open with correct parameters
        if self.camera.is_opened():
            self.camera.close()

        # Start the camera with the specified parameters & settings
        self._start_camera()

        # create reusable memory blocks for the measures
        # these will be allocated the first time they are used
        self.image_matrix = sl.Mat()
        self.image_matrix_right = sl.Mat()
        self.depth_image_matrix = sl.Mat()
        self.depth_matrix = sl.Mat()
        self.point_cloud_matrix = sl.Mat()
        self.pose = sl.Pose()
        self.spatial_map = sl.FusedPointCloud()

        self.confidence_matrix = sl.Mat()
        self.confidence_map = None

    def _start_camera(self) -> None:
        # try to open the camera
        N_OPEN_ATTEMPTS = 5
        for i in range(N_OPEN_ATTEMPTS):
            init_status = self.camera.open(self._zed_init_params)
            if init_status == sl.ERROR_CODE.SUCCESS:
                break
            logger.info(f"Opening Zed camera failed, attempt {i + 1}/{N_OPEN_ATTEMPTS}")
            if self.camera_init_params.serial_number:
                logger.info(f"Rebooting {self.camera_init_params.serial_number}")
                sl.Camera.reboot(self.camera_init_params.serial_number)
            time.sleep(2)
            logger.info(f"Available ZED cameras: {sl.Camera.get_device_list()}")
            self.camera = sl.Camera()

        if init_status != sl.ERROR_CODE.SUCCESS:
            logger.error(
                "Could not open Zed camera. Sometimes, unplugging the camera and plugging it back in helps.\n"
                "Alternatively, try to reboot the camera by running `ZED_Explorer --reboot` in a terminal."
            )
            raise RuntimeError(f"Could not open Zed camera, error = {init_status}")

        # Enable Positional tracking and Spatial Mapping if parameters are provided
        if self.camera_tracking_params:
            tracking_status = self.camera.enable_positional_tracking(self._zed_tracking_params)
            if tracking_status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Could not enable positional tracking on Zed camera, error = {tracking_status}")
            logger.info("Positional tracking enabled on Zed camera.")
        if self.camera_mapping_params:
            if not self.camera_tracking_params:
                raise RuntimeError("Positional tracking must be enabled before spatial mapping can be enabled.")
            mapping_status = self.camera.enable_spatial_mapping(self._zed_mapping_params)
            if mapping_status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Could not enable spatial mapping on Zed camera, error = {mapping_status}")
            logger.info("Spatial mapping enabled on Zed camera.")

    @property
    def fps(self) -> int:
        """The frame rate of the camera, in frames per second."""
        return self.camera_init_params.fps

    @property
    def resolution(self) -> CameraResolutionType:
        return self.camera_init_params.resolution

    def intrinsics_matrix(self, view: str = StereoRGBDCamera.LEFT_RGB) -> CameraIntrinsicsMatrixType:
        # get the 'rectified' intrinsics matrices.
        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1CameraParameters.html
        if view == self.LEFT_RGB:
            fx = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx
            fy = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.fy
            cx = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.cx
            cy = self.camera.get_camera_information().camera_configuration.calibration_parameters.left_cam.cy
        elif view == self.RIGHT_RGB:
            fx = self.camera.get_camera_information().camera_configuration.calibration_parameters.right_cam.fx
            fy = self.camera.get_camera_information().camera_configuration.calibration_parameters.right_cam.fy
            cx = self.camera.get_camera_information().camera_configuration.calibration_parameters.right_cam.cx
            cy = self.camera.get_camera_information().camera_configuration.calibration_parameters.right_cam.cy
        else:
            raise ValueError(f"view must be one of {self._VIEWS}")
        cam_matrix = np.zeros((3, 3))
        cam_matrix[0, 0] = fx
        cam_matrix[1, 1] = fy
        cam_matrix[2, 2] = 1
        cam_matrix[0, 2] = cx
        cam_matrix[1, 2] = cy
        return cam_matrix

    @property
    def pose_of_right_view_in_left_view(self) -> HomogeneousMatrixType:
        # get the 'rectified' pose of the right view wrt to the left view
        # should be approx a translation along the x-axis of 120mm (Zed2i camera), expressed in the unit of the coordinates, which we set to meters.
        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1CalibrationParameters.html
        # Note: the CalibrationParameters class changed for the SDK 4.0, the old .T attribute we used was removed.
        return self.camera.get_camera_information().camera_configuration.calibration_parameters.stereo_transform.m

    @property
    def depth_enabled(self) -> bool:
        """Runtime parameter to enable/disable the depth & point_cloud computation. This speeds up the RGB image capture."""
        return self.camera_runtime_params.depth_enabled

    @depth_enabled.setter
    def depth_enabled(self, value: bool) -> None:
        self.camera_runtime_params.depth_enabled = value
        self._zed_runtime_params.enable_depth = value

    def _grab_images(self) -> None:
        """grabs (and waits for) the latest image(s) from the camera, rectifies them and computes the depth information (based on the depth mode setting)"""
        # this is a blocking call
        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1Camera.html#a2338c15f49b5f132df373a06bd281822
        # we might want to consider running this in a seperate thread and using a queue to store the images?
        error_code = self.camera.grab(self._zed_runtime_params)
        if error_code != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Could not grab new camera frame")

    def _retrieve_rgb_image(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyFloatImageType:
        image = self._retrieve_rgb_image_as_int(view)
        # convert from int to float image
        # this can take up ~ ms for larger images (can impact FPS)
        image = ImageConverter.from_numpy_int_format(image).image_in_numpy_format
        return image

    def _retrieve_rgb_image_as_int(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyIntImageType:
        if view not in StereoRGBDCamera._VIEWS:
            raise ValueError(f"view must be one of {self._VIEWS}, but was {view}")
        image_bgra: OpenCVIntImageType
        if view == StereoRGBDCamera.RIGHT_RGB:
            self.camera.retrieve_image(self.image_matrix_right, sl.VIEW.RIGHT)
            image_bgra = self.image_matrix_right.get_data()
        else:
            self.camera.retrieve_image(self.image_matrix, sl.VIEW.LEFT)
            image_bgra = self.image_matrix.get_data()

        image = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGB)
        return image

    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        if self.camera_init_params.depth_mode == self.InitParams.NONE_DEPTH_MODE:
            raise RuntimeError("Cannot retrieve depth data if depth mode is NONE")
        if not self.camera_runtime_params.depth_enabled:
            raise RuntimeError("Cannot retrieve depth data if depth is disabled")
        self.camera.retrieve_measure(self.depth_matrix, sl.MEASURE.DEPTH)
        depth_map = self.depth_matrix.get_data()
        return depth_map

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        if self.camera_init_params.depth_mode == self.InitParams.NONE_DEPTH_MODE:
            raise RuntimeError("Cannot retrieve depth data if depth mode is NONE")
        if not self.camera_runtime_params.depth_enabled:
            raise RuntimeError("Cannot retrieve depth data if depth is disabled")
        self.camera.retrieve_image(self.depth_image_matrix, sl.VIEW.DEPTH)
        image_bgra = self.depth_image_matrix.get_data()
        # image = image[..., :3]
        image = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGB)
        return image

    def _retrieve_colored_point_cloud(self) -> PointCloud:
        if self.camera_init_params.depth_mode == self.InitParams.NONE_DEPTH_MODE:
            raise RuntimeError("Cannot retrieve depth data if depth mode is NONE")
        if not self.camera_runtime_params.depth_enabled:
            raise RuntimeError("Cannot retrieve depth data if depth is disabled")
        self.camera.retrieve_measure(self.point_cloud_matrix, sl.MEASURE.XYZ)
        # shape (width, height, 4) with the 4th dim being x,y,z,(rgba packed into float)
        # can be nan,nan,nan, nan (no point in the point_cloud on this pixel)
        # or x,y,z, nan (no color information on this pixel??)
        # or x,y,z, value (color information on this pixel)

        point_cloud_XYZ_ = self.point_cloud_matrix.get_data()

        positions = cv2.cvtColor(point_cloud_XYZ_, cv2.COLOR_BGRA2BGR).reshape(-1, 3)
        colors = self._retrieve_rgb_image_as_int().reshape(-1, 3)

        return PointCloud(positions, colors)

    def _retrieve_confidence_map(self) -> NumpyConfidenceMapType:
        self.camera.retrieve_measure(self.confidence_matrix, sl.MEASURE.CONFIDENCE)
        zed_confidence_map = self.confidence_matrix.get_data()  # single channel float32 image
        # The ZED confidence map is in the range [0, 100], where 0 is the highest confidence and 100 the lowest.
        # See: https://www.stereolabs.com/docs/depth-sensing/depth-settings#depth-confidence-filtering
        # We convert it to a more standard range [0, 1], where 0 is the lowest confidence and 1 the highest.
        confidence_map = 1 - zed_confidence_map / 100.0
        return confidence_map

    def _retrieve_camera_pose(
        self, coordinate_frame: sl.REFERENCE_FRAME = TrackingParams.REFERENCE_FRAME_WORLD
    ) -> HomogeneousMatrixType:
        """
        Retrieves the current camera pose as a 4x4 homogeneous transformation matrix in the specified coordinate frame.

        If the world frame is used, the pose is relative to the initial camera pose when tracking started.
        The initial world transformation is also taken into account.

        If the camera frame is used, the pose relates to the previous pose of the camera.
        """
        if not self.camera_tracking_params:
            raise RuntimeError("Cannot retrieve pose if positional tracking is not enabled.")

        positional_tracking_state = self.camera.get_position(self.pose, coordinate_frame)
        if positional_tracking_state != sl.POSITIONAL_TRACKING_STATE.OK:
            logger.warning(f"Positional tracking state is not OK: {positional_tracking_state}")
            logger.warning("Returning the last known pose.")
        pose_transform = self.pose.pose_data()
        pose_np = pose_transform.m
        return pose_np

    def _request_spatial_map_update(self) -> None:
        """
        Request a the spatial map update process in a non-blocking thread from the spatial mapping process.
        This function will trigger the generation of a mesh without blocking the program.
        Note: Only one mesh can be generated at a time, if the previous mesh generation is not over, new function calls will be ignored.
        """
        if not self.camera_mapping_params:
            raise RuntimeError("Cannot request spatial map update if spatial mapping is not enabled.")
        self.camera.request_spatial_map_async()

    def _get_spatial_map(self) -> list[tuple[PointCloud, bool]]:
        """
        Retrieves the current spatial map from the camera as a list of point clouds with update status.
        Returns:
            list[tuple[PointCloud, bool]]: A list of tuples, each containing a PointCloud object and a boolean
                                           indicating whether the corresponding chunk has been updated.
        Raises:
            RuntimeError: If spatial mapping is not enabled.
        Notes:
            - If the spatial map update is not ready or cannot be retrieved, the last known spatial map is returned.
            - Each chunk in the spatial map is processed to extract 3D points and their corresponding RGB values.
            - Chunks with no vertices are skipped.
        """
        if not self.camera_mapping_params:
            raise RuntimeError("Cannot retrieve spatial map if spatial mapping is not enabled.")

        if self.camera.get_spatial_map_request_status_async() != sl.ERROR_CODE.SUCCESS:
            logger.info("Spatial map update not ready yet.")
            logger.info("Returning last known spatial map.")
        elif self.camera.retrieve_spatial_map_async(self.spatial_map) != sl.ERROR_CODE.SUCCESS:
            logger.warning("Could not retrieve spatial map.")
            logger.warning("Returning last known spatial map.")

        spatial_map = []

        # process per chunk
        for chunk in self.spatial_map.chunks:
            if chunk.vertices.size == 0:
                continue

            # extract points
            points = cv2.cvtColor(chunk.vertices, cv2.COLOR_BGRA2BGR).reshape(-1, 3)

            # extract RGB values
            rgba_float = chunk.vertices[:, 3]
            rgba_uint32 = rgba_float.view(np.uint32)
            r = (rgba_uint32 >> 0) & 0xFF
            g = (rgba_uint32 >> 8) & 0xFF
            b = (rgba_uint32 >> 16) & 0xFF
            rgb_values = np.stack([r, g, b], axis=1).astype(np.uint8)

            # extend spatial map
            spatial_map.append((PointCloud(points, rgb_values), chunk.has_been_updated))

        return spatial_map

    def get_colored_point_cloud(self) -> PointCloud:
        if self.camera_init_params.depth_mode == self.InitParams.NONE_DEPTH_MODE:
            raise RuntimeError("Cannot retrieve depth data if depth mode is NONE")
        if not self.camera_runtime_params.depth_enabled:
            raise RuntimeError("Cannot retrieve depth data if depth is disabled")

        self._grab_images()
        return self._retrieve_colored_point_cloud()

    @staticmethod
    def list_camera_serial_numbers() -> List[str]:
        """
        List all connected ZED cameras
        can be used to select a device ID or to check if cameras are connected.
        """
        device_list = sl.Camera.get_device_list()
        return device_list

    # manage resources
    # this is important if you want to reuse the camera
    # multiple times within a python script, in which case you should release the camera before creating a new object.
    # cf. https://stackoverflow.com/questions/865115/how-do-i-correctly-clean-up-a-python-object
    def __enter__(self) -> StereoRGBDCamera:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.camera.close()


def _test_zed_implementation() -> None:
    """this script serves as a 'test' for the zed implementation."""
    from airo_camera_toolkit.cameras.manual_test_hw import manual_test_stereo_rgbd_camera, profile_rgb_throughput

    # zed specific tests:
    # - list all serial numbers of the cameras
    serial_numbers = Zed.list_camera_serial_numbers()
    print(serial_numbers)
    input("each camera connected to the pc should be listed, press enter to continue")

    with Zed(depth_mode=Zed.InitParams.NEURAL_DEPTH_MODE) as zed:
        print(zed.get_colored_point_cloud().points)  # TODO: test the point_cloud more explicity?
        manual_test_stereo_rgbd_camera(zed)

    # profile rgb throughput, should be at 60FPS, i.e. 0.017s
    with Zed(resolution=Zed.InitParams.RESOLUTION_720, fps=60) as zed:
        profile_rgb_throughput(zed)

    time.sleep(5)

    # Test pose retrieval and spatial mapping
    # Note that it is normal that no spatial map is retrieved the first few frames.
    print("Testing pose retrieval and spatial mapping...")
    tracking_params = Zed.TrackingParams(align_with_gravity=True)
    mapping_params = Zed.MappingParams()
    with Zed(
        depth_mode=Zed.InitParams.NEURAL_DEPTH_MODE,
        camera_tracking_params=tracking_params,
        camera_mapping_params=mapping_params,
    ) as zed:
        for _ in range(20):  # grab 20 frames
            zed._grab_images()
            pose = zed._retrieve_camera_pose()

            print(f"Current pose:\n{pose}")
            zed._request_spatial_map_update()
            time.sleep(0.1)  # wait a bit before next grab

            spatial_map = zed._get_spatial_map()
            if not spatial_map:
                logger.warning("Spatial map is empty.")

            num_points = 0
            for i, (chunk, updated) in enumerate(spatial_map):
                num_points += chunk.points.shape[0]
            print(f"Spatial map with {len(spatial_map)} chunks and {num_points} points in total retrieved.")


if __name__ == "__main__":
    _test_zed_implementation()
