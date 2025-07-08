from __future__ import annotations

from typing import Any, List, Optional

from loguru import logger

try:
    import pyzed.sl as sl

except ImportError:
    raise ImportError(
        "You should install the ZED SDK and pip install the python bindings in your environment first, see the installation README."
    )

# check SDK version
try:
    version = sl.Camera().get_sdk_version()
    assert version.split(".")[0] == "4"
except AssertionError:
    raise ImportError("You should install version 4.X of the SDK!")

import time

import cv2
import numpy as np
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    HomogeneousMatrixType,
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
    """

    # for more info on the different depth modes, see:
    # https://www.stereolabs.com/docs/api/group__Depth__group.html#ga391147e2eab8e101a7ff3a06cbed22da
    # keep in mind though that the depth map is calculated during the `grab`operation, so the depth mode also influences the
    # fps of the rgb images, which is why the default depth mode is None

    NEURAL_DEPTH_MODE = sl.DEPTH_MODE.NEURAL

    # no depth mode, higher troughput of the RGB images as the GPU has to do less work
    # can also turn depth off in the runtime params, which is recommended as it allows for switching at runtime.
    NONE_DEPTH_MODE = sl.DEPTH_MODE.NONE
    PERFORMANCE_DEPTH_MODE = sl.DEPTH_MODE.PERFORMANCE
    QUALITY_DEPTH_MODE = sl.DEPTH_MODE.QUALITY
    ULTRA_DEPTH_MODE = sl.DEPTH_MODE.ULTRA
    DEPTH_MODES = (NEURAL_DEPTH_MODE, NONE_DEPTH_MODE, PERFORMANCE_DEPTH_MODE, QUALITY_DEPTH_MODE, ULTRA_DEPTH_MODE)

    # for info on image resolution, pixel sizes, fov..., see:
    # https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view-
    # make sure to check the combination of frame rates and resolution is available.
    RESOLUTION_2K = (2208, 1242)
    RESOLUTION_1080 = (1920, 1080)
    RESOLUTION_720 = (1280, 720)
    RESOLUTION_VGA = (672, 376)

    resolution_to_identifier_dict = {
        RESOLUTION_2K: sl.RESOLUTION.HD2K,
        RESOLUTION_1080: sl.RESOLUTION.HD1080,
        RESOLUTION_720: sl.RESOLUTION.HD720,
        RESOLUTION_VGA: sl.RESOLUTION.VGA,
    }

    def __init__(  # type: ignore[no-any-unimported]
        self,
        resolution: CameraResolutionType = RESOLUTION_2K,
        fps: int = 15,
        depth_mode: sl.DEPTH_MODE = NONE_DEPTH_MODE,
        serial_number: Optional[str] = None,
        svo_filepath: Optional[str] = None,
    ) -> None:
        self._resolution = resolution
        self._fps = fps
        self.depth_mode = depth_mode
        self.serial_number = int(serial_number) if serial_number else None

        self.camera = sl.Camera()

        # TODO: create a configuration class for the camera parameters
        self.camera_params = sl.InitParameters()

        if self.serial_number:
            self.camera_params.set_from_serial_number(self.serial_number)

        if svo_filepath:
            input_type = sl.InputType()
            input_type.set_from_svo_file(svo_filepath)
            self.camera_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)

        self.camera_params.camera_resolution = Zed.resolution_to_identifier_dict[resolution]
        self.camera_params.camera_fps = fps
        # https://www.stereolabs.com/docs/depth-sensing/depth-settings/
        self.camera_params.depth_mode = depth_mode
        self.camera_params.coordinate_units = sl.UNIT.METER
        # objects closerby will have artifacts so they are filtered out (querying them will give a - Infinty)
        self.camera_params.depth_minimum_distance = 0.3
        self.camera_params.depth_maximum_distance = 10.0  # filter out far away objects

        if self.camera.is_opened():
            # close to open with correct params
            self.camera.close()

        N_OPEN_ATTEMPTS = 5
        for i in range(N_OPEN_ATTEMPTS):
            status = self.camera.open(self.camera_params)
            if status == sl.ERROR_CODE.SUCCESS:
                break
            logger.info(f"Opening Zed camera failed, attempt {i + 1}/{N_OPEN_ATTEMPTS}")
            if self.serial_number:
                logger.info(f"Rebooting {self.serial_number}")
                sl.Camera.reboot(self.serial_number)
            time.sleep(2)
            logger.info(f"Available ZED cameras: {sl.Camera.get_device_list()}")
            self.camera = sl.Camera()

        if status != sl.ERROR_CODE.SUCCESS:
            raise IndexError(f"Could not open Zed camera, error = {status}")

        # TODO: create a configuration class for the runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        # Enabling fill mode changed for SDK 4.0: https://www.stereolabs.com/developers/release/4.0/migration-guide/
        self.runtime_params.enable_fill_mode = False  # standard > fill for accuracy. See docs.
        self.runtime_params.texture_confidence_threshold = 100
        self.runtime_params.confidence_threshold = 100
        self.depth_enabled = True

        # Enable Positional tracking (mandatory for object detection)
        # positional_tracking_parameters = sl.PositionalTrackingParameters()
        # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
        # positional_tracking_parameters.set_as_static = True
        # self.camera.enable_positional_tracking(positional_tracking_parameters)

        # create reusable memory blocks for the measures
        # these will be allocated the first time they are used
        self.image_matrix = sl.Mat()
        self.image_matrix_right = sl.Mat()
        self.depth_image_matrix = sl.Mat()
        self.depth_matrix = sl.Mat()
        self.point_cloud_matrix = sl.Mat()

        self.confidence_matrix = sl.Mat()
        self.confidence_map = None

    @property
    def fps(self) -> int:
        """The frame rate of the camera, in frames per second."""
        return self._fps

    @property
    def resolution(self) -> CameraResolutionType:
        return self._resolution

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
        return self.runtime_params.enable_depth

    @depth_enabled.setter
    def depth_enabled(self, value: bool) -> None:
        self.runtime_params.enable_depth = value

    def _grab_images(self) -> None:
        """grabs (and waits for) the latest image(s) from the camera, rectifies them and computes the depth information (based on the depth mode setting)"""
        # this is a blocking call
        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1Camera.html#a2338c15f49b5f132df373a06bd281822
        # we might want to consider running this in a seperate thread and using a queue to store the images?
        error_code = self.camera.grab(self.runtime_params)
        if error_code != sl.ERROR_CODE.SUCCESS:
            raise IndexError("Could not grab new camera frame")

    def _retrieve_rgb_image(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyFloatImageType:
        image = self._retrieve_rgb_image_as_int(view)
        # convert from int to float image
        # this can take up ~ ms for larger images (can impact FPS)
        image = ImageConverter.from_numpy_int_format(image).image_in_numpy_format
        return image

    def _retrieve_rgb_image_as_int(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyIntImageType:
        assert view in StereoRGBDCamera._VIEWS
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
        assert self.depth_mode != self.NONE_DEPTH_MODE, "Cannot retrieve depth data if depth mode is NONE"
        assert self.depth_enabled, "Cannot retrieve depth data if depth is disabled"
        self.camera.retrieve_measure(self.depth_matrix, sl.MEASURE.DEPTH)
        depth_map = self.depth_matrix.get_data()
        return depth_map

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        assert self.depth_mode != self.NONE_DEPTH_MODE, "Cannot retrieve depth data if depth mode is NONE"
        assert self.depth_enabled, "Cannot retrieve depth data if depth is disabled"
        self.camera.retrieve_image(self.depth_image_matrix, sl.VIEW.DEPTH)
        image_bgra = self.depth_image_matrix.get_data()
        # image = image[..., :3]
        image = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGB)
        return image

    def _retrieve_colored_point_cloud(self) -> PointCloud:
        assert self.depth_mode != self.NONE_DEPTH_MODE, "Cannot retrieve depth data if depth mode is NONE"
        assert self.depth_enabled, "Cannot retrieve depth data if depth is disabled"
        self.camera.retrieve_measure(self.point_cloud_matrix, sl.MEASURE.XYZ)
        # shape (width, height, 4) with the 4th dim being x,y,z,(rgba packed into float)
        # can be nan,nan,nan, nan (no point in the point_cloud on this pixel)
        # or x,y,z, nan (no color information on this pixel??)
        # or x,y,z, value (color information on this pixel)

        point_cloud_XYZ_ = self.point_cloud_matrix.get_data()

        positions = cv2.cvtColor(point_cloud_XYZ_, cv2.COLOR_BGRA2BGR).reshape(-1, 3)
        colors = self._retrieve_rgb_image_as_int().reshape(-1, 3)

        return PointCloud(positions, colors)

    def _retrieve_confidence_map(self) -> NumpyFloatImageType:
        self.camera.retrieve_measure(self.confidence_matrix, sl.MEASURE.CONFIDENCE)
        return self.confidence_matrix.get_data()  # single channel float32 image

    def get_colored_point_cloud(self) -> PointCloud:
        assert self.depth_mode != self.NONE_DEPTH_MODE, "Cannot retrieve depth data if depth mode is NONE"
        assert self.depth_enabled, "Cannot retrieve depth data if depth is disabled"

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

    # test rgbd stereo camera

    with Zed(Zed.RESOLUTION_2K, fps=15, depth_mode=Zed.PERFORMANCE_DEPTH_MODE) as zed:
        print(zed.get_colored_point_cloud().points)  # TODO: test the point_cloud more explicity?
        manual_test_stereo_rgbd_camera(zed)

    # profile rgb throughput, should be at 60FPS, i.e. 0.017s

    zed = Zed(Zed.RESOLUTION_720, fps=60)
    profile_rgb_throughput(zed)


if __name__ == "__main__":
    _test_zed_implementation()
