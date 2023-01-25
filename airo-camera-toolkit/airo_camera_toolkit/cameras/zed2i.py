from __future__ import annotations

from typing import Any, List, Optional

try:
    import pyzed.sl as sl
except ImportError:
    raise ImportError(
        "You should install the ZED SDK and pip install the python bindings in your environment first, see the installation README."
    )

import numpy as np
from airo_camera_toolkit.cameras.test_hw import manual_test_stereo_rgbd_camera, profile_rgb_throughput
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.utils import ImageConverter
from airo_typing import (
    CameraIntrinsicsMatrixType,
    HomogeneousMatrixType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
    OpenCVIntImageType,
)


class Zed2i(StereoRGBDCamera):
    """
    Wrapper around the ZED2i SDK
    https://www.stereolabs.com/zed-2i/

    It is important to note that the ZED cameras are factory calibrated and hence provide undistorted images
    and corresponding intrinsics matrices.
    """

    # for more info on the different depth modes, see:
    # https://www.stereolabs.com/docs/api/group__Depth__group.html#ga391147e2eab8e101a7ff3a06cbed22da
    # keep in mind though that the depth map is calculated during the `grab`operation, so the depth mode also influences the
    # fps of the rgb images, which is why the default depth mode is None

    NEURAL_DEPTH_MODE = sl.DEPTH_MODE.NEURAL
    NONE_DEPTH_MODE = (
        sl.DEPTH_MODE.NONE
    )  # no depth mode, higher troughput of the RGB images as the GPU has to do less work
    PERFORMANCE_DEPTH_MODE = sl.DEPTH_MODE.QUALITY
    QUALITY_DEPTH_MODE = sl.DEPTH_MODE.QUALITY
    DEPTH_MODES = (NEURAL_DEPTH_MODE, NONE_DEPTH_MODE, PERFORMANCE_DEPTH_MODE, QUALITY_DEPTH_MODE)

    # for info on image resolution, pixel sizes, fov..., see:
    # https://support.stereolabs.com/hc/en-us/articles/360007395634-What-is-the-camera-focal-length-and-field-of-view-
    # make sure to check the combination of frame rates and resolution is available.
    RESOLUTION_720 = sl.RESOLUTION.HD720  # (1280x720)
    RESOLUTION_1080 = sl.RESOLUTION.HD1080  # (1920x1080)
    RESOLUTION_2K = sl.RESOLUTION.HD2K  # (2208 x 1242 )
    RESOLUTION_VGA = sl.RESOLUTION.VGA  # (672x376)
    RESOLUTIONS = (RESOLUTION_720, RESOLUTION_1080, RESOLUTION_2K, RESOLUTION_VGA)

    def __init__(  # type: ignore[no-any-unimported]
        self,
        resolution: sl.RESOLUTION = RESOLUTION_2K,
        fps: int = 15,
        depth_mode: str = NONE_DEPTH_MODE,
        serial_number: Optional[int] = None,
    ) -> None:
        self.resolution = resolution
        self.fps = fps
        self.depth_mode = depth_mode
        self.serial_number = serial_number

        self.camera = sl.Camera()
        self.camera_params = sl.InitParameters()
        self.camera_params.camera_resolution = resolution
        self.camera_params.camera_fps = fps
        if serial_number:
            self.camera_params.set_from_serial_number(serial_number)

        # https://www.stereolabs.com/docs/depth-sensing/depth-settings/
        self.camera_params.depth_mode = depth_mode
        self.camera_params.coordinate_units = sl.UNIT.METER
        self.camera_params.depth_minimum_distance = (
            0.3  # objects closerby will have artifacts so they are filtered out (querying them will give a - Infinty)
        )
        self.camera_params.depth_maximum_distance = 2.0  # filter out far away objects

        if self.camera.is_opened():
            # close to open with correct params
            self.camera.close()

        status = self.camera.open(self.camera_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise IndexError(f"could not open camera, error = {status}")

        self.runtime_params = sl.RuntimeParameters()
        self.runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD  # standard > fill for accuracy. See docs.

        self.image_matrix = sl.Mat()  # allocate memory for RGB view
        self.depth_matrix = sl.Mat()  # allocate memory for the depth map

    @property
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
        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1CalibrationParameters.html#a99ec1eeeb66c781c27b574fdc36881d2
        matrix = np.eye(4)
        matrix[:3, 3] = self.camera.get_camera_information().camera_configuration.calibration_parameters.T
        return matrix

    def _grab_latest_image(self) -> None:
        """grabs (and waits for) the latest image(s) from the camera, rectifies them and computes the depth information (based on the depth mode setting)"""
        # this is a blocking call
        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1Camera.html#a2338c15f49b5f132df373a06bd281822
        # we might want to consider running this in a seperate thread and using a queue to store the images?
        error_code = self.camera.grab(self.runtime_params)
        if error_code != sl.ERROR_CODE.SUCCESS:
            raise IndexError("Could not grab new camera frame")

    def get_depth_map(self) -> NumpyDepthMapType:
        assert self.depth_mode != self.NONE_DEPTH_MODE, "Cannot retrieve depth data if depth mode is NONE"
        self._grab_latest_image()
        self.camera.retrieve_measure(self.depth_matrix, sl.MEASURE.DEPTH)
        depth_map = self.depth_matrix.get_data()
        return depth_map

    def get_depth_image(self) -> NumpyIntImageType:
        assert self.depth_mode != self.NONE_DEPTH_MODE, "Cannot retrieve depth data if depth mode is NONE"
        self._grab_latest_image()
        self.camera.retrieve_image(self.image_matrix, sl.VIEW.DEPTH)
        image = self.image_matrix.get_data()
        image = image[..., :3]  # drop alpha channel
        image = image[..., ::-1]  # BGR to RGB
        return image

    def get_rgb_image(self, view: str = StereoRGBDCamera.LEFT_RGB) -> NumpyFloatImageType:
        assert view in StereoRGBDCamera._VIEWS

        self._grab_latest_image()
        if view == StereoRGBDCamera.RIGHT_RGB:
            view = sl.VIEW.RIGHT
        else:
            view = sl.VIEW.LEFT
        self.camera.retrieve_image(self.image_matrix, view)
        image: OpenCVIntImageType = self.image_matrix.get_data()
        image = image[..., :3]  # remove alpha channel
        # convert from int to float image
        # this can take up ~ ms for larger images (can impact FPS)
        return ImageConverter.from_opencv_format(image).image_in_numpy_format

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


if __name__ == "__main__":
    """this script serves as a 'test' for the zed implementation."""

    # zed specific tests:
    # - list all serial numbers of the cameras
    serial_numbers = Zed2i.list_camera_serial_numbers()
    print(serial_numbers)
    input("each camera connected to the pc should be listed, press enter to continue")

    # test rgbd stereo camera
    with Zed2i(Zed2i.RESOLUTION_2K, fps=15, depth_mode=Zed2i.PERFORMANCE_DEPTH_MODE) as zed:
        manual_test_stereo_rgbd_camera(zed)

    # profile rgb throughput, should be at 60FPS, i.e. 0.017s
    zed = Zed2i(Zed2i.RESOLUTION_720, fps=60)
    profile_rgb_throughput(zed)
