from __future__ import annotations

import time
from typing import Any, Optional

import cv2
import numpy as np
import pyrealsense2 as rs  # type: ignore
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    NumpyConfidenceMapType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
)
from loguru import logger


class Realsense(RGBDCamera):
    """Wrapper around the pyrealsense2 library to use the RealSense cameras (tested for the D415 and D435).

    Design decisions we made for this class:
    * Depth and color fps are the same
    * Depth resolution is automatically set
    * Depth frames are always aligned to color frames
    * Hole filling is enabled by default
    * Getting the confidence map is optional (disabled by default). The L515 supports a built-in confidence map, but the D415 and D435 do not. We do not support the L515 at this time.
    """

    # Built-in resolutions (16:9 aspect ratio) for convenience
    # for all resolutions see: realsense_scan_profiles.py
    RESOLUTION_1080 = (1920, 1080)
    RESOLUTION_720 = (1280, 720)
    RESOLUTION_540 = (960, 540)
    RESOLUTION_480 = (848, 480)

    def __init__(
        self,
        resolution: CameraResolutionType = RESOLUTION_1080,
        fps: int = 30,
        enable_depth: bool = True,
        enable_confidence_map: bool = False,
        enable_hole_filling: bool = True,
        serial_number: Optional[str] = None,
    ) -> None:
        self._resolution = resolution
        self._fps = fps
        self._depth_enabled = enable_depth
        self._confidence_enabled = enable_confidence_map
        if self._confidence_enabled and not self._depth_enabled:
            raise ValueError("enable_confidence_map can only be True if enable_depth is also True")
        self.hole_filling_enabled = enable_hole_filling
        self.serial_number = serial_number

        config = rs.config()

        if serial_number is not None:
            # Note: Invalid serial_number leads to RuntimeError for pipeline.start(config)
            config.enable_device(serial_number)

        config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, fps)

        # Use max resolution that can handle the fps for depth (will be change by align_transform)
        depth_resolution = Realsense.RESOLUTION_720 if fps <= 30 else Realsense.RESOLUTION_480
        if self._depth_enabled:
            config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, fps)

        if self._confidence_enabled:
            config.enable_stream(rs.stream.infrared, 1, depth_resolution[0], depth_resolution[1], rs.format.y8, fps)
            config.enable_stream(rs.stream.infrared, 2, depth_resolution[0], depth_resolution[1], rs.format.y8, fps)

        # Avoid having to reconnect the USB cable, see https://github.com/IntelRealSense/librealsense/issues/6628#issuecomment-646558144
        ctx = rs.context()
        devices = ctx.query_devices()
        for dev in devices:
            dev.hardware_reset()

        self.pipeline = rs.pipeline()

        self.pipeline.start(config)

        # Get intrinsics matrix
        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intrinsics = color_profile.get_intrinsics()
        self._intrinsics_matrix = np.array(
            [
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1],
            ]
        )

        if self._depth_enabled:
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self.depth_factor = depth_sensor.get_depth_scale()
            self._setup_depth_transforms()
            self.colorizer = rs.colorizer()
            self.colorizer.set_option(rs.option.color_scheme, 2)  # 2 = White to Black

    def __enter__(self) -> RGBDCamera:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.pipeline.stop()

    def _setup_depth_transforms(self) -> None:
        # Configure depth filters and transfrom, adapted from:
        # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
        # https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
        self.align_transform = rs.align(rs.stream.color)
        self.hole_filling = rs.hole_filling_filter()

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self._intrinsics_matrix

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def resolution(self) -> CameraResolutionType:
        return self._resolution

    def _grab_images(self) -> None:
        self._composite_frame = self.pipeline.wait_for_frames()

        if not self._depth_enabled:
            return

        try:
            aligned_frames = self.align_transform.process(self._composite_frame)
        except RuntimeError as e:
            # Sometimes, the realsense SDK throws an error withn aligning RGB and depth.
            # This can happen if the CPU is busy: https://github.com/IntelRealSense/librealsense/issues/6628#issuecomment-647379900
            # A solution is to try again. Here, we only try again once; if the error occurs again, we raise it
            # and let the user deal with it.
            logger.error(f"Error while grabbing images:\n{e}.\nWill retry in 1 second.")
            time.sleep(1)
            aligned_frames = self.align_transform.process(self._composite_frame)

        self._depth_frame = aligned_frames.get_depth_frame()

        if self.hole_filling_enabled:
            self._depth_frame = self.hole_filling.process(self._depth_frame)

        if not self._confidence_enabled:
            return

        self._infrared_frame_1 = aligned_frames.get_infrared_frame(1)
        self._infrared_frame_2 = aligned_frames.get_infrared_frame(2)

    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        image = self._retrieve_rgb_image_as_int()
        return ImageConverter.from_numpy_int_format(image).image_in_numpy_format

    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        assert isinstance(self._composite_frame, rs.composite_frame)
        color_frame = self._composite_frame.get_color_frame()
        image: NumpyIntImageType = np.asanyarray(color_frame.get_data())
        return image

    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        if not self._depth_enabled:
            raise RuntimeError("Cannot retrieve depth data if depth is disabled")
        frame = self._depth_frame
        image = np.asanyarray(frame.get_data()).astype(np.float32)
        return image * self.depth_factor

    def _retrieve_depth_image(self) -> NumpyIntImageType:
        if not self._depth_enabled:
            raise RuntimeError("Cannot retrieve depth data if depth is disabled")
        frame = self._depth_frame
        frame_colorized = self.colorizer.colorize(frame)
        image = np.asanyarray(frame_colorized.get_data())  # this is uint8 with 3 channels
        return image

    def _retrieve_confidence_map(self) -> NumpyConfidenceMapType:
        # Compute confidence map based on the disparity between the two IR images.
        if not self._confidence_enabled:
            raise RuntimeError("Cannot retrieve confidence data if confidence is disabled")
        if not isinstance(self._composite_frame, rs.composite_frame):
            raise RuntimeError("_grab_images must be called before retrieving images")
        ir1_frame = self._infrared_frame_1
        ir2_frame = self._infrared_frame_2

        # Convert images to numpy
        ir1 = np.asanyarray(ir1_frame.get_data())
        ir2 = np.asanyarray(ir2_frame.get_data())

        # default values for SGBM according to OpenCV docs
        wls_lambda = 8000.0
        wls_sigma = 1.5
        if not hasattr(self, "_stereo_sgbm"):
            max_disp = 160  # must be divisible by 16
            window_size = 3
            p1 = 216  # 24 * window_size ** 2
            p2 = 864  # 96 * window_size ** 2
            pre_filter_cap = 63

            self._stereo_sgbm = cv2.StereoSGBM.create(
                minDisparity=0,
                numDisparities=max_disp,
                blockSize=window_size,
                P1=p1,
                P2=p2,
                preFilterCap=pre_filter_cap,
                mode=cv2.StereoSGBM_MODE_SGBM_3WAY,
            )

        left_matcher = self._stereo_sgbm
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        left_disp = left_matcher.compute(ir1, ir2).astype(np.float32) / 16.0
        right_disp = right_matcher.compute(ir2, ir1).astype(np.float32) / 16.0
        wls_filter.setLambda(wls_lambda)
        wls_filter.setSigmaColor(wls_sigma)
        wls_filter.filter(left_disp, ir1, disparity_map_right=right_disp)
        confidence_map = wls_filter.getConfidenceMap()

        return confidence_map / 255.0


if __name__ == "__main__":
    import airo_camera_toolkit.cameras.manual_test_hw as test

    camera = Realsense(fps=30, resolution=Realsense.RESOLUTION_1080, enable_hole_filling=True)

    # Perform tests
    test.manual_test_camera(camera)
    test.manual_test_rgb_camera(camera)
    test.manual_test_depth_camera(camera)
    test.profile_rgb_throughput(camera)
    test.profile_rgbd_throughput(camera)

    # Live viewer
    cv2.namedWindow("RealSense RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RealSense Depth Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RealSense Depth Map", cv2.WINDOW_NORMAL)

    while True:
        color_image = camera.get_rgb_image_as_int()
        color_image = ImageConverter.from_numpy_int_format(color_image).image_in_opencv_format
        depth_image = camera._retrieve_depth_image()
        depth_map = camera._retrieve_depth_map()

        cv2.imshow("RealSense RGB", color_image)
        cv2.imshow("RealSense Depth Image", depth_image)
        cv2.imshow("RealSense Depth Map", depth_map)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
