from __future__ import annotations

from datetime import timedelta
from typing import Any, ClassVar, List, Optional

import cv2
import numpy as np

try:
    import depthai as dai
except ImportError:
    raise ImportError(
        "You should install the `depthai` package in your environment first, see the installation README."
    )

if int(dai.__version__.split(".")[0]) < 3:
    raise ImportError("You should install version 3.X of depthai!")

from airo_camera_toolkit.interfaces import RGBDCamera
from airo_camera_toolkit.utils.image_converter import ImageConverter
from airo_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
    PointCloud,
)
from loguru import logger


class Luxonis(RGBDCamera):
    """Wrapper around the depthai (v3) library to use the Luxonis OAK-D cameras (OAK-D, OAK-D Pro, OAK-D S2, OAK4-D, ...).

    Design decisions we made for this class:
    * The RGB image comes from the center color camera (CAM_A).
    * Depth is computed on-device from the left/right mono cameras (CAM_B/CAM_C) and is always aligned to the color frame.
    * Depth and color fps are the same, and RGB and depth are synchronized on-device into a single frame group.
    * Color images are undistorted on-device by default, so the intrinsics matrix is a valid pinhole model.
    * The point cloud is computed on the host by unprojecting the aligned depth map with the color intrinsics.
    * Invalid depth values are 0, as with the Realsense.
    * The confidence map uses the default (depth discontinuity based) implementation of `DepthCamera`,
      because the on-device stereo confidence map is not aligned to the color frame.
    """

    # Built-in resolutions for convenience. Color frames are produced by the ISP, so other resolutions are possible too,
    # but the width should be a multiple of 16 for on-device depth alignment.
    # Which resolutions are available depends on the color sensor: e.g. the IMX378 (OAK-D, OAK-D Pro) supports up to 4K,
    # while the global shutter OV9782 (OAK-D Pro W, OAK-D Pro with OV9782) supports at most 1280x800.
    RESOLUTION_4K: ClassVar[CameraResolutionType] = (3840, 2160)
    RESOLUTION_1080: ClassVar[CameraResolutionType] = (1920, 1080)
    RESOLUTION_800: ClassVar[CameraResolutionType] = (1280, 800)
    RESOLUTION_720: ClassVar[CameraResolutionType] = (1280, 720)
    RESOLUTION_480: ClassVar[CameraResolutionType] = (848, 480)

    # Resolutions for the left/right mono cameras used for stereo depth (OV9282 / OV9782 sensors).
    STEREO_RESOLUTION_800: ClassVar[CameraResolutionType] = (1280, 800)
    STEREO_RESOLUTION_400: ClassVar[CameraResolutionType] = (640, 400)

    # On-device stereo presets, for more info see:
    # https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/stereo_depth/
    DEPTH_PRESET_DEFAULT: ClassVar[Any] = dai.node.StereoDepth.PresetMode.DEFAULT
    DEPTH_PRESET_ROBOTICS: ClassVar[Any] = dai.node.StereoDepth.PresetMode.ROBOTICS
    DEPTH_PRESET_FAST_ACCURACY: ClassVar[Any] = dai.node.StereoDepth.PresetMode.FAST_ACCURACY
    DEPTH_PRESET_FAST_DENSITY: ClassVar[Any] = dai.node.StereoDepth.PresetMode.FAST_DENSITY
    DEPTH_PRESET_HIGH_DETAIL: ClassVar[Any] = dai.node.StereoDepth.PresetMode.HIGH_DETAIL

    _RGB_STREAM = "rgb"
    _DEPTH_STREAM = "depth"

    def __init__(
        self,
        resolution: CameraResolutionType = RESOLUTION_720,
        fps: int = 30,
        enable_depth: bool = True,
        enable_pointcloud: bool = True,
        depth_preset: Any = DEPTH_PRESET_DEFAULT,
        stereo_resolution: CameraResolutionType = STEREO_RESOLUTION_400,
        enable_undistortion: bool = True,
        serial_number: Optional[str] = None,
    ) -> None:
        """Initializes the Luxonis camera.

        Args:
            resolution: Resolution of the color (and aligned depth) images. Defaults to RESOLUTION_720, which is
                supported by all OAK-D color sensors.
            fps: Frames per second of the color and mono cameras. Defaults to 30.
            enable_depth: Whether to compute depth on-device. Defaults to True.
            enable_pointcloud: Whether to compute the colored point cloud on the host. Requires depth. Defaults to True.
            depth_preset: On-device stereo depth preset, one of the DEPTH_PRESET_* aliases. Defaults to DEPTH_PRESET_DEFAULT.
            stereo_resolution: Resolution of the left/right mono cameras used for stereo depth. Higher resolution gives
                more accurate depth at the cost of on-device compute. Defaults to STEREO_RESOLUTION_400.
            enable_undistortion: Whether to undistort the color images on-device. Defaults to True.
            serial_number: Device ID (MxID), IP address or USB name of the camera to use. If None, the first available
                camera is used. Defaults to None.
        """
        self._resolution = resolution
        self._fps = fps
        self._depth_enabled = enable_depth
        self._pointcloud_enabled = enable_pointcloud
        if self._pointcloud_enabled and not self._depth_enabled:
            raise ValueError("enable_pointcloud can only be True if enable_depth is also True")
        self.depth_preset = depth_preset
        self.stereo_resolution = stereo_resolution
        self.enable_undistortion = enable_undistortion
        self.serial_number = serial_number

        if serial_number is not None:
            # Note: an invalid serial_number leads to a RuntimeError here
            self.device = dai.Device(dai.DeviceInfo(serial_number))
        else:
            self.device = dai.Device()
        self._check_color_resolution(resolution)
        self.pipeline = dai.Pipeline(self.device)

        color_camera = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        color_output = color_camera.requestOutput(
            resolution, dai.ImgFrame.Type.BGR888i, fps=fps, enableUndistortion=enable_undistortion
        )

        sync = self.pipeline.create(dai.node.Sync)
        # Half a frame period is the largest threshold that cannot match frames from different capture instants.
        sync.setSyncThreshold(timedelta(seconds=0.5 / fps))
        color_output.link(sync.inputs[self._RGB_STREAM])

        if self._depth_enabled:
            left_camera = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right_camera = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = self.pipeline.create(dai.node.StereoDepth).build(
                left_camera.requestOutput(stereo_resolution, fps=fps),
                right_camera.requestOutput(stereo_resolution, fps=fps),
                presetMode=depth_preset,
            )
            if self.device.getPlatform() == dai.Platform.RVC4:
                # RVC4 (OAK4) does not align depth inside the StereoDepth node, a separate ImageAlign node is needed.
                align = self.pipeline.create(dai.node.ImageAlign)
                stereo.depth.link(align.input)
                color_output.link(align.inputAlignTo)
                align.outputAligned.link(sync.inputs[self._DEPTH_STREAM])
            else:
                color_output.link(stereo.inputAlignTo)
                stereo.depth.link(sync.inputs[self._DEPTH_STREAM])

        # Only keep the latest frame group, so grab_images() always returns a recent frame.
        self._queue = sync.out.createOutputQueue(maxSize=1, blocking=False)
        self.pipeline.start()

        calibration = self.device.readCalibration()
        self._intrinsics_matrix = np.array(
            calibration.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, resolution[0], resolution[1])
        )

        if self._pointcloud_enabled:
            # Precompute the ray direction (with z = 1) of every pixel, so unprojection is a single multiplication.
            width, height = resolution
            u, v = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
            fx, fy = self._intrinsics_matrix[0, 0], self._intrinsics_matrix[1, 1]
            cx, cy = self._intrinsics_matrix[0, 2], self._intrinsics_matrix[1, 2]
            self._pixel_rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1).reshape(-1, 3)

        logger.info(f"Opened Luxonis camera {self.device.getDeviceId()} ({self.device.getDeviceName()}).")

    def _check_color_resolution(self, resolution: CameraResolutionType) -> None:
        """Raise a clear error if the color sensor cannot produce the requested resolution.

        Without this check, depthai only fails when the pipeline is started, with a hard to interpret message.
        """
        for features in self.device.getConnectedCameraFeatures():
            if features.socket != dai.CameraBoardSocket.CAM_A:
                continue
            if resolution[0] > features.width or resolution[1] > features.height:
                sensor_resolutions = sorted({(config.width, config.height) for config in features.configs})
                device_name = self.device.getDeviceName()
                self.device.close()
                raise ValueError(
                    f"Resolution {resolution} is not supported by the {features.sensorName} color sensor of this "
                    f"{device_name}, its maximum is {(features.width, features.height)} "
                    f"(sensor modes: {sensor_resolutions}). Try Luxonis.RESOLUTION_720."
                )
            return

    def __enter__(self) -> Luxonis:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.pipeline.stop()
        self.device.close()

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        return self._intrinsics_matrix

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def resolution(self) -> CameraResolutionType:
        return self._resolution

    def grab_images(self) -> None:
        message_group = self._queue.get()  # this is a blocking call
        if not isinstance(message_group, dai.MessageGroup):
            raise RuntimeError("Could not grab new camera frame, the pipeline has stopped.")

        rgb_frame = message_group[self._RGB_STREAM]
        assert isinstance(rgb_frame, dai.ImgFrame)
        self._rgb_image = cv2.cvtColor(rgb_frame.getCvFrame(), cv2.COLOR_BGR2RGB)

        if not self._depth_enabled:
            return

        # Depth is sent in millimeters as uint16, we convert to meters.
        depth_frame = message_group[self._DEPTH_STREAM]
        assert isinstance(depth_frame, dai.ImgFrame)
        self._depth_map = depth_frame.getFrame().astype(np.float32) * 0.001

        if self._pointcloud_enabled:
            points = self._pixel_rays * self._depth_map.reshape(-1, 1)
            self._point_cloud = PointCloud(points, self._rgb_image.reshape(-1, 3))

    def retrieve_rgb_image(self) -> NumpyFloatImageType:
        image = self.retrieve_rgb_image_as_int()
        return ImageConverter.from_numpy_int_format(image).image_in_numpy_format

    def retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        return self._rgb_image

    def retrieve_depth_map(self) -> NumpyDepthMapType:
        if not self._depth_enabled:
            raise RuntimeError("Cannot retrieve depth data if depth is disabled")
        return self._depth_map

    def retrieve_depth_image(self) -> NumpyIntImageType:
        if not self._depth_enabled:
            raise RuntimeError("Cannot retrieve depth data if depth is disabled")
        # White (near) to black (far), with invalid pixels black, similar to the Realsense colorizer.
        depth_map = self._depth_map
        valid = depth_map > 0
        if not np.any(valid):
            return np.zeros((*depth_map.shape, 3), dtype=np.uint8)
        near, far = np.percentile(depth_map[valid], [1, 99])
        normalized = np.clip((far - depth_map) / max(far - near, 1e-6), 0.0, 1.0)
        image = (normalized * 255).astype(np.uint8)
        image[~valid] = 0
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    def retrieve_colored_point_cloud(self) -> PointCloud:
        if not self._pointcloud_enabled:
            raise RuntimeError("Cannot retrieve point cloud if point cloud is disabled")
        return self._point_cloud

    @staticmethod
    def list_camera_serial_numbers() -> List[str]:
        """List the device IDs (MxIDs) of all available Luxonis cameras.

        Can be used to select a camera or to check if cameras are connected.
        """
        return [device_info.getDeviceId() for device_info in dai.Device.getAllAvailableDevices()]


if __name__ == "__main__":
    import airo_camera_toolkit.cameras.manual_test_hw as test

    # Luxonis specific test: list all connected cameras
    print(Luxonis.list_camera_serial_numbers())
    input("each camera connected to the pc should be listed, press enter to continue")

    camera = Luxonis(fps=30, resolution=Luxonis.RESOLUTION_720)

    # Perform tests
    test.manual_test_camera(camera)
    test.manual_test_rgb_camera(camera)
    test.manual_test_depth_camera(camera)
    test.profile_rgb_throughput(camera)
    test.profile_rgbd_throughput(camera)

    # Live viewer
    cv2.namedWindow("Luxonis RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Luxonis Depth Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Luxonis Depth Map", cv2.WINDOW_NORMAL)

    while True:
        camera.grab_images()
        color_image = camera.retrieve_rgb_image_as_int()
        color_image = ImageConverter.from_numpy_int_format(color_image).image_in_opencv_format
        depth_image = camera.retrieve_depth_image()
        depth_map = camera.retrieve_depth_map()

        cv2.imshow("Luxonis RGB", color_image)
        cv2.imshow("Luxonis Depth Image", depth_image)
        cv2.imshow("Luxonis Depth Map", depth_map)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
