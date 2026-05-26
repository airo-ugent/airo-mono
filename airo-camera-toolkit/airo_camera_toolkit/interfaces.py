import abc

import cv2
import numpy as np
from airo_camera_toolkit.point_clouds.conversions import open3d_to_point_cloud
from airo_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    HomogeneousMatrixType,
    NumpyConfidenceMapType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
    PointCloud,
)
from loguru import logger
from typing_extensions import deprecated


class Camera(abc.ABC):
    """Base class for all cameras.

    Capture and retrieval are explicit and decoupled:

    - Call :meth:`grab_images` once to capture the latest frame from the camera
      into an internal buffer.
    - Call any number of ``retrieve_*`` methods afterwards to read individual
      fields (RGB, depth, point cloud, ...) from that same captured frame.

    All ``retrieve_*`` calls between two ``grab_images`` calls are guaranteed to
    return data from the same frame, so you can read e.g. a synchronized RGB +
    depth pair without ambiguity. Calling :meth:`grab_images` again overwrites
    the buffer with a new frame.

    We use the right-handed, y-down convention for the camera frame:
    - origin is at the camera lens center
    - z-axis points forward (towards the scene)
    - x-axis points to the right
    - y-axis points down
    cf. https://www.stereolabs.com/docs/positional-tracking/coordinate-frames/#selecting-a-coordinate-system for an overview of other conventions

    We use the (associated) top-left convention for the image plane:
    - origin is at the top left corner of the image
    - x-axis points to the right
    - y-axis points down

    keep in mind though that images are indexed column-row in numpy, which corresponds to y-x in the cartesian image coordinates
    so to get the value of the pixel at (u,v) you need to do image[v,u] and the shape of the numpy array is (height, width)
    cf https://scikit-image.org/docs/stable/user_guide/numpy_images.html#numpy-indexing
    """

    @abc.abstractmethod
    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        """returns the intrinsics matrix of the camera:

        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]]

        where all values are defined in pixels.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def grab_images(self) -> None:
        """Capture the latest frame from the camera into an internal buffer.

        Call this once per frame. All subsequent ``retrieve_*`` calls return
        data from this captured frame until ``grab_images`` is called again.
        """
        raise NotImplementedError

    @deprecated("Use the public grab_images() instead.")
    def _grab_images(self) -> None:
        """Deprecated alias for :meth:`grab_images`."""
        return self.grab_images()


class RGBCamera(Camera, abc.ABC):
    """Base class for all RGB cameras"""

    @property
    @abc.abstractmethod
    def resolution(self) -> CameraResolutionType:
        """The resolution of the camera, in pixels."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def fps(self) -> float:
        """The frames per second of the camera."""
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve_rgb_image(self) -> NumpyFloatImageType:
        """Return the RGB image from the most recently captured frame.

        Requires a prior call to :meth:`grab_images`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        """Return the RGB image from the most recently captured frame as uint8.

        This is typically the format in which the image is stored in the
        camera's memory buffer. Returning it directly avoids the overhead of
        converting it to floats first, which is what :meth:`retrieve_rgb_image`
        does.

        Requires a prior call to :meth:`grab_images`.
        """
        raise NotImplementedError

    @deprecated("Use grab_images() then retrieve_rgb_image() instead so capture and retrieval are explicit.")
    def get_rgb_image(self) -> NumpyFloatImageType:
        """Deprecated: use :meth:`grab_images` + :meth:`retrieve_rgb_image`."""
        self.grab_images()
        return self.retrieve_rgb_image()

    @deprecated("Use grab_images() then retrieve_rgb_image_as_int() instead so capture and retrieval are explicit.")
    def get_rgb_image_as_int(self) -> NumpyIntImageType:
        """Deprecated: use :meth:`grab_images` + :meth:`retrieve_rgb_image_as_int`."""
        self.grab_images()
        return self.retrieve_rgb_image_as_int()

    @deprecated("Use retrieve_rgb_image() instead.")
    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        """Deprecated alias for :meth:`retrieve_rgb_image`."""
        return self.retrieve_rgb_image()

    @deprecated("Use retrieve_rgb_image_as_int() instead.")
    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        """Deprecated alias for :meth:`retrieve_rgb_image_as_int`."""
        return self.retrieve_rgb_image_as_int()


class DepthCamera(Camera, abc.ABC):
    """Base class for all depth cameras"""

    @abc.abstractmethod
    def retrieve_depth_map(self) -> NumpyDepthMapType:
        """Return the depth map from the most recently captured frame.

        The depth map is a 2D array of floats that provide the estimated
        z-coordinate in the camera frame of that point on the image plane
        (pixel).

        Requires a prior call to :meth:`grab_images`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve_depth_image(self) -> NumpyIntImageType:
        """Return an 8-bit (int) quantization of the depth map from the most
        recently captured frame, which can be used for visualization.

        Requires a prior call to :meth:`grab_images`.
        """
        raise NotImplementedError

    def retrieve_confidence_map(self) -> NumpyConfidenceMapType:
        """Return a confidence map for the depth map of the most recently
        captured frame.

        The confidence map is a 2D array of floats that provide a measure of
        confidence in the depth estimate for each pixel. The values are between
        0 and 1, where 1 indicates high confidence and 0 indicates low
        confidence.

        Not all stereo depth cameras provide a confidence map. The default
        (naive) implementation computes one from depth discontinuities using
        OpenCV's Canny edge detection (depth estimates are assumed less
        reliable at depth discontinuities). Child classes may override this for
        a more accurate confidence map; for example:

        - StereoRGBDCamera uses disparity between left and right RGB images
          via OpenCV's SGBM algorithm.
        - Realsense uses disparity between left and right infrared images
          via OpenCV's SGBM algorithm.
        - ZED uses the camera's internal confidence measure.

        See also: https://github.com/opencv/opencv_contrib/blob/master/modules/ximgproc/samples/disparity_filtering.cpp

        Requires a prior call to :meth:`grab_images`.

        Returns:
            np.ndarray: the confidence map, a single channel float image of the
            same resolution as the depth map, with values between 0 and 1.
        """
        depth_map = self.retrieve_depth_map()
        depth_map_uint8 = np.empty_like(depth_map, dtype=np.uint8)
        depth_map_uint8 = cv2.normalize(depth_map, depth_map_uint8, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edges = cv2.Canny(depth_map_uint8, 100, 200)
        confidence_map = 1.0 - (edges / 255.0)  # edges are 255, so invert to get confidence
        return confidence_map

    @deprecated("Use grab_images() then retrieve_depth_map() instead so capture and retrieval are explicit.")
    def get_depth_map(self) -> NumpyDepthMapType:
        """Deprecated: use :meth:`grab_images` + :meth:`retrieve_depth_map`."""
        self.grab_images()
        return self.retrieve_depth_map()

    @deprecated("Use grab_images() then retrieve_depth_image() instead so capture and retrieval are explicit.")
    def get_depth_image(self) -> NumpyIntImageType:
        """Deprecated: use :meth:`grab_images` + :meth:`retrieve_depth_image`."""
        self.grab_images()
        return self.retrieve_depth_image()

    @deprecated("Use grab_images() then retrieve_confidence_map() instead so capture and retrieval are explicit.")
    def get_confidence_map(self) -> NumpyConfidenceMapType:
        """Deprecated: use :meth:`grab_images` + :meth:`retrieve_confidence_map`."""
        self.grab_images()
        return self.retrieve_confidence_map()

    @deprecated("Use retrieve_depth_map() instead.")
    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        """Deprecated alias for :meth:`retrieve_depth_map`."""
        return self.retrieve_depth_map()

    @deprecated("Use retrieve_depth_image() instead.")
    def _retrieve_depth_image(self) -> NumpyIntImageType:
        """Deprecated alias for :meth:`retrieve_depth_image`."""
        return self.retrieve_depth_image()

    @deprecated("Use retrieve_confidence_map() instead.")
    def _retrieve_confidence_map(self) -> NumpyConfidenceMapType:
        """Deprecated alias for :meth:`retrieve_confidence_map`."""
        return self.retrieve_confidence_map()


class RGBDCamera(RGBCamera, DepthCamera):
    """Base class for all RGBD cameras"""

    def retrieve_colored_point_cloud(self) -> PointCloud:
        """Return the colored point cloud from the most recently captured frame.

        The point cloud contains the estimated position in the camera frame of
        points on the image plane (pixels). Each point also has a color
        associated with it, which is the color of the corresponding pixel in
        the RGB image.

        Default implementation uses the depth map and RGB with open3d's
        ``create_from_rgbd_image()`` function. See:
        https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.create_from_rgbd_image

        Requires a prior call to :meth:`grab_images`.

        Returns:
            PointCloud: the points (= positions) and colors.
        """
        if not hasattr(self, "_logged_colored_point_cloud_warning"):
            logger.warning(
                """You are using an RGBDCamera which does not override retrieve_colored_point_cloud.
            We will use a default implementation based on Open3D, which is quite slow (several milliseconds per frame).
            Consider impementing this method if your camera supports point cloud processing."""
            )
            self._logged_colored_point_cloud_warning = True

        import open3d as o3d

        image_rgb_uint8 = self.retrieve_rgb_image_as_int()
        depth_map = self.retrieve_depth_map()
        intrinsics = self.intrinsics_matrix()

        # Convert airo-mono data types to open3d data types
        image_o3d = o3d.t.geometry.Image(image_rgb_uint8)
        depth_map_o3d = o3d.t.geometry.Image(depth_map)
        rgbd_o3d = o3d.t.geometry.RGBDImage(image_o3d, depth_map_o3d)

        # Note this is quite slow, > 100ms for a 2K image
        pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
            rgbd_o3d,
            intrinsics,
            depth_scale=1.0,
            depth_max=1000.0,
        )

        point_cloud = open3d_to_point_cloud(pcd)
        return point_cloud

    @deprecated("Use grab_images() then retrieve_colored_point_cloud() instead so capture and retrieval are explicit.")
    def get_colored_point_cloud(self) -> PointCloud:
        """Deprecated: use :meth:`grab_images` + :meth:`retrieve_colored_point_cloud`."""
        self.grab_images()
        return self.retrieve_colored_point_cloud()

    @deprecated("Use retrieve_colored_point_cloud() instead.")
    def _retrieve_colored_point_cloud(self) -> PointCloud:
        """Deprecated alias for :meth:`retrieve_colored_point_cloud`."""
        return self.retrieve_colored_point_cloud()


class StereoRGBDCamera(RGBDCamera):
    """Base class for all stereo RGBD cameras"""

    LEFT_RGB = "left"
    RIGHT_RGB = "right"
    _VIEWS = (LEFT_RGB, RIGHT_RGB)

    @abc.abstractmethod
    def retrieve_rgb_image(self, view: str = LEFT_RGB) -> NumpyFloatImageType:
        """Return the RGB image from the most recently captured frame for the
        given stereo view.

        Requires a prior call to :meth:`grab_images`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve_rgb_image_as_int(self, view: str = LEFT_RGB) -> NumpyIntImageType:
        """Return the RGB image from the most recently captured frame for the
        given stereo view, as uint8.

        Requires a prior call to :meth:`grab_images`.
        """
        raise NotImplementedError

    # TODO: check view argument value?
    @abc.abstractmethod
    def intrinsics_matrix(self, view: str = LEFT_RGB) -> CameraIntrinsicsMatrixType:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def pose_of_right_view_in_left_view(self) -> HomogeneousMatrixType:
        """
        get the transform of the right view frame in the left view frame,
        ususally this is simply a translation along the x-axis and referred to as the disparity.
        cf https://en.wikipedia.org/wiki/Binocular_disparity

        The left view is usually considered to be the 'camera frame', i.e. this is the frame that is used to define the camara extrinsics matrix
        """
        raise NotImplementedError

    def retrieve_confidence_map(self) -> NumpyConfidenceMapType:
        """Return a confidence map for the depth map of the most recently
        captured frame.

        Based on disparity between left and right RGB images using OpenCV's
        SGBM algorithm implementation.

        Requires a prior call to :meth:`grab_images`.
        """

        # default values for SGBM according to OpenCV docs
        max_disp = 160  # must be divisible by 16
        window_size = 3
        p1 = 216  # 24 * window_size ** 2
        p2 = 864  # 96 * window_size ** 2
        pre_filter_cap = 63
        wls_lambda = 8000.0
        wls_sigma = 1.5

        # get the left and right images
        left = self.retrieve_rgb_image_as_int(self.LEFT_RGB)
        right = self.retrieve_rgb_image_as_int(self.RIGHT_RGB)

        left_matcher = cv2.StereoSGBM.create(
            minDisparity=0,
            numDisparities=max_disp,
            blockSize=window_size,
            P1=p1,
            P2=p2,
            preFilterCap=pre_filter_cap,
            mode=cv2.StereoSGBM_MODE_SGBM_3WAY,
        )
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        left_disp = left_matcher.compute(left, right).astype(np.float32) / 16.0
        right_disp = right_matcher.compute(right, left).astype(np.float32) / 16.0
        wls_filter.setLambda(wls_lambda)
        wls_filter.setSigmaColor(wls_sigma)
        wls_filter.filter(left_disp, left, disparity_map_right=right_disp)
        confidence_map = wls_filter.getConfidenceMap()

        return confidence_map / 255.0

    @deprecated("Use grab_images() then retrieve_rgb_image() instead so capture and retrieval are explicit.")
    def get_rgb_image(self, view: str = LEFT_RGB) -> NumpyFloatImageType:
        """Deprecated: use :meth:`grab_images` + :meth:`retrieve_rgb_image`."""
        self.grab_images()
        return self.retrieve_rgb_image(view)

    @deprecated("Use grab_images() then retrieve_rgb_image_as_int() instead so capture and retrieval are explicit.")
    def get_rgb_image_as_int(self, view: str = LEFT_RGB) -> NumpyIntImageType:
        """Deprecated: use :meth:`grab_images` + :meth:`retrieve_rgb_image_as_int`."""
        self.grab_images()
        return self.retrieve_rgb_image_as_int(view)

    @deprecated("Use retrieve_rgb_image() instead.")
    def _retrieve_rgb_image(self, view: str = LEFT_RGB) -> NumpyFloatImageType:
        """Deprecated alias for :meth:`retrieve_rgb_image`."""
        return self.retrieve_rgb_image(view)

    @deprecated("Use retrieve_rgb_image_as_int() instead.")
    def _retrieve_rgb_image_as_int(self, view: str = LEFT_RGB) -> NumpyIntImageType:
        """Deprecated alias for :meth:`retrieve_rgb_image_as_int`."""
        return self.retrieve_rgb_image_as_int(view)
