import abc

from airo_camera_toolkit.point_clouds.conversions import open3d_to_point_cloud
from airo_typing import (
    CameraIntrinsicsMatrixType,
    CameraResolutionType,
    HomogeneousMatrixType,
    NumpyDepthMapType,
    NumpyFloatImageType,
    NumpyIntImageType,
    PointCloud,
)


class Camera(abc.ABC):
    """Base class for all cameras

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
    def _grab_images(self) -> None:
        """Transfer the latest camera data (RGB, stereo RGB, depth) from the camera to a memory buffer."""
        raise NotImplementedError


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

    def get_rgb_image(self) -> NumpyFloatImageType:
        """Get a new RGB image from the camera."""
        self._grab_images()
        return self._retrieve_rgb_image()

    def get_rgb_image_as_int(self) -> NumpyIntImageType:
        """Get a new RGB image from the camera as uint8.
        This is faster to retrieve as images are typically stored as ints in the buffer.
        It is also more compact, so recommended for communication."""
        self._grab_images()
        return self._retrieve_rgb_image_as_int()

    @abc.abstractmethod
    def _retrieve_rgb_image(self) -> NumpyFloatImageType:
        """Returns the current RGB image in the memory buffer."""
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_rgb_image_as_int(self) -> NumpyIntImageType:
        """Returns the current RGB image in the memory buffer as uint8.
        This is typically the format in which it is stored in memory.
        Returning it directly avoids the overhead of converting it to floats first, which is what _retrieve_rgb_image() does.
        """
        raise NotImplementedError


class DepthCamera(Camera, abc.ABC):
    """Base class for all depth cameras"""

    def get_depth_map(self) -> NumpyDepthMapType:
        """Get the latest depth map of the camera.
        The depth map is a 2D array of floats, that provide the estimated z-coordinate in the camera frame
        of that point on the image plane (pixel).

        Returns:
            np.ndarray: _description_
        """
        self._grab_images()
        return self._retrieve_depth_map()

    def get_depth_image(self) -> NumpyIntImageType:
        """an 8-bit (int) quantization of the latest depth map, which can be used for visualization"""
        self._grab_images()
        return self._retrieve_depth_image()

    @abc.abstractmethod
    def _retrieve_depth_map(self) -> NumpyDepthMapType:
        """Returns the current depth map in the memory buffer."""
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_depth_image(self) -> NumpyIntImageType:
        """Returns the current depth image in the memory buffer."""
        raise NotImplementedError


class RGBDCamera(RGBCamera, DepthCamera):
    """Base class for all RGBD cameras"""

    def get_colored_point_cloud(self) -> PointCloud:
        """Get the latest point cloud of the camera.
        The point cloud contains the estimated position in the camera frame of points on the image plane (pixels).
        Each point also has a color associated with it, which is the color of the corresponding pixel in the RGB image.

        Returns:
            PointCloud: the points (= positions) and colors
        """

        self._grab_images()
        return self._retrieve_colored_point_cloud()

    def _retrieve_colored_point_cloud(self) -> PointCloud:
        """Returns the current point cloud in the memory buffer.

        Default implementation uses the depth map and RGB with open3d's create_from_rgbd_image() function.
        See: https://www.open3d.org/docs/release/python_api/open3d.t.geometry.PointCloud.html#open3d.t.geometry.PointCloud.create_from_rgbd_image
        """
        import open3d as o3d

        image_rgb_uint8 = self._retrieve_rgb_image_as_int()
        depth_map = self._retrieve_depth_map()
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


class StereoRGBDCamera(RGBDCamera):
    """Base class for all stereo RGBD cameras"""

    LEFT_RGB = "left"
    RIGHT_RGB = "right"
    _VIEWS = (LEFT_RGB, RIGHT_RGB)

    def get_rgb_image(self, view: str = LEFT_RGB) -> NumpyFloatImageType:
        self._grab_images()
        return self._retrieve_rgb_image(view)

    def get_rgb_image_as_int(self, view: str = LEFT_RGB) -> NumpyIntImageType:
        self._grab_images()
        return self._retrieve_rgb_image_as_int(view)

    @abc.abstractmethod
    def _retrieve_rgb_image(self, view: str = LEFT_RGB) -> NumpyFloatImageType:
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_rgb_image_as_int(self, view: str = LEFT_RGB) -> NumpyIntImageType:
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
