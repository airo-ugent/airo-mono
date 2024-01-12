"""Contains code for testing implementations of the camera interfaces.
"""

from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from airo_camera_toolkit.interfaces import Camera, DepthCamera, RGBCamera, RGBDCamera, StereoRGBDCamera


def manual_test_camera(camera: Camera) -> None:
    intrinsics = camera.intrinsics_matrix()
    with np.printoptions(precision=1, suppress=True):
        print(f"the rectified intrinsics matrix = \n {intrinsics}")
    input(
        """Camera matrix should be as expected:
     cx,cy are +- 1/2 of the image resolution
     and fx,fy should match the camera documentation.
     All values are in pixels.
     Press a key if all seems reasonable."""
    )


def manual_test_rgb_camera(camera: RGBCamera) -> None:
    image = camera.get_rgb_image()
    plt.imshow(image)
    print(
        """
    The image resolution should match the expected camera resolution.
    Furthermore the RGB channels should be in the correct order (check with red/green objects).
    close the window if all seems fine.
    """
    )
    plt.show()

    int_image = camera.get_rgb_image_as_int()
    plt.imshow(int_image)
    print(
        """
        This should look the same as the previous image, as it is the same image, just in a different format.
        """
    )
    plt.show()


def manual_test_depth_camera(camera: DepthCamera) -> None:
    dept_image = camera.get_depth_image()
    plt.imshow(dept_image)
    print(
        """
    The image resolution should match the expected camera resolution.
    Furthermore the depth image should look reasonable (further away objects should be darker).
    close the window if all seems fine.
    """
    )
    plt.show()

    depth_map = camera.get_depth_map()
    with np.printoptions(precision=3, suppress=True):
        print(f"{depth_map=}")
    print(f"{depth_map.shape=}")
    input(
        "The depth map values should be reasonable and the resolution should match the camera resolution. Press a key if all seems fine."
    )


def manual_test_stereo_rgbd_camera(camera: StereoRGBDCamera) -> None:
    manual_test_camera(camera)
    manual_test_rgb_camera(camera)
    manual_test_depth_camera(camera)
    print(f"{camera.pose_of_right_view_in_left_view=}")
    input(
        "The pose of the right view in the left view should be as expected: a translation along the X-axis with the camera disparity distance. Press a key if all seems fine."
    )
    image = camera.get_rgb_image(view=camera.RIGHT_RGB)
    plt.imshow(image)
    print(
        """
    The right view image should be as expected.
    close the window if all seems fine.
    """
    )
    plt.show()


def profile(func: Callable, *args: Any, **kwargs: Any) -> None:
    """a wrapper around the python cProfiler

    https://docs.python.org/3/library/profile.html

    Args:
        func (_type_): the function to profile
    """
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    print(func(*args, **kwargs))
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    stats.print_stats(50)


def profile_rgb_throughput(camera: RGBCamera) -> None:
    """profile the throughput of the get_rgb_image() function"""

    def get_100_images() -> None:
        for _ in range(100):
            camera.get_rgb_image()

    profile(get_100_images)


def profile_rgbd_throughput(camera: RGBDCamera) -> None:
    """profile the throughput of the get_rgb_image() and _retrieve_depth_maps() functions"""

    def get_100_images() -> None:
        for _ in range(100):
            camera.get_rgb_image()
            camera._retrieve_depth_map()

    profile(get_100_images)
