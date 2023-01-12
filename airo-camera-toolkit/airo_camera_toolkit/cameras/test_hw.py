"""Contains code for manual testing of implementations of the camera interfaces.
"""

import matplotlib.pyplot as plt
from airo_camera_toolkit.interfaces import Camera, DepthCamera, RGBCamera, StereoRGBDCamera


def manual_test_camera(camera: Camera):
    intrinsics = camera.intrinsics_matrix
    print(f"the rectified intrinsics matrix = \n {intrinsics}")
    input(
        """Camera matrix should be as expected:
     cx,cy are +- 1/2 of the image resolution
     and fx,fy should match the camera documentation.
     All values are in pixels.
     Press a key if all seems reasonable."""
    )


def manual_test_rgb_camera(camera: RGBCamera):
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


def manual_test_depth_camera(camera: DepthCamera):
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
    print(f"{depth_map=}")
    print(f"{depth_map.shape=}")
    input(
        "The depth map values should be reasonable and the resolution should match the camera resolution. Press a key if all seems fine."
    )


def manual_test_stereo_rgbd_camera(camera: StereoRGBDCamera):
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
