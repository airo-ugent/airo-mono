"""Module to automatically resolve the desired connected camera.
Useful when you want to support both ZED and RealSense cameras in your scripts and CLIs.
"""
from enum import Enum
from typing import Any, Callable, Optional

from airo_camera_toolkit.interfaces import RGBDCamera
from loguru import logger


# Note: the order of this enum also determines the order in which the cameras are tried
class CameraBrand(Enum):
    ZED = "zed"
    REALSENSE = "realsense"


SUPPORTED_CAMERAS = [m.value for m in CameraBrand]


def discover_camera(brand: Optional[str], serial_number: Optional[str] = None, **kwargs: Any) -> RGBDCamera:
    """Find a connected camera, optionally filtering by brand and serial number.

    Args:
        brand: The brand of the camera to search for.
        serial_number: The serial number of the camera to use.
        **kwargs: Additional arguments to pass to the camera constructor.

    Returns:
        The camera instance.
    """

    logger.info(f"Resolving camera with brand {brand} and serial number {serial_number}.")

    if brand is None:
        for brand in SUPPORTED_CAMERAS:
            try:
                return discover_camera(brand, serial_number=serial_number, **kwargs)
            except Exception:
                pass

        raise RuntimeError("Could not find camera with the requested parameters.")

    brand = brand.lower()
    brand_enum = CameraBrand(brand)  # Attempt to convert to enum

    if brand_enum == CameraBrand.ZED:
        from airo_camera_toolkit.cameras.zed.zed import Zed

        camera = Zed(serial_number=serial_number, **kwargs)
    elif brand_enum == CameraBrand.REALSENSE:
        from airo_camera_toolkit.cameras.realsense.realsense import Realsense

        camera = Realsense(serial_number=serial_number, **kwargs)  # type: ignore
    else:
        raise RuntimeError(f"Camera brand {brand} not supported.")  # Should be unreachable due to enum

    logger.info(f"Found {brand_enum.value} camera.")

    return camera


def click_camera_options(f: Callable) -> Callable:
    """Function to be used as a decorator for click commands to add options for camera brand and serial number.

    Usage:
        @click.command()
        @click_camera_options
        def my_command(camera_brand: Optional[str] = None,
                       camera_serial_number: Optional[str] = None,):
            ...

    Args:
        f: function to decorate with the additional command line arguments

    Returns:
        f: decorated function
    """
    import click

    camera_brand_help = f"The brand of the camera to use, one of {SUPPORTED_CAMERAS}"
    camera_serial_number_help = "Serial number of the camera to use (if you have multiple cameras connected)."

    f = click.option("--camera_brand", help=camera_brand_help)(f)
    f = click.option("--camera_serial_number", default=None, type=str, help=camera_serial_number_help)(f)
    return f


if __name__ == "__main__":
    import click
    import cv2
    from airo_camera_toolkit.utils.image_converter import ImageConverter

    @click.command()
    @click_camera_options
    def show_camera_feed(camera_brand: str, camera_serial_number: str) -> None:
        """Script to test the automatic camera discovery."""
        camera = discover_camera(camera_brand, camera_serial_number)

        window_name = "Camera feed"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("Press Q to quit.")
        while True:
            image_rgb = camera.get_rgb_image_as_int()
            image = ImageConverter.from_numpy_int_format(image_rgb).image_in_opencv_format
            cv2.imshow(window_name, image)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break

    show_camera_feed()
