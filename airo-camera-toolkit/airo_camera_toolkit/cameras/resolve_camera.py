from enum import Enum
from typing import Optional

from airo_camera_toolkit.interfaces import RGBDCamera


class CameraBrand(Enum):
    ZED = "zed"
    REALSENSE = "realsense"


SUPPORTED_CAMERAS = [m.value for m in CameraBrand]


def resolve_camera(brand: Optional[CameraBrand], serial_number: Optional[str] = None, **kwargs) -> RGBDCamera:
    if CameraBrand(brand) == CameraBrand.ZED or brand is None:
        from airo_camera_toolkit.cameras.zed2i import Zed2i

        camera = Zed2i(**kwargs)
    elif CameraBrand(brand) == CameraBrand.REALSENSE:
        from airo_camera_toolkit.cameras.realsense import Realsense

        camera = Realsense(**kwargs)

    return camera


if __name__ == "__main__":
    """Script to test the automatic camera resolution."""

    import click
    import cv2
    from airo_camera_toolkit.utils import ImageConverter

    @click.command()
    @click.option(
        "--camera_brand", default="zed", help=f"The brand of the camera you are using, one of {SUPPORTED_CAMERAS}"
    )
    @click.option(
        "--camera_serial_number",
        default=None,
        type=str,
        help="Serial number of the camera to use (if you have multiple cameras connected).",
    )
    def show_camera_feed(camera_brand: str, camera_serial_number: str):
        camera = resolve_camera(CameraBrand(camera_brand), serial_number=camera_serial_number)

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
