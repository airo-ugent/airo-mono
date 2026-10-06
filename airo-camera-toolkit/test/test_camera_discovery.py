import airo_camera_toolkit.cameras as cameras
from airo_camera_toolkit.cameras.camera_discovery import SUPPORTED_CAMERAS, CameraBrand


def test_supported_camera_brands():
    assert SUPPORTED_CAMERAS == ["zed", "realsense", "luxonis"]
    assert CameraBrand("luxonis") == CameraBrand.LUXONIS


def test_lazy_camera_exports():
    assert {"Zed", "Realsense", "Luxonis"} <= set(dir(cameras))
