"""This file is present for backwards compatibility, when we only supported Zed2i cameras, and the Zed2i class was in this file."""
from airo_camera_toolkit.cameras.zed.zed import *  # noqa: F401, F403
from airo_camera_toolkit.cameras.zed.zed import _test_zed_implementation

if __name__ == "__main__":
    _test_zed_implementation()
