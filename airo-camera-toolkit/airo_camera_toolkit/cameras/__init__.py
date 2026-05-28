"""Camera implementations.

Hardware-dependent classes (`Zed`, `ZedSpatialMap`, `Realsense`) and the optional
`MultiprocessVideoRecorder` are exposed lazily so this package can be imported on
machines that don't have the corresponding hardware/SDK installed. The underlying module
is only loaded when the attribute is actually accessed; the resulting ImportError tells
the user which extra to install (or, for ZED, where to find the SDK install instructions).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

# Attribute name -> (submodule path to load, install hint shown in the ImportError).
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "Zed": (
        "airo_camera_toolkit.cameras.zed.zed",
        "the ZED SDK and its bundled `pyzed` bindings (see airo_camera_toolkit/cameras/zed/installation.md)",
    ),
    "Zed2i": (
        "airo_camera_toolkit.cameras.zed.zed2i",
        "the ZED SDK and its bundled `pyzed` bindings (see airo_camera_toolkit/cameras/zed/installation.md)",
    ),
    "ZedSpatialMap": (
        "airo_camera_toolkit.cameras.zed.zed",
        "the ZED SDK and its bundled `pyzed` bindings (see airo_camera_toolkit/cameras/zed/installation.md)",
    ),
    "Realsense": (
        "airo_camera_toolkit.cameras.realsense.realsense",
        "the realsense SDK and the `pyrealsense2` package (see airo_camera_toolkit/cameras/realsense/realsense_installation.md)",
    ),
    "OpenCVVideoCapture": (
        "airo_camera_toolkit.cameras.opencv_videocapture.opencv_videocapture",
        "[ERROR] this should not occur as OpenCV is a core dependency of airo-camera-toolkit. Please check your environment.",
    ),
    "MultiprocessVideoRecorder": (
        "airo_camera_toolkit.cameras.multiprocess.multiprocess_video_recorder",
        "the `recording` extra: `pip install 'airo-camera-toolkit[recording]'`",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module 'airo_camera_toolkit.cameras' has no attribute {name!r}")
    module_path, hint = _LAZY_ATTRS[name]
    try:
        module = import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import {name!r} from airo_camera_toolkit.cameras: "
            f"underlying module {module_path!r} failed to load ({exc}). Install {hint}."
        ) from exc
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(_LAZY_ATTRS)


if TYPE_CHECKING:
    from airo_camera_toolkit.cameras.multiprocess.multiprocess_video_recorder import MultiprocessVideoRecorder
    from airo_camera_toolkit.cameras.opencv_videocapture.opencv_videocapture import OpenCVVideoCapture
    from airo_camera_toolkit.cameras.realsense.realsense import Realsense
    from airo_camera_toolkit.cameras.zed.zed import Zed, ZedSpatialMap
    from airo_camera_toolkit.cameras.zed.zed2i import Zed2i

    __all__ = [
        "MultiprocessVideoRecorder",
        "OpenCVVideoCapture",
        "Realsense",
        "Zed",
        "Zed2i",
        "ZedSpatialMap",
    ]
