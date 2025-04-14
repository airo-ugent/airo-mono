# Changelog

All notable changes for the packages in the airo-mono repo are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses a [CalVer](https://calver.org/) versioning scheme with monthly releases, see [here](versioning.md)

## Unreleased

### Breaking changes

### Added

### Changed

### Fixed
- Fixed a bug when the KELO Robile platform was moving around multiples of 360 degrees, where the target angle would switch.

### Removed

## 2025.4.0

### Breaking changes
 - internal dependencies are now listed as regular dependencies in the `setup.py` file to overcome issues and make the installation process less complicated. This implies you need to install packages according to their dependencies and can no longer use the `external` tag as in `pip install airo-typing[external]`.
 see [issue #91](https://github.com/airo-ugent/airo-mono/issues/91) and
 [PR](https://github.com/airo-ugent/airo-mono/pull/108) for more details.
 - `PointCloud` dataclass replaces the `ColoredPointCloudType` to support point cloud attritubes

### Added
- add method `as_single_polygon` to combine disconnected parts of a binary mask into a single polygon to the `Mask` class, useful for data formats that only allow for a single polygon such as YOLO.
- `PointCloud` dataclass as the main data structure for point clouds in airo-mono
- Notebooks to get started with point clouds, checking performance and logging to rerun
- Functions to crop point clouds, filter points with a mask (e.g. low-confidence points), and transform point clouds
- Functions to convert from our numpy-based dataclass to and from open3d point clouds
- `BoundingBox3DType`
- `Zed.ULTRA_DEPTH_MODE` to enable the ultra depth setting for the Zed cameras
- `OpenCVVideoCapture` implementation of `RGBCamera` for working with arbitrary cameras
- `MultiprocessRGBRerunLogger` and `MultiprocessRGBDRerunLogger` now allow you to pass an `entity_path` value which determines where the RGB and depth images will be logged
- `MobileRobot` and `KELORobile` interface and subclass added, to control mobile robots via the `airo-tulip` package
- drivers for a Schunk Gripper using a USB connection, might require additional work to make them more stable. Remko is the go-to person.
- airo-mono packages are now on PyPI 🎉

### Changed
- `coco-to-yolo` conversion now creates a single polygon of all disconnected parts of the mask instead of simply taking the first polygon of the list.
- Dropped support for python 3.8 and added 3.11 to the testing matrix [#103](https://github.com/airo-ugent/airo-mono/issues/103).
- Set python version to 3.10 because of an issue with the `ur_rtde` wheels [#121](https://github.com/airo-ugent/airo-mono/issues/121). Updated README.md to reflect this change.
- `URrtde` will now try connecting to do control interface up to 3 times before raising a `RuntimeError`.
- Renamed `Zed2i` to `Zed` and `zed2i.py` to `zed.py`, but kept the old names as aliases for backwards compatibility
- Locked `numpy` to versions `<2.0` for compatibility with `opencv`, since we are using a locked version of `opencv` that is not compatible with newer versions of `numpy`.

### Fixed
- Fixed bug in `get_colored_point_cloud()` that removed some points see issue [#25](https://github.com/airo-ugent/airo-mono/issues/25).
- Fixed bug requiring unplug-and-plug of USB cable for Realsense: see issue [#109](https://github.com/airo-ugent/airo-mono/issues/109).
- Fixed bug with Realsense cameras raising `RuntimeErrors` in RGB-depth alignment when CPU is busy. The camera will now try again once after 1 second.
- Removed camera imports in `airo_camera_toolkit.cameras`: see issue [#110](https://github.com/airo-ugent/airo-mono/issues/).
- Added `__init__.py` to `realsense` and `utils` in `airo_camera_toolkit.cameras`, fixing installs with pip and issue #113.
- Fixed bug that returned a transposed resolution in `MultiprocessRGBReceiver`.
- Using `Zed.PERFORMANCE_DEPTH_MODE` will now correctly use the performance mode instead of the quality mode.
- Shared memory files that were not properly cleaned up are now unlinked and then recreated.
- The wait interval for shared memory files has been reduced to .5 seconds (from 5), to speed up application start times.

### Removed
- `ColoredPointCloudType`

## 2024.1.0

### Breaking changes

### Added
- Calendar-based versioning scheme and introduction of accompanying changelog.

### Changed

### Fixed

### Removed

