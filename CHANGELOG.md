# Changelog

All notable changes for the packages in the airo-mono repo are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses a [CalVer](https://calver.org/) versioning scheme with monthly releases, see [here](versioning.md)

## Unreleased

### Breaking changes
 - internal dependencies are now listed as regular dependencies in the `setup.py` file to overcome issues and make the installation process less complicated. This implies you need to install packages according to their dependencies and can no longer use the `external` tag as in `pip install airo-typing[external]`.
 see [issue #91](https://github.com/airo-ugent/airo-mono/issues/91) and
 [PR](https://github.com/airo-ugent/airo-mono/pull/108) for more details.
 - `PointCloud` dataclass replaces the `ColoredPointCloudType` to support point cloud attritubes


### Added
- `PointCloud` dataclass as the main data structure for point clouds in airo-mono
- Notebooks to get started with point clouds, checking performance and logging to rerun
- Functions to crop point clouds and filter points with a mask (e.g. low-confidence points)
- Functions to convert from our numpy-based dataclass to and from open3d point clouds
- `BoundingBox3DType`
- `Zed2i.ULTRA_DEPTH_MODE` to enable the ultra depth setting for the Zed2i cameras
- `OpenCVVideoCapture` implementation of `RGBCamera` for working with arbitrary cameras


### Changed
- Dropped support for python 3.8 and added 3.11 to the testing matrix [#103](https://github.com/airo-ugent/airo-mono/issues/103).
- Set python version to 3.10 because of an issue with the `ur_rtde` wheels [#121](https://github.com/airo-ugent/airo-mono/issues/121). Updated README.md to reflect this change.

### Fixed
- Fixed bug in `get_colored_point_cloud()` that removed some points see issue #25.
- Fixed bug requiring unplug-and-plug of USB cable for Realsense: see issue #109.
- Removed camera imports in `airo_camera_toolkit.cameras`: see issue #110.
- Added `__init__.py` to `realsense` and `utils` in `airo_camera_toolkit.cameras`, fixing installs with pip and issue #113.
- Fixed bug that returned a transposed resolution in `MultiprocessRGBReceiver`.
- Using `Zed2i.PERFORMANCE_DEPTH_MODE` will now correctly use the performance mode instead of the quality mode.

### Removed
- `ColoredPointCloudType`

## 2024.1.0

### Breaking changes

### Added
- Calendar-based versioning scheme and introduction of accompanying changelog.

### Changed

### Fixed

### Removed

