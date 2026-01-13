# Changelog

All notable changes for the packages in the airo-mono repo are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project uses a [CalVer](https://calver.org/) versioning scheme with monthly releases, see [here](versioning.md)

## Unreleased

### Breaking changes
- Code implementation for communicating with Schunk grippers has been changed to `SchunkGripperProcess`, `SchunkEGK40_USB` has been relegated to the schunk branch of this repo.
- Default manipulator specs in `URrtde` class were wrong, they are now multiplied by π -> **caution**: your robot may suddenly move 3x faster if you didn't set joint speed yourself.
- Old airo-teleop package was removed from the monorepo, in favor of a new [airo-teleop](https://github.com/airo-ugent/airo-teleop/), which is now considered a sister repository.
- Changed the usage interface of the parameter constants in `zed.py`. These constants are now accessible through their respective dataclass (e.g., NEURAL_DEPTH_MODE -> InitParams.NEURAL_DEPTH_MODE).

### Added
- Added `gc_disabled()` context manager to temporarily disable garbage collection for performance-critical sections.
- Added `get_model()` method to `URrtde` to automatically detect the UR model (UR3, UR3e, UR5e)
- Added `_retrieve_camera_pose()` method to the `Zed` class in `zed.py` to retrieve the pose matrix of the camera.
- Added `_request_spatial_map_update` and `_retrieve_spatial_map` methods to the `Zed` class in `zed.py` to respectively request and retrieve a spatial map update using VSLAM.
- Added dataclasses for the different kinds of Zed parameters (init parameters, positional tracking and spatial mapping) and a `ZedSpatialMap` dataclass.
- Added extra test code in the `__main__` block of `zed.py` to test the new methods.
- Added a new file `multiprocess_zed_camera.py`, implementing ZED-specific publisher and receiver classes that extend the StereoRGBD publisher and receiver with tracking and spatial mapping data exchange over shared memory.

### Changed
- Improved `execute_trajectory` reliability: we now temporarily disable the garbage collector in the hot loop to reduce latency using `gc_disabled()`.
- Deprecated `AsyncExecutor`. It is a very thin wrapper around [`ThreadPoolExecutor(max_workers=1)`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor), which should be used instead.
- `get_model()` (see Added) is now used in the URrtde constructor to automatically detect the UR model and set the manipulator specs appropriately.
- Dropped support for Python 3.9 and added support for 3.12. This only has an impact on our CI workflow and should not impact you directly, but it does mean that airo-mono could stop working without any warning on Python 3.9 in the future.
- Changed incorrect usages of the updated parameter constants in `zed.py` throughout the `airo-camera-toolkit` (see Breaking changes).

### Fixed
- Multiprocess cameras will now get an initial frame upon initialization using `_grab_images`, to avoid premature reads of invalid data (e.g., calling `intrinsics_matrix()` before `get_rgb_image_as_int()` - the latter triggers `_grab_images`).

### Removed

## 2025.11.0

### Breaking changes

### Added
- Added an `enable_pointcloud` flag to `Realsense` (default: `True`). When set to `False`, the point cloud will not be computed.
- Added an `enable_pointcloud` flag to `Multiprocess*RGBD[Publisher|Receiver]` classes (default: `True`). When set to `False`, the point cloud will not be transmitted.

### Changed
- Added a `Realsense._retrieve_colored_point_cloud` implementation, avoiding the costly default implementation. This improves performance of point cloud retrieval with Realsense cameras.
- Replace `time.time_ns()` with `time.perf_counter_ns()` in `execute_trajectory` implementations, for higher resolution timing and avoiding timing mistakes.

### Fixed
- Fixed timestamp in `MultiprocessRGBPublisher` and `MultiprocessRGBDPublisher` to represent the time when frames are captured from the camera, rather than when the message is created. This allows accurate latency measurement between image acquisition and processing.
- Fixed `extract_depth_from_depthmap_heuristic` function crashing when extracting depth values for points near image edges. The mask window now correctly clips to image boundaries and pads with NaN values when needed [#163](https://github.com/airo-ugent/airo-mono/issues/163).
- Fixed conversions between Open3D point clouds and airo-mono PointCloud types. They are now correctly casting between uint8 and float32. (Open3D uses floating point values between 0 and 1, but we use unsigned bytes between 0 and 255.)

### Removed

## 2025.9.0

### Breaking changes
- The ZED confidence map now returns values between 0 and 1 instead of 100 to 0, to be consistent with the newly added confidence maps in airo-mono.
- Assertions have been replaced with proper error handling and exceptions in several places to improve robustness and provide clearer error messages. This may affect existing code that relies on assertions for error checking.

### Added
- Add generic support for depth confidence maps in `airo-camera-toolkit`. Confidence maps are single-channel float32 images with values between 0 and 1, where 0 means no confidence and 1 means full confidence. A type has been added to `airo-typing`: `NumpyConfidenceMapType`.
  - The `DepthCamera` class has the most basic implementation, which naively assumes that confidence is lower around edges in the depth image. It uses Canny edge detection to find edges in the depth image and creates a confidence map based on the distance to the nearest edge.
  - The `StereoRGBDCamera` class has a more advanced implementation, which uses the left and right RGB images to compute a confidence map based on the disparity map. It uses the OpenCV `StereoSGBM` algorithm to compute the disparity map and then computes the confidence map based on the disparity values.
  - The `Realsense` class has a similar implementation to `StereoRGBDCamera`, but it uses a disparity map computed from the infrared images instead of the RGB images (because the D435 does not have a stereo RGB setup).
  - The `Zed` class has a built-in confidence map that is provided by the ZED SDK.
- `execute_trajectory` methods for executing time-parameterized trajectories on single and dual arm set-ups [#150](https://github.com/airo-ugent/airo-mono/issues/150)
- Unit tests for `execute_trajectory` methods

### Changed
- When an `AwaitableAction` timeouts, it will now print the file and line where it was created, as well as the file and line where `wait()` was called. This can help with debugging timeout issues.

### Fixed

### Removed

## 2025.8.0

### Breaking changes

#### airo-tulip

- Update airo-tulip to version 0.4.0, which returns odometry to the standard drive encoder based method.

#### NumPy 2

airo-mono is finally upgrading to NumPy 2.0! This is a major change that may break compatibility with some packages that depend on NumPy. Please read the notes below carefully.
In particular, the ZED SDK has been updated to version 5.0, which requires CUDA 12.8 to be installed.
You will need to update your CUDA installation and upgrade your ZED SDK to version 5.0 to use the ZED camera with airo-mono.

This has implications for downstream packages (or your own code). If you depend on certain software that requires an older version of NumPy, you may need to either:

- update that software to a version that is compatible with NumPy 2.0, or
- stick with airo-mono version 2025.7.0 or earlier until the software you depend on is updated.

The following changes have been made to airo-mono to support NumPy 2.0:

- Update NumPy to version > 2.0, which may break compatibility with some packages that depend on NumPy.
    - This forces downstream code to be compatible with the latest NumPy version.
    - With this change, we also update the OpenCV version to 4.10, which is compatible with NumPy 2.0.
    - With this change, we also update the Rerun version to 0.23, which is compatible with NumPy 2.0.
    - With this change, we also update the ZED SDK to version 5.0, which is compatible with NumPy 2.0.
- Update ZED SDK to version 5.0, which changes the API for the ZED camera provided by `airo-camera-toolkit` in the `Zed` class.
    - This requires CUDA 12.8 to be installed.
    - Depth modes have been replaced: you can now choose between `Zed.NEURAL_LIGHT_DEPTH_MODE`, `Zed.NEURAL_DEPTH_MODE`, and `Zed.NEURAL_PLUS_DEPTH_MODE`.
    - This improves depth quality, especially when using the neural plus model.
    - This also improves depth performance, especially when using the neural light model.
    - For more information, see the [ZED SDK 5.0 blog post](https://www.stereolabs.com/en-be/blog/introducing-zed-sdk-50).

### Added

- Add documentation on how to include custom sensors for odometry.

### Changed

### Fixed

- The `KELORobile` `move_platform_to_pose` method now calls the correct `get_odometry` method, allowing custom odometry implementations.

### Removed

## 2025.7.0

### Breaking changes

### Added

### Changed
- Update airo-tulip to version 0.3.0 for better orientation estimation.
- Use [`airo-ipc`](https://github.com/airo-ugent/airo-ipc) for multiprocessing in `airo-camera-toolkit`.

### Fixed
- Fixed a bug when the KELO Robile platform was moving around multiples of 360 degrees, where the target angle would switch.
- Fixed a bug where the KELO Robile platform would refuse to move to a pose if delta angle was close to 0.

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
