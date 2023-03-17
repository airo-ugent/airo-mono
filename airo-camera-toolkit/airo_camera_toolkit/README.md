## airo_blender_toolkit
A package for working with RGB(D) cameras and images.

most files have a script in the `__main__` block of the code, this is usually a good starting point for using the module.

Overview of the functionality and the structure:
```python
airo_camera_toolkit
├── interfaces.py               # Common interfaces for all cameras.
├── reprojection.py             # Projecting points to the image plane
│                               # and reprojecting points from image plane to world
├── utils.py                    # Conversion between image format e.g. BGR to RGB
│                               # or channel-first vs channel-last.
└── cameras                     # Implementation of the interfaces for real cameras
    ├── zed2i.py                # implementation using ZED SDK, run this file to test your ZED Installation
    └── manual_test_hw.py       # Used for manually testing in the above implementations.
└── calibration
    ├── fiducial_markers.py     # code for detecting and localising aruco markers and charuco boards
    └── hand_eye_calibration.py # camera-robot extrinsics calibration

```