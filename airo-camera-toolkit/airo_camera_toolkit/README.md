## airo_blender_toolkit
Overview of the functionality and the structure:
```python
airo_camera_toolkit
├── interfaces.py               # Common interfaces for all cameras.
├── reprojection.py             # Projecting points to the image plane
│                               # and reprojecting points from image plane to world
├── utils.py                    # Conversion between image format e.g. BGR to RGB
│                               # or channel-first vs channel-last.
└── cameras                     # Implementation of the interfaces for real cameras
    ├── zed2i.py                # Run this file to test your ZED Installation
    └── manual_test_hw.py       # Used for manually testing in the above implementations.

```