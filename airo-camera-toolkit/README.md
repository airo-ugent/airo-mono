# airo-camera-toolkit
This package contains code for working with RGB(D) cameras, images and point clouds.
Overview of the functionality and the structure:
```cs
airo_camera_toolkit
├── interfaces.py               # Common interfaces for all cameras.
├── reprojection.py             # Projecting points between the 3D world and images
│                               # and reprojecting points from image plane to world
├── utils.py                    # Conversion between image formats: BGR to RGB, int to float, etc.
│                               # or channel-first vs channel-last.
├── annotation_tools.py         # Tool for annotating images with keypoints, lines, etc.
└── cameras                     # Implementation of the interfaces for real cameras
│   ├── zed2i.py                # Implementation using ZED SDK, run this file to test your ZED Installation
│   ├── realsense.py            # Implementation using RealSense SDK
│   └── manual_test_hw.py       # Used for manually testing in the above implementations.
└── calibration
│   ├── fiducial_markers.py     # Detecting and localising aruco markers and charuco boards
│   └── hand_eye_calibration.py # Camera-robot extrinsics calibration, eye-in-hand and eye-to-hand
└── image_transforms            # Invertible transforms for cropping/scaling images with keypoints
└── point_clouds                # Point cloud processing, Open3D integration
    └── ...
└── multiprocess                # Multiprocessing
    └── ...

```

## Installation
The `airo_camera_toolkit` package can be installed with pip by running (from this directory):
```
pip install .
```
This will already allow you to use the hardare-independent functionality of this package, e.g. image conversion and projection.
Depending on the hardware you are using, you might need to complete additional installation.
Instructions can be found in the following files:
* [ZED Installation](airo_camera_toolkit/cameras/zed_installation.md)
* [RealSense Installation](airo_camera_toolkit/cameras/realsense_installation.md)

## Getting started with cameras
Camera can be accessed by instantiating the corresponding class:, e.g. for a ZED camera:
```python
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.utils import ImageConverter
import cv2

camera = Zed2i(Zed2i.RESOLUTION_720, fps=30)

while True:
    image_rgb_float = camera.get_rgb_image()
    image_bgr = ImageConverter.from_numpy_format(image_rgb_float).image_in_opencv_format
    cv2.imshow("Image", image_bgr)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
```

## Hand-eye calibration

Find the pose of a camera relative to a robot. In the [`calibration`](./airo_camera_toolkit/calibration/) folder, run this to get started:

```shell
airo-camera-toolkit hand-eye-calibration --help
```

See [calibration/README.md](./airo_camera_toolkit/calibration/README.md) for more details.


## Image format conversion
Camera by default return images as numpy 32-bit float RGB images with values between 0 to 1 through `get_rgb_image()`.
This is most convenient for subsequent processing, e.g. with neural networks.
For higher performance, 8-bit unsigned integer RGB images are also accessible through `get_rgb_image_as_int()`.

However, when using OpenCV, you will need conversion to BGR format.
For this you can use the `ImageConverter` class:
```python
from airo_camera_toolkit.utils import ImageConverter

image_rgb_int = camera.get_rgb_image_as_int()
image_bgr = ImageConverter.from_numpy_int_format(image_rgb_int).image_in_opencv_format
```


## Reprojection

See [reprojection.py](./airo_camera_toolkit/reprojection.py) for more details.

## Annotation tool

See [annotation_tool.md](./airo_camera_toolkit/annotation_tool.md) for usage instructions.

## Image Transforms

See the [README](./airo_camera_toolkit/image_transforms/README.md) in the `image_transforms` folder for more details.

## Real-time visualisation
For realtime visualisation of robotics data we  strongly encourage using [rerun.io](https://www.rerun.io/) instead of manually hacking something together with opencv/pyqt/... No wrappers are needed here, just pip install the SDK. An example notebook to get to know this tool and its potential can be found [here](notebooks/rerun-zed-tutorial.ipynb).
See this [README](./docs/rerun.md) for more details.

## Point clouds
See the tutorial notebook [here](notebooks/point_clouds_tutorial.ipynb) for an introduction.

## Multiprocessing
Camera processing can be computationally expensive.
If this is a problem for your application, see [multiprocess/README.md](./airo_camera_toolkit/cameras/multiprocess/README.md).

## References
For more background on cameras, in particular on the meaning of intrinsics, extrinics, distortion coefficients, pinhole (and other) camera models, see:
 - https://web.eecs.umich.edu/~justincj/teaching/eecs442/WI2021/schedule.html
 - https://learnopencv.com/geometry-of-image-formation/ (extrinsics & intrinsics)
 - http://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf (idem)
