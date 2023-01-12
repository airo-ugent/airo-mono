# airo-camera-toolkit
This package contains code for working with RGB(D) cameras and implementations of our interface for the cameras we use at the lab.

## 1. Installation
The general `airo_camera_toolkit` package can be installed with pip by running (from this directory):
```
pip install .
```
This will already allow you to use the hardare-independent functionality of this package, e.g. image conversion and projection.

> TODO: tutorial on how to get started with the tools.

### 1.1 Hardware Installation
Depending on the hardware you are using, you might need to complete additional installation.
Instructions can be found in the following files:
* [ZED Installation](airo_camera_toolkit/cameras/zed_installation.md)

## 2. References
For more background on cameras, in particular on the meaning of intrinsics, extrinics, distortion coefficients, pinhole (and other) camera models, see:
 - https://web.eecs.umich.edu/~justincj/teaching/eecs442/WI2021/schedule.html
 - https://learnopencv.com/geometry-of-image-formation/ (extrinsics & intrinsics)
 - http://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf (idem)
