# airo-camera-toolkit
This package contains code for working with RGB(D) cameras, images and pointclouds. It provides following functionality:

- interfacing with RGB(D) cameras
- (re)projecting between 3D world and images
- converting between different image formats
- detecting (ch)aruco markers
- extrinsics calibration: marker pose estimation,eye-in-hand and eye-to-hand
- invertible' transforms for cropping/scaling images, and obtaining the pixel in the original image that corresponds to pixels on the modified image (TODO)

#TODO: convert to a filetree.

## Installation
The `airo_camera_toolkit` package can be installed with pip by running (from this directory):
```
pip install .
```
This will already allow you to use the hardare-independent functionality of this package, e.g. image conversion and projection.

> TODO: tutorial on how to get started with the tools.

### 1.1 Hardware Installation
Depending on the hardware you are using, you might need to complete additional installation.
Instructions can be found in the following files:
* [ZED Installation](airo_camera_toolkit/cameras/zed_installation.md)

## Real-time visualisation
For realtime visualisation of robotics data we  strongly encourage using [rerun.io](https://www.rerun.io/) instead of manually hacking something together with opencv/pyqt/... No wrappers are needed here, just pip install the SDK. An example notebook to get to know this tool and its potential can be found [here](docs/rerun-zed-example.ipynb).

## Calibration
### hand-eye calibration
We use by default a charuco board for hand-eye calibration. You can find the board in the `test/data` folder. To match the size of the markers to the desired size, the board should be printed on a 300mm x 220mm surface. Using Charuco boards is highly recommended as they are a lot more robust and precise than individual aruco markers, if you do use an aruco marker, make sure that the whitespace around the marker is at least 25% of the marker dimensions. 
## References
For more background on cameras, in particular on the meaning of intrinsics, extrinics, distortion coefficients, pinhole (and other) camera models, see:
 - https://web.eecs.umich.edu/~justincj/teaching/eecs442/WI2021/schedule.html
 - https://learnopencv.com/geometry-of-image-formation/ (extrinsics & intrinsics)
 - http://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf (idem)
