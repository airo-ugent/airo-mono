# Generic OpenCV camera

This `RGBCamera` implementation allows testing arbitrary cameras through the OpenCV `VideoCapture` interface.

We currently do not support intrinsics calibration in airo-camera-toolkit. You can find the intrinsics of your camera
using [these instructions](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html).
