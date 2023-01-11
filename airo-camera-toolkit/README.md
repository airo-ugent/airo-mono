# airo-camera-toolkit

This package contains code for working with RGB(D) cameras and implementations of our interface for the cameras we use at the lab.

overview of the functionality and the structure:
```
airo_camera_toolkit/
    interfaces.py               # defines common interfaces for all cameras
    reprojection.py             # code for projecting points to the image plane
                                # and reprojecting points from image plane to world
    utils.py                    # a.o. code for converting images between
                                # differen formats, such as BGR - RGB or channel-first vs channel-last.

    cameras/                    # implementation of interface for real cameras
        zed2i.py
        manual_test_hw.py       # code for manually testing the above implementations.
```

some pointers for more background on cameras, in particular on the meaning of intrinsics, extrinics, distortion coefficients, pinhole (and other) camera models, see:
 - https://web.eecs.umich.edu/~justincj/teaching/eecs442/WI2021/schedule.html
 - https://learnopencv.com/geometry-of-image-formation/ (extrinsics & intrinsics)
 - http://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf (idem)

