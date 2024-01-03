# Pinhole Operations

This subpackage contains code to perform various geometric operations to convert between 3D positions and 2D image coordinates using the pinhole camera model.

Most notably these include:

- projection to go from 3D points to 2D image coordinates
- triangulation to go from a set of corresponding image coordinates to a 3D point
- unprojection to go from a 2D coordinate to a 3D point by intersecting the ray with depth information


For mathematical background:

- IRM course
- Sleziski, [Computer Vision and Algorithms](https://szeliski.org/Book/)
- Opencv page on [cameras](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#)