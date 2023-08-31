# airo-spatial-algebra

This package provides functionality to work with SE3 poses, transforms etc. The heavy lifting is done by Peter Corke's [spatial-math](https://github.com/petercorke/spatialmath-python) package. We simply wrap a subset of its features to make it more verbose and self-explanatory.

The package contains an `SE3Container` class for converting between different representations of SE3 poses (and hence SO3 rotations as well).
Furthermore a few common operations on points and poses (such as changing the frame in which they are represented) are provided for convenience.

