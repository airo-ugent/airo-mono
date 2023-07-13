# Reversible Image Transforms :framed_picture:

A few basics image transforms to help manipulate images into the shape, size and orientation needed for perception or visualization.
Our main use case is converting camera images to the shape needed for our keypoint detection models.
We also provide functionality to reverse transfrom the coordinates of the detected keypoints to the original image space. This is not available in image transform libraries such as Albumentations but is important for robotics applications: you often need to reproject the points using the camera intrinsics, which are known for the original but not for the transformed image.

Implemented transforms:
* [`Crop`](image_transforms/crop.py)
* [`Resize`](image_transforms/resize.py)
* [`Rotation90`](image_transforms/rotation.py)
* [`ComposedTransform`](image_transforms/composed_transform.py)

## Usage
See the [tutorial notebook](tutorial.ipynb).

Quick overview:
```python
from airo_camera_toolkit.image_transforms import Crop, Resize, Rotation90, ComposedTransform

image = cv2.imread("path/to/image.jpg")

# Available transforms
crop = Crop(image.shape, x=160, y=120, h=200, w=200)
resize = Resize(crop.shape, h=100, w=100, round_transformed_points=False)
rotation_clockwise = Rotation90(resize.shape, -1)

image_cropped = crop(image)
image_resized = resize(image_cropped)
image_rotated = rotation_clockwise(image_resized)

# Composing transforms
transform = ComposedTransform([crop, resize, rotation_clockwise])
image_transformed = transform(image) # Equivalent to image_rotated

# Transforming a point
point = (200, 200)
point_transformed = transform.transform_point(point)
point_reversed = transform.reverse_transform_point(point_transformed)

point_reversed_int = tuple(map(int, point_reversed))

point == point_reversed_int
```

## Image Coordinate System
The origin (0,0) of an image is generally chosen at the top left corner, and we also use this convention.
The x-axis is positive to the right and the y-axis is positive downwards.

![Coordinate system](https://documentation.euresys.com/Products/OPEN_EVISION/OPEN_EVISION/en-us/Content/Resources/Images/03-1-2_Manipulating/Image_Coordinate_Systems_integer.png)

> :information_source: Sometimes a difference is made between integer pixel coordinates and float image coordinates, illustrated [here](https://documentation.euresys.com/Products/OPEN_EVISION/OPEN_EVISION/en-us/Content/03_Using/1_Starting_Up/3_Manipulating/Image_Coordinate_Systems.htm?TocPath=Using%20Open%20eVision%7CManipulating%20Pixels%20Containers%20and%20Files%7C_____5). **TODO** [check](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html): when projecting from 3D to 2D, in which coordinate system are the 2D coordinates expressed?