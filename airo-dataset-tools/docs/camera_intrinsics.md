# Camera Intrinsics format

We currently have two use cases for storing camera intrinsics.
1. Setting up synthetic cameras (e.g. in Blender) with the intrinsics of our real ZED cameras.
2. Supplementing our image datasets with the intrinsics of the cameras that were used to capture them.

It seem natural to store intrinsics (focal length, pixel size) in phyiscal units e.g. millimeters and in a mostly resolution-independent way.
However in practice, intrinsics are often stored in pixels and are resolution-dependent.

The reason is twofold.
1. Not all physical parameters can be exacted with a calibration procedure (e.g. the size of a pixel).
2. Having everything in pixels makes it easier to create an intrinsics matrix K that can be used to convert 3D points to 2D image coordinates in pixels.

```json
{
  "image_resolution": {
    "width": 2208,
    "height": 1520
  },
  "focal_lengths_in_pixels": {
    "fx": 1067.91,
    "fy": 1068.05
  },
  "principal_point_in_pixels": {
    "cx": 1107.48,
    "cy": 629.675
  },
  "radial_distortion_coefficients": [-0.0542749, 0.0268096, -0.0104089],
  "tangential_distortion_coefficients": [0.000204483, -0.000310015]
}
```

## ZED camera intrinsics
For comparison, ZED stores intrinsics as follows (from `/usr/local/zed/settings/<serial_id>.conf`):

```conf
[LEFT_CAM_2K]
fx=1067.91
fy=1068.05
cx=1107.48
cy=629.675
k1=-0.0542749
k2=0.0268096
p1=0.000204483
p2=-0.000310015
k3=-0.0104089

[LEFT_CAM_FHD]
fx=1067.91
fy=1068.05
cx=963.48
cy=548.675
k1=-0.0542749
k2=0.0268096
p1=0.000204483
p2=-0.000310015
k3=-0.0104089

[LEFT_CAM_HD]
fx=533.955
fy=534.025
cx=640.24
cy=362.8375
k1=-0.0542749
k2=0.0268096
p1=0.000204483
p2=-0.000310015
k3=-0.0104089

...
```

The values above are for a ZED2i camera.
Note how the focal lengths are different for the 2K (2208 x 1520) and HD (1280 x 720) resolutions.
They are both expressed in "amount of pixels", however the phyiscal size of image pixel is different for the two resolutions.

The sensor pixels are 0.002 mm and the focal length is 2.12 mm, so the focal length expressed in sensor pixels is 2.12 / 0.002 = 1060.
This is close to the focal length expressed in the pixels for the 2K resolution.
This means that for that resolution, each image pixel corresponds to a sensor pixel.

For the HD resolution, we can see that focal lengths are halved, which means the pixel size should be twice as big.
So its likely that for HD 4 sensor pixels in a 2 x 2 square are binned to create an image pixel.

Beside focal length, principal point also differs between the resolutions.
The principal point shifts the image origin form the center to the top left corner of the image.
If the image center is aligned with the optical axis, we expect its values to be approximately half the resolution in pixels.

``` python
2208 / 2 = 1104
1242 / 2 = 621
1920 / 2 = 960
1080 / 2 = 540
1280 / 2 = 640
720 / 2 = 360
```
These are also pretty close to the principal point values we can see in calibration.
