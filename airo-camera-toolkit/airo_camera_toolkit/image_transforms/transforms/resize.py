import cv2
from airo_camera_toolkit.image_transforms.image_transform import (
    HWCImageType,
    ImagePointType,
    ImageShapeType,
    ImageTransform,
)


class Resize(ImageTransform):
    def __init__(self, input_shape: ImageShapeType, h: int, w: int, round_transformed_points: bool = True):
        """Create a new Resize transform.

        Note: Transforming a point to or from a resized image can lead to non-integer coordinates. Pixel coordinates
            are however often expected to be integers, e.g. by the OpenCV draw functions. So by default, this class
            will round transformed points to the nearest integer. If you want to avoid the errors introduced by
            rounding, you can set `round_transformed_points` to False to get the exact transformed points as floats.

        Args:
            input_shape: Shape of the images that will be resized.
            h: Height of the resized image.
            w: Width of the resized image.
            round_transformed_points: Whether to round transformed points to the nearest integer.
        """
        super().__init__(input_shape)
        self.h = h
        self.w = w
        self.round_transformed_points = round_transformed_points

    @property
    def shape(self) -> ImageShapeType:
        if len(self._input_shape) == 2:
            return self.h, self.w

        c = self._input_shape[2]
        return self.h, self.w, c

    def transform_image(self, image: HWCImageType) -> HWCImageType:
        return cv2.resize(image, (self.w, self.h))

    def transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        assert x >= 0 and x < self._input_w
        assert y >= 0 and y < self._input_h

        w_scale = self.w / self._input_w
        h_scale = self.h / self._input_h

        x_float = w_scale * x
        y_float = h_scale * y

        if self.round_transformed_points:
            return round(x_float), round(y_float)

        return x_float, y_float

    def reverse_transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        assert x >= 0 and x < self.w
        assert y >= 0 and y < self.h
        w_scale_inverse = self._input_w / self.w
        h_scale_inverse = self._input_h / self.h

        x_float = w_scale_inverse * x
        y_float = h_scale_inverse * y

        if self.round_transformed_points:
            return round(x_float), round(y_float)

        return x_float, y_float
