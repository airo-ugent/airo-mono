import cv2
from airo_camera_toolkit.image_transforms.image_transform import (
    HWCImageType,
    ImagePointType,
    ImageShapeType,
    ImageTransform,
)


class Resize(ImageTransform):
    def __init__(self, input_shape: ImageShapeType, h: int, w: int, round_transformed_points: bool = True):
        super().__init__(input_shape)
        self.h = h
        self.w = w
        self.round_transformed_points = round_transformed_points

    @property
    def shape(self) -> ImageShapeType:
        if len(self._input_shape) == 2:
            return self.h, self.w

        c = self._input_shape[2]  # type: ignore
        return self.h, self.w, c

    def transform_image(self, image: HWCImageType) -> HWCImageType:
        return cv2.resize(image, (self.w, self.h))

    def transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point

        w_scale = self.w / self._input_w
        h_scale = self.h / self._input_h

        x_float = w_scale * x
        y_float = h_scale * y

        if self.round_transformed_points:
            return round(x_float), round(y_float)

        return x_float, y_float

    def reverse_transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        w_scale_inverse = self._input_w / self.w
        h_scale_inverse = self._input_h / self.h

        x_float = w_scale_inverse * x
        y_float = h_scale_inverse * y

        if self.round_transformed_points:
            return round(x_float), round(y_float)

        return x_float, y_float
