import numpy as np
from airo_camera_toolkit.image_transforms.image_transform import (
    HWCImageType,
    ImagePointType,
    ImageShapeType,
    ImageTransform,
)


class Rotate90(ImageTransform):
    """Rotate an image by multiples of 90 degrees."""

    def __init__(
        self,
        input_shape: ImageShapeType,
        num_rotations: int = 1,
    ):
        """Create a new Rotate transform.

        Args:
            num_rotations: the number of 90-degree rotations to apply. Positive values rotate counter-clockwise.
        """
        super().__init__(input_shape)

        if not isinstance(num_rotations, int):
            raise TypeError("num_rotations must be an int")

        self._num_rotations = num_rotations % 4

    @property
    def shape(self) -> ImageShapeType:
        if self._num_rotations % 2 == 0:
            h, w = self._input_shape[:2]
        else:
            w, h = self._input_shape[:2]

        if len(self._input_shape) == 2:
            return h, w

        c = self._input_shape[2]
        return h, w, c

    def transform_image(self, image: HWCImageType) -> HWCImageType:
        # The copy here ensure the result is not a view into the original image.
        return np.rot90(image, self._num_rotations).copy()

    def transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        assert x >= 0 and x < self._input_w
        assert y >= 0 and y < self._input_h

        if self._num_rotations == 1:
            return y, self._input_w - x - 1
        elif self._num_rotations == 2:
            return self._input_w - x - 1, self._input_h - y - 1
        elif self._num_rotations == 3:
            return self._input_h - y - 1, x
        return x, y

    def reverse_transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        if self._num_rotations == 1:
            return self._input_w - y - 1, x
        elif self._num_rotations == 2:
            return self._input_w - x - 1, self._input_h - y - 1
        elif self._num_rotations == 3:
            return y, self._input_h - x - 1
        return x, y
