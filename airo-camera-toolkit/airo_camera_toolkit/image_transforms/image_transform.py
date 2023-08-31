from abc import ABC
from typing import Tuple, Union

from airo_typing import NumpyFloatImageType, NumpyIntImageType, OpenCVIntImageType

HWCImageType = Union[OpenCVIntImageType, NumpyFloatImageType, NumpyIntImageType]
"""an image with shape (H,W,C)"""

ImageShapeType = Union[Tuple[int, int, int], Tuple[int, int]]
ImagePointType = Union[Tuple[int, int], Tuple[float, float]]


class ImageTransform(ABC):
    def __init__(self, input_shape: ImageShapeType):
        self._input_shape = input_shape

    @property
    def _input_h(self) -> int:
        return self._input_shape[0]

    @property
    def _input_w(self) -> int:
        return self._input_shape[1]

    @property
    def shape(self) -> ImageShapeType:
        """The shape of the transformed image.

        Returns:
            ImageShapeType: The shape of the transformed image.
        """
        raise NotImplementedError

    def transform_image(self, image: HWCImageType) -> HWCImageType:
        """Apply the image transform to an image to get a new image.

        Args:
            image (HWCImageType): The original image, it will be unaffected by the transform.

        Raises:
            NotImplementedError: Subclasses must implement this method.

        Returns:
            HWCImageType: The new, transformed image.
        """
        raise NotImplementedError

    def transform_point(self, point: ImagePointType) -> ImagePointType:
        """Transform the coordinates of a point from original image to transformed image."""
        raise NotImplementedError

    def reverse_transform_point(self, point: ImagePointType) -> ImagePointType:
        """Transform the coordinates of a point in the transformed image back to the original image."""
        raise NotImplementedError

    def __call__(self, image: HWCImageType) -> HWCImageType:
        """Shorthand to transform an image."""
        return self.transform_image(image)
