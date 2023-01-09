from __future__ import annotations

import numpy as np
from airo_typing import NumpyFloatImageType, OpenCVImageType, TorchImageType


class ImageConverter:
    """
    Utility class to convert between numpy arrays of different image formats.

    Only supports cpu-located float images.
    Convert ints to floats if you want to convert int images.
    Convert cuda images to cpu images (if you can afford it) or re-implement with torch.
    """

    def __init__(self, image_in_numpy_float_format: NumpyFloatImageType) -> None:
        self._image_in_numpy_float_format = image_in_numpy_float_format

    @staticmethod
    def _is_valid_image(image: object) -> bool:
        if not isinstance(image, np.ndarray):
            return False
        valid = True
        valid = valid and image.ndim == 3
        valid = valid and np.max(image) <= 1.0
        valid = valid and np.min(image) >= 0.0
        valid = valid and image.dtype in (np.float32, np.float64, np.float16)
        return valid

    @classmethod
    def from_numpy_format(cls, image: NumpyFloatImageType) -> ImageConverter:
        assert image.shape[2] == 3
        assert cls._is_valid_image(image)
        return ImageConverter(image)

    @classmethod
    def from_opencv_format(cls, image: OpenCVImageType) -> ImageConverter:
        assert cls._is_valid_image(image)
        assert image.shape[2] == 3

        # convert BGR to RGB
        image = image[:, :, ::-1]
        return ImageConverter(image)

    @classmethod
    def from_torch_format(cls, image: TorchImageType) -> ImageConverter:
        assert cls._is_valid_image(image)
        assert image.shape[0] == 3

        # channel first to channel last
        image = np.transpose(image, (1, 2, 0))
        return ImageConverter(image)

    @property
    def image_in_numpy_format(self) -> NumpyFloatImageType:
        return self._image_in_numpy_float_format

    @property
    def image_in_opencv_format(self) -> OpenCVImageType:
        return self._image_in_numpy_float_format[:, :, ::-1]

    @property
    def image_in_torch_format(self) -> TorchImageType:
        return np.transpose(self._image_in_numpy_float_format, (2, 0, 1))
