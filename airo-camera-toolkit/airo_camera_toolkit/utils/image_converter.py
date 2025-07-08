from __future__ import annotations

import numpy as np
from airo_typing import NumpyFloatImageType, NumpyIntImageType, OpenCVIntImageType, TorchFloatImageType


def is_image_array(image: object) -> bool:
    """checks if an object is a numpy array with 3 dimensions, which is the only thing all image formats have in common"""
    if not isinstance(image, np.ndarray):
        return False
    valid = True
    valid = valid and image.ndim == 3
    return valid


def is_float_image_array(image: object) -> bool:
    """checks if an object is a valid  int image array
    by checking
    - if it is a valid image array
    - if it contains floats
    - and if the first element is in the right range"""
    valid = is_image_array(image)

    # make mypy happy but this check is already performed..
    assert isinstance(image, np.ndarray)

    valid = valid and image.dtype in (np.float32, np.float64, np.float16)
    # check first pixel instead of global max to reduce computational burden
    # doing this for a 6M float image (1000x 2000 x3) takes a few ms
    valid = valid and image[0, 0, 0] <= 1.0
    valid = valid and image[0, 0, 0] >= 0.0
    return valid


def is_int_image_array(image: np.ndarray) -> bool:
    """checks if an object is a valid  int image array
    by checking
    - if it is a valid image array
    - if it contains ints
    - and if the first element is in the right range"""
    valid = is_image_array(image)
    valid = valid and image.dtype in (np.uint8, np.uint16, np.uint32)
    # check first pixel instead of global max to reduce computational burden
    # doing this for a 6M int image (1000x 2000 x3) takes a few ms
    valid = valid and image[0, 0, 0] >= 0
    valid = valid and image[0, 0, 0] <= 255
    return valid


class ImageConverter:
    """
    Utility class to convert between numpy arrays of different image formats.

    Only supports cpu-located images.
    Convert cuda images to cpu images (if you can afford it) or re-implement with torch.

    **Note** that these conversions may not be optimal, because we use an intermediate numpy float format.
    So, there may be a conversion from type A to type B that is faster if you do it directly,
    but we don't implement that here to keep implementation complexity low.
    See also https://github.com/airo-ugent/airo-mono/issues/132.
    """

    def __init__(self, image_in_numpy_float_format: NumpyFloatImageType) -> None:
        assert is_float_image_array(image_in_numpy_float_format)
        assert image_in_numpy_float_format.shape[2] == 3

        self._image_in_numpy_float_format = image_in_numpy_float_format

    @classmethod
    def from_numpy_format(cls, image: NumpyFloatImageType) -> ImageConverter:
        assert is_float_image_array(image)
        assert image.shape[2] == 3
        # create copy to avoid altering the input image
        image = np.copy(image)
        return ImageConverter(image)

    @classmethod
    def from_numpy_int_format(cls, image: NumpyIntImageType) -> ImageConverter:
        assert is_int_image_array(image)
        assert image.shape[2] == 3
        # convert to floats (creates a copy)
        image = image.astype(np.float32) / 255.0
        return ImageConverter(image)

    @classmethod
    def from_opencv_format(cls, image: OpenCVIntImageType) -> ImageConverter:
        assert is_int_image_array(image)
        assert image.shape[2] == 3

        # convert to float (creates copy)
        # can take a few ms..

        image = image.astype(np.float32) / 255.0
        # convert BGR to RGB
        image = image[:, :, ::-1]

        return ImageConverter(image)

    @classmethod
    def from_torch_format(cls, image: TorchFloatImageType) -> ImageConverter:
        assert is_float_image_array(image)
        assert image.shape[0] == 3

        # create copy to avoid altering the input image
        image = np.copy(image)
        # channel first to channel last
        image = np.transpose(image, (1, 2, 0))
        return ImageConverter(image)

    @property
    def image_in_numpy_format(self) -> NumpyFloatImageType:
        return self._image_in_numpy_float_format

    @property
    def image_in_opencv_format(self) -> OpenCVIntImageType:
        image = self._image_in_numpy_float_format[:, :, ::-1]
        # can take up to a few ms..
        image *= 255.0
        return image.astype(np.uint8)

    @property
    def image_in_torch_format(self) -> TorchFloatImageType:
        return np.transpose(self._image_in_numpy_float_format, (2, 0, 1))

    @property
    def image_in_numpy_int_format(self) -> NumpyIntImageType:
        return (self._image_in_numpy_float_format * 255.0).astype(np.uint8)
