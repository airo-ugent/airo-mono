from __future__ import annotations

from airo_typing import NumpyFloatImageType, OpenCVImageType, TorchImageType


class ImageConverter:
    def __init__(self) -> None:
        self._image_in_numpy_float_format = None

    @classmethod
    def from_numpy_format(cls, image: NumpyFloatImageType) -> ImageConverter:
        pass

    @classmethod
    def from_opencv_format(cls, image: OpenCVImageType) -> ImageConverter:
        pass

    @classmethod
    def from_torch_format(cls, image: TorchImageType) -> ImageConverter:
        pass

    @property
    def as_numpy_float_image(self) -> NumpyFloatImageType:
        pass
