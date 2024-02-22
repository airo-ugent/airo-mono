from airo_camera_toolkit.image_transforms.image_transform import (
    HWCImageType,
    ImagePointType,
    ImageShapeType,
    ImageTransform,
)


def crop(image: HWCImageType, x: int, y: int, w: int, h: int) -> HWCImageType:
    """Crop a smaller rectangular part out of an image. We use the same rectangle convention as OpenCV.

    Args:
        image (HWCImageType): the image to crop
        x: the x-coordinate of the top-left corner of the crop, measured in pixels starting from the left edge of the image.
        y: the y-coordinate of the top-left corner of the crop, measured in pixels starting from the top edge of the image.
        w: the width of the crop in pixels.
        h: the height of the crop in pixels.
    """
    # Note that the first index of the array is the y-coordinate, because this indexes the rows of the image and the y-axis runs from top to bottom.
    if len(image.shape) == 2:
        return image[y : y + h, x : x + w].copy()

    return image[y : y + h, x : x + w, :].copy()


class Crop(ImageTransform):
    """"""

    def __init__(self, input_shape: ImageShapeType, x: int, y: int, w: int, h: int):
        super().__init__(input_shape)
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def shape(self) -> ImageShapeType:
        if len(self._input_shape) == 2:
            return self.h, self.w

        c = self._input_shape[2]
        return self.h, self.w, c

    def transform_image(self, image: HWCImageType) -> HWCImageType:
        return crop(image, self.x, self.y, self.w, self.h)

    def transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        assert x >= self.x and x < self.x + self.w
        assert y >= self.y and y < self.y + self.h
        return x - self.x, y - self.y

    def reverse_transform_point(self, point: ImagePointType) -> ImagePointType:
        x, y = point
        assert x >= 0 and x < self.w
        assert y >= 0 and y < self.h
        return x + self.x, y + self.y
