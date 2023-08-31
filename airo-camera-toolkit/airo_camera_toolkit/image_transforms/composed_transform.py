from typing import List

from airo_camera_toolkit.image_transforms.image_transform import (
    HWCImageType,
    ImagePointType,
    ImageShapeType,
    ImageTransform,
)


class ComposedTransform(ImageTransform):
    def __init__(self, transforms: List[ImageTransform]):
        if len(transforms) == 0:
            raise ValueError("transforms must be a non-empty list.")

        super().__init__(transforms[0]._input_shape)
        self.transforms = transforms

    @property
    def shape(self) -> ImageShapeType:
        return self.transforms[-1].shape

    def transform_image(self, image: HWCImageType) -> HWCImageType:
        for transform in self.transforms:
            image = transform.transform_image(image)
        return image

    def transform_point(self, point: ImagePointType) -> ImagePointType:
        for transform in self.transforms:
            point = transform.transform_point(point)
            print(point)
        return point

    def reverse_transform_point(self, point: ImagePointType) -> ImagePointType:
        for transform in reversed(self.transforms):
            point = transform.reverse_transform_point(point)
        return point
