import pytest
from airo_camera_toolkit.image_transforms import Crop, Resize, Rotation90
from airo_camera_toolkit.image_transforms.image_transform import ImageTransform


def _test_transform(point: tuple[int, int], transform: ImageTransform):
    transformed_point = transform.transform_point(point)
    reversed_point = transform.reverse_transform_point(transformed_point)
    assert point == reversed_point


@pytest.mark.parametrize("n_rotations", [1, 2, 3, 4, 5])
def test_rotate_transform_points(n_rotations: int):
    transform = Rotation90(input_shape=(101, 208, 3), num_rotations=n_rotations)
    original_point = (50, 75)
    _test_transform(original_point, transform)


def test_resize_transform_points():
    transform = Resize(input_shape=(101, 208, 3), h=51, w=109)
    original_point = (50, 75)
    _test_transform(original_point, transform)


def test_crop_transform_points():
    transform = Crop(input_shape=(101, 208, 3), x=10, y=10, h=51, w=109)
    original_point = (50, 75)
    _test_transform(original_point, transform)


# TODO: add tests for the image transforms as well?
