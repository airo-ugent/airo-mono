import pathlib
from typing import Tuple

import cv2
import numpy as np
import pytest
from airo_camera_toolkit.image_transforms import Crop, Resize, Rotate90
from airo_camera_toolkit.image_transforms.image_transform import ImageTransform

GRADIENT_IMAGE_PATH = pathlib.Path(__file__).parent / "data" / "gradient.jpg"


def _test_transform(point: Tuple[int, int], transform: ImageTransform):
    transformed_point = transform.transform_point(point)
    reversed_point = transform.reverse_transform_point(transformed_point)
    assert point == reversed_point


@pytest.mark.parametrize("n_rotations", [1, 2, 3, 4, 5])
def test_rotate_transform_points(n_rotations: int):
    transform = Rotate90(input_shape=(101, 208, 3), num_rotations=n_rotations)
    original_point = (50, 75)
    _test_transform(original_point, transform)


def test_rotate_off_by_one():
    rotate90 = Rotate90(input_shape=(720, 1280, 3))
    original_point = (1279, 0)
    rotate90_point = rotate90.transform_point(original_point)
    assert rotate90_point == (0, 0)

    rotate180 = Rotate90(input_shape=(720, 1280, 3), num_rotations=2)
    original_point = (1279, 719)
    rotate180_point = rotate180.transform_point(original_point)
    assert rotate180_point == (0, 0)


def test_resize_transform_points():
    transform = Resize(input_shape=(101, 208, 3), h=51, w=109)
    original_point = (50, 75)
    _test_transform(original_point, transform)


def test_crop_transform_points():
    transform = Crop(input_shape=(101, 208, 3), x=10, y=10, h=51, w=109)
    original_point = (50, 25)
    _test_transform(original_point, transform)


def test_crop_transform_image():
    test_pixel_coords = (19, 115)
    image = cv2.imread(str(GRADIENT_IMAGE_PATH))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = Crop(input_shape=image.shape, x=10, y=100, h=2000, w=2100)
    # test both int and float images
    for image in [image, image.astype(np.float32)]:
        transformed_image = transform.transform_image(image)
        assert transformed_image.shape == (2000, 2100, 3)
        transformed_point = transform.transform_point(test_pixel_coords)
        assert transformed_point[0] == test_pixel_coords[0] - 10
        assert transformed_point[1] == test_pixel_coords[1] - 100
        assert np.isclose(
            transformed_image[transformed_point[1], transformed_point[0]],
            image[test_pixel_coords[1], test_pixel_coords[0]],
        ).all()


def test_resize_transform_image():
    test_pixel_coords = (19, 15)
    image = cv2.imread(str(GRADIENT_IMAGE_PATH))
    transform = Resize(input_shape=image.shape, h=4000, w=3000)
    # test both int and float images
    for image in [image, image.astype(np.float32)]:
        transformed_image = transform.transform_image(image)
        assert transformed_image.shape == (4000, 3000, 3)
        # can be a little different due to interpolation!
        transformed_point = transform.transform_point(test_pixel_coords)
        assert np.isclose(
            transformed_image[transformed_point[1], transformed_point[0]],
            image[test_pixel_coords[1], test_pixel_coords[0]],
            rtol=0.1,
        ).all()


def test_rotate_transform_image():
    test_pixel_coords = (15, 19)
    image = cv2.imread(str(GRADIENT_IMAGE_PATH))
    transform = Rotate90(input_shape=image.shape, num_rotations=1)
    # test both int and float images
    for image in [image, image.astype(np.float32)]:
        transformed_image = transform.transform_image(image)
        assert transformed_image.shape[0] == image.shape[1]
        assert transformed_image.shape[1] == image.shape[0]
        transformed_point = transform.transform_point(test_pixel_coords)
        assert np.isclose(
            transformed_image[transformed_point[1], transformed_point[0]],
            image[test_pixel_coords[1], test_pixel_coords[0]],
        ).all()
