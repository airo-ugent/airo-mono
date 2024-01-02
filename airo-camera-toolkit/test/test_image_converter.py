import numpy as np
from airo_camera_toolkit.utils.image_converter import (
    ImageConverter,
    is_float_image_array,
    is_image_array,
    is_int_image_array,
)


def test_image_format_checks():
    float_image = np.random.rand(10, 10, 3)
    assert is_image_array(float_image)
    assert is_float_image_array(float_image)
    assert not is_int_image_array(float_image)

    int_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    assert is_image_array(int_image)
    assert not is_float_image_array(int_image)
    assert is_int_image_array(int_image)

    random_objects = [dict(), np.zeros((3, 2)), np.ones((4, 3, 2, 1))]
    for random_object in random_objects:
        assert not is_image_array(random_object)


def test_image_converter_conversions():
    # create blue images in each format
    opencv_shaped = np.zeros((10, 10, 3), dtype=np.uint8)
    opencv_shaped[..., 0] = 255
    numpy_shaped = np.zeros((10, 10, 3))
    numpy_shaped[..., 2] = 1.0
    torch_shaped = np.zeros((3, 10, 10))
    torch_shaped[2, ...] = 1.0

    assert np.isclose(ImageConverter.from_opencv_format(opencv_shaped).image_in_numpy_format, numpy_shaped).all()
    assert np.isclose(ImageConverter.from_numpy_format(numpy_shaped).image_in_torch_format, torch_shaped).all()
    assert np.isclose(ImageConverter.from_torch_format(torch_shaped).image_in_opencv_format, opencv_shaped).all()
    assert np.isclose(
        ImageConverter.from_numpy_int_format(numpy_shaped.astype(np.uint8)).image_in_numpy_int_format,
        numpy_shaped.astype(np.uint8),
    ).all()

    # check conversion keeps int values the same
    opencv_shaped = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    numpy_shaped = ImageConverter.from_opencv_format(opencv_shaped).image_in_numpy_format
    assert np.isclose(opencv_shaped, ImageConverter.from_numpy_format(numpy_shaped).image_in_opencv_format).all()
