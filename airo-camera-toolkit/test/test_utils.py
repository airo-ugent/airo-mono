import numpy as np 
from airo_camera_toolkit.utils import ImageConverter

def test_image_congerter_format_checks():
    random_image = np.random.rand(10,10,3)
    assert ImageConverter._is_valid_image(random_image)

    int_image = np.random.randint(0, 255, (10,10,3))
    assert not ImageConverter._is_valid_image(int_image)

def test_image_converter_conversions():
    # create blue images in each format
    opencv_shaped = np.zeros((10,10,3))
    opencv_shaped[...,0] = 1.0
    numpy_shaped = np.zeros((10,10,3))
    numpy_shaped[...,2] = 1.0
    torch_shaped = np.zeros((3,10,10))
    torch_shaped[2,...] = 1.0

    assert np.isclose(ImageConverter.from_opencv_format(opencv_shaped).image_in_numpy_format, numpy_shaped).all()
    assert np.isclose(ImageConverter.from_numpy_format(numpy_shaped).image_in_torch_format, torch_shaped).all()
    assert np.isclose(ImageConverter.from_torch_format(torch_shaped).image_in_opencv_format, opencv_shaped).all()
