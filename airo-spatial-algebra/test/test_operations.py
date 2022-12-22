import numpy as np
import pytest
from airo_spatial_algebra.operations import _HomogeneousPoints, transform_points
from airo_spatial_algebra.se3 import SE3Container


def test_helper_class_creation():
    point = np.array([1.0, 2, 3])
    hpoints = _HomogeneousPoints(point)
    assert hpoints._homogeneous_points.shape == (1, 4)
    assert hpoints._homogeneous_points[0, -1] == 1.0

    homogeneous_point = np.array([1, 2, 3, 1])
    with pytest.raises(ValueError):
        _HomogeneousPoints(homogeneous_point)

    points = np.arange(6).reshape(2, 3)
    hpoints = _HomogeneousPoints(points)
    assert hpoints._homogeneous_points.shape == (2, 4)
    # check that the scale is 1.0
    assert hpoints._homogeneous_points[0, -1] == 1.0

    wronglyshaped_points = np.arange(6).reshape(3, 2)
    with pytest.raises(ValueError):
        _HomogeneousPoints(wronglyshaped_points)


@pytest.mark.parametrize("points", [np.arange(6).astype(np.float32).reshape(2, 3), np.array([1.0, 2, 3])])
def test_helper_class_properties(points):
    hpoints = _HomogeneousPoints(points)
    assert np.isclose(points, hpoints.points).all()
    assert hpoints.homogeneous_points.shape == (points.size // 3, 4)


def test_transform_points():
    points = np.arange(6).astype(np.float32).reshape(2, 3)
    transform = SE3Container.random()
    transformed_points = transform_points(transform.homogeneous_matrix, points)
    assert np.isclose(transformed_points[0], transform.rotation_matrix @ points[0] + transform.translation).all()
