import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from spatialmath import SE3, SO3

from airo_core.spatial_algebra.se3 import SE3Container


@pytest.fixture(autouse=True)
def seed():
    """fixture that will run before each test in this module to make them deterministic.
    Random poses etc are generated with spatialmath,
    which usesnp.random"""
    np.random.seed(2022)


def test_from_translation_and_homogeneous():
    pose = np.eye(4)
    translation = np.array([1, 2, 3.0])
    pose[:3, 3] = translation
    se3 = SE3Container.from_translation(translation)
    assert np.isclose(se3.homogeneous_matrix, pose).all()


def test_from_hom_matrix():
    pose = SE3.Rand()
    se3 = SE3Container.from_homogeneous_matrix(pose)
    assert np.isclose(se3.homogeneous_matrix, pose).all()


def test_from_rot_mat():
    # no translation
    rot_matrix = SO3.Rand()
    pose = np.eye(4)
    pose[:3, :3] = rot_matrix
    se3 = SE3Container.from_rotation_matrix_and_translation(rot_matrix)
    assert np.isclose(se3.homogeneous_matrix, pose).all()

    # with translation
    trans = np.array([1, 2, 3.2])
    pose[:3, 3] = trans
    se3 = SE3Container.from_rotation_matrix_and_translation(rot_matrix, trans)
    assert np.isclose(se3.homogeneous_matrix, pose).all()


def test_from_base_vectors():
    rot_matrix = np.eye(3)
    se3 = SE3Container.from_orthogonal_base_vectors_and_translation(
        rot_matrix[:, 0], rot_matrix[:, 1], rot_matrix[:, 2], np.array([1, 2, 3])
    )
    assert np.isclose(se3.rotation_matrix, rot_matrix).all()


def test_quaternions():
    quat = [0, 0, 0, 1.0]  # no rotation scalar-last
    se3 = SE3Container.from_quaternion_and_translation(quat)
    assert np.isclose(se3.rotation_matrix, np.eye(3)).all()
    assert np.isclose(se3.translation, np.zeros(3)).all()
    assert np.isclose(quat, se3.get_orientation_as_quaternion()).all()


def test_euler():
    euler = [np.pi, np.pi / 4, np.pi / 5]  # 90 degs of X
    rot_matrix = Rotation.from_euler("xyz", euler).as_matrix()
    se3 = SE3Container.from_euler_angles_and_translation(euler)
    assert np.isclose(se3.rotation_matrix, rot_matrix).all()
    assert np.isclose(se3.get_orientation_as_euler_angles(), euler).all()


def test_get_axes_functions():
    se3 = SE3Container.random()
    hom_matrix = se3.homogeneous_matrix
    for i, func in enumerate([se3.get_x_axis, se3.get_y_axis, se3.get_z_axis]):
        assert np.isclose(func(), hom_matrix[:3, i]).all()


def test_repr():
    se3 = SE3Container.random()
    print(se3)


def test_all_orientation_reprs_are_equivalent():
    se3 = SE3Container.random()
    rot_matrix = se3.rotation_matrix
    assert np.isclose(Rotation.from_rotvec(se3.get_orientation_as_rotation_vector()).as_matrix(), rot_matrix).all()
    assert np.isclose(Rotation.from_euler("xyz", se3.get_orientation_as_euler_angles()).as_matrix(), rot_matrix).all()
    assert np.isclose(Rotation.from_quat(se3.get_orientation_as_quaternion()).as_matrix(), rot_matrix).all()
