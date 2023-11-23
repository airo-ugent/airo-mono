import json
import pathlib

from airo_dataset_tools.data_parsers.camera_intrinsics import (
    CameraIntrinsics,
    FocalLengths,
    PrincipalPoint,
    Resolution,
)


def test_camera_intrinsics_save_and_load(tmp_path: pathlib.Path):
    """
    Based on the follwing intrinsics of one of our ZED2i cameras:
    [LEFT_CAM_2K]
    fx=1067.91
    fy=1068.05
    cx=1107.48
    cy=629.675
    k1=-0.0542749
    k2=0.0268096
    p1=0.000204483
    p2=-0.000310015
    k3=-0.0104089
    """
    camera_intrinsics = CameraIntrinsics(
        image_resolution=Resolution(width=2208, height=1520),
        focal_lengths_in_pixels=FocalLengths(fx=1067.91, fy=1068.05),
        principal_point_in_pixels=PrincipalPoint(cx=1107.48, cy=629.675),
        radial_distortion_coefficients=[-0.0542749, 0.0268096, -0.0104089],
        tangential_distortion_coefficients=[0.000204483, -0.000310015],
    )

    with open(tmp_path / "camera_intrinsics.json", "w") as file:
        json.dump(camera_intrinsics.model_dump(exclude_none=True), file, indent=4)

    with open(tmp_path / "camera_intrinsics.json", "r") as file:
        camera_intrinsics2 = CameraIntrinsics.model_validate_json(file.read())

    assert camera_intrinsics2.image_resolution.width == 2208
    assert camera_intrinsics2.image_resolution.height == 1520
    assert camera_intrinsics2.focal_lengths_in_pixels.fx == 1067.91
    assert camera_intrinsics2.focal_lengths_in_pixels.fy == 1068.05
    assert camera_intrinsics2.principal_point_in_pixels.cx == 1107.48
    assert camera_intrinsics2.principal_point_in_pixels.cy == 629.675
    assert camera_intrinsics2.radial_distortion_coefficients[0] == -0.0542749
    assert camera_intrinsics2.radial_distortion_coefficients[1] == 0.0268096
    assert camera_intrinsics2.radial_distortion_coefficients[2] == -0.0104089
    assert camera_intrinsics2.tangential_distortion_coefficients[0] == 0.000204483
    assert camera_intrinsics2.tangential_distortion_coefficients[1] == -0.000310015


if __name__ == "__main__":
    test_camera_intrinsics_save_and_load(pathlib.Path("."))
