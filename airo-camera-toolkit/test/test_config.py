from pathlib import Path

import numpy as np


class _ImageTestValues:
    """This class contains camera parameters and positions for an image that was rendered
    with pybulllet to test the camera toolkit functionality. The values are obtained through pybullet and considered to be ground truth
    for testing in this package.

    The scene contains a 0.2x0.2x0.2 cube centered at (0,0,0.1). The camera is placed at (0, 0.31,0.5) and looks at the origin.
    """

    _intrinsics_matrix = np.array([[282.1089698, 0.0, 200.0], [0.0, 282.1089698, 150.0], [0.0, 0.0, 1.0]])

    _extrinsics_matrix = np.array(
        [
            [-1.0, -0.0, -0.0, -0.0],
            [0.0, 0.84990275, -0.52693971, 0.31000001],
            [-0.0, -0.52693971, -0.84990275, 0.50000001],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    _positions_in_world_frame = np.array([[0.1, 0.1, 0.2], [-0.1, 0.1, 0.2]])
    _positions_in_camera_frame = np.array([[-0.1, -0.02039766, 0.36562812], [0.1, -0.02039766, 0.36562812]])
    _positions_on_image_plane = np.array([[122.84, 134.26], [277.16, 134.26]])
    _depth_z_values = np.array([0.368, 0.368])
    _rgb_image_path = Path(__file__).parent / "data" / "test_rgb.png"
    _depth_image_path = Path(__file__).parent / "data" / "test_depth_map.png"
    _depth_map_path = Path(__file__).parent / "data" / "test_depth_map.npy"
    _image_dims = (400, 300)


class _CalibrationTest:
    _empty_image_path = Path(__file__).parent / "data" / "empty_marker.png"
    _default_charuco_board_path = Path(__file__).parent / "data" / "default_charuco_board.png"
