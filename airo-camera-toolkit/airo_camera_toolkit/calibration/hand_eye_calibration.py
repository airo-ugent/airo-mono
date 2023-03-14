import time
from typing import List, Optional

import cv2
import numpy as np
from airo_camera_toolkit.calibration.fiducial_markers import (
    detect_aruco_markers,
    detect_charuco_corners,
    draw_frame_on_image,
    get_pose_of_charuco_board,
    visualize_aruco_detections,
    visualize_charuco_detection,
)
from airo_camera_toolkit.interfaces import RGBCamera
from airo_camera_toolkit.utils import ImageConverter
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType
from cv2 import aruco


def eye_in_hand_pose_estimation(
    tcp_poses_in_base: List[HomogeneousMatrixType], marker_poses_in_camera: List[HomogeneousMatrixType]
) -> Optional[HomogeneousMatrixType]:
    tcp_orientations_as_rotvec_in_base = [
        SE3Container.from_homogeneous_matrix(tcp_pose).orientation_as_rotation_vector for tcp_pose in tcp_poses_in_base
    ]
    tcp_positions_in_base = [
        SE3Container.from_homogeneous_matrix(tcp_pose).translation for tcp_pose in tcp_poses_in_base
    ]

    marker_orientations_as_rotvec_in_camera = [
        SE3Container.from_homogeneous_matrix(marker_pose).orientation_as_rotation_vector
        for marker_pose in marker_poses_in_camera
    ]
    marker_positions_in_camera = [
        SE3Container.from_homogeneous_matrix(marker_pose).translation for marker_pose in marker_poses_in_camera
    ]

    camera_rotation_matrix, camera_translation = cv2.calibrateHandEye(
        tcp_orientations_as_rotvec_in_base,
        tcp_positions_in_base,
        marker_orientations_as_rotvec_in_camera,
        marker_positions_in_camera,
        None,
        None,
        # method=cv2.CALIB_HAND_EYE_HORAUD
    )

    if camera_rotation_matrix is None or camera_translation is None:
        return None

    camera_pose_in_tcp_frame = SE3Container.from_rotation_matrix_and_translation(
        camera_rotation_matrix, camera_translation
    ).homogeneous_matrix
    return camera_pose_in_tcp_frame


def eye_to_hand_pose_estimation(
    tcp_poses_in_base: List[HomogeneousMatrixType], marker_poses_in_camera: List[HomogeneousMatrixType]
) -> Optional[HomogeneousMatrixType]:

    # the AX=XB problem for the eye-to-hand is equivalent to the AX=XB problem for the eye-in-hand if you invert the poses
    # cf https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gaebfc1c9f7434196a374c382abf43439b
    # cf https://forum.opencv.org/t/eye-to-hand-calibration/5690/2

    base_pose_in_tcp_frame = [np.linalg.inv(tcp_pose) for tcp_pose in tcp_poses_in_base]

    camera_pose_in_base = eye_in_hand_pose_estimation(base_pose_in_tcp_frame, marker_poses_in_camera)
    if camera_pose_in_base is None:
        return None
    return camera_pose_in_base


if __name__ == "__main__":  # noqa C901 - ignore complexity warning
    from airo_camera_toolkit.cameras.zed2i import Zed2i
    from airo_robots.manipulators.hardware.ur_rtde import UR_RTDE

    # TODO: do we want this package to depend on airo-robots?

    def do_camera_robot_calibration(mode, aruco_dict, charuco_board, camera: RGBCamera, robot: UR_RTDE):
        # TODO: this function is not tested yet.
        tcp_poses_in_base = []
        marker_poses_in_camera = []
        camera_pose = None

        print(
            "Press S to capture pose, press F to finish. Make sure the detections look good (corners/contours are accurate) before capturing."
        )
        robot.rtde_control.teachMode()
        while True:
            image = camera.get_rgb_image()
            image = ImageConverter.from_numpy_format(image).image_in_opencv_format

            aruco_result = detect_aruco_markers(image, aruco_dict)
            if not aruco_result:
                continue
            charuco_result = detect_charuco_corners(image, aruco_result, charuco_board)
            if not charuco_result:
                continue

            charuco_pose = get_pose_of_charuco_board(charuco_result, charuco_board, camera.intrinsics_matrix(), None)

            # visualize
            image = visualize_aruco_detections(image, aruco_result)
            image = visualize_charuco_detection(image, charuco_result)
            if charuco_pose is not None:
                image = draw_frame_on_image(image, charuco_pose, camera.intrinsics_matrix())

            image = cv2.resize(image, (1024, 768))
            cv2.imshow("image", image)
            key = cv2.waitKey(1)
            if key == ord("s"):
                robot.rtde_control.endTeachMode()
                time.sleep(0.5)
                marker_poses_in_camera.append(charuco_pose)
                tcp_poses_in_base.append(robot.get_tcp_pose())
                print(f"{len(tcp_poses_in_base)} poses captured")
                time.sleep(0.5)
                robot.rtde_control.teachMode()

            elif key == ord("f"):
                robot.rtde_control.endTeachMode()
                break

            if len(tcp_poses_in_base) >= 4:
                if mode == "eye_in_hand":
                    camera_pose = eye_in_hand_pose_estimation(tcp_poses_in_base, marker_poses_in_camera)
                elif mode == "eye_to_hand":
                    camera_pose = eye_to_hand_pose_estimation(tcp_poses_in_base, marker_poses_in_camera)

        return camera_pose

    robot = UR_RTDE("10.42.0.162", UR_RTDE.UR3_CONFIG)
    camera = Zed2i()
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    charuco_board = aruco.CharucoBoard((6, 4), 0.04, 0.031, aruco_dict)

    pose = do_camera_robot_calibration("eye_in_hand", aruco_dict, charuco_board, camera, robot)
    print(pose)
