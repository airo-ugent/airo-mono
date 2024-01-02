import cv2
from airo_camera_toolkit.calibration.fiducial_markers import detect_charuco_board, draw_frame_on_image
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.image_converter import ImageConverter

camera = Zed2i(fps=30)
intrinsics = camera.intrinsics_matrix()

window_name = "Charuco Pose"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    image = camera.get_rgb_image_as_int()
    image = ImageConverter.from_numpy_int_format(image).image_in_opencv_format
    pose = detect_charuco_board(image, intrinsics)
    if pose is not None:
        draw_frame_on_image(image, pose, intrinsics)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
