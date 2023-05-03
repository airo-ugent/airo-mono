from airo_camera_toolkit.cameras.dds_publisher import DDSCameraIntrinsics, DDSImage
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_typing import CameraIntrinsicsMatrixType, NumpyDepthMapType, NumpyFloatImageType, NumpyIntImageType
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic


class RGBDCameraDDSReceiver(RGBDCamera):
    def __init__(self, name: str):
        self.dds_participant = DomainParticipant()
        self.name = name
        self.intrinsics_topic = Topic(
            self.dds_participant, f"{name}/intrinsics", DDSCameraIntrinsics, qos=Qos(Policy.Reliability.Reliable(0))
        )
        self.intrinsics_subscriber = DataReader(self.dds_participant, self.intrinsics_topic)
        self.rgb_topic = Topic(self.dds_participant, f"{name}/rgb", DDSImage, qos=Qos(Policy.Reliability.Reliable(0)))
        self.rgb_subscriber = DataReader(self.dds_participant, self.rgb_topic)
        self.depth_map_topic = Topic(
            self.dds_participant, f"{name}/depth_map", DDSImage, qos=Qos(Policy.Reliability.Reliable(0))
        )
        self.depth_map_subscriber = DataReader(self.dds_participant, self.depth_map_topic)

    def get_rgb_image(self) -> NumpyFloatImageType:
        latest_message: DDSImage = self.rgb_subscriber.read_next()
        if latest_message is None:
            return None
        return latest_message.as_np_array

    def intrinsics_matrix(self) -> CameraIntrinsicsMatrixType:
        raise NotImplementedError

    def get_depth_image(self) -> NumpyIntImageType:
        raise NotImplementedError

    def get_depth_map(self) -> NumpyDepthMapType:
        raise NotImplementedError


if __name__ == "__main__":
    import cv2

    receiver = RGBDCameraDDSReceiver("zed1")
    while True:
        img = receiver.get_rgb_image()
        if img is not None:
            cv2.imshow("img", img)
            cv2.waitKey(1)
