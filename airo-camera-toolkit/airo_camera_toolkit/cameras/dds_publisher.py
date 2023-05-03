from dataclasses import dataclass
from multiprocessing import Process

import cyclonedds.idl.types as idl
import numpy as np
from airo_camera_toolkit.interfaces import RGBDCamera
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.pub import DataWriter
from cyclonedds.topic import Topic


@dataclass
class DDSImage(IdlStruct, typename="DDSImage"):
    image: idl.sequence[float]
    width: int
    height: int
    channels: int

    @property
    def as_np_array(self):
        return np.array(self.image).reshape((self.height, self.width, self.channels))

    @classmethod
    def from_numpy_array(cls, array: np.ndarray):
        return cls(
            name="name",
            image=array.flatten().tolist(),
            width=array.shape[1],
            height=array.shape[0],
            channels=array.shape[2],
        )


@dataclass
class DDSCameraIntrinsics(IdlStruct, typename="DDSCameraIntrinsics"):
    matrix: idl.array[float, 9]

    @property
    def as_np_array(self):
        return np.array(self.matrix).reshape((3, 3))

    @classmethod
    def from_numpy_array(cls, array: np.ndarray):
        return cls(matrix=array.flatten().tolist())


class RGBDCameraDDSPublisher(Process):
    def __init__(self, name: str, camera: RGBDCamera) -> None:
        super().__init__()

        self.dds_participant = DomainParticipant()
        self.name = name
        # TODO: can we check if namespace already in use? this is not desired, only one publisher.
        self.intrinsics_topic = Topic(
            self.dds_participant, f"{name}/intrinsics", DDSCameraIntrinsics, qos=Qos(Policy.Reliability.Reliable(0))
        )
        self.intrinsics_writer = DataWriter(self.dds_participant, self.intrinsics_topic)
        self.rgb_topic = Topic(self.dds_participant, f"{name}/rgb", DDSImage, qos=Qos(Policy.Reliability.Reliable(0)))
        self.rgb_writer = DataWriter(self.dds_participant, self.rgb_topic)
        self.depth_map_topic = Topic(
            self.dds_participant, f"{name}/depth_map", DDSImage, qos=Qos(Policy.Reliability.Reliable(0))
        )
        self.depth_map_writer = DataWriter(self.dds_participant, self.depth_map_topic)
        self.camera = camera

    def run(self):
        while True:
            rgb = self.camera.get_rgb_image()
            # depth_map = self.camera.get_depth_map()
            # intrinsics = self.camera.intrinsics_matrix()

            self.rgb_writer.write(DDSImage(rgb.flatten().tolist(), rgb.shape[1], rgb.shape[0], rgb.shape[2]))
            # self.depth_map_writer.write(DDSImage.from_numpy_array(depth_map[...,np.newaxis]))
            # self.intrinsics_writer.write(DDSCameraIntrinsics.from_numpy_array(intrinsics))


if __name__ == "__main__":
    pass

    from airo_camera_toolkit.cameras.zed2i import Zed2i

    camera = Zed2i(depth_mode=Zed2i.NONE_DEPTH_MODE, resolution=Zed2i.RESOLUTION_720)

    publisher = RGBDCameraDDSPublisher("zed1", camera)
    publisher.run()
