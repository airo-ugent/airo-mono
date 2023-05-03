from dataclasses import dataclass
from multiprocessing import Process

import cyclonedds.idl.types as idl
import numpy as np
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_typing import CameraIntrinsicsMatrixType, NumpyDepthMapType, NumpyFloatImageType, NumpyIntImageType
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.annotations import key
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic


@dataclass
class DDSImage(IdlStruct, typename="DDSImage"):
    name: str
    key("name")
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
    def __init__(self, name: str, camera_cls: type, camera_kwargs: dict) -> None:
        super().__init__()
        # assert issubclass(type, RGBDCamera)
        self.camera_cls = camera_cls
        self.camera_kwargs = camera_kwargs

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
        self.camera = None

    def run(self):
        # self.camera = self.camera_cls(**self.camera_kwargs)
        # assert isinstance(self.camera, RGBDCamera)
        while True:
            print("publishing")
            # rgb = self.camera.get_rgb_image()
            # depth_map = self.camera.get_depth_map()
            # intrinsics = self.camera.intrinsics_matrix()
            self.rgb_writer.write(DDSImage("test", [1.0, 2.0], 1, 2, 1))
            # self.depth_map_writer.write(DDSImage.from_numpy_array(depth_map[...,np.newaxis]))
            # self.intrinsics_writer.write(DDSCameraIntrinsics.from_numpy_array(intrinsics))
            time.sleep(0.4)


class RGBDCameraDDSReceiver(RGBDCamera):
    def __init__(self, name: str):
        self.dds_participant = DomainParticipant()
        self.name = name
        self.intrinsics_topic = Topic(
            self.dds_participant, f"{name}/intrinsics", DDSCameraIntrinsics, qos=Qos(Policy.Reliability.Reliable(0))
        )
        self.intrinsics_subscriber = DataReader(self.dds_participant, self.intrinsics_topic)
        self.rgb_topic = Topic(self.dds_participant, "rgb", DDSImage, qos=Qos(Policy.Reliability.Reliable(0)))
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
    import time

    from airo_camera_toolkit.cameras.zed2i import Zed2i

    publisher = RGBDCameraDDSPublisher("zed1", Zed2i, {"depth_mode": Zed2i.PERFORMANCE_DEPTH_MODE})
    # publisher.start()

    camera = RGBDCameraDDSReceiver("zed1")
    publisher.rgb_writer.write(DDSImage("test", [0.0, 0.1], 1, 2, 1))
    publisher.run()
