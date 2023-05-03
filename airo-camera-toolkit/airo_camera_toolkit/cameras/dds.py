from dataclasses import dataclass
from time import sleep

import numpy as np
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.annotations import key
from cyclonedds.idl.types import sequence
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic


# C, C++ require using IDL, Python doesn't
@dataclass
class Chatter(IdlStruct, typename="Chatter"):
    name: str
    key("name")
    message: sequence[float]


class Writer:
    def __init__(self) -> None:
        self.dds_participant = DomainParticipant()
        self.dds_topic = Topic(self.dds_participant, "Hello", Chatter, qos=Qos(Policy.Reliability.Reliable(0)))


rng = np.random.default_rng()
dp = DomainParticipant()
tp = Topic(dp, "Hello", Chatter, qos=Qos(Policy.Reliability.Reliable(0)))
dw = DataWriter(dp, tp)
dr = DataReader(dp, tp)
count = 0
while True:
    sample = Chatter(name="test", message=[1, 2.0, 3.0])
    count = count + 1
    print("Writing ", sample)
    dw.write(sample)
    # for sample in dr.take(10):
    #     print("Read ", sample)
    sleep(rng.exponential())
