import numpy as np
from airo_robots.manipulators.force_torque_sensor import ForceTorqueSensor
from airo_typing import WrenchType
from rtde_receive import RTDEReceiveInterface


class UReForceTorqueSensor(ForceTorqueSensor):
    """
    Class for reading the built-in FT sensor of UR e-series robots through the RTDE interface.

    Gravity compensation is (in theory) handled by the UR control box.
    To make it work you need to specify the center of gravity and weight of your payload
    (gripper + other grasped items).

    cf https://assets.robotiq.com/website-assets/support_documents/document/2F-85_2F-140_Instruction_Manual_e-Series_PDF_20190206.pdf
    for the CoG, M etc. of the Robotiq grippers.
    """

    def __init__(self, robot_ip_address: str) -> None:
        self.ip_address = robot_ip_address
        self.rtde_receive = RTDEReceiveInterface(self.ip_address)
        self.wrench_lowpass_filter = ExponentialLowPassFilter(0.95)

    def get_wrench(self) -> WrenchType:
        gravity_compensated_tcp_wrench_in_base_frame = np.array(self.rtde_receive.getActualTCPForce())
        filtered_wrench = self.wrench_lowpass_filter.update_filter(gravity_compensated_tcp_wrench_in_base_frame)
        return filtered_wrench


class ExponentialLowPassFilter:
    def __init__(self, filter_coefficient: float = 0.95) -> None:
        self.filter_coefficient = filter_coefficient
        self.filtered_value = None

    def update_filter(self, latest_value: np.ndarray) -> np.ndarray:
        if not isinstance(self.filtered_value, np.ndarray):
            self.filtered_value = np.zeros_like(latest_value, dtype=np.float32)
            return self.filtered_value

        self.filtered_value = (
            self.filter_coefficient * self.filtered_value + (1 - self.filter_coefficient) * latest_value
        )
        return self.filtered_value


if __name__ == "__main__":
    """script for testing the FT sensor readout
    e.g. python airo-robots/airo_robots/manipulators/hardware/ur_force_torque_rtde.py --ip-address 10.42.0.162
    """
    import time

    import click
    from airo_robots.manipulators.hardware.ur_rtde import UR_RTDE

    @click.command()
    @click.option("--ip-address", type=str, help="ip address of the e-series robot to read the FT sensor from.")
    def test_ur3e_ft_sensor(ip_address):
        robot = UR_RTDE(ip_address, UR_RTDE.UR3E_CONFIG)
        print(f"robot joint configuration = {robot.get_joint_configuration()}")
        ft_sensor = UReForceTorqueSensor(ip_address)
        while True:
            print(ft_sensor.get_wrench())
            time.sleep(1)

    test_ur3e_ft_sensor()
