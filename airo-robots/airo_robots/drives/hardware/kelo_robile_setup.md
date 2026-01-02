## General
For information on how to set up the KELO Robile, please refer to the documentation of [`airo-tulip`](https://pypi.org/project/airo-tulip/).

## Odometry
The odometry supplied by `airo-tulip` is based on the drive encoders of the KELO Robile. This odometry is not very accurate.
To improve the odometry, you may want to add additional sensors to the KELO Robile, such as optical flow sensors and a compass.
`airo-tulip` used to implement this functionality (more details below) but this was removed: having the code related to additional sensors
in this library instead of user code was too limiting. We do not want to update external libraries when we want to add new sensors.
It is now expected of users of `airo-tulip`/`airo-robots` to perform sensor fusion themselves, using the odometry provided by `airo-tulip` as a starting point.

### Custom odometry with `KELORobile` in `airo-robots`
If you want to implement your own odometry, you should create a subclass of `airo_robots.drives.hardware.kelo_robile.KELORobile`
and override `get_odometry()`. The closed loop movement method will call this method (and use your odometry implementation) internally.
You can find an example of this [here](https://github.com/m-decoster/tulip_user_peripherals).

### Adding external sensors
Below is information on how `airo-tulip` used to manage external sensors, so that you can replicate this in your codebase.

#### Reading out sensors
We connect a Teensy 4.1 microcontroller to the robot via USB and communicate over a serial interface. The Teensy runs [this code](https://github.com/airo-ugent/airo-tulip/blob/4c8f9b1e0b71621627086df5445861b0f819a372/peripherals/peripherals_server_teensy/peripherals_server_teensy.ino).
Hence, flow data can be obtained by sending `"FLOW"` to the Teensy over the serial interface, and it will respond with a string
```dx1,dy1,dx2,dy2``` where `dx1` and `dy1` are the flow data of the first sensor, and `dx2` and `dy2` are the flow data of the second sensor.
The compass data can be obtained by sending `"BNO"` to the Teensy, which will respond with a string
```x,y,z```, where `x`, `y`, and `z` are the compass data in the x, y, and z axes respectively. The `x` value is of interest for the heading of the robot.

#### Using sensor data in Python
To read out the data in Python with `pyserial`, you can use [this code](https://github.com/airo-ugent/airo-tulip/blob/4c8f9b1e0b71621627086df5445861b0f819a372/airo-tulip/airo_tulip/hardware/peripheral_client.py).

To extract odometry from the flow data, you should look at the [`PlatformPoseEstimatorPeripherals`](https://github.com/airo-ugent/airo-tulip/blob/4c8f9b1e0b71621627086df5445861b0f819a372/airo-tulip/airo_tulip/hardware/platform_monitor.py#L138) class copied below.

```python
class PlatformPoseEstimatorPeripherals:
    def __init__(self):
        self._time_last_update = None
        self._pose = np.array([0.0, 0.0, 0.0])

    def _calculate_velocities(self, delta_t: float, raw_flow: List[float]):
        [flow_x_1, flow_y_1, flow_x_2, flow_y_2] = raw_flow

        T_X = 0.348  # mounting position of the flow sensor on robot
        T_Y = 0.232  # mounting position of the flow sensor on robot
        R = np.sqrt(T_X**2 + T_Y**2)
        beta = np.arctan2(T_Y, T_X)

        v_x_1 = (flow_x_1 - flow_y_1) * np.sqrt(2) / 2 / delta_t
        v_y_1 = (-flow_x_1 - flow_y_1) * np.sqrt(2) / 2 / delta_t
        v_a_1 = (-flow_x_1 * np.cos(beta) - flow_y_1 * np.sin(beta)) / R / delta_t
        v_x_2 = (-flow_x_2 + flow_y_2) * np.sqrt(2) / 2 / delta_t
        v_y_2 = (flow_x_2 + flow_y_2) * np.sqrt(2) / 2 / delta_t
        v_a_2 = (-flow_x_2 * np.cos(beta) - flow_y_2 * np.sin(beta)) / R / delta_t

        v_x = (v_x_1 + v_x_2) / 2
        v_y = (v_y_1 + v_y_2) / 2
        v_a = (v_a_1 + v_a_2) / 2

        return v_x, v_y, v_a

    def _update_pose(self, delta_t: float, v_x, v_y, p_a):
        self._pose[0] += (v_x * np.cos(p_a) - v_y * np.sin(p_a)) * delta_t
        self._pose[1] += (v_x * np.sin(p_a) + v_y * np.cos(p_a)) * delta_t
        self._pose[2] = p_a

    def get_pose(self, raw_flow: List[float], raw_orientation_x: float) -> np.ndarray:
        if self._time_last_update is None:
            self._time_last_update = time.time()
            return np.array([0.0, 0.0, 0.0])

        delta_time = time.time() - self._time_last_update
        self._time_last_update = time.time()

        v_x, v_y, v_a = self._calculate_velocities(delta_time, raw_flow)
        self._update_pose(delta_time, v_x, v_y, -raw_orientation_x)

        return self._pose
```

You should call this class's method `get_pose` with
```python
flow = np.array(peripheral_client.get_flow(), dtype=np.float64)
flow /= 12750.0  # conversion from dimensionless to meters - this is specific to how high you mounted the flow sensors
orientation = np.array(peripheral_client.get_orientation(), dtype=np.float64)
orientation *= np.pi / 180.0  # conversion from degrees to radians
peripheral_pose_estimator.get_pose(flow, orientation[0])
```

We found that using flow and compass information (and no longer the drive encoders) provided us with accurate odometry values (< 1% error).