# Schunk Gripper with RS-485 physical connection

## Important notes

The Schunk gripper differs from the Robotiq in a couple ways.
1. It does not have an automatic width calibration to account for custom fingertips. A custom calibrate_width() routine is provided in this class, but given the Schunk's high minimum gripping force, it is recommended to manually set the max_stroke_setting in the constructor appropriately (refer to docstring in constructor for further explanation).
2. bks_tools.bks_lib.bks_modbus OVERWRITES Python's serial.Serial.read() function which is absolutely ludicrous. One issue is that serial.Serial.read(size=num_bytes), i.e. a call with size passed as a keyword argument, is not handled properly. serial.Serial.read(num_bytes), i.e. no keyword argument, seems to work.
3. The connection to the Schunk gripper requires constant synchronisation, meaning that the connection is lost if your main code executes any other code for more than a couple seconds. For example, time.sleep(2) would cause the connection to be dropped. To account for this, bkstools.bks_lib.bks_base provides functions such as keep_communication_alive_sleep, as a wrapper around time.sleep.
This class handles it differently, making use of the BKSModule.MakeReady() cmd. This command will "refresh" the
connection to the Schunk. It is hence executed before every move() command. However, it takes about 30ms to complete,
so in addition servo() commands are provided: first call servo_start(), which will execute MakeReady(), then use
servo() in a loop. MakeReady() doesn't have to be executed in the loop, since the movement commands themselves
keep the communication alive.

## Hardware installation



## Software installation

1. `git clone https://github.com/SCHUNK-SE-Co-KG/bkstools.git`

2. In your project environment, run `pip install -e <path-to-bkstools-clone>`. In later steps we need to adjust one of the project files, so the regular `pip install bkstools` from PyPI is not applicable.

3. Find the name of the USB interface where your Schunk is connected. Run `sudo dmesg | grep tty` and you should see a line like `ch341-uart converter now attached to ttyUSB0`. In this case, the USB interface is ttyUSB0.

4. Find the ID of the Schunk gripper: run `python ./bkstools/scripts/bks_scan.py -H /dev/ttyUSB*` with `*` the appropriate index.

5. Change slave_id (line 30) in ./bkstools/bks_lib/bks_base.py to the ID you found in step 4.

6. Testing demo `python ./bkstools/demo/demo_bks_grip_outside_inside.py`

## Debugging

If the Schunk doesn't show up as /dev/ttyUSB*, but `lsusb` lists `QinHeng Electronics CH340 serial converter`, and
`sudo dmesg | grep brltty` shows `usbfs: interface 0 claimed by ch341 while 'brltty' sets config #1`, then:
`sudo apt remove brltty`, replug the usb cable, the Schunk should now show up as /dev/ttyUSB*.

Could be necessary (unconfirmed): `sudo apt-get install build-essential linux-source`
