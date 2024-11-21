## For Schunk Gripper with RS-485 physical connection

1. `git clone https://github.com/SCHUNK-SE-Co-KG/bkstools.git`

2. Follow the bkstools' installing step (get .whl source as bkstools)

3. Find the name of the USB interface where your Schunk is connected. Run `sudo dmesg | grep tty` and you should see a line like `ch341-uart converter now attached to ttyUSB0`. In this case, the USB interface is ttyUSB0.

4. Find the ID of the Schunk gripper: run `python ./bkstools/scripts/bks_scan.py -H /dev/ttyUSB*` with `*` the appropriate index.

5. Change slave_id (line 30) in ./bkstools/bks_lib/bks_base.py to the ID you found in step 4.

6. Testing demo `python ./bkstools/demo/demo_bks_grip_outside_inside.py`

## Debugging

If the Schunk doesn't show up as /dev/ttyUSB*, but `lsusb` lists `QinHeng Electronics CH340 serial converter`, and
`sudo dmesg | grep brltty` shows `usbfs: interface 0 claimed by ch341 while 'brltty' sets config #1`, then:
`sudo apt remove brltty`, replug the usb cable, the Schunk should now show up as /dev/ttyUSB*.

Could be necessary (unconfirmed): `sudo apt-get install build-essential linux-source`
