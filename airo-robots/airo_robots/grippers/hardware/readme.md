## For Schunk Gripper with RS-485 physical connection

1. git clone https://github.com/SCHUNK-SE-Co-KG/bkstools.git

2. follow the bkstools' installing step (get .whl source as bkstools)

3. find the ID for schunk: python ./bkstools/scripts/bks_scan.py

4. change the defualt ID in ./bkstools/bks_lib/bks_base.py (change slave_id to the scan_id)

testing demo: python ./bkstools/demo/demo_bks_grip_outside_inside.py

## Debugging

sudo apt-get install build-essential linux-source
if 
`sudo dmesg | grep brltty` shows `usbfs: interface 0 claimed by ch341 while 'brltty' sets config #1`
run `sudo apt remove brltty`

