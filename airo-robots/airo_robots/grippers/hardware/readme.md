## For Schunk Gripper with RS-485 physical connection

1. git clone https://github.com/SCHUNK-SE-Co-KG/bkstools.git

2. follow the bkstools' installing step (get .whl source as bkstools)

3. find the ID for schunk: python ./bkstools/scripts/bks_scan.py

4. change the defualt ID in ./bkstools/bks_lib/bks_base.py (slave_id to the scaned id)
