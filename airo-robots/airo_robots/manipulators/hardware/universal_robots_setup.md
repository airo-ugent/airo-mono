# Universal Robots Setup

New UR robots require some additional setup to get remote control working.
First you need to ensure you the **ethernet connection** works.
Then you will need to enable **remote control mode**.

## Ethernet Connection
Establish a Ethernet connection between the control box and the external computer:
* Connect an UTP cable from the control box to the external computer.
* Create a new network profile on the external computer in `Settings > Network > Wired > +`. Give it any name e.g. `UR` and in the IPv4 tab select `Shared to other computers`.
* On the control box, go to `Settings > System > Network` and select Static Address and set:
    * IP address: `10.42.0.162`
    * Subnet mask: `255.255.255.0`
    * Gateway: `10.42.0.1`
    * Preferred DNS server: `10.42.0.1`

On MacOS, this corresponds to creating a network with IPv4 configured manually at 10.42.0.1 (subnet mask 255.255.0.0) and configure IPv4 automatically.

If you're lucky, the control box will already say "Network is connected".
If pinging the control box from the external computer works, you're done and can read the next section to Enable remote control mode:
```bash
ping 10.42.0.162
```
If not, you can try to manually bringing up the network profile you created:
```bash
nmcli connection up UR
```
If pinging still doesn't, try restarting the robot control box.
If still not successful, try swapping ethernet cables, ports or computers.


## Remote Control Mode
To be able to conmmand the robot from an external computer, the control box must be in Remote Control mode.
To get access to it, set first the following passwords:

1. In `Settings > Password` set the **Safety** passwords. The default password by UR is `easybot`.
(No need to set the Mode password, as this will add an Automatic mode which we don't need).
2. In `Settings > Remote Control` enable remote control.

Now you should see an new icon in the top right corner of the control box screen.
Click this icon to switch between **Manual** and **Remote Control** mode.

## Testing
To test that everything works, try running the [ur_rtde.py](ur_rdte.py) script.
Be sure to set the robot in **Remote Control** mode before running the script.
If the robot has a gripper, also set the TCP to something reasonable to avoid collisions.
```bash
python ur_rtde.py --ip_address 10.42.0.162
```
