import time
import sys
import os
sys.path.append("/home/ag/ef_ws/g1/scripts/dev")
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from safety.hanger_boot_sequence import hanger_boot_sequence
iface="enp1s0"
from ef_client import Robot
#client = hanger_boot_sequence(iface=iface)

# Walk forward ~1m (0.3 m/s x 3.3s)
#for _ in range(10):
#    client.Move(0.4, 0, 0)
#client.StopMove()
#time.sleep(0.5)

# Turn ~90 deg (0.5 rad/s x 3.14s = pi/2 rad)
#for _ in range(10):
#    client.Move(0, 0, 0.6)
#client.StopMove()
#time.sleep(0.5)

# Walk forward ~1m
#for _ in range(10):
#    client.Move(0.4, 0, 0)
#client.StopMove()

g1 = Robot("enp1s0")
g1.stop_moving()
g1.loco_move(0.3, 0, 0)
time.sleep(0.33)
g1.stop_moving()
time.sleep(0.5)
g1.loco_move(0,0,0.5)
time.sleep(3.14)
g1.stop_moving()
time.sleep(0.5)
g1.loco_move(0.3, 0, 0)
time.sleep(0.33)
g1.stop_moving()

