from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient

iface = "enp1s0"
domain_id = 0

ChannelFactoryInitialize(domain_id, iface)

msc = MotionSwitcherClient()
msc.Init()

code, data = msc.CheckMode()
print("CheckMode before:", code, data)

code, _ = msc.ReleaseMode()
print("ReleaseMode:", code)

code, data = msc.CheckMode()
print("CheckMode after:", code, data)

sport = SportClient()
sport.SetTimeout(10.0)
sport.Init()

print("StandUp:", sport.StandUp())
print("BalanceStand:", sport.BalanceStand())
