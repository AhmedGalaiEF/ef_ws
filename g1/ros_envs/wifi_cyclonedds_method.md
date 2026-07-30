# ROS 2 over Wi-Fi with CycloneDDS

This note documents the method for running ROS 2 tools such as `ros2 topic echo`
from an external laptop over Wi-Fi while the robot boards remain on the
`192.168.123.0/24` Ethernet subnet.

## Network layout

Known Jetson addresses:

- Jetson Wi-Fi: `192.168.2.41`
- Jetson Ethernet `eth0`: `192.168.123.164`
- Robot ROS 2 subnet: `192.168.123.0/24`
- External laptop Wi-Fi: `192.168.2.X`

The Jetson must act as the router between the laptop Wi-Fi network and the
robot Ethernet network.

## 1. Add a laptop route to the robot subnet

Run this on the external laptop:

```bash
sudo ip route add 192.168.123.0/24 via 192.168.2.41
```

If the route may already exist, replace it instead:

```bash
sudo ip route replace 192.168.123.0/24 via 192.168.2.41
```

Check it with:

```bash
ip route get 192.168.123.164
ping 192.168.123.164
```

Then test a real robot board IP:

```bash
ping 192.168.123.xxx
```

## 2. Enable IPv4 forwarding on the Jetson

Run this on the Jetson:

```bash
sudo sysctl -w net.ipv4.ip_forward=1
```

To make it persistent across reboot, create a sysctl config file:

```bash
echo 'net.ipv4.ip_forward=1' | sudo tee /etc/sysctl.d/99-ros2-wifi-routing.conf
sudo sysctl --system
```

## 3. Ensure return routing from the robot subnet

The robot-side boards need a route back to the laptop Wi-Fi subnet. Without the
return route, the laptop may reach `192.168.123.xxx`, but replies and DDS data
may not make it back to `192.168.2.X`.

On each relevant robot-side board, add:

```bash
sudo ip route add 192.168.2.0/24 via 192.168.123.164
```

If the route may already exist:

```bash
sudo ip route replace 192.168.2.0/24 via 192.168.123.164
```

If changing routes on the robot boards is not practical, NAT on the Jetson can
be used as a fallback, but explicit routes are preferred because they preserve
real source addresses for DDS.

## 4. Configure CycloneDDS explicit peers

ROS 2 discovery usually relies on UDP multicast. Multicast normally does not
cross the Wi-Fi/Ethernet routed boundary, so configure CycloneDDS static peers
instead of relying on multicast discovery.

Create this file on the external laptop, for example at
`$HOME/cyclonedds-wifi.xml`:

```xml
<CycloneDDS>
  <Domain>
    <General>
      <NetworkInterfaceAddress>192.168.2.X</NetworkInterfaceAddress>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <Peers>
        <Peer Address="192.168.123.164"/>
        <Peer Address="192.168.123.xxx"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
```

Replace:

- `192.168.2.X` with the laptop Wi-Fi IP.
- `192.168.123.xxx` with each robot board IP that runs ROS 2 nodes the laptop
  needs to discover.

If the Jetson itself runs the ROS 2 nodes of interest, `192.168.123.164` may be
enough. If topics are published by other robot boards, add those board IPs as
additional peers.

## 5. Run ROS 2 commands from the laptop

On the laptop:

```bash
source /opt/ros/foxy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$HOME/cyclonedds-wifi.xml
export ROS_DOMAIN_ID=<same_as_robot>

ros2 topic list
ros2 topic echo /some_topic
```

Use the same `ROS_DOMAIN_ID` as the robot system. If the robot uses the default
domain, this is usually `0`.

## 6. Debug checklist

Basic IP routing:

```bash
ping 192.168.2.41
ping 192.168.123.164
ping 192.168.123.xxx
```

Laptop route:

```bash
ip route get 192.168.123.xxx
```

Jetson forwarding:

```bash
sysctl net.ipv4.ip_forward
```

Robot board return route:

```bash
ip route get 192.168.2.X
```

ROS 2 environment:

```bash
echo $RMW_IMPLEMENTATION
echo $CYCLONEDDS_URI
echo $ROS_DOMAIN_ID
```

If `ping` works but `ros2 topic list` does not, the most likely causes are:

- Missing CycloneDDS peer entries.
- Wrong `ROS_DOMAIN_ID`.
- Robot-side nodes are bound only to a specific interface.
- Firewall rules blocking UDP DDS traffic.
- Missing return route from `192.168.123.0/24` back to `192.168.2.0/24`.

## Summary

The laptop can run ROS 2 over Wi-Fi by routing through the Jetson:

```text
laptop 192.168.2.X -> Jetson Wi-Fi 192.168.2.41 -> Jetson eth0 192.168.123.164 -> robot boards 192.168.123.xxx
```

For reliable `ros2 topic echo`, use explicit CycloneDDS peers because multicast
discovery is not expected to work reliably across the routed subnet boundary.
