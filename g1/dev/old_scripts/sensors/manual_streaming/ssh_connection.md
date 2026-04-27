# SSH connection to Unitree

## Connect
Use this command from your local machine:

```bash
ssh unitree@192.168.123.164
```

When prompted for password, enter:

```text
123
```

Optional (non-interactive) test:

```bash
ssh unitree@192.168.123.164 'hostname && whoami'
```

## Very short ROS (Foxy) environment notes
After login, source ROS and your workspace (adjust path if different):

```bash
source /opt/ros/foxy/setup.bash
source ~/foxy_ws/install/setup.bash
```

Quick checks:

```bash
printenv | grep -E 'ROS|RMW'
ros2 doctor
```

## ROS commands cheat sheet

### Topics
```bash
ros2 topic list
ros2 topic info /topic_name
ros2 topic echo /topic_name
ros2 topic hz /topic_name
ros2 topic pub /topic_name std_msgs/msg/String '{data: hello}'
```

### Nodes
```bash
ros2 node list
ros2 node info /node_name
```

### Packages
```bash
ros2 pkg list
ros2 pkg prefix <package_name>
ros2 pkg executables <package_name>
```

### Messages / interfaces
```bash
ros2 interface list
ros2 interface show std_msgs/msg/String
ros2 msg list
ros2 msg show geometry_msgs/msg/Twist
```

### Services
```bash
ros2 service list
ros2 service type /service_name
ros2 service call /service_name <srv_type> '{}'
```

### Actions
```bash
ros2 action list
ros2 action info /action_name
```

### Parameters
```bash
ros2 param list
ros2 param get /node_name param_name
ros2 param set /node_name param_name value
```

### Logs
```bash
ros2 run rqt_console rqt_console
journalctl -u <service_name> -f
tail -f ~/.ros/log/latest/*
```
