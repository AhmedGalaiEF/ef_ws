# Dashboard Service

Install the ROS2 dashboard as a user-specific templated service:

```bash
sudo cp dashboards/systemd/g1-dashboard-ros2@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now g1-dashboard-ros2@$USER
```

Default assumptions:

- workspace path: `~/ef_ws/g1`
- ROS distro: `foxy`
- dashboard port: `8000`

Logs:

```bash
journalctl -u g1-dashboard-ros2@$USER -f
```
