# g1_approval_ros

Sample ROS 2 Python package that puts an approval gate in front of robot commands.

## What it demonstrates

- A ROS topic for inbound command requests: `/g1/command_request`
- A ROS topic for approval prompts: `/g1/approval_request`
- A ROS topic for operator decisions: `/g1/approval_response`
- A ROS topic for final command results: `/g1/command_result`
- Append-only audit logging to JSONL
- A strict allowlist policy that rejects arbitrary script execution

## Why this model is safer

This package deliberately does not accept raw shell commands or arbitrary Python.
Instead, the caller requests a named action such as `walk_distance` or `hand_open`.
The gateway classifies the action, records requester metadata such as `source_ip`,
and either executes it, requests operator approval, or rejects it.

## Package layout

- `g1_approval_ros/command_gateway.py`: main approval and execution flow
- `g1_approval_ros/approval_console.py`: simple terminal-based approver
- `g1_approval_ros/executor.py`: stub where you wire calls into `sdk_client.Robot`
- `g1_approval_ros/policy.py`: allowlist and risk policy
- `g1_approval_ros/audit.py`: JSONL audit logger
- `g1_approval_ros/request_demo.py`: example request publisher

## Example request payload

```json
{
  "action": "walk_distance",
  "parameters": {
    "distance_m": 0.4
  },
  "source_ip": "192.168.1.50",
  "requester": "teleop-laptop-2",
  "submitted_at": "2026-04-30T12:00:00+00:00",
  "code_summary": "Mapped gateway action 'walk_distance' to a fixed robot SDK call."
}
```

## Build

From the workspace root that contains this package:

```bash
colcon build --packages-select g1_approval_ros
```

## Run

In one terminal:

```bash
ros2 launch g1_approval_ros g1_approval_demo.launch.py
```

In another terminal:

```bash
ros2 run g1_approval_ros request_demo walk_distance 192.168.1.50 teleop-laptop-2
```

## Next integration step

Replace `RobotCommandExecutor.execute()` with direct calls into your local robot wrapper,
for example creating one `sdk_client.Robot` instance and mapping:

- `balanced_stand` -> `robot.balanced_stand()`
- `walk_distance` -> `robot.walk_for(...)`
- `turn_angle` -> `robot.turn_for(...)`
- `hand_open` -> `robot.hand_open(...)`
- `hand_close` -> `robot.hand_close(...)`

Keep the allowlist model. Do not replace it with arbitrary code execution.
