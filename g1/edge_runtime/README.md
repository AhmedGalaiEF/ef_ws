# G1 Edge Runtime

This directory contains the edge runtime for the Unitree G1 robot, designed to integrate with a cloud marketplace platform. The runtime manages device identity, hardware fingerprinting, skill execution, and telemetry.

## Architecture Overview

```
Robot / Edge Runtime
        │
        ▼
Robot Gateway + Device Identity
        │
        ▼
Compatibility Engine
        │
        ▼
Marketplace Platform
```

## Core Components

- **Device Agent**: Manages communication with the cloud platform.
- **Skill Runtime Manager**: Handles installation and execution of skills.
- **Local Safety Monitor**: Ensures safe operation of skills.
- **Telemetry Collector**: Gathers and reports system metrics.
- **Secure Update Client**: Manages software updates.

## Directory Structure

- `config/`: Configuration files for the runtime.
- `device_identity/`: Hardware and software fingerprinting.
- `skills/`: Skill management and execution.
- `telemetry/`: Data collection and reporting.
- `gateway/`: Communication with the cloud platform.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Device Identity**:
   - Edit `device_identity/hardware_fingerprint.json` to match your robot's specifications.
   - Fill in any placeholders in `device_identity/user_config.json`.

3. **Run the Runtime**:
   ```bash
   python -m edge_runtime.main
   ```

## Device Identity

The runtime automatically identifies robot hardware capabilities. This fingerprint determines which skills can be discovered and installed.

Example fingerprint:
```json
{
  "manufacturer": "Unitree",
  "model": "G1",
  "os_version": "3.2.1",
  "arm_dof": 6,
  "hand_type": "5_finger",
  "sensors": ["rgbd_camera", "lidar"],
  "compute": "Nvidia Orin",
  "payload_kg": 3
}
```

## Compatibility Engine

The compatibility engine evaluates whether a robot can install and run a specific skill based on hardware, software, safety constraints, and regulatory conditions.

Example compatibility check:
```json
{
  "skill_id": "warehouse_pick_pro",
  "device_id": "robot_123",
  "status": "compatible_with_conditions",
  "required_dependencies": ["vision_runtime_v2"],
  "constraints": ["requires_5_finger_hand", "requires_depth_camera"]
}
```

## License

MIT
