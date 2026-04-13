# PBD Scripts Documentation

This document explains how `pbd_demonstrate.py` and `pbd_reproduce.py` work, how they use `unitree_sdk2py`, and how to run them with all available flags.

## 1) Overview

The two scripts implement a simple Programming by Demonstration (PBD) workflow:

1. `pbd_demonstrate.py`: put the robot in a safe standing state, switch to zero-torque, and record arm joint trajectories from `rt/lowstate`.
2. `pbd_reproduce.py`: load recorded trajectories and replay them by publishing `LowCmd` on `rt/arm_sdk`.

Both scripts rely on `safety/hanger_boot_sequence.py` to bring the robot to balanced stand (FSM 200) before teaching or replay.

## 2) Shared Safety/Bring-up: `hanger_boot_sequence`

Both scripts call:

```python
from safety.hanger_boot_sequence import hanger_boot_sequence
```

`hanger_boot_sequence(...)` does the following with `LocoClient`:

1. Initializes DDS (`ChannelFactoryInitialize(0, iface)`).
2. Creates/initializes `LocoClient`.
3. Early-exits if robot is already in balanced stand (`FSM 200` and feet loaded mode).
4. Otherwise runs a boot sequence:
   - `Damp()`
   - `SetFsmId(4)` (stand-up helper)
   - Incrementally increases stand height until feet-loaded condition is detected
   - `BalanceStand(0)`
   - `Start()` to enter FSM 200

This means both PBD scripts assume the robot can be safely brought to a standing-ready condition first.

## 3) `pbd_demonstrate.py` Implementation

### 3.1 High-level flow

1. Parse CLI args.
2. Call `hanger_boot_sequence(iface=args.iface)`.
3. Call `bot.ZeroTorque()` so arms can be physically guided by hand.
4. Initialize DDS channel factory on the selected interface.
5. Resolve `LowState_` type from either:
   - `unitree_sdk2py.idl.unitree_hg.msg.dds_`
   - `unitree_sdk2py.idl.unitree_go.msg.dds_`
6. Build joint list from `--arm` plus waist yaw (`joint 12`).
7. Subscribe to `rt/lowstate` and keep latest joint values in a thread-safe recorder.
8. Poll snapshots at `--poll-s` period, append samples to memory, and stream rows to CSV.
9. Stop on Enter, Ctrl+C, or duration timeout.
10. Save NPZ output (`joints`, `ts`, `qs`, plus compatibility fields).

### 3.2 Joint selection

Constants:

- Left arm joints: `15..21`
- Right arm joints: `22..28`
- Waist yaw: `12` (always appended)

`--arm left`: left arm + waist yaw
`--arm right`: right arm + waist yaw
`--arm both`: left + right + waist yaw

### 3.3 Low-state recording mechanism

`Recorder` stores latest sample from subscriber callback:

- Callback reads `msg.motor_state[j].q` for configured joints.
- Values are protected by `threading.Lock`.
- Main loop samples this latest snapshot at fixed polling interval (`--poll-s`), not at raw DDS callback frequency.

So the logged trajectory is a regularly sampled stream from most recent lowstate data.

### 3.4 Output formats

#### CSV log (`--log` or default `<out>.csv`)

Header:

```text
t_s,j15,j16,...
```

Rows contain sampled relative time and joint positions.

#### NPZ (`--out`, default `/tmp/pbd_motion.npz`)

Saved keys:

- `joints`: `int32` array of joint indices in column order
- `ts`: `float32` time stamps (seconds from recording start)
- `qs`: `float32` joint matrix `[N, num_joints]`
- `fk_qs`: same as `qs` (compatibility for replay modes)
- `poll_s`: scalar array with polling period
- `representation`: `"joint_space"`

### 3.5 Flags for `pbd_demonstrate.py`

- `--iface` (default `enp1s0`)
  - DDS network interface name.
- `--arm {left,right,both}` (default `both`)
  - Which arm joints to record.
- `--duration FLOAT` (default `15.0`)
  - Max recording time in seconds. `0` means unlimited until Enter/Ctrl+C.
- `--poll-s FLOAT` (default `0.02`)
  - Sampling period (seconds). Effective minimum is clamped to `1e-3`.
- `--out PATH` (default `/tmp/pbd_motion.npz`)
  - NPZ output path.
- `--log PATH` (default empty -> `<out_basename>.csv`)
  - CSV log path.

### 3.6 Example commands

```bash
# Record both arms for 20 seconds
python pbd_demonstrate.py --iface enp1s0 --arm both --duration 20 --out /tmp/wave.npz

# Record only right arm until Enter is pressed
python pbd_demonstrate.py --arm right --duration 0 --out /tmp/right_demo.npz --log /tmp/right_demo.csv
```

## 4) `pbd_reproduce.py` Implementation

### 4.1 High-level flow

1. Parse CLI args.
2. Call `hanger_boot_sequence(iface=args.iface)`.
3. Load motion data from file (`.npz`, `.csv`, `.pkl/.pickle`, or fallback auto-parse).
4. Resolve replay matrix based on `--mode`:
   - `joint`: use `qs`
   - `fk`: use `fk_qs` if available, else `qs`
   - `ik`: use `ik_qs` if available, else compute IK online
5. Validate shape consistency (`ts`, `qs`, `joints`).
6. Filter columns to requested arm (`--arm`) and optional waist yaw.
7. Initialize `LowCmdController`.
8. Seed initial command from lowstate (optional best-effort wait).
9. Ramp smoothly to first trajectory point (`--start-ramp`).
10. Replay in real time (or scaled by `--speed`) at `--cmd-hz` with linear interpolation.

### 4.2 How Unitree SDK is used in replay

#### DDS setup and topics

- `ChannelFactoryInitialize(0, iface)`
- Publish on topic: `rt/arm_sdk`
- Subscribe (for seeding) on topic: `rt/lowstate`

#### Message types

- Command message type: `LowCmd_` with default-initialized storage `unitree_hg_msg_dds__LowCmd_()`.
- `NOT_USED_IDX = 29` is set with `q=1` to enable arm SDK path.

#### Per-joint command fields set each cycle

For each commanded joint index:

- `motor_cmd[j].q = target_position`
- `motor_cmd[j].kp = --kp`
- `motor_cmd[j].kd = --kd`
- `motor_cmd[j].tau = 0.0`

Then CRC is computed (`CRC().Crc(cmd)`) and message is written.

#### Seeding and ramping

- `seed_from_lowstate(...)` reads current `q` from `rt/lowstate` when available.
- If available, initial internal command state starts from current robot pose.
- `ramp_to(...)` linearly interpolates from current command state to first trajectory sample to reduce initial jump.

### 4.3 Replay modes

- `joint`
  - Replay recorded joints exactly from `qs`.
- `fk`
  - Uses `fk_qs` if present; otherwise identical to `joint`.
- `ik`
  - Uses `ik_qs` if present.
  - If missing, computes IK per frame using internal FK/IK solver:
    - Approximate 7-DoF arm model
    - Damped least-squares numeric Jacobian solve
    - Warm-started from previous frame solution

### 4.4 Input motion file formats

#### NPZ
Expected keys include `joints`, `ts`, `qs` (and optionally `fk_qs`, `ik_qs`).

#### CSV
Expected:

- A time column: one of `t_s`, `ts`, `time_s`, `time`
- Joint columns named like `j22`, `j23`, ...

#### Pickle (`.pkl` / `.pickle`)
Expected object: `dict` whose values can be converted to NumPy arrays.

### 4.5 Arm/joint filtering at replay time

`--arm` selects required sets:

- `left`: requires all joints `15..21`
- `right`: requires all joints `22..28`
- `both`: requires both sets

If required joints are missing, script exits with explicit error.

Waist yaw (`12`) is appended if present in the motion file.

### 4.6 Timing behavior

- Command period: `dt = 1 / cmd_hz`
- Replay duration target: `t_final = ts[-1] / speed`
- Interpolation uses linear blend between adjacent samples in scaled time base (`ts / speed`)

So:

- `speed > 1.0` replays faster.
- `speed < 1.0` replays slower.

### 4.7 Flags for `pbd_reproduce.py`

- `--iface` (default `enp1s0`)
  - DDS network interface.
- `--file PATH` (default `/tmp/pbd_motion.npz`)
  - Motion input file (`.npz`, `.csv`, `.pkl/.pickle`).
- `--arm {left,right,both}` (default `both`)
  - Arm subset to replay.
- `--mode {joint,fk,ik}` (default `fk`)
  - Which trajectory representation to replay.
- `--speed FLOAT` (default `1.0`)
  - Time scale. `2.0` is twice as fast, `0.5` is half speed.
- `--cmd-hz FLOAT` (default `50.0`)
  - Command publish rate.
- `--seed-timeout FLOAT` (default `0.8`)
  - Wait time for lowstate-based initial pose seeding.
- `--start-ramp FLOAT` (default `0.8`)
  - Duration for initial ramp to first target sample.
- `--kp FLOAT` (default `40.0`)
  - Position stiffness gain for commanded joints.
- `--kd FLOAT` (default `1.0`)
  - Damping gain for commanded joints.

### 4.8 Example commands

```bash
# Replay default file in FK mode, real-time
python pbd_reproduce.py --iface enp1s0 --file /tmp/pbd_motion.npz --mode fk --speed 1.0

# Replay only right arm, slower, with softer gains
python pbd_reproduce.py --arm right --speed 0.7 --kp 25 --kd 0.8 --file /tmp/right_demo.npz

# Replay CSV trajectory at 100 Hz in joint mode
python pbd_reproduce.py --file /tmp/right_demo.csv --mode joint --cmd-hz 100
```

## 5) End-to-end Usage

1. Record a demonstration:

```bash
python pbd_demonstrate.py --iface enp1s0 --arm both --duration 15 --out /tmp/my_motion.npz
```

2. Replay it:

```bash
python pbd_reproduce.py --iface enp1s0 --file /tmp/my_motion.npz --mode fk --speed 1.0
```

3. Tune replay behavior as needed:

- Increase `--start-ramp` for gentler startup.
- Lower `--kp` / increase `--kd` if motion feels too stiff.
- Use `--mode ik` only when IK behavior is specifically needed.

## 6) Notes and Practical Constraints

- Both scripts expect `unitree_sdk2py` to be installed and importable.
- Both scripts assume correct DDS interface (`--iface`) for the robot network.
- Recording stores position trajectories (`q`) only; torque/feedforward is not recorded.
- Replay is position-PD control over selected arm joints via `rt/arm_sdk`.
- If no lowstate is received during recording, `pbd_demonstrate.py` exits with an error.

