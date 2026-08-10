# Unitree G1 SDK Usage Notes

This file is loaded automatically as documentary RAG for the OpenAI API CLI agent.

## High-level arm gestures

The SDK high-level arm action client supports canned actions by name. Useful action names include:

- `face wave`
- `high wave`
- `clap`
- `left kiss`
- `shake hand`
- `release arm`

These actions use the high-level arm action service, not `/arm_sdk` IK control.

## Whole-body FSM modes

The locomotion client exposes finite-state-machine mode changes:

- `zero_torque`: FSM id 0, used for zero-torque mode.
- `damp`: FSM id 1, used for damping mode.
- `prepare`: FSM id 4, used for prepare/stand-ready mode.
- `walk`: FSM id 501.
- `run`: FSM id 802.

The local `sdk_client.Robot` wrapper provides `zero_torque()`, `damp()`, `prepare()`, `walk_mode()`, `run_mode()`, and `dev_mode()`.

## Developer mode

`dev_mode()` enters low-command developer mode through the low-command mode service. It should be treated as an operator-controlled mode change.

## Dexterous hands

Dex3 hand topics may be unavailable when the dexterous hands are not connected. Missing left or right hand state should only block hand-specific actions such as opening or closing the hand, or reading tactile sensors. It should not block locomotion, SLAM, speech, high-level gestures, or normal arm SDK motions.

## SLAM and mapping

The SLAM web app backend controls mapping, relocation, selected-pose navigation, queued navigation tasks, pause, resume, and stop. If `start_mapping` returns error code 501 with `Lack of lidar or imu data`, the command reached the SLAM service but the robot-side SLAM backend did not accept it because required lidar or IMU input was missing.

## Audio input

The robot microphone ASR transcript topic is `/audio_msg`. When `audio.input_enabled` and `audio.asr_enabled` are true, the CLI should treat fresh ASR transcripts as normal user prompts.

## Vision input

RGB-D input is configured through `vision.rgbd_enabled`, `vision.rgbd_host`, `vision.rgbd_port`, and `vision.rgbd_topic`. OpenAI vision answers use `vision.openai_model`, currently defaulting to `gpt-4o-mini`.
