[
  {
    "title": "Unitree G1 high-level arm gestures",
    "category": "sdk gestures arm actions",
    "text": "The SDK high-level arm action client supports canned actions named face wave, high wave, clap, left kiss, shake hand, and release arm. These actions use the high-level arm action service, not /arm_sdk IK control."
  },
  {
    "title": "Unitree G1 whole-body FSM modes",
    "category": "sdk locomotion fsm modes",
    "text": "The locomotion client exposes FSM modes: zero_torque uses FSM id 0, damp uses FSM id 1, prepare uses FSM id 4, walk uses FSM id 501, and run uses FSM id 802. sdk_client.Robot provides zero_torque(), damp(), prepare(), walk_mode(), run_mode(), and dev_mode()."
  },
  {
    "title": "Developer mode and ai_sport",
    "category": "sdk developer mode low command ai_sport",
    "text": "dev_mode() enters low-command developer mode by turning the ai_sport service off. Leaving developer mode should re-enable ai_sport with leave_lowcmd_dev_mode(). Developer mode should be treated as an operator-controlled mode change."
  },
  {
    "title": "Dex3 hand faults",
    "category": "dex3 hands tactile faults",
    "text": "Dex3 hand topics may be unavailable when the dexterous hands are not connected. Missing left or right hand state should only block hand-specific actions such as opening or closing the hand, or reading tactile sensors. It should not block locomotion, SLAM, speech, high-level gestures, or normal arm SDK motions."
  },
  {
    "title": "SLAM and mapping",
    "category": "slam mapping navigation lidar imu",
    "text": "The SLAM web app backend controls mapping, relocation, selected-pose navigation, queued navigation tasks, pause, resume, and stop. start_mapping calls SLAM API 1801. If start_mapping returns error code 501 with Lack of lidar or imu data, the command reached the SLAM service but the robot-side SLAM backend did not accept it because required lidar or IMU input was missing."
  },
  {
    "title": "Named SLAM points",
    "category": "slam navigation named points labels",
    "text": "The nav bot stores named map points in a JSON file containing map_path, updated timestamp, and points keyed by normalized names. Each point stores x, y, z, and yaw. Commands can save the current point under a name, list saved points, clear points, and navigate to a named point after relocation."
  },
  {
    "title": "Audio input",
    "category": "asr microphone audio_msg",
    "text": "The robot microphone ASR transcript topic is /audio_msg. When audio.input_enabled and audio.asr_enabled are true, the CLI should treat fresh ASR transcripts as normal user prompts."
  },
  {
    "title": "Vision input",
    "category": "rgbd vision openai model",
    "text": "RGB-D input is configured through vision.rgbd_enabled, vision.rgbd_host, vision.rgbd_port, and vision.rgbd_topic. OpenAI vision answers use vision.openai_model, currently defaulting to gpt-4o-mini."
  }
]
