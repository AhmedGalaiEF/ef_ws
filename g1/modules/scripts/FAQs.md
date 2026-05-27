# Unitree SDK Setup

Question: How do I initialize the Unitree Python SDK in these scripts?

Answer: Most scripts initialize DDS with `ChannelFactoryInitialize(int(domain_id), iface)`. The usual defaults in this workspace are interface `eth0` and DDS domain ID `0`. For scripts that use the local helper client, create `Robot(iface="eth0", domain_id=0)`.

# Unitree SDK Import Error

Question: What should I do if `unitree_sdk2py` is not installed?

Answer: Install the Unitree Python SDK in editable mode with `pip install -e <path-to-unitree_sdk2_python>`, then run the script again from the workspace environment.

# DDS Environment

Question: What DDS environment variables are used here?

Answer: Several scripts set `CYCLONEDDS_HOME` to `/home/unitree/cyclonedds_ws/install/cyclonedds` and `CYCLONEDDS_URI` to `/home/unitree/cyclonedds_ws/cyclonedds.xml`. Use these when the SDK cannot discover robot topics.

# Network Interface

Question: Which network interface should I use for Unitree SDK commands?

Answer: Use `--iface eth0` unless the robot network is on a different adapter. The SDK DDS participant must bind to the interface connected to the robot.

# Domain ID

Question: Which DDS domain ID should I use?

Answer: Use `--domain-id 0` by default. The robot, ROS bridge, and Unitree SDK scripts must use the same domain ID to see each other.

# Discovering DDS Topics

Question: How can I discover active robot DDS topics?

Answer: Run `python3 dds_discover_topics.py --iface eth0 --domain-id 0 --seconds 6`. This uses CycloneDDS built-in discovery to list publishers, subscribers, and topic names.

# Arm SDK Topic

Question: Which topic controls the G1 arm SDK commands?

Answer: The arm control scripts publish `LowCmd_` messages on `rt/arm_sdk`. They usually create a `ChannelPublisher("rt/arm_sdk", LowCmd_)` after calling `ChannelFactoryInitialize`.

# Arm Command Message

Question: Which Unitree message type is used for arm low-level commands?

Answer: The scripts use `unitree_hg_msg_dds__LowCmd_()` from `unitree_sdk2py.idl.default` and publish it as `LowCmd_` from `unitree_sdk2py.idl.unitree_hg.msg.dds_`.

# Arm CRC

Question: Do arm SDK commands need a CRC?

Answer: Yes. Before publishing low-level arm commands, update the command CRC with `CRC().Crc(cmd)`. Without the CRC the robot may ignore the command.

# Arm Boot Sequence

Question: When should I use the hanger boot sequence?

Answer: Some arm-control scripts call `hanger_boot_sequence(iface=args.iface, domain_id=args.domain_id)` before publishing. Use it when the robot needs to enter the correct arm SDK control mode before accepting `rt/arm_sdk` commands.

# Releasing Arms

Question: How do I release or reengage the G1 arms from these scripts?

Answer: Use the local `sdk_client.Robot` helper. The dashboard calls `robot.release_arms()` to release the arms and `robot.unrelease_arms()` to reengage them.

# Text To Speech

Question: How does the robot speak a generated answer?

Answer: The chat scripts call `robot_say_once.py`, which creates `Robot(iface=args.iface, domain_id=args.domain_id)` and sends one spoken message through the robot audio path.

# Headlight Control

Question: How is the robot headlight controlled?

Answer: The chat scripts call `robot_headlight_once.py` at startup. It uses `sdk_client.Robot` with the selected interface and domain ID, then sends one headlight color and intensity command.

# ASR Chat Topic

Question: Which ROS topic does the chat script listen to?

Answer: `chat.py` and `chat_with_FAQs.py` subscribe to `/audio_msg` by default. The message is a ROS 2 `std_msgs/String` containing JSON or raw text from ASR.

# FAQ Chat Startup

Question: How do I start the FAQ robot chat?

Answer: Run `python3 chat_with_FAQs.py --faq-file FAQs.md --iface eth0 --domain-id 0`. If `FAQs.md` is next to the script, it is loaded automatically when `--faq-file` is omitted.

# FAQ Chat Logs

Question: Where does FAQ chat save logs?

Answer: By default, FAQ chat writes JSONL events to `/tmp/robot_chat_faqs.jsonl` and a plain transcript to `/tmp/robot_chat_faqs.txt`.

# Safety Placeholder

Question: What should the robot say if the FAQ does not contain an SDK answer?

Answer: It should say it does not know yet instead of guessing. Add the missing verified SDK detail to this FAQ file before relying on the answer.
