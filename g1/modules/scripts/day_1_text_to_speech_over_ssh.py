#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import PurePosixPath


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from unitree_sdk2py.core import channel as channel_module

from sdk_client import Robot


DEFAULT_WIFI_IFACE = "wlp2s0"
DEFAULT_JETSON_IP = "10.34.0.11"

REMOTE_TTS_SCRIPT = r"""
import argparse
import audioop
import shutil
import sys
import tempfile
import wave
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--volume", type=int)
    parser.add_argument("--text", required=True)
    return parser.parse_args()


def convert_wav_for_robot(src_path: Path, dst_path: Path) -> Path:
    with wave.open(str(src_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    if channels == 2:
        pcm = audioop.tomono(pcm, sample_width, 0.5, 0.5)
        channels = 1
    elif channels != 1:
        raise ValueError(f"WAV must be mono or stereo PCM, got {channels} channels")

    if sample_width != 2:
        pcm = audioop.lin2lin(pcm, sample_width, 2)
        sample_width = 2

    if frame_rate != 16000:
        pcm, _state = audioop.ratecv(pcm, sample_width, channels, frame_rate, 16000, None)

    with wave.open(str(dst_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm)

    return dst_path


def main() -> int:
    args = parse_args()
    ChannelFactoryInitialize(int(args.domain_id), args.iface or None)

    client = AudioClient()
    client.SetTimeout(5.0)
    client.Init()

    if args.volume is not None:
        volume_code = int(client.SetVolume(int(args.volume)))
        if volume_code != 0:
            print(f"Remote volume set failed with code {volume_code}.", file=sys.stderr)
            return volume_code

    if shutil.which("espeak") is None:
        print("Remote text-to-speech requires espeak on the Jetson.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="g1_remote_say_") as td:
        wav_path = Path(td) / "speech.wav"
        robot_wav_path = Path(td) / "speech_robot.wav"

        import subprocess

        subprocess.run(["espeak", "-w", str(wav_path), args.text], check=True)
        convert_wav_for_robot(wav_path, robot_wav_path)

        with wave.open(str(robot_wav_path), "rb") as wf:
            pcm = wf.readframes(wf.getnframes())

        code, _data = client.PlayStream("sdk_client", "sdk-client-ssh", pcm)
        code = int(code)
        print(f"Remote speech command completed with code {code}.")
        return code


if __name__ == "__main__":
    raise SystemExit(main())
"""


def configure_cyclonedds(iface: str, peer_ip: str | None) -> None:
    discovery = ""
    if peer_ip:
        discovery = (
            "    <Discovery>\n"
            "      <Peers>\n"
            f'        <Peer Address="{peer_ip}"/>\n'
            "      </Peers>\n"
            "    </Discovery>\n"
        )

    channel_module.ChannelConfigHasInterface = f"""<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="{iface}" priority="default" multicast="default"/>
      </Interfaces>
    </General>
{discovery}  </Domain>
</CycloneDDS>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt for text and play it as speech on the G1 robot."
    )
    parser.add_argument(
        "--transport",
        choices=("ssh", "direct"),
        default="ssh",
        help="Use SSH to run the audio request on the Jetson, or direct DDS from this machine.",
    )
    parser.add_argument(
        "--iface",
        default=DEFAULT_WIFI_IFACE,
        help="Local network interface for direct DDS transport.",
    )
    parser.add_argument(
        "--remote-iface",
        default=os.environ.get("JETSON_IFACE", ""),
        help="Optional network interface to use on the Jetson for SSH transport. Default is auto-detect.",
    )
    parser.add_argument(
        "--jetson-ip",
        default=DEFAULT_JETSON_IP,
        help="Jetson Wi-Fi IP address for SSH transport.",
    )
    parser.add_argument(
        "--jetson-user",
        default=os.environ.get("JETSON_USER") or getpass.getuser(),
        help="SSH username for the Jetson.",
    )
    parser.add_argument(
        "--remote-python",
        default=os.environ.get("JETSON_PYTHON"),
        help="Python executable to use on the Jetson for SSH transport.",
    )
    parser.add_argument(
        "--remote-cyclonedds-home",
        default=os.environ.get("JETSON_CYCLONEDDS_HOME"),
        help="Optional CYCLONEDDS_HOME to export on the Jetson for SSH transport.",
    )
    parser.add_argument("--domain-id", type=int, default=0, help="DDS domain id.")
    parser.add_argument(
        "--safety-boot",
        action="store_true",
        help="Run the hanged safety boot sequence during initialization. Use only when the robot is properly supported.",
    )
    parser.add_argument(
        "--no-safety-boot",
        action="store_true",
        help="Deprecated no-op. Safety boot is skipped by default.",
    )
    parser.add_argument(
        "--volume",
        type=int,
        help="Optional robot playback volume to set before speech playback.",
    )
    return parser.parse_args()


def read_text() -> str | None:
    try:
        text = input("Enter text for the robot to speak: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nInput cancelled.")
        return None

    if not text:
        print("No text provided. Nothing to play.")
        return None
    return text


def run_remote_tts(args: argparse.Namespace, text: str) -> int:
    target = f"{args.jetson_user}@{args.jetson_ip}"
    remote_python = args.remote_python or str(
        PurePosixPath("/home") / args.jetson_user / "ef_ws" / "ef_ws" / "bin" / "python"
    )
    remote_cyclonedds_home = args.remote_cyclonedds_home or str(
        PurePosixPath("/home") / args.jetson_user / "cyclonedds_ws" / "install" / "cyclonedds"
    )
    command = [
        "ssh",
        target,
        "env",
        f"CYCLONEDDS_HOME={remote_cyclonedds_home}",
        remote_python,
        "-",
        "--domain-id",
        str(args.domain_id),
        "--text",
        text,
    ]
    if args.remote_iface:
        command.extend(["--iface", args.remote_iface])
    if args.volume is not None:
        command.extend(["--volume", str(args.volume)])

    try:
        result = subprocess.run(
            command,
            input=REMOTE_TTS_SCRIPT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("SSH client not found. Install ssh or use --transport direct.")
        return 1

    if result.returncode != 0:
        print(
            f"Remote text-to-speech failed via SSH to {target} with exit code {result.returncode}."
        )
        return result.returncode
    return 0


def run_direct_tts(args: argparse.Namespace, text: str) -> int:
    peer_ip = args.jetson_ip.strip() or None
    configure_cyclonedds(args.iface, peer_ip)

    try:
        robot = Robot(
            iface=args.iface,
            domain_id=args.domain_id,
            safety_boot=args.safety_boot,
            auto_start_sensors=True,
        )
    except Exception as exc:
        print(f"Failed to connect to robot: {exc}")
        return 1

    try:
        code = robot.say(text, volume=args.volume)
    except Exception as exc:
        print(f"Text-to-speech failed: {exc}")
        return 1

    print(f"Speech command completed with code {code}.")
    return 0 if code == 0 else int(code)


def main() -> int:
    args = parse_args()
    text = read_text()
    if text is None:
        return 1

    if args.transport == "ssh":
        return run_remote_tts(args, text)
    return run_direct_tts(args, text)


if __name__ == "__main__":
    raise SystemExit(main())
