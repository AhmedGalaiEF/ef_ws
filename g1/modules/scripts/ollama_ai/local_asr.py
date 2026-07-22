#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import select
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import urllib.parse
import wave
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen to a local headset mic, transcribe locally, and POST text to chatbot.py /asr."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8095/asr")
    parser.add_argument("--token", default="")
    parser.add_argument("--backend", choices=("auto", "faster-whisper", "whisper", "vosk"), default="auto")
    parser.add_argument("--model", default="tiny.en",
                        help="faster-whisper/openai-whisper model name, or Vosk model directory.")
    parser.add_argument("--device", default=None,
                        help="sounddevice input device id/name. Use --list-devices to inspect.")
    parser.add_argument("--level-meter", action="store_true",
                        help="Print per-slice RMS while recording to debug microphone/device selection.")
    parser.add_argument("--compute-device", default="auto",
                        help="faster-whisper compute device: auto, cpu, cuda.")
    parser.add_argument("--compute-type", default="int8",
                        help="faster-whisper compute type, e.g. int8, int8_float16, float16, float32.")
    parser.add_argument("--download-root", default=None,
                        help="Optional faster-whisper model download/cache directory.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--min-rms", type=float, default=0.008,
                        help="Skip chunks quieter than this RMS level.")
    parser.add_argument("--language", default="en")
    parser.add_argument("--once", action="store_true", help="Transcribe one chunk and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print transcripts but do not POST.")
    parser.add_argument("--toggle-key", default="space",
                        help="Key used to stop/start recording. Use 'space', 'enter', or a single character.")
    parser.add_argument("--start-paused", action="store_true",
                        help="Start paused; press the toggle key to begin recording.")
    parser.add_argument("--slice-seconds", type=float, default=0.25,
                        help="Small recording slice used so the toggle key responds quickly.")
    return parser.parse_args()


def normalize_post_url(url: str) -> str:
    text = str(url or "").strip()
    if not text:
        raise SystemExit("--url must not be empty")
    if "://" not in text:
        text = "http://" + text
    parsed = urllib.parse.urlparse(text)
    if not parsed.hostname:
        raise SystemExit(f"Invalid --url: {url!r}")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/asr"
    if path not in {"/asr", "/command"}:
        print(
            f"Warning: --url path is {path!r}; navbot usually expects /asr.",
            file=sys.stderr,
            flush=True,
        )
    netloc = parsed.netloc
    if parsed.port is None:
        default_port = 8097 if path in {"/asr", "/command"} else 80
        netloc = f"{parsed.hostname}:{default_port}"
    normalized = urllib.parse.urlunparse((
        parsed.scheme or "http",
        netloc,
        path,
        "",
        parsed.query,
        "",
    ))
    return normalized


def describe_post_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return (
        f"POST target: {url} "
        f"(host={parsed.hostname or 'unknown'} port={parsed.port or 'default'} path={parsed.path or '/'})"
    )


def require_module(name: str) -> Any:
    try:
        return __import__(name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing Python package '{name}'. Install dependencies on the headset computer, for example:\n"
            "  python3 -m pip install sounddevice faster-whisper\n"
            "or:\n"
            "  python3 -m pip install sounddevice vosk\n"
        ) from exc


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    for backend, module in (("faster-whisper", "faster_whisper"), ("whisper", "whisper"), ("vosk", "vosk")):
        try:
            __import__(module)
            return backend
        except ImportError:
            continue
    raise SystemExit(
        "No local ASR backend found. Install one:\n"
        "  python3 -m pip install sounddevice faster-whisper\n"
        "or:\n"
        "  python3 -m pip install sounddevice vosk"
    )


def list_devices() -> None:
    sd = require_module("sounddevice")
    print(sd.query_devices())


def resolve_input_device(device_arg: str | None) -> int | str | None:
    if device_arg is None:
        return None
    device_text = str(device_arg).strip()
    return int(device_text) if device_text.isdigit() else device_text


def validate_input_device(device: int | str | None) -> None:
    sd = require_module("sounddevice")
    try:
        info = sd.query_devices(device, "input")
    except Exception as exc:
        raise SystemExit(
            f"Invalid input device {device!r}: {exc}\n"
            "Run: py .\\local_asr.py --list-devices\n"
            "Choose a device with inputs, e.g. one marked '(2 in, 0 out)', not headphones/output."
        ) from exc
    max_inputs = int(info.get("max_input_channels", 0) or 0)
    if max_inputs < 1:
        raise SystemExit(
            f"Device {device!r} is not an input device: {info.get('name', device)!r} "
            f"has {max_inputs} input channels.\n"
            "Choose a microphone device, not a headphones/output device."
        )


def record_chunk(args: argparse.Namespace) -> tuple[Any, float]:
    sd = require_module("sounddevice")
    np = require_module("numpy")
    frames = int(float(args.sample_rate) * float(args.chunk_seconds))
    device: int | str | None
    device = resolve_input_device(args.device)
    audio = sd.rec(
        frames,
        samplerate=int(args.sample_rate),
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    audio = np.asarray(audio).reshape(-1)
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    return audio, rms


def record_seconds(args: argparse.Namespace, seconds: float) -> tuple[Any, float]:
    old_chunk = args.chunk_seconds
    args.chunk_seconds = float(seconds)
    try:
        return record_chunk(args)
    finally:
        args.chunk_seconds = old_chunk


def write_wav(audio: Any, sample_rate: int) -> Path:
    np = require_module("numpy")
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(prefix="local_asr_", suffix=".wav", delete=False)
    tmp.close()
    path = Path(tmp.name)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return path


class FasterWhisperBackend:
    def __init__(self, model_name: str, language: str, *, compute_device: str, compute_type: str, download_root: str | None) -> None:
        fw = require_module("faster_whisper")
        self.language = language
        kwargs: dict[str, Any] = {
            "device": str(compute_device),
            "compute_type": str(compute_type),
        }
        if download_root:
            kwargs["download_root"] = str(Path(download_root).expanduser())
        print(
            "Loading faster-whisper model. The first run may download model files and can take a while.",
            flush=True,
        )
        self.model = fw.WhisperModel(model_name, **kwargs)

    def transcribe(self, wav_path: Path) -> str:
        segments, _info = self.model.transcribe(str(wav_path), language=self.language, vad_filter=True)
        return " ".join(seg.text.strip() for seg in segments).strip()


class OpenAIWhisperBackend:
    def __init__(self, model_name: str, language: str) -> None:
        whisper = require_module("whisper")
        self.language = language
        self.model = whisper.load_model(model_name)

    def transcribe(self, wav_path: Path) -> str:
        result = self.model.transcribe(str(wav_path), language=self.language, fp16=False)
        return str(result.get("text", "")).strip()


class VoskBackend:
    def __init__(self, model_path: str, _language: str) -> None:
        vosk = require_module("vosk")
        model_dir = Path(model_path).expanduser()
        if not model_dir.exists():
            raise SystemExit(f"Vosk model directory not found: {model_dir}")
        self.vosk = vosk
        self.model = vosk.Model(str(model_dir))

    def transcribe(self, wav_path: Path) -> str:
        rec = self.vosk.KaldiRecognizer(self.model, 16000)
        with wave.open(str(wav_path), "rb") as wf:
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                rec.AcceptWaveform(data)
        result = json.loads(rec.FinalResult())
        return str(result.get("text", "")).strip()


def make_backend(args: argparse.Namespace) -> Any:
    backend = choose_backend(str(args.backend))
    print(f"ASR backend: {backend} model={args.model}", flush=True)
    if backend == "faster-whisper":
        return FasterWhisperBackend(
            str(args.model),
            str(args.language),
            compute_device=str(args.compute_device),
            compute_type=str(args.compute_type),
            download_root=args.download_root,
        )
    if backend == "whisper":
        return OpenAIWhisperBackend(str(args.model), str(args.language))
    return VoskBackend(str(args.model), str(args.language))


def post_text(args: argparse.Namespace, text: str) -> None:
    if args.dry_run:
        return
    body = json.dumps({"text": text}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    request = urllib.request.Request(str(args.url), data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5.0) as response:
            print(f"POST {response.status}: {response.read().decode('utf-8', errors='replace')}", flush=True)
    except urllib.error.URLError as exc:
        print(f"POST failed: {exc}", file=sys.stderr, flush=True)


def normalize_toggle_key(key: str) -> str:
    key = str(key).strip().lower()
    if key in {"space", "spacebar", ""}:
        return " "
    if key in {"enter", "return"}:
        return "\r"
    return key[:1]


def start_toggle_listener(toggle_key: str) -> queue.Queue[str]:
    events: queue.Queue[str] = queue.Queue()
    wanted = normalize_toggle_key(toggle_key)

    def run_windows() -> None:
        import msvcrt
        while True:
            ch = msvcrt.getwch()
            if ch in {"\x03", "\x1a"}:
                events.put("interrupt")
                return
            if ch == wanted or (wanted == "\r" and ch in {"\r", "\n"}):
                events.put("toggle")

    def run_posix() -> None:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not readable:
                    continue
                ch = sys.stdin.read(1)
                if ch in {"\x03", "\x04"}:
                    events.put("interrupt")
                    return
                if ch == wanted or (wanted == "\r" and ch in {"\r", "\n"}):
                    events.put("toggle")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    target = run_windows if os.name == "nt" else run_posix
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return events


def transcribe_and_post(args: argparse.Namespace, backend: Any, audio: Any, rms: float) -> None:
    np = require_module("numpy")
    if audio.size == 0:
        return
    if rms < float(args.min_rms):
        print(f"quiet rms={rms:.4f}", flush=True)
        return
    wav_path = write_wav(audio, int(args.sample_rate))
    try:
        text = backend.transcribe(wav_path).strip()
    finally:
        try:
            wav_path.unlink()
        except OSError:
            pass
    if text:
        print(f"heard: {text}", flush=True)
        post_text(args, text)
    else:
        print(f"no transcript rms={rms:.4f}", flush=True)


def main() -> int:
    args = parse_args()
    args.url = normalize_post_url(str(args.url))
    print(describe_post_url(str(args.url)), flush=True)
    if args.list_devices:
        list_devices()
        return 0
    validate_input_device(resolve_input_device(args.device))

    try:
        backend = make_backend(args)
    except KeyboardInterrupt:
        print(
            "\nInterrupted while loading/downloading the ASR model. Re-run the same command to resume the cache download.",
            file=sys.stderr,
            flush=True,
        )
        return 130
    np = require_module("numpy")
    toggles = start_toggle_listener(str(args.toggle_key))
    recording = not bool(args.start_paused)
    chunks: list[Any] = []
    print(
        f"{'Recording' if recording else 'Paused'}. Press {args.toggle_key!r} to "
        f"{'stop and transcribe' if recording else 'start recording'}; Ctrl-C exits.",
        flush=True,
    )
    try:
        while True:
            while not toggles.empty():
                event = toggles.get_nowait()
                if event == "interrupt":
                    raise KeyboardInterrupt
                if event != "toggle":
                    continue
                recording = not recording
                if recording:
                    chunks = []
                    print("recording...", flush=True)
                else:
                    if chunks:
                        audio = np.concatenate(chunks)
                        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
                        print(f"stopped; transcribing {audio.size / float(args.sample_rate):.1f}s rms={rms:.4f}", flush=True)
                        transcribe_and_post(args, backend, audio, rms)
                    chunks = []
                    print("paused", flush=True)
            if not recording:
                time.sleep(0.02)
                continue
            audio, slice_rms = record_seconds(args, max(0.05, float(args.slice_seconds)))
            if bool(args.level_meter):
                print(f"rms={slice_rms:.4f}", flush=True)
            chunks.append(audio)
            if args.once:
                audio = np.concatenate(chunks)
                rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
                transcribe_and_post(args, backend, audio, rms)
                break
    except KeyboardInterrupt:
        if recording and chunks:
            audio = np.concatenate(chunks)
            rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
            print(f"\nstopped; transcribing {audio.size / float(args.sample_rate):.1f}s rms={rms:.4f}", flush=True)
            transcribe_and_post(args, backend, audio, rms)
        print("\nexiting", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
