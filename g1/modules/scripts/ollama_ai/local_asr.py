#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
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
        description="Listen to a local headset mic, transcribe locally, and POST text to nav_bot.py /asr."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8096/asr")
    parser.add_argument("--token", default="")
    parser.add_argument("--text-input", action="store_true",
                        help="Disable microphone/ASR and send typed console lines to the POST target.")
    parser.add_argument("--backend", choices=("auto", "faster-whisper", "whisper", "vosk"), default="auto")
    parser.add_argument("--model", default=os.environ.get("LOCAL_ASR_MODEL", "small"),
                        help="faster-whisper/openai-whisper model name, or Vosk model directory.")
    parser.add_argument("--fast", action="store_true",
                        help="Low-latency preset: tiny model, beam size 1, and shorter maximum recordings unless explicitly overridden.")
    parser.add_argument("--device", default=os.environ.get("LOCAL_ASR_DEVICE", "20" if os.name == "nt" else None),
                        help="sounddevice input device id/name. Use --list-devices to inspect.")
    parser.add_argument("--level-meter", action="store_true",
                        help="Print per-slice RMS while recording to debug microphone/device selection.")
    parser.add_argument("--record-dtype", choices=("float32", "int16"),
                        default=os.environ.get("LOCAL_ASR_RECORD_DTYPE", "int16" if os.name == "nt" else "float32"),
                        help="PortAudio capture dtype. Try int16 on Windows devices that return silent float32 samples.")
    parser.add_argument("--save-debug-wav", default="",
                        help="Optional path to save the next transcribed/quiet chunk exactly as Python recorded it.")
    parser.add_argument("--compute-device", default="auto",
                        help="faster-whisper compute device: auto, cpu, cuda.")
    parser.add_argument("--compute-type", default="int8",
                        help="faster-whisper compute type, e.g. int8, int8_float16, float16, float32.")
    parser.add_argument("--download-root", default=None,
                        help="Optional faster-whisper model download/cache directory.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--sample-rate", type=int, default=int(os.environ.get("LOCAL_ASR_SAMPLE_RATE", "0" if os.name == "nt" else "16000")),
                        help="Input sample rate. Use 0 to use the device default; invalid rates fall back automatically.")
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--min-rms", type=float, default=0.00005,
                        help="Skip chunks quieter than this RMS level.")
    parser.add_argument("--gain", type=float, default=float(os.environ.get("LOCAL_ASR_GAIN", "8.0" if os.name == "nt" else "1.0")),
                        help="Multiply recorded audio before transcription. Useful for very quiet Windows inputs.")
    parser.add_argument("--normalize-rms", type=float, default=0.04,
                        help="Normalize non-quiet chunks to this RMS before writing WAV; use 0 to disable.")
    parser.add_argument("--language", "--lang", dest="language", default=os.environ.get("LOCAL_ASR_LANGUAGE", "de"),
                        help="ASR language code, for example de or en.")
    parser.add_argument("--asr-context", default=os.environ.get(
        "LOCAL_ASR_CONTEXT",
        "EF Robotics, Unitree G1, AutoXing, CenoBots, Navigation, Kartierung, Lokalisierung, Gesten, winken, klatschen, Punkt eins, Punkt zwei"
    ), help="Hotwords for ASR, used only when the backend supports hotwords.")
    parser.add_argument("--no-brand-corrections", dest="brand_corrections", action="store_false", default=True,
                        help="Disable local correction of common brand-name ASR mistakes.")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size for faster-whisper decoding; higher is slower but usually more accurate.")
    parser.add_argument("--max-record-seconds", type=float, default=6.0,
                        help="Automatically stop and transcribe after this many seconds; use 0 to disable.")
    parser.add_argument("--once", action="store_true", help="Transcribe one chunk and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print transcripts but do not POST.")
    parser.add_argument("--toggle-key", default="space",
                        help="Key used to stop/start recording. Use 'space', 'enter', or a single character.")
    parser.add_argument("--start-paused", action="store_true",
                        help="Start paused; press the toggle key to begin recording.")
    parser.add_argument("--slice-seconds", type=float, default=0.25,
                        help="Small recording slice used so the toggle key responds quickly.")
    args = parser.parse_args()
    if bool(args.fast):
        if str(args.model) == str(os.environ.get("LOCAL_ASR_MODEL", "small")):
            args.model = "tiny"
        if int(args.beam_size) == 5:
            args.beam_size = 1
        if float(args.max_record_seconds) == 6.0:
            args.max_record_seconds = 4.0
        if float(args.slice_seconds) == 0.25:
            args.slice_seconds = 0.15
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    """Reject bad capture settings before importing audio/model backends."""
    positive = (
        ("chunk-seconds", args.chunk_seconds),
        ("slice-seconds", args.slice_seconds),
    )
    non_negative = (
        ("sample-rate", args.sample_rate),
        ("min-rms", args.min_rms),
        ("normalize-rms", args.normalize_rms),
        ("max-record-seconds", args.max_record_seconds),
    )
    for label, raw in positive:
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise SystemExit(f"--{label} must be a positive finite number")
    for label, raw in non_negative:
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise SystemExit(f"--{label} must be a non-negative finite number")
    if int(args.sample_rate) != float(args.sample_rate):
        raise SystemExit("--sample-rate must be an integer")
    gain = float(args.gain)
    if not math.isfinite(gain) or gain <= 0.0:
        raise SystemExit("--gain must be a positive finite number")
    if int(args.beam_size) < 1:
        raise SystemExit("--beam-size must be at least 1")
    if not str(args.model).strip():
        raise SystemExit("--model must not be empty")
    if not str(args.language).strip():
        raise SystemExit("--language must not be empty")


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
    try:
        port = parsed.port
    except ValueError as exc:
        raise SystemExit(f"Invalid --url port: {url!r}") from exc
    if port is None:
        default_port = 8096 if path in {"/asr", "/command"} else 80
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


def validate_input_device(device: int | str | None, sample_rate: int, record_dtype: str = "float32") -> int:
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
    default_rate = int(round(float(info.get("default_samplerate", 0) or 0)))
    requested_rate = int(sample_rate)
    if requested_rate <= 0:
        if default_rate <= 0:
            raise SystemExit(f"Device {device!r} did not report a usable default sample rate.")
        print(
            f"Using device default sample rate {default_rate} Hz for {info.get('name', device)!r}.",
            flush=True,
        )
        return default_rate
    try:
        sd.check_input_settings(device=device, channels=1, samplerate=requested_rate, dtype=record_dtype)
        return requested_rate
    except Exception as exc:
        if default_rate <= 0 or default_rate == requested_rate:
            raise SystemExit(
                f"Device {device!r} cannot open at {requested_rate} Hz: {exc}\n"
                "Try --sample-rate 0 or another microphone device."
            ) from exc
        try:
            sd.check_input_settings(device=device, channels=1, samplerate=default_rate, dtype=record_dtype)
        except Exception as default_exc:
            raise SystemExit(
                f"Device {device!r} cannot open at requested {requested_rate} Hz or default {default_rate} Hz.\n"
                f"Requested error: {exc}\nDefault error: {default_exc}"
            ) from default_exc
        print(
            f"Device {device!r} rejected {requested_rate} Hz; using default {default_rate} Hz.",
            flush=True,
        )
        return default_rate


def audio_stats(audio: Any) -> tuple[float, float]:
    np = require_module("numpy")
    arr = np.asarray(audio).reshape(-1)
    rms = float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    return rms, peak


def record_chunk(args: argparse.Namespace) -> tuple[Any, float, float]:
    sd = require_module("sounddevice")
    np = require_module("numpy")
    frames = int(float(args.sample_rate) * float(args.chunk_seconds))
    device: int | str | None
    device = resolve_input_device(args.device)
    try:
        channels = 2
        try:
            info = sd.query_devices(device, "input")
            channels = max(1, min(2, int(info.get("max_input_channels", 1) or 1)))
        except Exception:
            channels = 1
        audio = sd.rec(
            frames,
            samplerate=int(args.sample_rate),
            channels=channels,
            dtype=str(args.record_dtype),
            device=device,
        )
        sd.wait()
    except Exception as exc:
        raise SystemExit(
            f"Could not record from input device {device!r} at {int(args.sample_rate)} Hz: {exc}\n"
            "Run --list-devices and choose a microphone input device, or try --sample-rate 0."
        ) from exc
    audio = np.asarray(audio)
    if audio.dtype.kind in {"i", "u"}:
        max_value = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / max_value
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2 and audio.shape[1] > 1:
        rms_by_channel = np.sqrt(np.mean(audio * audio, axis=0))
        channel = int(np.argmax(rms_by_channel))
        audio = audio[:, channel]
    else:
        audio = audio.reshape(-1)
    gain = float(args.gain)
    if gain != 1.0:
        audio = np.clip(audio * gain, -1.0, 1.0)
    rms, peak = audio_stats(audio)
    return audio, rms, peak


def record_seconds(args: argparse.Namespace, seconds: float) -> tuple[Any, float, float]:
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
    def __init__(
        self,
        model_name: str,
        language: str,
        *,
        compute_device: str,
        compute_type: str,
        download_root: str | None,
        beam_size: int,
        asr_context: str,
    ) -> None:
        fw = require_module("faster_whisper")
        self.language = language
        self.beam_size = max(1, int(beam_size))
        self.asr_context = asr_context.strip()
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
        kwargs: dict[str, Any] = {
            "language": self.language,
            "vad_filter": True,
            "beam_size": self.beam_size,
            "condition_on_previous_text": False,
            "vad_parameters": {
                "min_silence_duration_ms": 450,
                "speech_pad_ms": 250,
            },
        }
        if self.asr_context:
            kwargs["hotwords"] = self.asr_context
        try:
            segments, _info = self.model.transcribe(str(wav_path), **kwargs)
        except TypeError:
            kwargs.pop("hotwords", None)
            segments, _info = self.model.transcribe(str(wav_path), **kwargs)
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
        with wave.open(str(wav_path), "rb") as wf:
            sample_rate = wf.getframerate()
            if sample_rate <= 0:
                raise RuntimeError(f"WAV file has invalid sample rate: {sample_rate}")
            rec = self.vosk.KaldiRecognizer(self.model, sample_rate)
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
            beam_size=int(args.beam_size),
            asr_context=str(args.asr_context),
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


def run_text_input(args: argparse.Namespace) -> int:
    print("Text input mode. Type a prompt and press Enter; Ctrl-C or an empty line exits.", flush=True)
    try:
        while True:
            try:
                text = input("> ")
            except EOFError:
                break
            text = text.strip()
            if not text:
                break
            if bool(args.brand_corrections):
                corrected = apply_brand_corrections(text)
                if corrected != text:
                    print(f"corrected: {text} -> {corrected}", flush=True)
                text = corrected
            post_text(args, text)
    except KeyboardInterrupt:
        print("\nexiting", flush=True)
    return 0


def apply_brand_corrections(text: str) -> str:
    corrected = str(text)
    replacements = (
        (r"\bVastion\s+Robotics\b", "EF Robotics"),
        (r"^\s*NO\s*,?\s*AutoXing\s*$", "Was ist AutoXing?"),
        (r"\bUnitree\s+1\b", "Unitree G1"),
        (r".*\b(?:Kannst|Kamst)\s+du\s+das\s+Publikum\s*,?\s*begrüße\s+ich\.?\s*$", "winken"),
        (r".*\bPublikum\b.*\bbegrüß.*$", "winken"),
        (r"^\s*Lokalisierung\s*$", "Lokalisiere dich neu"),
        (r"^\s*Punkt\s*,?\s*(?:Speichern|Spanchen)\s*$", "Punkt speichern"),
        (r"^\s*Punktspeicher\s*$", "Punkt speichern"),
        (r"^\s*Punkt\s+ins\s+Field\s*$", "Punkt eins"),
        (r"^\s*und\s+geben\s*$", "Hand geben"),
        (r"^\s*(?:Vater|Fahrer|Gear)\s*,?\s*(?:2\s*,?\s*0\s*,?\s*)?1\s*$", "Fahre zu Punkt 1"),
        (r"^\s*Gear\s+2\s+und\s+1\s*$", "Fahre zu Punkt 1"),
        (r"^\s*Fahrer\s+Zoom\s+0\.?1\s*$", "Fahre zu Punkt 1"),
        (r"^\s*Nach\s+uns\s*,?\s*eins\s*,?\s*gehen\.?!?\s*$", "nach Punkt eins gehen"),
        (r"\bFahrer\s+(?=zu|zum|zur|nach)\b", "Fahre "),
        (r"\bFehre\s+(?=zu|zum|zur|nach)\b", "Fahre "),
        (r"\bFähre\s+(?=zu|zum|zur|nach)\b", "Fahre "),
        (r"\bLaufen\s+(?=zu|zum|zur|nach)\b", "Laufe "),
        (r"\b(?:Pukt|Pukts|Punks|Punkts)\b", "Punkt"),
        (r"\bHeinz\b", "eins"),
        (r"\beinfather\b", "eins"),
        (r"\bein\s+father\b", "eins"),
        (r"\b2002\.1\b", "zu Punkt 1"),
        (r"\bD\s*F\s+Robotics\b", "EF Robotics"),
        (r"\bE\s*F\s+Robotics\b", "EF Robotics"),
        (r"\bF[\s-]?Botix\b", "EF Robotics"),
        (r"\bF[\s-]?Robotics\b", "EF Robotics"),
        (r"\bE[\s-]?Robotics\b", "EF Robotics"),
        (r"\bE[\s-]?Accordion\b", "EF Robotics"),
        (r"\bEF\s+Robotik\b", "EF Robotics"),
        (r"\bDF\s+Robotics\b", "EF Robotics"),
        (r"\bUnitree\s+G\s*1\b", "Unitree G1"),
        (r"\bUnitree\s+gee\s+one\b", "Unitree G1"),
        (r"\bG\s*One\b", "G1"),
        (r"\bAuto\s*Xing\b", "AutoXing"),
        (r"\bAuto\s*Crossing\b", "AutoXing"),
        (r"\bCeno\s*Bots\b", "CenoBots"),
        (r"\bZeno\s*Bots\b", "CenoBots"),
    )
    for pattern, replacement in replacements:
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
    return corrected


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
    debug_path = str(args.save_debug_wav or "").strip()
    if debug_path:
        try:
            saved = Path(debug_path).expanduser()
            original_debug = write_wav(audio, int(args.sample_rate))
            saved.write_bytes(original_debug.read_bytes())
            try:
                original_debug.unlink()
            except OSError:
                pass
            print(f"saved debug wav: {saved}", flush=True)
        except Exception as exc:
            print(f"failed to save debug wav: {exc}", file=sys.stderr, flush=True)
        args.save_debug_wav = ""
    if rms < float(args.min_rms):
        _rms, peak = audio_stats(audio)
        print(f"quiet rms={rms:.6f} peak={peak:.6f}", flush=True)
        return
    normalize_rms = float(args.normalize_rms)
    if normalize_rms > 0.0 and rms > 0.0:
        scale = min(200.0, normalize_rms / rms)
        if scale > 1.0:
            audio = np.clip(audio * scale, -1.0, 1.0)
            new_rms, new_peak = audio_stats(audio)
            print(f"normalized audio gain={scale:.1f} rms={new_rms:.6f} peak={new_peak:.6f}", flush=True)
    wav_path = write_wav(audio, int(args.sample_rate))
    try:
        text = backend.transcribe(wav_path).strip()
    finally:
        try:
            wav_path.unlink()
        except OSError:
            pass
    if text:
        original_text = text
        if bool(args.brand_corrections):
            text = apply_brand_corrections(text)
        print(f"heard: {text}", flush=True)
        if text != original_text:
            print(f"corrected: {original_text} -> {text}", flush=True)
        post_text(args, text)
    else:
        print(f"no transcript rms={rms:.4f}", flush=True)


def main() -> int:
    args = parse_args()
    args.url = normalize_post_url(str(args.url))
    print(describe_post_url(str(args.url)), flush=True)
    if bool(args.text_input):
        return run_text_input(args)
    if args.list_devices:
        list_devices()
        return 0
    args.sample_rate = validate_input_device(resolve_input_device(args.device), int(args.sample_rate), str(args.record_dtype))
    print(
        f"Audio input: device={args.device!r} sample_rate={int(args.sample_rate)} "
        f"dtype={args.record_dtype} gain={float(args.gain):g}",
        flush=True,
    )

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
    recording_started = time.time() if recording else 0.0
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
                    recording_started = time.time()
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
            audio, slice_rms, slice_peak = record_seconds(args, max(0.05, float(args.slice_seconds)))
            if bool(args.level_meter):
                print(f"rms={slice_rms:.6f} peak={slice_peak:.6f}", flush=True)
            chunks.append(audio)
            max_record_s = float(args.max_record_seconds)
            if max_record_s > 0.0 and chunks and time.time() - recording_started >= max_record_s:
                audio = np.concatenate(chunks)
                rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
                print(f"auto-stopped; transcribing {audio.size / float(args.sample_rate):.1f}s rms={rms:.4f}", flush=True)
                transcribe_and_post(args, backend, audio, rms)
                chunks = []
                recording = False
                print("paused", flush=True)
                continue
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
