import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
except ImportError:
    DeepLabV3_MobileNet_V3_Large_Weights = None


VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

PERSON_CLASS_ID = 15


@dataclass
class SegmentationResult:
    mask: np.ndarray
    labels: list[str]
    updated_at: float


class AsyncSegmenter:
    def __init__(self, device: str, input_size: int):
        self.device = torch.device(device)
        self.input_size = input_size
        self.model = self._build_model()
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.result_lock = threading.Lock()
        self.latest_result: SegmentationResult | None = None
        self.latest_error: str | None = None
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)

    def _build_model(self):
        if DeepLabV3_MobileNet_V3_Large_Weights is not None:
            model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        else:
            model = deeplabv3_mobilenet_v3_large(pretrained=True)
        model.eval().to(self.device)
        return model

    def start(self):
        self.worker.start()

    def stop(self):
        self.stop_event.set()
        self.worker.join(timeout=2.0)

    def submit(self, frame_bgr: np.ndarray):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait(frame_bgr.copy())
        except queue.Full:
            pass

    def get_latest(self) -> SegmentationResult | None:
        with self.result_lock:
            return self.latest_result

    def get_latest_error(self) -> str | None:
        with self.result_lock:
            return self.latest_error

    def _run(self):
        while not self.stop_event.is_set():
            try:
                frame_bgr = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                tensor = self.transform(rgb).unsqueeze(0).to(self.device)

                with torch.inference_mode():
                    output = self.model(tensor)["out"][0]

                mask_small = output.argmax(0).detach().cpu().numpy().astype(np.uint8)
                mask = cv2.resize(
                    mask_small,
                    (frame_bgr.shape[1], frame_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                with self.result_lock:
                    self.latest_result = SegmentationResult(
                        mask=mask,
                        labels=self._labels_from_mask(mask),
                        updated_at=time.time(),
                    )
                    self.latest_error = None
            except Exception as exc:
                with self.result_lock:
                    self.latest_error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _labels_from_mask(mask: np.ndarray) -> list[str]:
        ids = sorted(int(v) for v in np.unique(mask) if v != 0)
        return [VOC_CLASSES[i] for i in ids if i < len(VOC_CLASSES)]


def build_segmented_view(
    frame_bgr: np.ndarray,
    seg_result: SegmentationResult | None,
    seg_error: str | None,
) -> np.ndarray:
    if seg_result is None:
        placeholder = frame_bgr.copy()
        line1 = "Segmentation model is starting..."
        line2 = "This can take a few seconds on CPU."
        color = (0, 255, 255)
        if seg_error:
            line1 = "Segmentation failed"
            line2 = seg_error[:80]
            color = (0, 0, 255)
        cv2.putText(
            placeholder,
            line1,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            placeholder,
            line2,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        return placeholder

    mask = seg_result.mask
    human_mask = mask == PERSON_CLASS_ID
    object_mask = (mask != 0) & (mask != PERSON_CLASS_ID)
    background_mask = mask == 0

    output = np.zeros_like(frame_bgr)
    output[background_mask] = (35, 25, 15)
    output[human_mask] = (40, 220, 40)
    output[object_mask] = (0, 170, 255)

    outlines = np.zeros_like(mask)
    outlines[human_mask | object_mask] = 255
    contours, _ = cv2.findContours(outlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (255, 255, 255), 1)

    edges = cv2.Canny(frame_bgr, 80, 160)
    output[edges > 0] = (220, 220, 220)

    object_labels = [label for label in seg_result.labels if label != "person"]
    human_state = "yes" if np.any(human_mask) else "no"
    cv2.putText(
        output,
        f"Human: {human_state}  Objects: {', '.join(object_labels) if object_labels else 'none'}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "Green=human  Orange=objects  Brown=background",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def put_fps(image: np.ndarray, fps: float, prefix: str):
    cv2.putText(
        image,
        f"{prefix}: {fps:5.1f} FPS",
        (20, image.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def compose_views(views: list[np.ndarray]) -> np.ndarray:
    if not views:
        raise ValueError("At least one view must be enabled.")
    if len(views) == 1:
        return views[0]
    return np.hstack(views)


def colorize_depth_raw(depth_raw: np.ndarray) -> np.ndarray:
    depth = depth_raw.astype(np.float32)
    valid = depth > 0
    if not np.any(valid):
        return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    lo = np.percentile(depth[valid], 2)
    hi = np.percentile(depth[valid], 98)
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    depth_u8 = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(255 - depth_u8, cv2.COLORMAP_TURBO)


def list_realsense_devices() -> list[str]:
    if rs is None:
        return []

    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        try:
            name = dev.get_info(rs.camera_info.name)
        except RuntimeError:
            name = "Unknown device"
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
        except RuntimeError:
            serial = "unknown-serial"
        devices.append(f"{name} ({serial})")
    return devices


def prepare_console():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def dshow_capture(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap
    return cv2.VideoCapture(index)


def probe_uvc_devices(max_index: int) -> list[str]:
    found = []
    for index in range(max_index):
        cap = dshow_capture(index)
        if not cap.isOpened():
            continue
        ok, frame = cap.read()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if ok and frame is not None:
            channels = 1 if frame.ndim == 2 else frame.shape[2]
            found.append(f"index={index} size={width}x{height} channels={channels} dtype={frame.dtype}")
        else:
            found.append(f"index={index} opened but frame read failed")
        cap.release()
    return found


def decode_depth_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2 and frame.dtype == np.uint16:
        return frame
    if frame.ndim == 2:
        return frame.astype(np.uint16)
    if frame.ndim == 3 and frame.shape[2] == 2 and frame.dtype == np.uint8:
        return frame.view(np.uint16).reshape(frame.shape[0], frame.shape[1])
    if frame.ndim == 3 and frame.shape[2] >= 1:
        if frame.dtype == np.uint16:
            return frame[:, :, 0]
        return frame[:, :, 0].astype(np.uint16)
    raise ValueError(f"Unsupported depth frame format: shape={frame.shape}, dtype={frame.dtype}")


def open_uvc_rgb(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = dshow_capture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RGB camera index {index}.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def open_uvc_depth(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = dshow_capture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open depth camera index {index}.")
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def run_uvc(args):
    rgb_cap = open_uvc_rgb(args.rgb_index, args.width, args.height, args.fps)
    depth_cap = open_uvc_depth(args.depth_index, args.width, args.height, args.fps)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter = AsyncSegmenter(device=device, input_size=args.seg_input_size)
    segmenter.start()

    last_time = time.perf_counter()
    display_fps = 0.0
    seg_fps = 0.0
    last_seg_timestamp = 0.0

    try:
        while True:
            ok_rgb, rgb_frame = rgb_cap.read()
            ok_depth, depth_frame = depth_cap.read()
            if not ok_rgb or rgb_frame is None:
                raise RuntimeError("Failed to read RGB frame from UVC camera.")
            if not ok_depth or depth_frame is None:
                raise RuntimeError("Failed to read depth frame from UVC camera.")

            depth_raw = decode_depth_frame(depth_frame)
            depth_view = colorize_depth_raw(depth_raw)

            segmenter.submit(rgb_frame)
            seg_result = segmenter.get_latest()
            seg_error = segmenter.get_latest_error()
            segmented_view = build_segmented_view(rgb_frame, seg_result, seg_error)

            now = time.perf_counter()
            display_fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            if seg_result is not None and seg_result.updated_at != last_seg_timestamp:
                seg_fps = 1.0 / max(seg_result.updated_at - last_seg_timestamp, 1e-6) if last_seg_timestamp else 0.0
                last_seg_timestamp = seg_result.updated_at

            rgb_view = rgb_frame.copy()
            put_fps(rgb_view, display_fps, "RGB")
            put_fps(depth_view, display_fps, "Depth")
            put_fps(segmented_view, seg_fps, "Seg")

            views = []
            if not args.no_rgb:
                views.append(rgb_view)
            if not args.no_depth:
                views.append(depth_view)
            if not args.no_seg:
                views.append(segmented_view)

            combined = compose_views(views)
            cv2.imshow("RGB | Depth | Segmentation", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        segmenter.stop()
        rgb_cap.release()
        depth_cap.release()
        cv2.destroyAllWindows()


def run_realsense(args):
    if rs is None:
        raise RuntimeError("pyrealsense2 is not installed.")

    visible_devices = list_realsense_devices()
    if not visible_devices:
        raise RuntimeError("No RealSense devices detected by pyrealsense2.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter = AsyncSegmenter(device=device, input_size=args.seg_input_size)
    segmenter.start()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    last_time = time.perf_counter()
    display_fps = 0.0
    seg_fps = 0.0
    last_seg_timestamp = 0.0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            rgb_frame = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale * 1000.0
            depth_view = colorize_depth_raw(depth_raw.astype(np.uint16))

            segmenter.submit(rgb_frame)
            seg_result = segmenter.get_latest()
            seg_error = segmenter.get_latest_error()
            segmented_view = build_segmented_view(rgb_frame, seg_result, seg_error)

            now = time.perf_counter()
            display_fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            if seg_result is not None and seg_result.updated_at != last_seg_timestamp:
                seg_fps = 1.0 / max(seg_result.updated_at - last_seg_timestamp, 1e-6) if last_seg_timestamp else 0.0
                last_seg_timestamp = seg_result.updated_at

            rgb_view = rgb_frame.copy()
            put_fps(rgb_view, display_fps, "RGB")
            put_fps(depth_view, display_fps, "Depth")
            put_fps(segmented_view, seg_fps, "Seg")

            views = []
            if not args.no_rgb:
                views.append(rgb_view)
            if not args.no_depth:
                views.append(depth_view)
            if not args.no_seg:
                views.append(segmented_view)

            combined = compose_views(views)
            cv2.imshow("RGB | Depth | Segmentation", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        segmenter.stop()
        pipeline.stop()
        cv2.destroyAllWindows()


def list_berxel_devices() -> list[str]:
    from berxel_py_warpper import BerxelHawkContext

    ctx = BerxelHawkContext()
    ctx.initCamera()
    try:
        devices = ctx.getDeviceList()
        return [
            f"vid={dev.vendorId} pid={dev.productId} addr={dev.deviceAddress.decode(errors='ignore')}"
            for dev in devices
        ]
    finally:
        ctx.destroyCamera()


def run_berxel(args):
    from berxel_py_warpper import (
        BerxelHawkContext,
        BerxelHawkStreamFlagMode,
        BerxelHawkStreamType,
    )

    ctx = BerxelHawkContext()
    ctx.initCamera()

    device = None
    segmenter = None
    stream_mask = (
        BerxelHawkStreamType.forward_dict["BERXEL_HAWK_DEPTH_STREAM"]
        | BerxelHawkStreamType.forward_dict["BERXEL_HAWK_COLOR_STREAM"]
    )

    try:
        devices = ctx.getDeviceList()
        if not devices:
            raise RuntimeError("No Berxel camera detected by the Berxel SDK.")

        device = ctx.openDevice(devices[0])
        if device is None:
            raise RuntimeError("Berxel SDK detected the camera but failed to open it.")

        device.setDenoiseStatus(False)
        device.setStreamFlagMode(
            BerxelHawkStreamFlagMode.forward_dict["BERXEL_HAWK_MIX_STREAM_FLAG_MODE"]
        )
        device.setRegistrationEnable(True)

        depth_mode = device.getCurrentFrameMode(
            BerxelHawkStreamType.forward_dict["BERXEL_HAWK_DEPTH_STREAM"]
        )
        if depth_mode is not None:
            device.setFrameMode(
                BerxelHawkStreamType.forward_dict["BERXEL_HAWK_DEPTH_STREAM"],
                depth_mode,
            )

        ret = device.startStreams(stream_mask)
        if ret != 0:
            raise RuntimeError("Berxel SDK failed to start color/depth streams.")

        time.sleep(args.warmup_sec)

        compute_device = "cuda" if torch.cuda.is_available() else "cpu"
        segmenter = AsyncSegmenter(device=compute_device, input_size=args.seg_input_size)
        segmenter.start()

        last_time = time.perf_counter()
        display_fps = 0.0
        seg_fps = 0.0
        last_seg_timestamp = 0.0

        while True:
            depth_frame = device.readDepthFrame(args.read_timeout_ms)
            color_frame = device.readColorFrame(args.read_timeout_ms)

            if depth_frame is None or color_frame is None:
                if color_frame is not None:
                    device.releaseFrame(color_frame)
                if depth_frame is not None:
                    device.releaseFrame(depth_frame)
                continue

            try:
                depth_width = depth_frame.getWidth()
                depth_height = depth_frame.getHeight()
                depth_buffer = depth_frame.getDataAsUint16()
                depth_raw = np.ndarray(
                    shape=(depth_height, depth_width),
                    dtype=np.uint16,
                    buffer=depth_buffer,
                ).copy()

                color_width = color_frame.getWidth()
                color_height = color_frame.getHeight()
                color_buffer = color_frame.getDataAsUint8()
                color_rgb = np.ndarray(
                    shape=(color_height, color_width, 3),
                    dtype=np.uint8,
                    buffer=color_buffer,
                ).copy()
                color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            finally:
                device.releaseFrame(color_frame)
                device.releaseFrame(depth_frame)

            depth_view = colorize_depth_raw(depth_raw)
            segmenter.submit(color_bgr)
            seg_result = segmenter.get_latest()
            seg_error = segmenter.get_latest_error()
            segmented_view = build_segmented_view(color_bgr, seg_result, seg_error)

            now = time.perf_counter()
            display_fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            if seg_result is not None and seg_result.updated_at != last_seg_timestamp:
                seg_fps = 1.0 / max(seg_result.updated_at - last_seg_timestamp, 1e-6) if last_seg_timestamp else 0.0
                last_seg_timestamp = seg_result.updated_at

            rgb_view = color_bgr.copy()
            put_fps(rgb_view, display_fps, "RGB")
            put_fps(depth_view, display_fps, "Depth")
            put_fps(segmented_view, seg_fps, "Seg")

            views = []
            if not args.no_rgb:
                views.append(rgb_view)
            if not args.no_depth:
                views.append(depth_view)
            if not args.no_seg:
                views.append(segmented_view)

            combined = compose_views(views)
            cv2.imshow("Berxel RGB | Depth | Segmentation", combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        if segmenter is not None:
            segmenter.stop()
        if device is not None:
            try:
                device.stopStream(stream_mask)
            except Exception:
                pass
            ctx.clsoeDevice(device)
        ctx.destroyCamera()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show RGB, depth, and semantic segmentation from an RGB-D camera."
    )
    parser.add_argument(
        "--backend",
        choices=["berxel", "uvc", "realsense"],
        default="berxel",
        help="Camera backend. Use 'berxel' for Berxel iHawk cameras.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Probe generic UVC camera indices and exit.",
    )
    parser.add_argument(
        "--list-berxel",
        action="store_true",
        help="List devices visible to the Berxel SDK and exit.",
    )
    parser.add_argument(
        "--list-realsense",
        action="store_true",
        help="List RealSense devices visible to pyrealsense2 and exit.",
    )
    parser.add_argument("--rgb-index", type=int, default=0, help="UVC RGB camera index.")
    parser.add_argument("--depth-index", type=int, default=1, help="UVC depth camera index.")
    parser.add_argument("--probe-max-index", type=int, default=10, help="Max UVC index to probe.")
    parser.add_argument("--width", type=int, default=640, help="Stream width in pixels.")
    parser.add_argument("--height", type=int, default=400, help="Stream height in pixels.")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS target.")
    parser.add_argument("--no-rgb", action="store_true", help="Hide the RGB pane.")
    parser.add_argument("--no-depth", action="store_true", help="Hide the depth pane.")
    parser.add_argument("--no-seg", action="store_true", help="Hide the segmentation pane.")
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=1.0,
        help="Seconds to wait after opening Berxel streams before reading frames.",
    )
    parser.add_argument(
        "--read-timeout-ms",
        type=int,
        default=200,
        help="Berxel frame read timeout in milliseconds.",
    )
    parser.add_argument(
        "--seg-input-size",
        type=int,
        default=320,
        help="Segmentation model input size. Lower is faster, higher is sharper.",
    )
    return parser.parse_args()


def main():
    prepare_console()
    args = parse_args()

    if args.no_rgb and args.no_depth and args.no_seg:
        raise SystemExit("All panes are disabled. Enable at least one feed.")

    if args.probe:
        devices = probe_uvc_devices(args.probe_max_index)
        if devices:
            print("UVC devices:")
            for dev in devices:
                print(f"  - {dev}")
        else:
            print("No readable UVC devices found.")
        return

    if args.list_berxel:
        devices = list_berxel_devices()
        if devices:
            print("Berxel devices:")
            for dev in devices:
                print(f"  - {dev}")
        else:
            print("No Berxel devices detected by the Berxel SDK.")
        return

    if args.list_realsense:
        devices = list_realsense_devices()
        if devices:
            print("RealSense devices:")
            for dev in devices:
                print(f"  - {dev}")
        else:
            print("No RealSense devices detected by pyrealsense2.")
        return

    if args.backend == "berxel":
        run_berxel(args)
        return

    if args.backend == "uvc":
        run_uvc(args)
        return

    run_realsense(args)


if __name__ == "__main__":
    main()
