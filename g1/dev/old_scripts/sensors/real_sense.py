"""
stream_realsense.py
====================

A small convenience wrapper around Intel RealSense SDK (librealsense) that
demonstrates how to:

1. Detect a connected RealSense device.
2. Stream depth + colour (RGB) frames at the same resolution / FPS.
3. Optionally stream the two infrared (IR) channels and the IMU (gyroscope & accelerometer) if the
   selected model supports them (e.g. D435i).
4. Display the images live using OpenCV.
5. Exit cleanly when the user presses the **ESC** or **q** key.

This file is 100 % self-contained – the only runtime dependencies are:

* pyrealsense2  (``pip install pyrealsense2``)
* opencv-python  (``pip install opencv-python``)

No additional helper libraries or ROS runtimes are required.

There is *no* hardware connected inside the execution environment that runs
this script during CI, therefore the *main* clause is guarded so that the
file can be imported without throwing an exception if no camera is present.
When you actually run the script on a machine with a RealSense camera
connected, an OpenCV window will pop up and display the live feed.

Author: OpenAI Codex-CLI helper
"""

from __future__ import annotations

import os
import struct
import sys
import time
from typing import Optional

import cv2
import numpy as np
try:
    import zmq  # type: ignore
except ImportError:
    zmq = None  # type: ignore

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError as exc:  # pragma: no cover – only happens if dependency missing
    raise SystemExit(
        "pyrealsense2 is not installed. Install it with 'pip install pyrealsense2'"
    ) from exc


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def colourise_depth(depth_frame: rs.depth_frame) -> cv2.Mat:
    """Converts a depth frame (16-bit, in millimetres) into an 8-bit BGR image.

    The function normalises the depth range to 0-255 and applies the OpenCV
    *JET* colour map so that closer objects appear red and farther objects
    blue.
    """

    depth_image = np.asanyarray(depth_frame.get_data())
    return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)


def get_first_device(context: rs.context) -> Optional[rs.device]:
    """Return the first RealSense device if any, otherwise *None*."""

    devices = context.query_devices()
    if len(devices) == 0:
        return None
    return devices[0]


def has_display() -> bool:
    """Return True when OpenCV GUI windows are likely to work."""

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def encode_rgbd_payload(
    colour_image: np.ndarray,
    depth_image: np.ndarray,
    depth_scale: float,
) -> tuple[bytes, bytes, bytes]:
    """Encode RGBD data for rgbd_client.py multipart ZeroMQ transport."""

    ok_color, color_jpg = cv2.imencode(".jpg", colour_image)
    ok_depth, depth_png = cv2.imencode(".png", depth_image)
    if not ok_color or not ok_depth:
        raise RuntimeError("Failed to encode RGBD frames for network streaming.")
    return color_jpg.tobytes(), depth_png.tobytes(), struct.pack("f", float(depth_scale))


def build_stream_config(
    width: int,
    height: int,
    fps: int,
    serial: Optional[str],
    enable_infra: bool,
    enable_imu: bool,
) -> rs.config:
    """Create a RealSense stream config for one RGBD profile."""

    config = rs.config()
    if serial:
        config.enable_device(serial)

    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    if enable_infra:
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)

    if enable_imu:
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)

    return config


def fallback_profiles(width: int, height: int, fps: int) -> list[tuple[int, int, int]]:
    """Return requested profile plus conservative fallback RGBD profiles."""

    profiles = [
        (width, height, fps),
        (640, 480, 15),
        (424, 240, 30),
        (424, 240, 15),
        (320, 240, 30),
        (320, 240, 15),
    ]
    unique_profiles: list[tuple[int, int, int]] = []
    for profile in profiles:
        if profile not in unique_profiles:
            unique_profiles.append(profile)
    return unique_profiles


def start_pipeline_with_fallback(
    pipeline: rs.pipeline,
    rgb_width: int,
    rgb_height: int,
    fps: int,
    serial: Optional[str],
    enable_infra: bool,
    enable_imu: bool,
    auto_profile_fallback: bool,
) -> tuple[rs.pipeline_profile, tuple[int, int, int]]:
    """Start a pipeline, optionally trying lower profiles after a failure."""

    profiles = (
        fallback_profiles(rgb_width, rgb_height, fps)
        if auto_profile_fallback
        else [(rgb_width, rgb_height, fps)]
    )
    errors: list[str] = []

    for width, height, profile_fps in profiles:
        config = build_stream_config(width, height, profile_fps, serial, enable_infra, enable_imu)
        try:
            print(f"Trying RealSense streams: {width}x{height}@{profile_fps}")
            return pipeline.start(config), (width, height, profile_fps)
        except RuntimeError as err:
            errors.append(f"{width}x{height}@{profile_fps}: {err}")
            print(f"Profile failed: {width}x{height}@{profile_fps}: {err}")

    raise RuntimeError(
        "Could not start any requested RealSense stream profile. Tried: "
        + "; ".join(errors)
        + ". Check that no other process is using the camera, try --reset, or reconnect the camera."
    )


# ---------------------------------------------------------------------------
# Main streaming routine
# ---------------------------------------------------------------------------


def run(
    rgb_width: int = 640,
    rgb_height: int = 480,
    fps: int = 30,
    enable_infra: bool = False,
    enable_imu: bool = False,
    serial: Optional[str] = None,
    timeout_ms: int = 15000,
    reset: bool = False,
    display: str = "auto",
    publish: bool = False,
    publish_host: str = "*",
    publish_port: int = 5555,
    auto_profile_fallback: bool = True,
):
    """Open a pipeline, start streaming, and display frames."""

    ctx = rs.context()
    device = get_first_device(ctx)

    if device is None:
        raise RuntimeError("No RealSense device found. Plug in a camera and try again.")

    print("Found device:", device.get_info(rs.camera_info.name))
    print("  Serial number:", device.get_info(rs.camera_info.serial_number))
    print("  Firmware ver.:", device.get_info(rs.camera_info.firmware_version))
    if device.supports(rs.camera_info.usb_type_descriptor):
        print("  USB type    :", device.get_info(rs.camera_info.usb_type_descriptor))

    if reset:
        print("Resetting camera hardware; waiting for USB reconnect ...")
        device.hardware_reset()
        time.sleep(5.0)

    show_windows = display == "on" or (display == "auto" and has_display())
    if not show_windows:
        print("Display mode: off (no GUI display detected). Printing frame stats.")

    pipeline = rs.pipeline(ctx)

    # Apply some recommended depth-postprocessing options to improve quality.
    spatial_filter = rs.spatial_filter()  # edge-preserving smoothing
    temporal_filter = rs.temporal_filter()  # reduces depth noise over time

    align_to = rs.stream.color  # align depth to colour coordinate system
    align = rs.align(align_to)

    # Start streaming
    print("Starting pipeline …")
    profile, active_profile = start_pipeline_with_fallback(
        pipeline,
        rgb_width,
        rgb_height,
        fps,
        serial,
        enable_infra,
        enable_imu,
        auto_profile_fallback,
    )
    rgb_width, rgb_height, fps = active_profile
    print(f"Active streams : {rgb_width}x{rgb_height}@{fps}")

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    print(f"Depth scale    : {depth_scale:.8f} m/unit")

    zmq_context = None
    zmq_socket = None
    if publish:
        if zmq is None:
            raise RuntimeError("pyzmq is not installed. Install it with 'pip install pyzmq'.")
        zmq_context = zmq.Context()
        zmq_socket = zmq_context.socket(zmq.PUB)
        endpoint = f"tcp://{publish_host}:{publish_port}"
        zmq_socket.bind(endpoint)
        print(f"ZeroMQ publish : {endpoint}")

    print("Camera intrinsics (colour stream):")
    colour_intr: rs.video_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = colour_intr.get_intrinsics()
    print(f"  Resolution    : {intr.width} × {intr.height}")
    print(f"  Focal length  : fx={intr.fx:.1f}  fy={intr.fy:.1f}")
    print(f"  Principal pt. : cx={intr.ppx:.1f} cy={intr.ppy:.1f}")

    # Main loop ----------------------------------------------------------------
    last_time = time.perf_counter()

    try:
        # Some hosts, especially embedded machines or cameras after hot-plug,
        # need longer than the SDK's 5000 ms default before the first frame set.
        print(f"Waiting for first frameset (timeout {timeout_ms} ms) ...")
        try:
            pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as err:
            raise RuntimeError(
                "No RealSense frames arrived during startup. Common causes: "
                "USB2 connection or weak cable, another process using the camera, "
                "unsupported stream profile, or a camera that needs --reset. "
                "Try: python stream_realsense.py --timeout-ms 30000 --fps 15 "
                "--width 424 --height 240"
            ) from err
        last_time = time.perf_counter()

        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms)
            except RuntimeError as err:
                raise RuntimeError(
                    "No RealSense frames arrived before the timeout. Common causes: "
                    "USB2 connection or weak cable, another process using the camera, "
                    "unsupported stream profile, or a camera that needs --reset. "
                    "Try: python stream_realsense.py --timeout-ms 30000 --fps 15 "
                    "--width 424 --height 240"
                ) from err

            # Align depth to colour so that pixel (u,v) matches
            aligned_frames = align.process(frames)

            depth_frame: rs.depth_frame = aligned_frames.get_depth_frame()
            colour_frame: rs.video_frame = aligned_frames.get_color_frame()

            if not depth_frame or not colour_frame:
                # Should rarely happen, but continue gracefully.
                continue

            # Post-process depth
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)

            # Convert RealSense frames to numpy arrays
            colour_image = np.asanyarray(colour_frame.get_data())  # BGR order
            depth_image = np.asanyarray(depth_frame.get_data())

            if zmq_socket is not None:
                color_jpg, depth_png, depth_scale_bytes = encode_rgbd_payload(
                    colour_image,
                    depth_image,
                    depth_scale,
                )
                zmq_socket.send_multipart([color_jpg, depth_png, depth_scale_bytes])

            # FPS counter
            now = time.perf_counter()
            fps_calc = 1.0 / (now - last_time)
            last_time = now

            if show_windows:
                depth_coloured = colourise_depth(depth_frame)

                # Combine side-by-side for display (make sure both are same height)
                combined = cv2.hconcat([colour_image, depth_coloured])

                cv2.putText(
                    combined,
                    f"FPS: {fps_calc:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("RealSense RGB + Depth", combined)
            else:
                centre_depth_mm = int(depth_image[depth_image.shape[0] // 2, depth_image.shape[1] // 2])
                print(
                    f"FPS: {fps_calc:5.1f} | centre depth: {centre_depth_mm:5d} mm",
                    end="\r",
                    flush=True,
                )

            if enable_infra:
                ir_left = aligned_frames.get_infrared_frame(1)
                ir_right = aligned_frames.get_infrared_frame(2)
                if ir_left and ir_right:
                    ir_left_img = ir_left.get_data()
                    ir_right_img = ir_right.get_data()
                    if show_windows:
                        cv2.imshow("IR-left", ir_left_img)
                        cv2.imshow("IR-right", ir_right_img)

            if enable_imu:
                gyro: rs.motion_frame = frames.first_or_default(rs.stream.gyro)
                accel: rs.motion_frame = frames.first_or_default(rs.stream.accel)
                if gyro and accel:
                    g_data = gyro.as_motion_frame().get_motion_data()
                    a_data = accel.as_motion_frame().get_motion_data()
                    print(
                        f"Gyro [rad/s]: x={g_data.x:+.3f} y={g_data.y:+.3f} z={g_data.z:+.3f} | "
                        f"Accel [m/s²]: x={a_data.x:+.3f} y={a_data.y:+.3f} z={a_data.z:+.3f}",
                        end="\r",
                    )

            if show_windows:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # ESC or q to quit
                    break
    finally:
        print("\nStopping pipeline, closing windows …")
        pipeline.stop()
        if zmq_socket is not None:
            zmq_socket.close(0)
        if zmq_context is not None:
            zmq_context.term()
        if show_windows:
            cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# "python stream_realsense.py" entry-point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple RealSense viewer (colour + depth) written in Python",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--width", type=int, default=640, help="Width of the RGB/depth stream")
    parser.add_argument("--height", type=int, default=480, help="Height of the RGB/depth stream")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--infra", action="store_true", help="Also display the two IR streams")
    parser.add_argument("--imu", action="store_true", help="Print IMU (gyro + accel) readings")
    parser.add_argument("--serial", type=str, default=None, help="Camera serial number to open")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=15000,
        help="How long to wait for each RealSense frameset",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset camera hardware before opening streams",
    )
    parser.add_argument(
        "--display",
        choices=("auto", "on", "off"),
        default="auto",
        help="Open OpenCV windows, disable them, or auto-detect GUI availability",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish RGB + raw depth for rgbd_client.py over ZeroMQ",
    )
    parser.add_argument(
        "--publish-host",
        default="*",
        help="ZeroMQ bind host/address, e.g. '*' or a specific interface IP",
    )
    parser.add_argument(
        "--publish-port",
        type=int,
        default=5555,
        help="ZeroMQ bind TCP port for RGBD publishing",
    )
    parser.add_argument(
        "--no-profile-fallback",
        action="store_true",
        help="Fail immediately instead of trying lower stream profiles after startup errors",
    )

    args = parser.parse_args()

    try:
        run(
            rgb_width=args.width,
            rgb_height=args.height,
            fps=args.fps,
            enable_infra=args.infra,
            enable_imu=args.imu,
            serial=args.serial,
            timeout_ms=args.timeout_ms,
            reset=args.reset,
            display=args.display,
            publish=args.publish,
            publish_host=args.publish_host,
            publish_port=args.publish_port,
            auto_profile_fallback=not args.no_profile_fallback,
        )
    except RuntimeError as err:
        sys.exit(str(err))
