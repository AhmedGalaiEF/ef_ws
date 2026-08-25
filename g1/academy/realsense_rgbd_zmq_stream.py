#!/usr/bin/env python3
"""Publish RealSense RGB-D as JPEG, PNG, depth-scale ZMQ multipart frames."""
import argparse
import struct
import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bind', default='tcp://0.0.0.0:5555')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    pipe = rs.pipeline(); config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipe.start(config)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    context = zmq.Context.instance(); publisher = context.socket(zmq.PUB); publisher.bind(args.bind)
    print('Publishing RGB-D on', args.bind, 'depth scale=', scale)
    try:
        while True:
            frames = pipe.wait_for_frames()
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame: continue
            color = np.asanyarray(color_frame.get_data()); depth = np.asanyarray(depth_frame.get_data())
            ok_color, color_jpeg = cv2.imencode('.jpg', color, [cv2.IMWRITE_JPEG_QUALITY, 85])
            ok_depth, depth_png = cv2.imencode('.png', depth)
            if ok_color and ok_depth: publisher.send_multipart([color_jpeg.tobytes(), depth_png.tobytes(), struct.pack('f', scale)])
    except KeyboardInterrupt: pass
    finally:
        publisher.close(0); context.term(); pipe.stop()

if __name__ == '__main__':
    main()
