#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class FakeImage(Node):
    def __init__(self):
        super().__init__("fake_image_pub")
        self.pub = self.create_publisher(Image, "/fake_image", 10)
        self.timer = self.create_timer(0.5, self.tick)

    def tick(self):
        w, h = 160, 120
        data = bytearray(w * h * 3)
        for y in range(h):
            for x in range(w):
                i = (y * w + x) * 3
                data[i + 0] = (x * 2) % 256
                data[i + 1] = (y * 2) % 256
                data[i + 2] = 0
        msg = Image()
        msg.width = w
        msg.height = h
        msg.encoding = "rgb8"
        msg.step = w * 3
        msg.data = data
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FakeImage()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
