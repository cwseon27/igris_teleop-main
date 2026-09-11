#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2

class ShowCompressed(Node):
    def __init__(self):
        super().__init__('show_compressed')
        self.sub_l = self.create_subscription(
            CompressedImage, '/left/image_rect/compressed', self.cb_l, 10
        )
        self.sub_r = self.create_subscription(
            CompressedImage, '/right/image_rect/compressed', self.cb_r, 10
        )

    def _decode(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img

    def cb_l(self, msg):
        img = self._decode(msg)
        if img is not None:
            cv2.imshow("LEFT (compressed)", img)
            cv2.waitKey(1)

    def cb_r(self, msg):
        img = self._decode(msg)
        if img is not None:
            cv2.imshow("RIGHT (compressed)", img)
            cv2.waitKey(1)

def main():
    rclpy.init()
    node = ShowCompressed()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
