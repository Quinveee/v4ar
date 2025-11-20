#!/usr/bin/env python3
"""
OAK-D Multi Depth Explorer
Compares multiple depth topics side by side
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import zlib
import struct

def decode_compressed_depth(msg):
    # decompress zlib
    compressed_data = msg.data
    decompressed = zlib.decompress(compressed_data)

    # convert to numpy array of 16-bit unsigned
    depth_array = np.frombuffer(decompressed, dtype=np.uint16)

    # reshape according to msg.height and msg.width
    depth_array = depth_array.reshape((msg.height, msg.width))
    return depth_array


class OAKDMultiDepth(Node):
    def __init__(self):
        super().__init__('oakd_multi_depth')
        self.bridge = CvBridge()

        self.topics = [
            '/stereo/depth',
            '/stereo/converted_depth',
            '/stereo/depth/compressed',
            '/stereo/depth/compressedDepth'
        ]

        self.depth_images = {topic: None for topic in self.topics}

        # Subscribers
        for topic in self.topics:
            self.create_subscription(
                Image, topic, lambda msg, t=topic: self.depth_callback(msg, t), 10
            )

        # Timer for display
        self.timer = self.create_timer(0.033, self.display_callback)

        cv2.namedWindow("Multi-Depth Comparison", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Multi-Depth Comparison", 1200, 800)
        self.get_logger().info("Subscribed to multiple depth topics")

    def depth_callback(self, msg, topic):
        try:
            if topic == '/stereo/depth/compressed':
                # JPEG mono8
                np_arr = np.frombuffer(msg.data, np.uint8)
                depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            elif topic == '/stereo/depth/compressedDepth':
                depth = decode_compressed_depth(msg)
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_images[topic] = depth
        except Exception as e:
            self.get_logger().error(f"Error converting {topic}: {e}")

    def normalize_depth(self, depth):
        if depth is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        depth_vis = depth.copy()

        if depth_vis.dtype == np.uint16:
            depth_vis = depth_vis.astype(np.float32) / 1000.0
        elif depth_vis.dtype == np.uint8:
            depth_vis = depth_vis.astype(np.float32)

        depth_vis = np.clip(depth_vis / 5.0 * 255, 0, 255).astype(np.uint8)
        depth_vis_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        if len(depth_vis_color.shape) == 2:
            depth_vis_color = cv2.cvtColor(depth_vis_color, cv2.COLOR_GRAY2BGR)

        return depth_vis_color

    def display_callback(self):
        vis_list = [self.normalize_depth(self.depth_images[t]) for t in self.topics]

        # Ensure all images have the same size (resize to the smallest height and width)
        min_height = min(img.shape[0] for img in vis_list)
        min_width = min(img.shape[1] for img in vis_list)
        vis_list = [cv2.resize(img, (min_width, min_height)) for img in vis_list]

        # Arrange in 2x2 grid
        top = np.hstack(vis_list[:2])
        bottom = np.hstack(vis_list[2:])
        combined = np.vstack([top, bottom])

        cv2.imshow("Multi-Depth Comparison", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Quitting...")
            raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    node = OAKDMultiDepth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
