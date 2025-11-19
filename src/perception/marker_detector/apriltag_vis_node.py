#!/usr/bin/env python3
"""
AprilTag Detection Visualization Node (using correct apriltag API)
"""

import time
import rclpy
from rclpy.time import Time as RclTime
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime

# --- your rover's apriltag library ---
from apriltag import apriltag


class AprilTagVisualizationNode(Node):
    def __init__(self, output_dir='apriltag_screenshots',
                 no_gui=False, tag_family='tagStandard41h12'):

        super().__init__('apriltag_visualization')

        self.bridge = CvBridge()
        self.no_gui = no_gui

        self.proc_ema = 0.0
        self.last_msg_stamp = None
        self.last_log_t = time.perf_counter()

        self.screenshot_count = 0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.current_original = None
        self.current_detected = None
        self.frame_count = 0
        self.first_detection = True  # Flag to print keys once

        # parameters
        self.declare_parameter('tag_family', tag_family)
        self.declare_parameter('image_topic', '/image_rect')
        self.declare_parameter('output_dir', output_dir)
        self.declare_parameter('no_gui', no_gui)

        self.tag_family = self.get_parameter('tag_family').value
        image_topic = self.get_parameter('image_topic').value
        self.no_gui = self.get_parameter('no_gui').value

        # --- FIXED: correct detector initialization ---
        self.detector = apriltag(
            self.tag_family,
            threads=4,
            maxhamming=2,
            decimate=0.25,
            blur=0.8,
            refine_edges=True,
            debug=False
        )

        self.get_logger().info(
            f"AprilTag detector initialized: {self.tag_family}")

        self.sub = self.create_subscription(
            Image, image_topic, self.image_callback, 50
        )

        if not self.no_gui:
            cv2.namedWindow("1. Original Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("2. Detected AprilTags", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("1. Original Image", 640, 480)
            cv2.resizeWindow("2. Detected AprilTags", 640, 480)

    def save_screenshots(self):
        if self.current_original is None:
            self.get_logger().warn("No frames yet!")
            return

        self.screenshot_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        f1 = os.path.join(
            self.output_dir, f"shot_{self.screenshot_count:03d}_{ts}_orig.png")
        f2 = os.path.join(
            self.output_dir, f"shot_{self.screenshot_count:03d}_{ts}_det.png")

        cv2.imwrite(f1, self.current_original)
        cv2.imwrite(f2, self.current_detected)

        self.get_logger().info(f"Saved screenshots:\n  {f1}\n  {f2}")

    def image_callback(self, msg):
        self.frame_count += 1

        now = self.get_clock().now()
        msg_t = RclTime.from_msg(msg.header.stamp)

        image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- detect with correct API ---
        detections = self.detector.detect(gray)

        self.current_original = image.copy()
        detected_vis = image.copy()

        for d in detections:
            # Debug: print keys on first detection
            if self.first_detection:
                self.get_logger().info(f"Detection keys: {d.keys()}")
                self.get_logger().info(f"Detection structure: {d}")
                self.first_detection = False

            # Try to extract corners - handle different possible formats
            try:
                # Check if corners are in separate keys 'lb', 'rb', 'rt', 'lt'
                if 'lb' in d:
                    lb = tuple(d['lb'].astype(int))
                    rb = tuple(d['rb'].astype(int))
                    rt = tuple(d['rt'].astype(int))
                    lt = tuple(d['lt'].astype(int))
                # Or if they're in 'lb-rb-rt-lt' format (single key)
                elif 'lb-rb-rt-lt' in d:
                    corners = d['lb-rb-rt-lt']
                    lb = tuple(corners[0].astype(int))
                    rb = tuple(corners[1].astype(int))
                    rt = tuple(corners[2].astype(int))
                    lt = tuple(corners[3].astype(int))
                else:
                    self.get_logger().error(
                        f"Unknown corner format. Keys: {d.keys()}")
                    continue

                # Draw bounding box
                cv2.line(detected_vis, lt, rt, (0, 255, 0), 2)
                cv2.line(detected_vis, rt, rb, (0, 255, 0), 2)
                cv2.line(detected_vis, rb, lb, (0, 255, 0), 2)
                cv2.line(detected_vis, lb, lt, (0, 255, 0), 2)

                # Draw center
                center = tuple(d['center'].astype(int))
                cv2.circle(detected_vis, center, 5, (0, 0, 255), -1)

                # Draw tag info
                tag_id = d['id']
                cv2.putText(
                    detected_vis,
                    f"{self.tag_family} id={tag_id}",
                    (lt[0], lt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2
                )
            except Exception as e:
                self.get_logger().error(f"Error processing detection: {e}")
                continue

        self.current_detected = detected_vis.copy()

        if not self.no_gui:
            cv2.imshow("1. Original Image", self.current_original)
            cv2.imshow("2. Detected AprilTags", self.current_detected)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                self.save_screenshots()
            if key == ord("q"):
                raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag_family", type=str, default="tagStandard41h12")
    parser.add_argument("--output_dir", type=str,
                        default="apriltag_screenshots")
    parser.add_argument("--no-gui", action="store_true")
    parsed, _ = parser.parse_known_args()

    node = AprilTagVisualizationNode(
        output_dir=parsed.output_dir,
        no_gui=parsed.no_gui,
        tag_family=parsed.tag_family
    )

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
