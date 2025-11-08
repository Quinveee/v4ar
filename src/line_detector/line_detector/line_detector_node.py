import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from line_msgs.msg import DetectedLine, DetectedLines
from .skeleton_detector import SkeletonLineDetector
from .canny_detector import CannyLineDetector
from .brightness_detector import BrightnessLineDetector
from .gradient_detector import GradientLineDetector
from .custom_detector import CustomLineDetector
import cv2
import math
import argparse
import sys

# Available detector classes
DETECTOR_CLASSES = {
    'custom': CustomLineDetector,
    'canny': CannyLineDetector,
    'brightness': BrightnessLineDetector,
    'gradient': GradientLineDetector,
    'skeleton': SkeletonLineDetector,
}


class LineDetectorNode(Node):
    def __init__(self, display_window=False, vignette=False):
        super().__init__('line_detector')
        self.bridge = CvBridge()
        self.display_window = display_window
        self.vignette = vignette

        # Runtime parameter for choosing detector
        self.declare_parameter('detector_type', 'canny')
        detector_type = self.get_parameter('detector_type').value
        self.detector = DETECTOR_CLASSES.get(
            detector_type, CannyLineDetector)(vignette=self.vignette)

        
        # Subscriptions & Publications
        self.sub = self.create_subscription(
            Image,
            '/processed_image',
            self.image_callback,
            10
        )
        self.pub = self.create_publisher(DetectedLines, '/detected_lines', 10)

        self.get_logger().info(
            f" LineDetectorNode started with '{type(self.detector)}' detector "
            f"(Display window = {'ON' if display_window else 'OFF'})"
        )

    # -------------------------------------------------------------------------
    # Callback for image input
    # -------------------------------------------------------------------------
    def image_callback(self, msg: Image):
        image = self.bridge.imgmsg_to_cv2(msg)
        detected_lines = self.detector.detect(image)

        if not detected_lines:
            self.get_logger().debug("No lines detected.")
            if self.display_window:
                cv2.imshow("Detected Lines", image)
                cv2.waitKey(1)
            return

        # Create DetectedLines message
        lines_msg = DetectedLines()
        lines_msg.header = msg.header

        for (x1, y1, x2, y2) in detected_lines:
            det = DetectedLine()
            det.x1, det.y1, det.x2, det.y2 = int(x1), int(y1), int(x2), int(y2)
            line_center_x = (x1 + x2) / 2
            det.offset_x = float(line_center_x - image.shape[1] / 2)
            dx, dy = x2 - x1, y2 - y1
            det.angle = math.atan2(dy, dx)
            lines_msg.lines.append(det)

            # Draw the line on image for visualization
            if self.display_window:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Publish all detected lines
        self.pub.publish(lines_msg)
        self.get_logger().info(
            f"Published {len(lines_msg.lines)} lines on /detected_lines")

        # Display visualization window if enabled
        if self.display_window:
            cv2.imshow("Detected Lines", image)
            cv2.waitKey(1)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main(args=None):
    parser = argparse.ArgumentParser(description="Line Detector Node")
    parser.add_argument("--display_window", action="store_true",
                        help="Display image with detected lines (OpenCV window)")
    parser.add_argument("--vignette", type=bool, default=False,
                        help="Vignette parameter for CustomLineDetector only")
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = LineDetectorNode(
        display_window=parsed_args.display_window, vignette=parsed_args.vignette)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
