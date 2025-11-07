import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from line_msgs.msg import DetectedLines, DetectedLine
from cv_bridge import CvBridge
import cv2
import numpy as np


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        self.bridge = CvBridge()

        # Subscriptions
        self.image_sub = self.create_subscription(Image, '/processed_image', self.image_callback, 10)
        self.lines_sub = self.create_subscription(DetectedLines, '/detected_lines', self.lines_callback, 10)
        self.selected_line_sub = self.create_subscription(DetectedLine, '/selected_line', self.selected_line_callback, 10)
        self.cmd_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)

        # Buffers for latest data
        self.latest_image = None
        self.detected_lines = []
        self.selected_line = None
        self.current_twist = Twist()

        self.get_logger().info("✅ VisualizationNode started — listening to /processed_image, /detected_lines, /selected_line, /cmd_vel")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg)
        self.render()

    def lines_callback(self, msg):
        self.detected_lines = msg.lines

    def selected_line_callback(self, msg):
        self.selected_line = msg

    def cmd_callback(self, msg):
        self.current_twist = msg

    # -------------------------------------------------------------------------
    def render(self):
        """Render visualization overlays on the incoming camera image."""
        if self.latest_image is None:
            return

        frame = self.latest_image.copy()
        h, w = frame.shape[:2]
        center_x = w // 2

        # --- Draw detected lines (green)
        for line in self.detected_lines:
            x1, y1, x2, y2 = int(line.x1), int(line.y1), int(line.x2), int(line.y2)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # --- Draw selected line (red, thicker)
        if self.selected_line is not None:
            sl = self.selected_line
            cv2.line(frame, (int(sl.x1), int(sl.y1)), (int(sl.x2), int(sl.y2)), (0, 0, 255), 4)
            cv2.putText(frame, "Selected Line", (int(sl.x1), int(sl.y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- Draw image center line (blue)
        cv2.line(frame, (center_x, 0), (center_x, h), (255, 0, 0), 1)

        # --- Draw steering direction arrow (from Twist)
        steering = self.current_twist.angular.z
        forward = self.current_twist.linear.x

        arrow_length = int(100 * abs(steering) + 50)
        start_point = (center_x, h - 50)
        end_point = (
            int(center_x + arrow_length * np.sin(-steering)),
            int(h - 50 - arrow_length * np.cos(-steering))
        )
        cv2.arrowedLine(frame, start_point, end_point, (0, 165, 255), 3, tipLength=0.3)

        # --- Overlay telemetry text
        cv2.putText(frame, f"Speed: {forward:.2f} m/s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Steering: {steering:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detected lines: {len(self.detected_lines)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Show image
        cv2.imshow("Line Visualization", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
