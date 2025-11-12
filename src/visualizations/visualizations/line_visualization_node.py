import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from line_msgs.msg import DetectedLines, DetectedLine
from cv_bridge import CvBridge
import cv2
import numpy as np
import os


class VisualizationNode(Node):
    def __init__(self, record_video=None):
        super().__init__('visualization_node')
        self.bridge = CvBridge()
        self.record_path = record_video

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

        # Video recording setup
        self.video_writer = None
        self.video_fps = 20  # estimated capture rate

        if self.record_path:
            os.makedirs(os.path.dirname(self.record_path) or ".", exist_ok=True)
            self.get_logger().info(f"Current working directory: {os.getcwd()}")
            self.get_logger().info(f"Recording will be saved to: {self.record_path}")

        self.get_logger().info("VisualizationNode started — listening to /processed_image, /detected_lines, /selected_line, /cmd_vel")

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
        # arrow for selected line direction
        if self.selected_line:
            ang = self.selected_line.angle
            cv2.arrowedLine(frame, (center_x, h-100),
                    (int(center_x + 80*np.cos(ang)),
                     int(h-100 + 80*np.sin(ang))),
                    (0,255,255), 2, tipLength=0.3)

        # arrow for steering
        steer = - self.current_twist.angular.z
        cv2.arrowedLine(frame, (center_x, h-50),
                (int(center_x + 80*np.sin(steer)),
                 int(h-50 - 80*np.cos(steer))),
                (0,165,255), 3, tipLength=0.3)


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

        # --- Record if requested
        if self.record_path:
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(self.record_path, fourcc, self.video_fps, (w, h))
                self.get_logger().info("Recording started.")
            self.video_writer.write(frame)

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f"Saved recorded video to {self.record_path}")
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description="ROS2 Windowed Visualization")
    parser.add_argument(
        "--record_path",
        type=str,
        default=None,
        help="Path to save the recorded visualization video (e.g., '/tmp/output.avi')."
    )
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = VisualizationNode(record_video=parsed_args.record_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
