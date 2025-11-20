import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from perception_msgs.msg import MarkerPose, MarkerPoseArray
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
        self.image_sub = self.create_subscription(
            Image, '/processed_image', self.image_callback, 10)
        self.marker_sub = self.create_subscription(
            MarkerPoseArray, '/detected_markers', self.markers_callback, 10)

        # Buffers for latest data
        self.latest_image = None
        self.detected_markers = []
        self.current_twist = Twist()

        # Video recording setup
        self.video_writer = None
        self.video_fps = 30  # estimated capture rate

        if self.record_path:
            os.makedirs(os.path.dirname(self.record_path)
                        or ".", exist_ok=True)
            self.get_logger().info(f"Current working directory: {os.getcwd()}")
            self.get_logger().info(
                f"Recording will be saved to: {self.record_path}")

        self.get_logger().info(
            "VisualizationNode started — listening to /processed_image, /detected_markers, /selected_marker, /cmd_vel")

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg)
        self.render()

    def markers_callback(self, msg):
        self.detected_markers = msg.markers

    def selected_marker_callback(self, msg):
        self.selected_marker = msg

    def cmd_callback(self, msg):
        self.current_twist = msg

    # -------------------------------------------------------------------------

    def render(self):
        if self.latest_image is None:
            return

        frame = self.latest_image.copy()
        h, w = frame.shape[:2]
        center_x = w // 2

        # --- Telemetry ---
        steering = self.current_twist.angular.z
        forward = self.current_twist.linear.x

        cv2.putText(frame, f"Speed: {forward:.2f} m/s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Steering: {steering:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detected markers: {len(self.detected_markers)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Draw markers ---
        for m in self.detected_markers:
            c = m.corners
            lt = (int(c[0]), int(c[1]))
            rt = (int(c[2]), int(c[3]))
            rb = (int(c[4]), int(c[5]))
            lb = (int(c[6]), int(c[7]))

            cv2.line(frame, lt, rt, (0, 255, 0), 2)
            cv2.line(frame, rt, rb, (0, 255, 0), 2)
            cv2.line(frame, rb, lb, (0, 255, 0), 2)
            cv2.line(frame, lb, lt, (0, 255, 0), 2)

            center = (int(m.center_x), int(m.center_y))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"ID {m.id}  Z={m.pose.position.z:.2f}  Dist={m.distance:.2f}",
                (lt[0], lt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            end_point = (
                int(center[0] + 80 * m.pose.position.x),
                int(center[1] - 80 * m.pose.position.z)
            )
            cv2.arrowedLine(frame, center, end_point,
                            (255, 0, 0), 2, tipLength=0.3)

        # --- Show ---
        cv2.imshow("Visualization", frame)
        cv2.waitKey(1)

        # --- Record ---
        if self.record_path:
            h, w = frame.shape[:2]
            if self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(
                    self.record_path, fourcc, self.video_fps, (w, h))
                self.get_logger().info("Recording started.")
            self.video_writer.write(frame)

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(
                f"Saved recorded video to {self.record_path}")
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
