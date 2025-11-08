import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import argparse
import os


class ImageSubscriber(Node):
    def __init__(self, topic_name, display_window=False, record_path=None):
        super().__init__('image_subscriber')
        self.br = CvBridge()
        self.display_window = display_window
        self.record_path = record_path
        self.video_writer = None
        self.video_fps = 20

        # Automatically detect whether topic is compressed
        self.use_compressed = "compressed" in topic_name.lower()
        msg_type = CompressedImage if self.use_compressed else Image

        # Create subscriber
        self.subscription = self.create_subscription(
            msg_type,
            topic_name,
            self.listener_callback,
            10
        )

        self.image_pub = self.create_publisher(Image, "processed_image", 10)

        self.get_logger().info(
            f"Subscribed to: {topic_name} "
            f"({'Compressed' if self.use_compressed else 'Raw'})"
        )

        # Handle recording setup
        if self.record_path:
            os.makedirs(os.path.dirname(self.record_path) or ".", exist_ok=True)
            self.get_logger().info(f"Recording enabled, saving to: {self.record_path}")
        else:
            self.get_logger().info("Running without video recording.")

        self.get_logger().info(f"Current working directory: {os.getcwd()}")

    def listener_callback(self, data):
        """Handle incoming image frames."""
        try:
            if self.use_compressed:
                np_arr = np.frombuffer(data.data, np.uint8)
                current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                current_frame = self.br.imgmsg_to_cv2(data)

            if current_frame is not None and self.display_window:
                cv2.imshow("camera", current_frame)
                cv2.waitKey(1)

            if current_frame is not None:
                ros_image = self.br.cv2_to_imgmsg(current_frame)
                self.image_pub.publish(ros_image)

            # --- Record frame if enabled
            if self.record_path:
                h, w = current_frame.shape[:2]
                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv2.VideoWriter(self.record_path, fourcc, self.video_fps, (w, h))
                    self.get_logger().info("Recording started.")
                self.video_writer.write(current_frame)

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def destroy_node(self):
        """Clean up video writer and OpenCV windows."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f"?? Video saved to {self.record_path}")
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description="ROS2 Image Subscriber")
    parser.add_argument(
        "--topic",
        type=str,
        default="/image_rect/compressed",
        help="Image topic to subscribe to (e.g. /image_raw/compressed or video_frames)"
    )
    parser.add_argument("--display_window", action="store_true",
                        help="Display the image in a window")
    parser.add_argument(
        "--record_path",
        type=str,
        default=None,
        help="Path to save recorded video (e.g. /tmp/camera_feed.avi)"
    )
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = ImageSubscriber(topic_name=parsed_args.topic,
                           display_window=parsed_args.display_window,
                           record_path=parsed_args.record_path)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
