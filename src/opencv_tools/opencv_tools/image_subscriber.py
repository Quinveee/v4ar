import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import argparse


class ImageSubscriber(Node):
    def __init__(self, topic_name, display_window=False):
        super().__init__('image_subscriber')
        self.br = CvBridge()
        self.display_window = display_window

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

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")


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
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = ImageSubscriber(topic_name=parsed_args.topic,
                           display_window=parsed_args.display_window)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
