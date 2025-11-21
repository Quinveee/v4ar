import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import argparse
import os


class ImageSubscriber(Node):
    def __init__(self, topic_name, display_window=False, record_path=None,
                 custom_rect=False, frame_rate=30.0):
        super().__init__('image_subscriber')
        self.br = CvBridge()
        self.display_window = display_window
        self.record_path = record_path
        self.video_writer = None
        self.video_fps = frame_rate
        self.custom_rect = custom_rect

        self.use_compressed = "compressed" in topic_name.lower()
        msg_type = CompressedImage if self.use_compressed else Image

        self.publish_period = 1.0 / frame_rate
        self.last_pub_time = self.get_clock().now()

        self.subscription = self.create_subscription(
            msg_type,
            topic_name,
            self.mono_callback,
            10
        )

        if self.custom_rect:
            self.K = np.array([
                [286.9896545092208, 0.0, 311.7114840273407],
                [0.0, 290.8395992360502, 249.9287049631703],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

            self.D = np.array([[
                -0.17995587161461585, 0.020688274841999105,
                -0.005297672531455161, 0.003378882156848116, 0.0
            ]], dtype=np.float64)

            self.P = np.array([
                [190.74984649646845, 0.0, 318.0141593176815, 0.0],
                [0.0, 247.35103262891005, 248.37293105876694, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ], dtype=np.float64)

            self.map1_custom = None
            self.map2_custom = None
            self.new_K_custom = None

        self.map1_oak = None
        self.map2_oak = None
        self.new_K_oak = None

        self.image_pub = self.create_publisher(Image, "processed_image", 10)
        self.oak_image_pub = self.create_publisher(
            Image, "/color/rect_image", 10)

        topics = [t[0] for t in self.get_topic_names_and_types()]

        oak_color_topic = '/color/image'
        oak_color_compressed = '/color/image/compressed'
        oak_info_topic = '/color/camera_info'

        if oak_info_topic in topics:
            self.oak_info_sub = self.create_subscription(
                CameraInfo,
                oak_info_topic,
                self.oak_camera_info_callback,
                10
            )
            self.get_logger().info(
                f"Subscribed to OAK CameraInfo: {oak_info_topic}")

            if oak_color_compressed in topics:
                self.oak_use_compressed = True
                self.oak_sub = self.create_subscription(
                    CompressedImage,
                    oak_color_compressed,
                    self.oak_callback,
                    10
                )
                self.get_logger().info(
                    f"Subscribed to OAK image: {oak_color_compressed}")
            elif oak_color_topic in topics:
                self.oak_use_compressed = False
                self.oak_sub = self.create_subscription(
                    Image,
                    oak_color_topic,
                    self.oak_callback,
                    10
                )
                self.get_logger().info(
                    f"Subscribed to OAK image: {oak_color_topic}")
            else:
                self.get_logger().warn("No OAK color image topic found")
        else:
            self.get_logger().warn("OAK camera not available")

        self.get_logger().info(
            f"Subscribed to mono camera: {topic_name} "
            f"({'Compressed' if self.use_compressed else 'Raw'})"
        )

        if self.record_path:
            os.makedirs(os.path.dirname(self.record_path)
                        or ".", exist_ok=True)
            self.get_logger().info(
                f"Recording enabled, saving to: {self.record_path}")
        else:
            self.get_logger().info("Running without video recording.")

    def oak_camera_info_callback(self, msg):
        if self.map1_oak is not None and self.map2_oak is not None:
            return
        try:
            K = np.array(msg.k).reshape(3, 3)
            D = np.array(msg.d)
            P = np.array(msg.p).reshape(3, 4)
            width, height = msg.width, msg.height
            new_K = P[:3, :3]
            self.map1_oak, self.map2_oak = cv2.initUndistortRectifyMap(
                K, D, None, new_K, (width, height), cv2.CV_32FC1
            )
            self.get_logger().info("OAK rectification initialized from CameraInfo.")
        except Exception as e:
            self.get_logger().error(f"Failed OAK rectification: {e}")

    def oak_callback(self, data):
        if self.map1_oak is None or self.map2_oak is None:
            return

        try:
            if self.oak_use_compressed:
                np_arr = np.frombuffer(data.data, np.uint8)
                raw_oak = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                raw_oak = self.br.imgmsg_to_cv2(data, 'bgr8')

            rect_oak = cv2.remap(raw_oak, self.map1_oak,
                                 self.map2_oak, cv2.INTER_LINEAR)

            self.oak_image_pub.publish(
                self.br.cv2_to_imgmsg(rect_oak, encoding='bgr8'))

        except Exception as e:
            self.get_logger().error(f"Error processing OAK frame: {e}")

    def mono_callback(self, data):
        current_time = self.get_clock().now()
        if (current_time - self.last_pub_time).nanoseconds < self.publish_period * 1e9:
            return
        self.last_pub_time = current_time

        try:
            if self.use_compressed and not self.custom_rect:
                np_arr = np.frombuffer(data.data, np.uint8)
                current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            elif not self.use_compressed and not self.custom_rect:
                current_frame = self.br.imgmsg_to_cv2(
                    data, desired_encoding='bgr8')
            elif self.custom_rect:
                if self.use_compressed:
                    np_arr = np.frombuffer(data.data, np.uint8)
                    raw = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                else:
                    raw = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')

                h, w = raw.shape[:2]

                if self.map1_custom is None or self.map2_custom is None:
                    if self.P is not None:
                        self.new_K_custom = self.P[:, :3]
                    else:
                        self.new_K_custom, _ = cv2.getOptimalNewCameraMatrix(
                            self.K, self.D, (w, h), 0.0, (w, h)
                        )
                    self.map1_custom, self.map2_custom = cv2.initUndistortRectifyMap(
                        self.K, self.D, None, self.new_K_custom, (
                            w, h), cv2.CV_32FC1
                    )
                current_frame = cv2.remap(
                    raw, self.map1_custom, self.map2_custom, cv2.INTER_LINEAR)

            if current_frame is not None and self.display_window:
                cv2.imshow("camera", current_frame)
                cv2.waitKey(1)

            if current_frame is not None:
                ros_image = self.br.cv2_to_imgmsg(current_frame)
                self.image_pub.publish(ros_image)

            if self.record_path and current_frame is not None:
                h, w = current_frame.shape[:2]
                if self.video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv2.VideoWriter(
                        self.record_path, fourcc, self.video_fps, (w, h)
                    )
                self.video_writer.write(current_frame)

        except Exception as e:
            self.get_logger().error(f"Error processing mono frame: {e}")

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f"Video saved to {self.record_path}")
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser(description="ROS2 Image Subscriber")
    parser.add_argument("--topic", type=str, default="/image_raw",
                        help="Image topic to subscribe to")
    parser.add_argument("--display_window", action="store_true",
                        help="Display the image in a window")
    parser.add_argument("--record_path", type=str, default=None,
                        help="Path to save recorded video")
    parser.add_argument("--custom_rect", action="store_true",
                        help="Whether to apply custom mono rectification")
    parser.add_argument("--frame_rate", type=float, default=30.0,
                        help="Frame rate for recording video")
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = ImageSubscriber(topic_name=parsed_args.topic,
                           display_window=parsed_args.display_window,
                           record_path=parsed_args.record_path,
                           custom_rect=parsed_args.custom_rect,
                           frame_rate=parsed_args.frame_rate)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, CompressedImage
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import argparse
# import os


# class ImageSubscriber(Node):
#     def __init__(self, topic_name, display_window=False, record_path=None, custom_rect=False, frame_rate=30.0):
#         super().__init__('image_subscriber')
#         self.br = CvBridge()
#         self.display_window = display_window
#         self.record_path = record_path
#         self.video_writer = None
#         self.video_fps = frame_rate
#         self.custom_rect = custom_rect

#         # Automatically detect whether topic is compressed
#         self.use_compressed = "compressed" in topic_name.lower()
#         msg_type = CompressedImage if self.use_compressed else Image

#         self.publish_period = 1.0 / frame_rate
#         self.last_pub_time = self.get_clock().now()

#         # Create subscriber
#         self.subscription = self.create_subscription(
#             msg_type,
#             topic_name,
#             self.listener_callback,
#             10
#         )

#         if self.custom_rect:
#             # Updated calibration parameters
#             self.K = np.array([
#                 [286.9896545092208, 0.0, 311.7114840273407],
#                 [0.0, 290.8395992360502, 249.9287049631703],
#                 [0.0, 0.0, 1.0]
#             ], dtype=np.float64)

#             self.D = np.array([[
#                 -0.17995587161461585, 0.020688274841999105, -
#                 0.005297672531455161, 0.003378882156848116, 0.0
#             ]], dtype=np.float64)

#             # Projection matrix from calibration (3x4)
#             self.P = np.array([
#                 [190.74984649646845, 0.0, 318.0141593176815, 0.0],
#                 [0.0, 247.35103262891005, 248.37293105876694, 0.0],
#                 [0.0, 0.0, 1.0, 0.0]
#             ], dtype=np.float64)

#             # rectification maps
#             self.map1 = None
#             self.map2 = None
#             self.new_K = None

#         self.image_pub = self.create_publisher(Image, "processed_image", 10)

#         self.get_logger().info(
#             f"Subscribed to: {topic_name} "
#             f"({'Compressed' if self.use_compressed else 'Raw'})"
#         )

#         # Handle recording setup
#         if self.record_path:
#             os.makedirs(os.path.dirname(self.record_path)
#                         or ".", exist_ok=True)
#             self.get_logger().info(
#                 f"Recording enabled, saving to: {self.record_path}")
#         else:
#             self.get_logger().info("Running without video recording.")

#         self.get_logger().info(f"Current working directory: {os.getcwd()}")

#     # ------------------------------------------------------------------ #
#     # TODO: implement if nessacery -> Calibration loader
#     # ------------------------------------------------------------------ #
#     def load_calibration(self, yaml_file: str):
#         """
#         Load calibration from YAML.
#         Supports:
#           - nav2 / camera_calibration style:
#               camera_matrix: {data: [...]}
#               distortion_coefficients: {data: [...]}
#               projection_matrix: {data: [...]}
#           - oST style:
#               K: [...]
#               D: [...]
#               P: [...]
#         """
#         try:
#             with open(yaml_file, 'r') as f:
#                 data = yaml.safe_load(f)

#             K = None
#             D = None
#             P = None

#             if 'camera_matrix' in data:
#                 K = np.array(data['camera_matrix']['data'],
#                              dtype=np.float64).reshape(3, 3)
#                 D = np.array(data['distortion_coefficients']['data'],
#                              dtype=np.float64).reshape(1, -1)
#                 if 'projection_matrix' in data:
#                     P = np.array(data['projection_matrix']['data'],
#                                  dtype=np.float64).reshape(3, 4)
#             elif 'K' in data:
#                 K = np.array(data['K'], dtype=np.float64).reshape(3, 3)
#                 D = np.array(data['D'], dtype=np.float64).reshape(1, -1)
#                 if 'P' in data:
#                     P = np.array(data['P'], dtype=np.float64).reshape(3, 4)

#             if K is not None:
#                 self.K = K
#             if D is not None:
#                 self.D = D
#             if P is not None:
#                 self.P = P

#             self.get_logger().info(f"Loaded calibration from {yaml_file}")
#             self.get_logger().info(f"K:\n{self.K}")
#             self.get_logger().info(f"D: {self.D}")
#             if self.P is not None:
#                 self.get_logger().info(f"P:\n{self.P}")
#         except Exception as e:
#             self.get_logger().error(
#                 f"Failed to load calibration from {yaml_file}: {e}")

#     def listener_callback(self, data):
#         """Handle incoming image frames."""
#         current_time = self.get_clock().now()
#         if (current_time - self.last_pub_time).nanoseconds < self.publish_period * 1e9:
#             return  # Skip frame to maintain desired frame rate
#         self.last_pub_time = current_time

#         try:
#             if self.use_compressed and not self.custom_rect:
#                 np_arr = np.frombuffer(data.data, np.uint8)
#                 current_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#             elif not self.use_compressed and not self.custom_rect:
#                 current_frame = self.br.imgmsg_to_cv2(
#                     data, desired_encoding='bgr8')
#             elif self.custom_rect:
#                 # Custom rectification
#                 if self.use_compressed:
#                     np_arr = np.frombuffer(data.data, np.uint8)
#                     raw = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#                 else:
#                     raw = self.br.imgmsg_to_cv2(data, desired_encoding='bgr8')
#                 self.current_original = raw.copy()

#                 h, w = raw.shape[:2]

#                 # Init rectification maps once
#                 if self.map1 is None or self.map2 is None:
#                     if self.P is not None:
#                         # Use projection matrix from calibration as new K
#                         self.new_K = self.P[:, :3]
#                         self.get_logger().info(
#                             "Using P[:,:3] as new camera matrix for rectification.")
#                     else:
#                         # Fallback: compute new_K with alpha=0 (crop to valid region)
#                         self.new_K, _ = cv2.getOptimalNewCameraMatrix(
#                             self.K, self.D, (w, h), 0.0, (w, h)
#                         )
#                         self.get_logger().warn(
#                             "No P found in calibration, using getOptimalNewCameraMatrix(alpha=0).")

#                     self.map1, self.map2 = cv2.initUndistortRectifyMap(
#                         self.K, self.D, None, self.new_K, (w, h), cv2.CV_32FC1
#                     )

#                 current_frame = cv2.remap(
#                     raw, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

#             if current_frame is not None and self.display_window:
#                 cv2.imshow("camera", current_frame)
#                 cv2.waitKey(1)

#             if current_frame is not None:
#                 ros_image = self.br.cv2_to_imgmsg(current_frame)
#                 self.image_pub.publish(ros_image)

#             # --- Record frame if enabled
#             if self.record_path:
#                 h, w = current_frame.shape[:2]
#                 if self.video_writer is None:
#                     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#                     self.video_writer = cv2.VideoWriter(
#                         self.record_path, fourcc, self.video_fps, (w, h))
#                     self.get_logger().info("Recording started.")
#                 self.video_writer.write(current_frame)

#         except Exception as e:
#             self.get_logger().error(f"Error processing frame: {e}")

#     def destroy_node(self):
#         """Clean up video writer and OpenCV windows."""
#         if self.video_writer is not None:
#             self.video_writer.release()
#             self.get_logger().info(f"?? Video saved to {self.record_path}")
#         cv2.destroyAllWindows()
#         super().destroy_node()


# def main(args=None):
#     parser = argparse.ArgumentParser(description="ROS2 Image Subscriber")
#     parser.add_argument(
#         "--topic",
#         type=str,
#         default="/image_raw",
#         help="Image topic to subscribe to (e.g. /image_raw/compressed or video_frames)"
#     )
#     parser.add_argument("--display_window", action="store_true",
#                         help="Display the image in a window")
#     parser.add_argument(
#         "--record_path",
#         type=str,
#         default=None,
#         help="Path to save recorded video (e.g. /tmp/camera_feed.avi)"
#     )
#     parser.add_argument(
#         "--custom_rect",
#         action="store_true",
#         help="Whether to apply custom rectification"
#     )
#     parser.add_argument(
#         "--frame_rate",
#         type=float,
#         default=30.0,
#         help="Frame rate for recording video"
#     )
#     parsed_args, ros_args = parser.parse_known_args()

#     rclpy.init(args=ros_args)

#     node = ImageSubscriber(topic_name=parsed_args.topic,
#                            display_window=parsed_args.display_window,
#                            record_path=parsed_args.record_path, custom_rect=parsed_args.custom_rect,
#                            frame_rate=parsed_args.frame_rate)
#     rclpy.spin(node)

#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()
