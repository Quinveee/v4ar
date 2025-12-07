"""Visualization utilities for displaying ROS image topics with RGB and depth.

Now shows: RGB, Raw Depth (normalized), CompressedDepth (normalized), CompressedDepth (filtered)

Usage: ros2 run var_mapping display_cam
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import argparse


class ImageVisualizationNode(Node):
    """Node to display ROS image topics (RGB and depth) in a 2x2 window."""

    def __init__(
        self,
        rgb_topic: str = "/oak/rgb/image_rect",
        depth_topic: str = "/oak/stereo/image_raw",
    ):
        super().__init__("image_visualization_node")
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.br = CvBridge()
        self.start_time = None
        self.latest_timestamp = None

        # Store latest images for 4 screens
        self.latest_rgb = None
        self.latest_depth_raw = None
        self.latest_depth_compressed = None
        self.latest_depth_filtered = None
        
        self.depth_callback_count = 0
        self.frame_count = 0

        # Create subscriptions
        self.rgb_subscription = self.create_subscription(
            Image, self.rgb_topic, self.rgb_callback, 10
        )

        self.depth_raw_subscription = self.create_subscription(
            Image, "/oak/stereo/image_raw", self.depth_raw_callback, 10
        )

        self.depth_compressed_subscription = self.create_subscription(
            CompressedImage,
            "/oak/stereo/image_raw/compressedDepth",
            self.depth_compressed_callback,
            10
        )

        self.depth_filtered_subscription = self.create_subscription(
            CompressedImage,
            "/oak/stereo/image_raw/compressedDepth/filtered",
            self.depth_filtered_callback,
            10
        )

        cv2.namedWindow("Robot View - 4 Screens", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot View - 4 Screens", 1600, 1200)

        self.get_logger().info("Image visualization node started with 4 screens")

    def rgb_callback(self, msg: Image) -> None:
        try:
            if self.start_time is None:
                self.start_time = msg.header.stamp
            self.latest_timestamp = msg.header.stamp
            self.latest_rgb = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.update_display()
        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")

    def depth_raw_callback(self, msg: Image) -> None:
        try:
            if self.start_time is None:
                self.start_time = msg.header.stamp
            self.latest_timestamp = msg.header.stamp

            depth_image = self.br.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # Apply percentile-based normalization
            depth_float = depth_image.astype(np.float32)
            depth_float[depth_float == 0] = np.nan
            valid_depths = depth_float[~np.isnan(depth_float)]
            
            if len(valid_depths) == 0:
                return
            
            min_depth = np.percentile(valid_depths, 1)
            max_depth = np.percentile(valid_depths, 99)
            
            self.frame_count += 1
            if self.frame_count % 30 == 1:
                self.get_logger().info(
                    f"Raw depth range: {min_depth:.0f}mm to {max_depth:.0f}mm"
                )
            
            depth_float = np.nan_to_num(depth_float, nan=max_depth)
            depth_clipped = np.clip(depth_float, min_depth, max_depth)
            depth_normalized = cv2.normalize(
                depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_normalized = cv2.medianBlur(depth_normalized, 5)
            self.latest_depth_raw = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            self.update_display()

        except Exception as e:
            self.get_logger().error(f"Error processing raw depth: {e}")

    def depth_compressed_callback(self, msg: CompressedImage) -> None:
        """Process compressedDepth with normalization."""
        try:
            depth_image = self.decompress_depth(msg)
            if depth_image is None:
                return
            
            self.latest_depth_compressed = self.normalize_depth(depth_image)
            self.update_display()
            
        except Exception as e:
            self.get_logger().error(f"Error processing compressed depth: {e}")

    def depth_filtered_callback(self, msg: CompressedImage) -> None:
        """Process filtered compressedDepth with normalization."""
        try:
            depth_image = self.decompress_depth(msg)
            if depth_image is None:
                return
            
            self.latest_depth_filtered = self.normalize_depth(depth_image)
            self.update_display()
            
        except Exception as e:
            self.get_logger().error(f"Error processing filtered depth: {e}")

    def normalize_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """Apply percentile-based normalization to depth image."""
        depth_float = depth_image.astype(np.float32)
        depth_float[depth_float == 0] = np.nan
        valid_depths = depth_float[~np.isnan(depth_float)]
        
        if len(valid_depths) == 0:
            return None
        
        min_depth = np.percentile(valid_depths, 1)
        max_depth = np.percentile(valid_depths, 99)
        
        depth_float = np.nan_to_num(depth_float, nan=max_depth)
        depth_clipped = np.clip(depth_float, min_depth, max_depth)
        depth_normalized = cv2.normalize(
            depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_normalized = cv2.medianBlur(depth_normalized, 5)
        return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

    def decompress_depth(self, msg: CompressedImage) -> np.ndarray:
        """
        Decompress compressedDepth format.
        Handles both PNG without header and PNG with 12-byte header.
        """
        try:
            # Convert to bytes
            if isinstance(msg.data, np.ndarray):
                if msg.data.size == 0:
                    return None
                data_bytes = msg.data.tobytes() if msg.data.dtype == np.uint8 else msg.data.astype(np.uint8).tobytes()
            elif isinstance(msg.data, (list, tuple)):
                if len(msg.data) == 0:
                    return None
                data_bytes = bytes(msg.data)
            elif isinstance(msg.data, bytes):
                data_bytes = msg.data
            elif isinstance(msg.data, bytearray):
                data_bytes = bytes(msg.data)
            elif hasattr(msg.data, 'tobytes'):
                if len(msg.data) == 0:
                    return None
                data_bytes = msg.data.tobytes()
            else:
                try:
                    np_arr = np.array(msg.data, dtype=np.uint8)
                    if np_arr.size == 0:
                        return None
                    data_bytes = np_arr.tobytes()
                except:
                    return None
            
            if len(data_bytes) == 0:
                return None
            
            # Try PNG decompression WITHOUT header first
            try:
                np_arr = np.frombuffer(data_bytes, np.uint8)
                if len(np_arr) > 0:
                    png_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    if png_image is not None and len(png_image.shape) == 2:
                        if png_image.dtype == np.uint8:
                            png_image = (png_image.astype(np.uint16) * 256).astype(np.uint16)
                        return png_image
            except:
                pass
            
            # If that fails, try WITH 12-byte header (ROS 1 style)
            if len(data_bytes) >= 12:
                compressed_data = data_bytes[12:]
                
                try:
                    np_arr = np.frombuffer(compressed_data, np.uint8)
                    png_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    if png_image is not None and len(png_image.shape) == 2:
                        if png_image.dtype == np.uint8:
                            png_image = (png_image.astype(np.uint16) * 256).astype(np.uint16)
                        return png_image
                except:
                    pass
            
            return None
            
        except Exception as e:
            self.get_logger().error(f"Error decompressing: {e}")
            return None

    def update_display(self) -> None:
        """Update the display window with 4 screens in a 2x2 grid."""
        if (self.latest_rgb is None and self.latest_depth_raw is None and 
            self.latest_depth_compressed is None and self.latest_depth_filtered is None):
            return

        try:
            # Calculate elapsed time
            if self.start_time is not None and self.latest_timestamp is not None:
                current_nanos = self.latest_timestamp.sec * int(1e9) + self.latest_timestamp.nanosec
                start_nanos = self.start_time.sec * int(1e9) + self.start_time.nanosec
                elapsed_seconds = (current_nanos - start_nanos) / 1e9
            else:
                elapsed_seconds = 0.0

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            target_size = (480, 640)

            # 1. RGB (top-left)
            if self.latest_rgb is not None:
                rgb_display = cv2.resize(self.latest_rgb, (target_size[1], target_size[0]))
                time_text = f"Time: {elapsed_seconds:.2f}s"
                cv2.putText(rgb_display, time_text, (10, 30), font, font_scale, (0, 255, 0), thickness)
                cv2.putText(rgb_display, "RGB", (10, 470), font, font_scale, (0, 255, 0), thickness)
            else:
                rgb_display = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                cv2.putText(rgb_display, "Waiting for RGB...", (50, 240), font, 1, (255, 255, 255), 2)

            # 2. Raw depth (top-right)
            if self.latest_depth_raw is not None:
                depth_raw_display = cv2.resize(self.latest_depth_raw, (target_size[1], target_size[0]))
                cv2.putText(depth_raw_display, "Raw Depth (Normalized)", (10, 470), font, font_scale, (255, 255, 0), thickness)
            else:
                depth_raw_display = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                cv2.putText(depth_raw_display, "Waiting...", (50, 240), font, 1, (255, 255, 255), 2)

            # 3. CompressedDepth (bottom-left)
            if self.latest_depth_compressed is not None:
                depth_comp_display = cv2.resize(self.latest_depth_compressed, (target_size[1], target_size[0]))
                cv2.putText(depth_comp_display, "CompressedDepth (Normalized)", (10, 470), font, font_scale, (255, 0, 255), thickness)
            else:
                depth_comp_display = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                cv2.putText(depth_comp_display, "Waiting...", (50, 240), font, 1, (255, 255, 255), 2)

            # 4. Filtered depth (bottom-right)
            if self.latest_depth_filtered is not None:
                depth_filt_display = cv2.resize(self.latest_depth_filtered, (target_size[1], target_size[0]))
                cv2.putText(depth_filt_display, "CompressedDepth (Filtered)", (10, 470), font, font_scale, (0, 255, 255), thickness)
            else:
                depth_filt_display = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                cv2.putText(depth_filt_display, "Waiting for Filtered...", (50, 240), font, 1, (255, 255, 255), 2)

            # Create 2x2 grid
            top_row = np.hstack([rgb_display, depth_raw_display])
            bottom_row = np.hstack([depth_comp_display, depth_filt_display])
            combined = np.vstack([top_row, bottom_row])

            cv2.imshow("Robot View - 4 Screens", combined)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error updating display: {e}")


def main(args=None):
    parser = argparse.ArgumentParser(description='Display RGB and depth camera feeds')
    parser.add_argument('--rgb-topic', type=str, default='/oak/rgb/image_rect')
    parser.add_argument('--depth-topic', type=str, default='/oak/stereo/image_raw')

    if args is not None:
        filtered_args = [arg for arg in args if not arg.startswith('--ros-args')]
        parsed_args = parser.parse_args(filtered_args)
    else:
        parsed_args = parser.parse_args()

    rclpy.init(args=args)
    node = ImageVisualizationNode(
        rgb_topic=parsed_args.rgb_topic,
        depth_topic=parsed_args.depth_topic
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()