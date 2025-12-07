#!/usr/bin/env python3
"""
Depth filtering comparison viewer.
Shows 6 screens: RGB + 4 filter levels + info panel
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np


class DepthComparisonNode(Node):
    """Node to compare different depth filtering methods."""

    def __init__(self):
        super().__init__("depth_comparison_node")
        self.br = CvBridge()
        self.start_time = None
        self.latest_timestamp = None

        self.latest_rgb = None
        self.latest_depth_compressed = None
        
        self.frame_count = 0
        self.rgb_count = 0
        self.depth_count = 0

        self.rgb_subscription = self.create_subscription(
            Image, "/oak/rgb/image_rect", self.rgb_callback, 10
        )

        self.depth_subscription = self.create_subscription(
            CompressedImage,
            "/oak/stereo/image_raw/compressedDepth",
            self.depth_callback,
            10
        )

        cv2.namedWindow("Depth Filter Comparison", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth Filter Comparison", 1920, 1080)

        self.get_logger().info("=" * 60)
        self.get_logger().info("Depth comparison viewer started")
        self.get_logger().info("=" * 60)
        
        self.status_timer = self.create_timer(5.0, self.status_callback)

    def status_callback(self):
        has_rgb = self.latest_rgb is not None
        has_depth = self.latest_depth_compressed is not None
        
        self.get_logger().info(
            f"Status: RGB: {self.rgb_count}, Depth: {self.depth_count}, "
            f"Displaying: {'YES' if has_rgb and has_depth else 'NO'}"
        )

    def rgb_callback(self, msg: Image) -> None:
        try:
            self.rgb_count += 1
            
            if self.rgb_count == 1:
                self.get_logger().info("✅ First RGB frame received!")
            
            if self.start_time is None:
                self.start_time = msg.header.stamp
            self.latest_timestamp = msg.header.stamp
            self.latest_rgb = self.br.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            self.update_display()
        except Exception as e:
            self.get_logger().error(f"Error processing RGB: {e}")

    def depth_callback(self, msg: CompressedImage) -> None:
        try:
            self.depth_count += 1
            
            if self.depth_count == 1:
                self.get_logger().info("✅ First depth frame received!")
            
            if self.start_time is None:
                self.start_time = msg.header.stamp
            self.latest_timestamp = msg.header.stamp
            
            depth_decompressed = self.decompress_depth(msg)
            
            if depth_decompressed is not None:
                self.latest_depth_compressed = depth_decompressed
                
                if self.depth_count == 1:
                    self.get_logger().info(f"✅ Depth decompressed successfully!")
                    self.get_logger().info(f"   Shape: {depth_decompressed.shape}, dtype: {depth_decompressed.dtype}")
                
                self.update_display()
            else:
                if self.depth_count <= 3:
                    self.get_logger().error(f"❌ Failed to decompress depth frame {self.depth_count}")
                
        except Exception as e:
            self.get_logger().error(f"Error in depth callback: {e}")

    def decompress_depth(self, msg: CompressedImage) -> np.ndarray:
        """Decompress compressedDepth format."""
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
                    if self.depth_count <= 3:
                        self.get_logger().error(f"Cannot convert msg.data type: {type(msg.data)}")
                    return None
            
            if data_bytes is None or len(data_bytes) == 0:
                return None
            
            # Try PNG without header first
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
            
            # Try with 12-byte header
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
            if self.depth_count <= 3:
                self.get_logger().error(f"Exception in decompress_depth: {e}")
            return None

    # ========== FILTERING METHODS (FIXED) ==========

    def filter_level_0(self, depth_image: np.ndarray) -> np.ndarray:
        """Level 0: No filtering."""
        return depth_image.copy()

    def filter_level_1(self, depth_image: np.ndarray) -> np.ndarray:
        """Level 1: Median filter 5x5."""
        return cv2.medianBlur(depth_image, 5)

    def filter_level_2(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Level 2: Median + Bilateral.
        FIXED: Convert to uint8 for bilateral filter.
        """
        # Median filter (works with uint16)
        depth_median = cv2.medianBlur(depth_image, 5)
        
        # Convert to uint8 for bilateral filter (divide by 256)
        depth_8u = (depth_median // 256).astype(np.uint8)
        
        # Bilateral filter (requires uint8 or float32)
        depth_bilateral_8u = cv2.bilateralFilter(
            depth_8u, d=9, sigmaColor=75, sigmaSpace=75
        )
        
        # Convert back to uint16 (multiply by 256)
        depth_bilateral = (depth_bilateral_8u.astype(np.uint16) * 256)
        
        return depth_bilateral

    def filter_level_3(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Level 3: Statistical + Median + Bilateral + Inpaint.
        FIXED: Convert to uint8 for bilateral filter and inpainting.
        """
        # Statistical outlier removal
        depth_float = depth_image.astype(np.float32)
        depth_float[depth_float == 0] = np.nan
        
        kernel = np.ones((5, 5), np.float32) / 25
        depth_mean = cv2.filter2D(depth_float, -1, kernel)
        
        diff = np.abs(depth_float - depth_mean)
        valid_diff = diff[~np.isnan(diff)]
        if len(valid_diff) > 0:
            threshold = np.std(valid_diff) * 2
            depth_float[diff > threshold] = np.nan
        
        depth_clean = np.nan_to_num(depth_float, nan=0).astype(np.uint16)
        
        # Median filter
        depth_median = cv2.medianBlur(depth_clean, 5)
        
        # Convert to uint8 for bilateral filter
        depth_8u = (depth_median // 256).astype(np.uint8)
        
        # Bilateral filter
        depth_bilateral_8u = cv2.bilateralFilter(
            depth_8u, d=9, sigmaColor=75, sigmaSpace=75
        )
        
        # Convert back to uint16
        depth_bilateral = (depth_bilateral_8u.astype(np.uint16) * 256)
        
        # Hole filling
        mask = (depth_bilateral == 0).astype(np.uint8)
        if np.any(mask):
            # Convert to uint8 for inpainting
            depth_8u_for_inpaint = (depth_bilateral // 256).astype(np.uint8)
            depth_filled_8u = cv2.inpaint(
                depth_8u_for_inpaint, mask, inpaintRadius=3, flags=cv2.INPAINT_NS
            )
            # Convert back to uint16
            depth_filled = (depth_filled_8u.astype(np.uint16) * 256)
        else:
            depth_filled = depth_bilateral
        
        return depth_filled

    def normalize_depth(self, depth_image: np.ndarray) -> np.ndarray:
        """Normalize depth for visualization."""
        if depth_image is None:
            return None
            
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
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        return depth_colored

    def update_display(self) -> None:
        """Update display with 6 screens."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        target_size = (360, 640)

        try:
            elapsed_seconds = 0.0
            if self.start_time is not None and self.latest_timestamp is not None:
                current_nanos = self.latest_timestamp.sec * int(1e9) + self.latest_timestamp.nanosec
                start_nanos = self.start_time.sec * int(1e9) + self.start_time.nanosec
                elapsed_seconds = (current_nanos - start_nanos) / 1e9

            # RGB
            if self.latest_rgb is not None:
                rgb_display = cv2.resize(self.latest_rgb, (target_size[1], target_size[0]))
                time_text = f"Time: {elapsed_seconds:.2f}s"
                cv2.putText(rgb_display, time_text, (10, 30), font, font_scale, (0, 255, 0), thickness)
                cv2.putText(rgb_display, "RGB", (10, 350), font, font_scale, (0, 255, 0), thickness)
            else:
                rgb_display = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                cv2.putText(rgb_display, "Waiting for RGB", (50, 180), font, 0.8, (255, 255, 255), 2)

            # Depth filters
            if self.latest_depth_compressed is not None:
                depth_level_0 = self.filter_level_0(self.latest_depth_compressed)
                depth_level_1 = self.filter_level_1(self.latest_depth_compressed)
                depth_level_2 = self.filter_level_2(self.latest_depth_compressed)
                depth_level_3 = self.filter_level_3(self.latest_depth_compressed)

                depth_vis_0 = self.normalize_depth(depth_level_0)
                depth_vis_1 = self.normalize_depth(depth_level_1)
                depth_vis_2 = self.normalize_depth(depth_level_2)
                depth_vis_3 = self.normalize_depth(depth_level_3)
            else:
                depth_vis_0 = depth_vis_1 = depth_vis_2 = depth_vis_3 = None

            # Display depth levels
            if depth_vis_0 is not None:
                depth_0_display = cv2.resize(depth_vis_0, (target_size[1], target_size[0]))
                cv2.putText(depth_0_display, "Level 0: No Filter", (10, 30), font, font_scale, (255, 255, 255), thickness)
            else:
                depth_0_display = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                cv2.putText(depth_0_display, "Waiting", (50, 180), font, 0.8, (255, 255, 255), 2)

            if depth_vis_1 is not None:
                depth_1_display = cv2.resize(depth_vis_1, (target_size[1], target_size[0]))
                cv2.putText(depth_1_display, "Level 1: Minimal", (10, 30), font, font_scale, (0, 255, 255), thickness)
                cv2.putText(depth_1_display, "(Median 5x5)", (10, 350), font, font_scale, (0, 255, 255), thickness)
            else:
                depth_1_display = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

            if depth_vis_2 is not None:
                depth_2_display = cv2.resize(depth_vis_2, (target_size[1], target_size[0]))
                cv2.putText(depth_2_display, "Level 2: Standard", (10, 30), font, font_scale, (255, 255, 0), thickness)
                cv2.putText(depth_2_display, "(Median+Bilateral)", (10, 350), font, font_scale, (255, 255, 0), thickness)
            else:
                depth_2_display = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

            if depth_vis_3 is not None:
                depth_3_display = cv2.resize(depth_vis_3, (target_size[1], target_size[0]))
                cv2.putText(depth_3_display, "Level 3: Advanced", (10, 30), font, font_scale, (255, 0, 255), thickness)
                cv2.putText(depth_3_display, "(Full Pipeline)", (10, 350), font, 0.5, (255, 0, 255), 1)
            else:
                depth_3_display = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

            # Info panel
            info_display = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            info_text = [
                "FILTER COMPARISON",
                "",
                f"RGB: {self.rgb_count}",
                f"Depth: {self.depth_count}",
                "",
                "Level 0: Baseline",
                "  No filtering",
                "",
                "Level 1: ~5ms",
                "  Median 5x5",
                "",
                "Level 2: ~20ms",
                "  Median+Bilateral",
                "",
                "Level 3: ~40ms",
                "  Full pipeline",
            ]
            y_offset = 25
            for i, line in enumerate(info_text):
                color = (0, 255, 0) if "COMPARISON" in line else (255, 255, 255)
                scale = 0.7 if "COMPARISON" not in line else 0.8
                cv2.putText(info_display, line, (10, y_offset + i * 20), font, scale, color, 1)

            # Create grid
            row1 = np.hstack([rgb_display, depth_0_display])
            row2 = np.hstack([depth_1_display, depth_2_display])
            row3 = np.hstack([depth_3_display, info_display])
            combined = np.vstack([row1, row2, row3])

            cv2.imshow("Depth Filter Comparison", combined)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error updating display: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    rclpy.init(args=args)
    node = DepthComparisonNode()

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