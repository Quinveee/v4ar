#!/usr/bin/env python3
"""
Depth filtering node with multiple filter options.

Subscribes to: /oak/stereo/image_raw/compressedDepth (CompressedImage)
Publishes to: /oak/stereo/image_raw/compressedDepth/filtered (Image - Raw 16UC1)

Parameters:
    filter_level: 0 (none), 1 (minimal), 2 (standard), 3 (advanced)

Usage:
    ros2 run var_mapping depth_filter --ros-args -p filter_level:=1
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np


class DepthFilterNode(Node):
    """Node to filter depth images with configurable filter level."""

    def __init__(self):
        super().__init__('depth_filter_node')
        
        # Declare parameter
        self.declare_parameter('filter_level', 1)  # Default: minimal
        
        self.filter_level = self.get_parameter('filter_level').value
        self.br = CvBridge()
        self.frame_count = 0
        
        # Subscribe to compressedDepth
        self.subscription = self.create_subscription(
            CompressedImage,
            '/oak/stereo/image_raw/compressedDepth',
            self.depth_callback,
            10
        )
        
        # === CRITICAL CHANGE: Publish RAW IMAGE to the specific topic requested ===
        # Even though the name says "compressedDepth", the type is now Image (raw)
        self.publisher = self.create_publisher(
            Image,
            '/oak/stereo/image_raw/compressedDepth/filtered',
            10
        )
        
        filter_names = {
            0: "None (passthrough)",
            1: "Minimal (median 5x5)",
            2: "Standard (median + bilateral)",
            3: "Advanced (statistical + bilateral + inpaint)"
        }
        
        self.get_logger().info(
            f"Depth filter node started with filter level {self.filter_level}: "
            f"{filter_names.get(self.filter_level, 'Unknown')}"
        )

    def depth_callback(self, msg: CompressedImage) -> None:
        """Process and filter depth image."""
        try:
            self.frame_count += 1
            
            # 1. Decompress depth using the ROBUST logic
            depth_image = self.decompress_depth(msg)
            
            if depth_image is None:
                # Only warn occasionally to avoid spamming logs
                if self.frame_count % 30 == 0:
                    self.get_logger().warn("Failed to decompress depth image")
                return
            
            # 2. Apply filtering based on level
            if self.filter_level == 0:
                depth_filtered = depth_image  # No filtering
            elif self.filter_level == 1:
                depth_filtered = self.filter_minimal(depth_image)
            elif self.filter_level == 2:
                depth_filtered = self.filter_standard(depth_image)
            elif self.filter_level == 3:
                depth_filtered = self.filter_advanced(depth_image)
            else:
                self.get_logger().error(f"Invalid filter_level: {self.filter_level}")
                return
            
            # 3. Publish as RAW Image (16UC1) to the requested topic
            # This makes the topic '/oak/stereo/image_raw/compressedDepth/filtered' compatible with SLAM
            filtered_msg = self.br.cv2_to_imgmsg(depth_filtered, encoding="16UC1")
            filtered_msg.header = msg.header # Preserve timestamp and frame_id
            self.publisher.publish(filtered_msg)
            
            # Log periodically
            if self.frame_count % 100 == 1:
                self.get_logger().info(
                    f"Processed {self.frame_count} frames with filter level {self.filter_level}"
                )
                
        except Exception as e:
            self.get_logger().error(f"Error in depth callback: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def decompress_depth(self, msg: CompressedImage) -> np.ndarray:
        """
        Decompress compressedDepth format.
        Matched exactly to your working DepthComparisonNode logic.
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
            
            if data_bytes is None or len(data_bytes) == 0:
                return None
            
            # Try PNG without header first (Standard ROS 2 approach)
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
            
            # If that failed, try with 12-byte header (Legacy/ROS 1 approach)
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
            # Only log errors occasionally
            return None

    def filter_minimal(self, depth_image: np.ndarray) -> np.ndarray:
        """Level 1: Minimal filtering (Median 5x5)."""
        return cv2.medianBlur(depth_image, 5)

    def filter_standard(self, depth_image: np.ndarray) -> np.ndarray:
        """Level 2: Standard filtering (Median + Bilateral)."""
        # Median filter
        depth_median = cv2.medianBlur(depth_image, 5)
        
        # Convert to uint8 for bilateral (divide by 256)
        depth_8u = (depth_median // 256).astype(np.uint8)
        
        # Bilateral filter
        depth_bilateral_8u = cv2.bilateralFilter(
            depth_8u, d=9, sigmaColor=75, sigmaSpace=75
        )
        
        # Convert back to uint16
        depth_bilateral = (depth_bilateral_8u.astype(np.uint16) * 256)
        
        return depth_bilateral

    def filter_advanced(self, depth_image: np.ndarray) -> np.ndarray:
        """Level 3: Advanced filtering (Statistical + Median + Bilateral + Inpaint)."""
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
        
        # Bilateral filter
        depth_8u = (depth_median // 256).astype(np.uint8)
        depth_bilateral_8u = cv2.bilateralFilter(
            depth_8u, d=9, sigmaColor=75, sigmaSpace=75
        )
        depth_bilateral = (depth_bilateral_8u.astype(np.uint16) * 256)
        
        # Hole filling (Inpainting)
        mask = (depth_bilateral == 0).astype(np.uint8)
        if np.any(mask):
            depth_8u_for_inpaint = (depth_bilateral // 256).astype(np.uint8)
            depth_filled_8u = cv2.inpaint(
                depth_8u_for_inpaint, mask, inpaintRadius=3, flags=cv2.INPAINT_NS
            )
            depth_filled = (depth_filled_8u.astype(np.uint16) * 256)
        else:
            depth_filled = depth_bilateral
        
        return depth_filled


def main(args=None):
    rclpy.init(args=args)
    node = DepthFilterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()