#!/usr/bin/env python3
"""
OAK-D Camera Explorer
Visualizes all OAK-D topics with comprehensive info display
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct


class OAKDExplorer(Node):
    def __init__(self):
        super().__init__('oakd_explorer')
        
        self.bridge = CvBridge()
        
        # Storage for latest messages
        self.depth_image = None
        self.converted_depth = None
        self.camera_info = None
        self.pointcloud = None
        
        # Statistics
        self.depth_stats = {'min': 0, 'max': 0, 'mean': 0, 'valid_pixels': 0}
        self.frame_count = 0
        
        # Subscribers
        self.depth_sub = self.create_subscription(
            Image, '/stereo/depth', self.depth_callback, 10)
        self.converted_depth_sub = self.create_subscription(
            Image, '/stereo/converted_depth', self.converted_depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/stereo/camera_info', self.camera_info_callback, 10)
        self.points_sub = self.create_subscription(
            PointCloud2, '/stereo/points', self.pointcloud_callback, 10)
        
        # Timer for display
        self.timer = self.create_timer(0.033, self.display_callback)  # 30 FPS
        
        # Create windows
        cv2.namedWindow("OAK-D Depth Visualization", cv2.WINDOW_NORMAL)
        cv2.namedWindow("OAK-D Info Dashboard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("OAK-D Depth Visualization", 800, 600)
        cv2.resizeWindow("OAK-D Info Dashboard", 600, 800)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("OAK-D Explorer Started")
        self.get_logger().info("=" * 60)
        self.get_logger().info("Subscribed to:")
        self.get_logger().info("  - /stereo/depth")
        self.get_logger().info("  - /stereo/converted_depth")
        self.get_logger().info("  - /stereo/camera_info")
        self.get_logger().info("  - /stereo/points")
        self.get_logger().info("")
        self.get_logger().info("Press 's' to save images")
        self.get_logger().info("Press 'q' to quit")
        self.get_logger().info("=" * 60)
    
    # ================================================================
    # CALLBACKS
    # ================================================================
    
    def depth_callback(self, msg: Image):
        """Process raw depth image"""
        try:
            # Convert depth image
            if msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            elif msg.encoding == '32FC1':
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            self.depth_image = depth
            self.frame_count += 1
            
            # Calculate statistics
            valid_depth = depth[depth > 0]
            if len(valid_depth) > 0:
                self.depth_stats = {
                    'min': float(np.min(valid_depth)),
                    'max': float(np.max(valid_depth)),
                    'mean': float(np.mean(valid_depth)),
                    'median': float(np.median(valid_depth)),
                    'valid_pixels': len(valid_depth),
                    'total_pixels': depth.size,
                    'encoding': msg.encoding,
                    'width': msg.width,
                    'height': msg.height
                }
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")
    
    def converted_depth_callback(self, msg: Image):
        """Process converted depth image"""
        try:
            self.converted_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Error in converted_depth_callback: {e}")
    
    def camera_info_callback(self, msg: CameraInfo):
        """Process camera info"""
        self.camera_info = msg
    
    def pointcloud_callback(self, msg: PointCloud2):
        """Process point cloud"""
        self.pointcloud = msg
    
    # ================================================================
    # VISUALIZATION
    # ================================================================
    
    def create_depth_visualization(self, depth_image):
        """Create colorized depth visualization"""
        if depth_image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Normalize depth for visualization
        depth_normalized = depth_image.copy()
        
        # Handle different encodings
        if depth_image.dtype == np.uint16:
            # Convert mm to meters, then to 0-255 range
            depth_normalized = depth_normalized.astype(np.float32) / 1000.0  # to meters
            max_depth = 5.0  # Show up to 5 meters
            depth_normalized = np.clip(depth_normalized / max_depth * 255, 0, 255).astype(np.uint8)
        elif depth_image.dtype == np.float32:
            # Already in meters
            max_depth = 5.0
            depth_normalized = np.clip(depth_normalized / max_depth * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # Make invalid pixels (0 depth) black
        mask = depth_image == 0
        depth_colored[mask] = [0, 0, 0]
        
        return depth_colored
    
    def create_info_dashboard(self):
        """Create information dashboard"""
        # Create blank canvas
        dashboard = np.ones((800, 600, 3), dtype=np.uint8) * 240
        
        y = 30
        line_height = 25
        
        def draw_text(text, y_pos, color=(0, 0, 0), font_scale=0.6, thickness=1):
            cv2.putText(dashboard, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            return y_pos + line_height
        
        def draw_header(text, y_pos):
            cv2.putText(dashboard, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return y_pos + line_height + 5
        
        # Title
        cv2.putText(dashboard, "OAK-D Camera Information", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        y += 40
        
        # Depth Image Info
        y = draw_header("DEPTH IMAGE", y)
        if self.depth_image is not None:
            y = draw_text(f"Resolution: {self.depth_stats.get('width', 'N/A')}x{self.depth_stats.get('height', 'N/A')}", y)
            y = draw_text(f"Encoding: {self.depth_stats.get('encoding', 'N/A')}", y)
            y = draw_text(f"Frame Count: {self.frame_count}", y)
            y += 5
            y = draw_text(f"Min Depth: {self.depth_stats.get('min', 0):.3f} {'mm' if self.depth_stats.get('encoding') == '16UC1' else 'm'}", y)
            y = draw_text(f"Max Depth: {self.depth_stats.get('max', 0):.3f} {'mm' if self.depth_stats.get('encoding') == '16UC1' else 'm'}", y)
            y = draw_text(f"Mean Depth: {self.depth_stats.get('mean', 0):.3f} {'mm' if self.depth_stats.get('encoding') == '16UC1' else 'm'}", y)
            y = draw_text(f"Median Depth: {self.depth_stats.get('median', 0):.3f} {'mm' if self.depth_stats.get('encoding') == '16UC1' else 'm'}", y)
            y += 5
            valid = self.depth_stats.get('valid_pixels', 0)
            total = self.depth_stats.get('total_pixels', 1)
            percent = (valid / total * 100) if total > 0 else 0
            y = draw_text(f"Valid Pixels: {valid:,} / {total:,} ({percent:.1f}%)", y)
        else:
            y = draw_text("No depth data received yet", y, color=(0, 0, 255))
        
        y += 10
        
        # Camera Info
        y = draw_header("CAMERA INFO", y)
        if self.camera_info is not None:
            info = self.camera_info
            y = draw_text(f"Frame ID: {info.header.frame_id}", y)
            y = draw_text(f"Resolution: {info.width}x{info.height}", y)
            y += 5
            
            # Intrinsics
            y = draw_text("Intrinsic Matrix (K):", y, font_scale=0.7, thickness=2)
            K = np.array(info.k).reshape(3, 3)
            y = draw_text(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}", y, font_scale=0.5)
            y = draw_text(f"  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}", y, font_scale=0.5)
            y += 5
            
            # Distortion
            y = draw_text("Distortion Coefficients:", y, font_scale=0.7, thickness=2)
            D = info.d
            if len(D) >= 5:
                y = draw_text(f"  k1={D[0]:.4f}  k2={D[1]:.4f}", y, font_scale=0.5)
                y = draw_text(f"  p1={D[2]:.4f}  p2={D[3]:.4f}", y, font_scale=0.5)
                y = draw_text(f"  k3={D[4]:.4f}", y, font_scale=0.5)
            y += 5
            
            # Projection Matrix
            y = draw_text("Projection Matrix (P):", y, font_scale=0.7, thickness=2)
            P = np.array(info.p).reshape(3, 4)
            y = draw_text(f"  [{P[0,0]:.1f}, {P[0,1]:.1f}, {P[0,2]:.1f}, {P[0,3]:.1f}]", y, font_scale=0.4)
            y = draw_text(f"  [{P[1,0]:.1f}, {P[1,1]:.1f}, {P[1,2]:.1f}, {P[1,3]:.1f}]", y, font_scale=0.4)
            y = draw_text(f"  [{P[2,0]:.1f}, {P[2,1]:.1f}, {P[2,2]:.1f}, {P[2,3]:.1f}]", y, font_scale=0.4)
        else:
            y = draw_text("No camera info received yet", y, color=(0, 0, 255))
        
        y += 10
        
        # Point Cloud Info
        y = draw_header("POINT CLOUD", y)
        if self.pointcloud is not None:
            pc = self.pointcloud
            y = draw_text(f"Points: {pc.width * pc.height:,}", y)
            y = draw_text(f"Dimensions: {pc.width}x{pc.height}", y)
            y = draw_text(f"Point Step: {pc.point_step} bytes", y)
            y = draw_text(f"Row Step: {pc.row_step} bytes", y)
            y = draw_text(f"Is Dense: {pc.is_dense}", y)
            y += 5
            y = draw_text("Fields:", y, font_scale=0.7, thickness=2)
            for field in pc.fields:
                y = draw_text(f"  {field.name} ({field.datatype})", y, font_scale=0.5)
        else:
            y = draw_text("No point cloud received yet", y, color=(0, 0, 255))
        
        y += 10
        
        # Instructions
        y = draw_header("CONTROLS", y)
        y = draw_text("'s' - Save images", y, color=(0, 100, 0))
        y = draw_text("'q' - Quit", y, color=(0, 100, 0))
        
        return dashboard
    
    def display_callback(self):
        """Main display loop"""
        # Create depth visualization
        depth_vis = self.create_depth_visualization(self.depth_image)
        
        # Add overlay text
        if self.depth_image is not None:
            # Add depth range indicator
            cv2.putText(depth_vis, "Depth Range: 0m (Black) -> 5m (Red)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add statistics
            stats_text = f"Frame: {self.frame_count} | Valid: {self.depth_stats.get('valid_pixels', 0):,} pixels"
            cv2.putText(depth_vis, stats_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            mean_depth = self.depth_stats.get('mean', 0)
            if self.depth_stats.get('encoding') == '16UC1':
                mean_depth /= 1000.0  # Convert mm to m
            cv2.putText(depth_vis, f"Mean Depth: {mean_depth:.2f}m", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Create info dashboard
        dashboard = self.create_info_dashboard()
        
        # Display
        cv2.imshow("OAK-D Depth Visualization", depth_vis)
        cv2.imshow("OAK-D Info Dashboard", dashboard)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            self.save_images(depth_vis, dashboard)
        elif key == ord('q'):
            self.get_logger().info("Quitting...")
            raise KeyboardInterrupt
    
    def save_images(self, depth_vis, dashboard):
        """Save current visualizations"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        depth_file = f"oakd_depth_{timestamp}.png"
        dashboard_file = f"oakd_info_{timestamp}.png"
        
        cv2.imwrite(depth_file, depth_vis)
        cv2.imwrite(dashboard_file, dashboard)
        
        self.get_logger().info(f"Saved: {depth_file}, {dashboard_file}")


def main(args=None):
    rclpy.init(args=args)
    node = OAKDExplorer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()