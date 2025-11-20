#!/usr/bin/env python3
"""
Complete OAK-D Explorer
Shows: RGB, Depth, Rectified comparison, YOLOv4 detections
Saves everything to files for viewing on laptop
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime
from collections import deque


class CompleteOAKDExplorer(Node):
    def __init__(self):
        super().__init__('complete_oakd_explorer')
        
        # Parameters
        self.declare_parameter('output_dir', 'oakd_complete_exploration')
        self.declare_parameter('save_interval', 3.0)
        self.declare_parameter('show_rectification', True)
        
        self.output_dir = self.get_parameter('output_dir').value
        self.save_interval = self.get_parameter('save_interval').value
        self.show_rectification = self.get_parameter('show_rectification').value
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.bridge = CvBridge()
        
        # Storage
        self.rgb_image = None
        self.rgb_preview = None
        self.depth_image = None
        self.rgb_camera_info = None
        self.depth_camera_info = None
        self.yolo_detections = None
        
        # Rectification maps (will be computed once)
        self.rgb_map1 = None
        self.rgb_map2 = None
        
        # Statistics
        self.frame_count = {'rgb': 0, 'depth': 0, 'yolo': 0}
        self.save_count = 0
        self.depth_stats_history = deque(maxlen=30)
        
        # Subscribers - RGB
        self.rgb_sub = self.create_subscription(
            Image, '/color/image', self.rgb_callback, 10)
        self.rgb_preview_sub = self.create_subscription(
            Image, '/color/preview/image', self.rgb_preview_callback, 10)
        self.rgb_info_sub = self.create_subscription(
            CameraInfo, '/color/camera_info', self.rgb_info_callback, 10)
        
        # Subscribers - Depth
        self.depth_sub = self.create_subscription(
            Image, '/stereo/depth', self.depth_callback, 10)
        self.depth_info_sub = self.create_subscription(
            CameraInfo, '/stereo/camera_info', self.depth_info_callback, 10)
        
        # Subscribers - YOLO
        self.yolo_sub = self.create_subscription(
            Detection2DArray, '/color/yolov4_Spatial_detections', 
            self.yolo_callback, 10)
        
        # Timer for saving/printing
        self.save_timer = self.create_timer(self.save_interval, self.save_all)
        self.stats_timer = self.create_timer(2.0, self.print_stats)
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("Complete OAK-D Explorer Started")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info(f"Save interval: {self.save_interval}s")
        self.get_logger().info(f"Rectification comparison: {self.show_rectification}")
        self.get_logger().info("")
        self.get_logger().info("Subscribed to:")
        self.get_logger().info("  RGB:")
        self.get_logger().info("    - /color/image")
        self.get_logger().info("    - /color/preview/image")
        self.get_logger().info("    - /color/camera_info")
        self.get_logger().info("  Depth:")
        self.get_logger().info("    - /stereo/depth")
        self.get_logger().info("    - /stereo/camera_info")
        self.get_logger().info("  AI:")
        self.get_logger().info("    - /color/yolov4_Spatial_detections")
        self.get_logger().info("=" * 80)
    
    # ================================================================
    # CALLBACKS - RGB
    # ================================================================
    
    def rgb_callback(self, msg: Image):
        """Full resolution RGB"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.frame_count['rgb'] += 1
        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")
    
    def rgb_preview_callback(self, msg: Image):
        """Preview resolution RGB"""
        try:
            self.rgb_preview = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            pass
    
    def rgb_info_callback(self, msg: CameraInfo):
        """RGB camera calibration"""
        if self.rgb_camera_info is None:
            self.rgb_camera_info = msg
            self.get_logger().info(f"RGB Camera: {msg.width}x{msg.height}")
            self.print_camera_calibration("RGB", msg)
            
            # Initialize rectification maps
            if self.show_rectification:
                self.initialize_rectification(msg)
    
    # ================================================================
    # CALLBACKS - DEPTH
    # ================================================================
    
    def depth_callback(self, msg: Image):
        """Depth image"""
        try:
            depth_array = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width)
            self.depth_image = depth_array
            self.frame_count['depth'] += 1
            
            # Calculate statistics
            valid_depth = depth_array[depth_array > 0]
            if len(valid_depth) > 0:
                self.depth_stats_history.append({
                    'min': int(np.min(valid_depth)),
                    'max': int(np.max(valid_depth)),
                    'mean': int(np.mean(valid_depth)),
                    'median': int(np.median(valid_depth)),
                    'valid_percent': len(valid_depth) / depth_array.size * 100
                })
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")
    
    def depth_info_callback(self, msg: CameraInfo):
        """Depth camera calibration"""
        if self.depth_camera_info is None:
            self.depth_camera_info = msg
            self.get_logger().info(f"Depth Camera: {msg.width}x{msg.height}")
            self.print_camera_calibration("DEPTH", msg)
    
    # ================================================================
    # CALLBACKS - YOLO
    # ================================================================
    
    def yolo_callback(self, msg: Detection2DArray):
        """YOLO detections"""
        self.yolo_detections = msg
        self.frame_count['yolo'] += 1
    
    # ================================================================
    # RECTIFICATION
    # ================================================================
    
    def initialize_rectification(self, camera_info):
        """Initialize rectification maps from camera info"""
        K = np.array(camera_info.k).reshape(3, 3)
        D = np.array(camera_info.d)
        P = np.array(camera_info.p).reshape(3, 4)
        R = np.array(camera_info.r).reshape(3, 3)
        
        size = (camera_info.width, camera_info.height)
        
        # Get new camera matrix for rectification
        new_K = P[:3, :3]
        
        # Initialize undistortion maps
        self.rgb_map1, self.rgb_map2 = cv2.initUndistortRectifyMap(
            K, D, R, new_K, size, cv2.CV_32FC1)
        
        self.get_logger().info("Rectification maps initialized")
    
    def rectify_image(self, image):
        """Apply rectification to image"""
        if self.rgb_map1 is None or self.rgb_map2 is None:
            return image
        
        return cv2.remap(image, self.rgb_map1, self.rgb_map2, cv2.INTER_LINEAR)
    
    # ================================================================
    # VISUALIZATION
    # ================================================================
    
    def create_depth_visualization(self):
        """Create colorized depth image with statistics"""
        if self.depth_image is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Normalize depth (mm to meters, 0-5m range)
        depth_normalized = self.depth_image.astype(np.float32) / 1000.0
        depth_normalized = np.clip(depth_normalized / 5.0 * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # Mark invalid pixels black
        mask = self.depth_image == 0
        depth_colored[mask] = [0, 0, 0]
        
        # Add text overlay
        if self.depth_stats_history:
            stats = self.depth_stats_history[-1]
            y = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(depth_colored, f"Depth Range: {stats['min']}-{stats['max']} mm", 
                       (10, y), font, 0.7, (255, 255, 255), 2)
            y += 30
            cv2.putText(depth_colored, f"Mean: {stats['mean']} mm ({stats['mean']/1000:.2f}m)", 
                       (10, y), font, 0.7, (255, 255, 255), 2)
            y += 30
            cv2.putText(depth_colored, f"Valid: {stats['valid_percent']:.1f}%", 
                       (10, y), font, 0.7, (255, 255, 255), 2)
        
        # Add colorbar legend
        cv2.putText(depth_colored, "0m", (10, 700), font, 0.5, (255, 255, 255), 1)
        cv2.putText(depth_colored, "5m", (1220, 700), font, 0.5, (255, 255, 255), 1)
        
        return depth_colored
    
    def create_comparison_view(self):
        """Create side-by-side comparison: original vs rectified"""
        if self.rgb_image is None:
            return np.zeros((480, 1280, 3), dtype=np.uint8)
        
        # Original
        original = self.rgb_image.copy()
        cv2.putText(original, "ORIGINAL (with distortion)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw grid to show distortion
        for i in range(0, original.shape[1], 100):
            cv2.line(original, (i, 0), (i, original.shape[0]), (0, 255, 0), 1)
        for i in range(0, original.shape[0], 100):
            cv2.line(original, (0, i), (original.shape[1], i), (0, 255, 0), 1)
        
        # Rectified
        rectified = self.rectify_image(self.rgb_image.copy())
        cv2.putText(rectified, "RECTIFIED (distortion corrected)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw grid to show straightness
        for i in range(0, rectified.shape[1], 100):
            cv2.line(rectified, (i, 0), (i, rectified.shape[0]), (0, 0, 255), 1)
        for i in range(0, rectified.shape[0], 100):
            cv2.line(rectified, (0, i), (rectified.shape[1], i), (0, 0, 255), 1)
        
        # Combine side by side
        # Resize if needed to fit
        h = min(original.shape[0], rectified.shape[0], 480)
        w = min(original.shape[1], rectified.shape[1], 640)
        
        orig_resized = cv2.resize(original, (w, h))
        rect_resized = cv2.resize(rectified, (w, h))
        
        combined = np.hstack([orig_resized, rect_resized])
        
        return combined
    
    def create_yolo_visualization(self):
        """Create RGB with YOLO detections overlay"""
        if self.rgb_image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        vis = self.rgb_image.copy()
        
        if self.yolo_detections is not None:
            for detection in self.yolo_detections.detections:
                # Get bounding box
                bbox = detection.bbox
                x = int(bbox.center.position.x - bbox.size_x / 2)
                y = int(bbox.center.position.y - bbox.size_y / 2)
                w = int(bbox.size_x)
                h = int(bbox.size_y)
                
                # Draw box
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Get label and confidence
                if detection.results:
                    label = detection.results[0].hypothesis.class_id
                    score = detection.results[0].hypothesis.score
                    text = f"{label}: {score:.2f}"
                    cv2.putText(vis, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add detection count
        count = len(self.yolo_detections.detections) if self.yolo_detections else 0
        cv2.putText(vis, f"Detections: {count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return vis
    
    # ================================================================
    # SAVING & PRINTING
    # ================================================================
    
    def save_all(self):
        """Save all visualizations"""
        if self.rgb_image is None and self.depth_image is None:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_count += 1
        prefix = f"{self.save_count:04d}_{timestamp}"
        
        saved_files = []
        
        # Save RGB original
        if self.rgb_image is not None:
            file = os.path.join(self.output_dir, f"{prefix}_rgb_original.jpg")
            cv2.imwrite(file, self.rgb_image)
            saved_files.append(file)
        
        # Save rectified
        if self.rgb_image is not None and self.show_rectification:
            rectified = self.rectify_image(self.rgb_image)
            file = os.path.join(self.output_dir, f"{prefix}_rgb_rectified.jpg")
            cv2.imwrite(file, rectified)
            saved_files.append(file)
        
        # Save comparison
        if self.show_rectification:
            comparison = self.create_comparison_view()
            file = os.path.join(self.output_dir, f"{prefix}_comparison.jpg")
            cv2.imwrite(file, comparison)
            saved_files.append(file)
        
        # Save depth visualization
        if self.depth_image is not None:
            depth_vis = self.create_depth_visualization()
            file = os.path.join(self.output_dir, f"{prefix}_depth_colored.jpg")
            cv2.imwrite(file, depth_vis)
            saved_files.append(file)
            
            # Save raw depth
            file = os.path.join(self.output_dir, f"{prefix}_depth_raw.png")
            cv2.imwrite(file, self.depth_image)
            saved_files.append(file)
        
        # Save YOLO visualization
        yolo_vis = self.create_yolo_visualization()
        file = os.path.join(self.output_dir, f"{prefix}_yolo_detections.jpg")
        cv2.imwrite(file, yolo_vis)
        saved_files.append(file)
        
        self.get_logger().info(f"Saved {len(saved_files)} files: {prefix}_*")
    
    def print_stats(self):
        """Print statistics"""
        self.get_logger().info("")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"FRAME COUNTS: RGB={self.frame_count['rgb']}, "
                              f"Depth={self.frame_count['depth']}, "
                              f"YOLO={self.frame_count['yolo']}")
        
        if self.depth_stats_history:
            stats = self.depth_stats_history[-1]
            self.get_logger().info(f"DEPTH: {stats['min']}-{stats['max']}mm, "
                                  f"mean={stats['mean']}mm, "
                                  f"valid={stats['valid_percent']:.1f}%")
        
        if self.yolo_detections:
            count = len(self.yolo_detections.detections)
            self.get_logger().info(f"YOLO: {count} objects detected")
        
        self.get_logger().info(f"SAVED: {self.save_count} image sets to {self.output_dir}/")
        self.get_logger().info("=" * 80)
    
    def print_camera_calibration(self, name, info):
        """Print camera calibration parameters"""
        K = np.array(info.k).reshape(3, 3)
        D = np.array(info.d)
        P = np.array(info.p).reshape(3, 4)
        
        self.get_logger().info("")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"{name} CAMERA CALIBRATION")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Resolution: {info.width}x{info.height}")
        self.get_logger().info(f"Distortion model: {info.distortion_model}")
        self.get_logger().info("")
        self.get_logger().info("Intrinsic Matrix K:")
        self.get_logger().info(f"  fx = {K[0,0]:.2f} (focal length X)")
        self.get_logger().info(f"  fy = {K[1,1]:.2f} (focal length Y)")
        self.get_logger().info(f"  cx = {K[0,2]:.2f} (principal point X)")
        self.get_logger().info(f"  cy = {K[1,2]:.2f} (principal point Y)")
        self.get_logger().info("")
        self.get_logger().info("Distortion Coefficients D:")
        if len(D) >= 5:
            self.get_logger().info(f"  k1 = {D[0]:.6f} (radial)")
            self.get_logger().info(f"  k2 = {D[1]:.6f} (radial)")
            self.get_logger().info(f"  p1 = {D[2]:.6f} (tangential)")
            self.get_logger().info(f"  p2 = {D[3]:.6f} (tangential)")
            self.get_logger().info(f"  k3 = {D[4]:.6f} (radial)")
        self.get_logger().info("")
        self.get_logger().info("Why these matter:")
        self.get_logger().info("  - K: Converts world coords → pixel coords")
        self.get_logger().info("  - D: Corrects lens distortion (fisheye effect)")
        self.get_logger().info("  - Used for: distance calc, triangulation, rectification")
        self.get_logger().info("=" * 80)


def main(args=None):
    rclpy.init(args=args)
    node = CompleteOAKDExplorer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f"\nSaved {node.save_count} image sets")
        node.get_logger().info(f"Check: {node.output_dir}/")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()