#!/usr/bin/env python3
"""
OAK-D Rover Detector with Pose Publishing
Detects black rovers and publishes their 3D pose with distance and angles
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion
from perception_msgs.msg import ObjectPose, ObjectPoseArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


class RoverDetectorWithPose(Node):
    def __init__(self):
        super().__init__('rover_detector_with_pose')
        self.bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('processing_scale', 0.5)
        self.declare_parameter('display_scale', 1.0)
        self.declare_parameter('processing_fps', 15.0)
        self.declare_parameter('min_rover_area', 200)
        self.declare_parameter('depth_roi_size', 10)
        self.declare_parameter('black_threshold', 60)
        self.declare_parameter('publish_visualization', True)
        
        # Get parameters
        self.processing_scale = self.get_parameter('processing_scale').value
        self.display_scale = self.get_parameter('display_scale').value
        self.processing_fps = self.get_parameter('processing_fps').value
        self.min_rover_area = self.get_parameter('min_rover_area').value
        self.depth_roi_size = self.get_parameter('depth_roi_size').value
        self.black_threshold = self.get_parameter('black_threshold').value
        self.publish_visualization = self.get_parameter('publish_visualization').value

        # Camera intrinsics (will be updated from camera_info)
        self.fx = None  # Focal length X
        self.fy = None  # Focal length Y
        self.cx = None  # Principal point X
        self.cy = None  # Principal point Y
        self.camera_info_received = False

        self.current_rgb = None
        self.current_depth = None
        self.cached_display_image = None
        
        # Rover tracking
        self.next_rover_id = 0
        self.tracked_rovers = {}  # {id: last_center}

        # Subscribers
        self.create_subscription(Image, '/color/image', self.rgb_callback, 10)
        self.create_subscription(Image, '/stereo/depth', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/color/camera_info', self.camera_info_callback, 10)

        # Publishers
        self.rover_pose_pub = self.create_publisher(ObjectPoseArray, '/detected_rovers', 10)
        
        if self.publish_visualization:
            self.viz_pub = self.create_publisher(Image, '/rover_detection/visualization', 10)

        # Timers
        processing_period = 1.0 / self.processing_fps
        self.processing_timer = self.create_timer(processing_period, self.process_frame)
        
        if self.publish_visualization:
            self.display_timer = self.create_timer(0.033, self.display_frame)

        # Window (only if not publishing visualization)
        if not self.publish_visualization:
            cv2.namedWindow("Rover Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Rover Detection", 960, 540)
        
        self.frame_count = 0
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("🤖 Rover Detector with Pose Publishing")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Publishing to: /detected_rovers")
        self.get_logger().info(f"Visualization: {'Enabled' if self.publish_visualization else 'Disabled'}")
        self.get_logger().info("=" * 80)

    def camera_info_callback(self, msg: CameraInfo):
        """Extract camera intrinsics for 3D pose calculation"""
        if not self.camera_info_received:
            K = np.array(msg.k).reshape(3, 3)
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]
            self.camera_info_received = True
            
            self.get_logger().info(f"Camera intrinsics received:")
            self.get_logger().info(f"  fx={self.fx:.2f}, fy={self.fy:.2f}")
            self.get_logger().info(f"  cx={self.cx:.2f}, cy={self.cy:.2f}")

    def rgb_callback(self, msg):
        try:
            self.current_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")

    def depth_callback(self, msg):
        try:
            self.current_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width)
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def pixel_to_3d_point(self, u, v, depth_mm):
        """
        Convert pixel coordinates + depth to 3D point in camera frame
        
        Args:
            u, v: Pixel coordinates
            depth_mm: Depth in millimeters
            
        Returns:
            (x, y, z) in meters in camera frame
            z: forward, x: right, y: down
        """
        if not self.camera_info_received or depth_mm == 0:
            return None
        
        z = depth_mm / 1000.0  # Convert mm to meters
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return (x, y, z)

    def calculate_angles(self, u, v):
        """
        Calculate horizontal and vertical angles from camera center
        
        Args:
            u, v: Pixel coordinates
            
        Returns:
            (angle_h, angle_v) in radians
            Positive angle_h = right
            Positive angle_v = up
        """
        if not self.camera_info_received:
            return (0.0, 0.0)
        
        # Angle from optical axis
        angle_h = math.atan((u - self.cx) / self.fx)
        angle_v = math.atan((v - self.cy) / self.fy)
        
        return (angle_h, -angle_v)  # Negate vertical because image Y is down

    def assign_rover_id(self, center, distance):
        """
        Assign persistent ID to rovers using simple nearest-neighbor tracking
        
        Args:
            center: (x, y) pixel coordinates
            distance: Distance in meters
            
        Returns:
            rover_id: Unique integer ID
        """
        # Match to existing rover if close enough (50 pixels)
        best_id = None
        best_dist = float('inf')
        
        for rover_id, last_center in list(self.tracked_rovers.items()):
            dx = center[0] - last_center[0]
            dy = center[1] - last_center[1]
            pixel_dist = math.sqrt(dx**2 + dy**2)
            
            if pixel_dist < 50 and pixel_dist < best_dist:
                best_dist = pixel_dist
                best_id = rover_id
        
        if best_id is not None:
            # Update existing rover
            self.tracked_rovers[best_id] = center
            return best_id
        else:
            # Create new rover
            new_id = self.next_rover_id
            self.next_rover_id += 1
            self.tracked_rovers[new_id] = center
            return new_id

    def create_depth_overlay(self, depth_image, mask):
        """Create colored depth visualization for detected pixels only"""
        depth_normalized = depth_image.astype(np.float32) / 1000.0
        depth_normalized = np.clip(depth_normalized / 5.0 * 255, 0, 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        overlay = np.zeros_like(depth_colored)
        overlay[mask > 0] = depth_colored[mask > 0]
        return overlay

    def process_frame(self):
        """Main processing loop"""
        if self.current_rgb is None or self.current_depth is None:
            return
        
        if not self.camera_info_received:
            self.get_logger().warn("Waiting for camera info...", throttle_duration_sec=5.0)
            return

        self.frame_count += 1

        # ===== DOWNSCALE FOR PROCESSING =====
        rgb_full = self.current_rgb.copy()
        depth_full = self.current_depth.copy()
        
        if self.processing_scale != 1.0:
            proc_height = int(rgb_full.shape[0] * self.processing_scale)
            proc_width = int(rgb_full.shape[1] * self.processing_scale)
            rgb_proc = cv2.resize(rgb_full, (proc_width, proc_height), 
                                 interpolation=cv2.INTER_AREA)
            depth_proc = cv2.resize(depth_full, (proc_width, proc_height), 
                                   interpolation=cv2.INTER_NEAREST)
        else:
            rgb_proc = rgb_full
            depth_proc = depth_full

        # ===== ROVER DETECTION =====
        hsv = cv2.cvtColor(rgb_proc, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, self.black_threshold])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        kernel_size = max(3, int(5 * self.processing_scale))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # ===== FIND CONTOURS =====
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_rovers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_rover_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
            
            if 0 <= cy < depth_proc.shape[0] and 0 <= cx < depth_proc.shape[1]:
                # Get average depth in ROI
                roi_size = int(self.depth_roi_size * self.processing_scale)
                roi_y1 = max(0, cy - roi_size)
                roi_y2 = min(depth_proc.shape[0], cy + roi_size)
                roi_x1 = max(0, cx - roi_size)
                roi_x2 = min(depth_proc.shape[1], cx + roi_size)
                depth_roi = depth_proc[roi_y1:roi_y2, roi_x1:roi_x2]
                valid_depths = depth_roi[depth_roi > 0]
                
                if len(valid_depths) > 0:
                    avg_depth_mm = np.mean(valid_depths)
                    
                    # Scale coordinates back to full resolution
                    scale_factor = 1.0 / self.processing_scale
                    cx_full = int(cx * scale_factor)
                    cy_full = int(cy * scale_factor)
                    x_full = int(x * scale_factor)
                    y_full = int(y * scale_factor)
                    w_full = int(w * scale_factor)
                    h_full = int(h * scale_factor)
                    
                    # Calculate 3D position
                    point_3d = self.pixel_to_3d_point(cx_full, cy_full, avg_depth_mm)
                    
                    if point_3d is not None:
                        # Calculate angles
                        angle_h, angle_v = self.calculate_angles(cx_full, cy_full)
                        
                        # Calculate Euclidean distance
                        x_3d, y_3d, z_3d = point_3d
                        distance = math.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
                        
                        # Assign persistent ID
                        rover_id = self.assign_rover_id((cx_full, cy_full), distance)
                        
                        detected_rovers.append({
                            'id': rover_id,
                            'bbox': (x_full, y_full, w_full, h_full),
                            'center': (cx_full, cy_full),
                            'point_3d': point_3d,
                            'distance': distance,
                            'angle_h': angle_h,
                            'angle_v': angle_v,
                            'area': area * (scale_factor ** 2),
                            'confidence': min(1.0, area / 5000.0)  # Simple confidence based on size
                        })

        # ===== PUBLISH ROVER POSES =====
        rover_array_msg = ObjectPoseArray()
        rover_array_msg.header.stamp = self.get_clock().now().to_msg()
        rover_array_msg.header.frame_id = "oak_rgb_camera_optical_frame"
        
        for rover in detected_rovers:
            rover_msg = ObjectPose()
            rover_msg.id = rover['id']
            
            # Pose in camera frame
            pose = Pose()
            pose.position.x = float(rover['point_3d'][0])
            pose.position.y = float(rover['point_3d'][1])
            pose.position.z = float(rover['point_3d'][2])
            pose.orientation.w = 1.0  # No orientation estimation
            rover_msg.pose = pose
            
            # Distance and angles
            rover_msg.distance = float(rover['distance'])
            rover_msg.angle_horizontal = float(rover['angle_h'])
            rover_msg.angle_vertical = float(rover['angle_v'])
            
            # Pixel coordinates
            x, y, w, h = rover['bbox']
            rover_msg.center_x = float(rover['center'][0])
            rover_msg.center_y = float(rover['center'][1])
            rover_msg.bbox_x_min = float(x)
            rover_msg.bbox_y_min = float(y)
            rover_msg.bbox_x_max = float(x + w)
            rover_msg.bbox_y_max = float(y + h)
            
            # Additional info
            rover_msg.confidence = float(rover['confidence'])
            rover_msg.area = float(rover['area'])
            
            rover_array_msg.rovers.append(rover_msg)
            
            # Log with detailed info
            angle_h_deg = math.degrees(rover['angle_h'])
            angle_v_deg = math.degrees(rover['angle_v'])
            self.get_logger().info(
                f"Rover {rover['id']}: {rover['distance']:.2f}m "
                f"at ({angle_h_deg:+.1f}°, {angle_v_deg:+.1f}°) "
                f"[x={rover['point_3d'][0]:+.2f}, y={rover['point_3d'][1]:+.2f}, z={rover['point_3d'][2]:.2f}]",
                throttle_duration_sec=1.0
            )
        
        self.rover_pose_pub.publish(rover_array_msg)

        # ===== CREATE VISUALIZATION =====
        if self.processing_scale != 1.0:
            mask_full = cv2.resize(mask, (rgb_full.shape[1], rgb_full.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
        else:
            mask_full = mask
        
        depth_overlay = self.create_depth_overlay(depth_full, mask_full)
        alpha = 0.5
        rgb_with_depth = cv2.addWeighted(rgb_full, 1 - alpha, depth_overlay, alpha, 0)
        
        # Draw detections
        for rover in detected_rovers:
            x, y, w, h = rover['bbox']
            cx, cy = rover['center']
            
            # Bounding box
            cv2.rectangle(rgb_with_depth, (x, y), (x + w, y + h), (0, 0, 255), 3)
            
            # Center crosshair
            cross_size = 15
            cv2.line(rgb_with_depth, (cx - cross_size, cy), (cx + cross_size, cy), 
                    (0, 255, 255), 2)
            cv2.line(rgb_with_depth, (cx, cy - cross_size), (cx, cy + cross_size), 
                    (0, 255, 255), 2)
            cv2.circle(rgb_with_depth, (cx, cy), 5, (0, 255, 255), -1)
            
            # Label with distance and angle
            angle_h_deg = math.degrees(rover['angle_h'])
            label = f"Rover {rover['id']}: {rover['distance']:.2f}m @ {angle_h_deg:+.1f}°"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            
            label_x = x
            label_y = y - 15
            if label_y < text_h + 10:
                label_y = y + h + text_h + 10
            
            cv2.rectangle(rgb_with_depth, 
                         (label_x, label_y - text_h - 5),
                         (label_x + text_w + 10, label_y + 5),
                         (0, 0, 0), -1)
            cv2.putText(rgb_with_depth, label, (label_x + 5, label_y),
                       font, font_scale, (0, 0, 255), thickness)
        
        # Info overlay
        info_y = 30
        cv2.putText(rgb_with_depth, f"Rovers: {len(detected_rovers)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Downscale for display
        if self.display_scale != 1.0:
            disp_height = int(rgb_with_depth.shape[0] * self.display_scale)
            disp_width = int(rgb_with_depth.shape[1] * self.display_scale)
            rgb_with_depth = cv2.resize(rgb_with_depth, (disp_width, disp_height),
                                       interpolation=cv2.INTER_AREA)
        
        self.cached_display_image = rgb_with_depth
        
        # Publish visualization
        if self.publish_visualization:
            try:
                viz_msg = self.bridge.cv2_to_imgmsg(rgb_with_depth, encoding='bgr8')
                viz_msg.header.stamp = rover_array_msg.header.stamp
                viz_msg.header.frame_id = rover_array_msg.header.frame_id
                self.viz_pub.publish(viz_msg)
            except Exception as e:
                self.get_logger().error(f"Visualization publish error: {e}")

    def display_frame(self):
        """Display cached result"""
        if self.cached_display_image is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera data...", 
                       (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Rover Detection", blank)
            cv2.waitKey(1)
            return
        
        cv2.imshow("Rover Detection", self.cached_display_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    node = RoverDetectorWithPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()