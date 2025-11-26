#!/usr/bin/env python3
"""
OAK-D Rover Detector with Pose Publishing - COMPUTE OPTIMIZED
Multi-scale detection with adaptive frequency and distance filtering
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from perception_msgs.msg import ObjectPose, ObjectPoseArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import copy


class RoverDetectorWithPose(Node):
    def __init__(self):
        super().__init__('rover_detector_with_pose')
        self.bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('low_res_fps', 15.0)
        self.declare_parameter('high_res_fps', 5.0)
        self.declare_parameter('min_rover_area', 200)
        self.declare_parameter('depth_roi_size', 10)
        self.declare_parameter('black_threshold', 60)
        self.declare_parameter('max_distance', 4.0)
        self.declare_parameter('enable_buffer', True)
        self.declare_parameter('buffer_max_age', 0.5)
        self.declare_parameter('buffer_alpha', 0.7)
        self.declare_parameter('no_gui', False)  # NEW: GUI control
        
        # Get parameters
        self.low_res_fps = self.get_parameter('low_res_fps').value
        self.high_res_fps = self.get_parameter('high_res_fps').value
        self.min_rover_area = self.get_parameter('min_rover_area').value
        self.depth_roi_size = self.get_parameter('depth_roi_size').value
        self.black_threshold = self.get_parameter('black_threshold').value
        self.max_distance = self.get_parameter('max_distance').value
        self.enable_buffer = self.get_parameter('enable_buffer').value
        self.buffer_max_age = self.get_parameter('buffer_max_age').value
        self.buffer_alpha = self.get_parameter('buffer_alpha').value
        self.no_gui = self.get_parameter('no_gui').value

        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_info_received = False

        self.current_rgb_msg = None
        self.current_depth = None
        self.cached_visualization = None  # Store for GUI display
        
        # Detection state
        self.current_detections = {}
        self.rover_buffer = {}
        self.next_rover_id = 0

        # Subscribers
        self.create_subscription(Image, '/oak/stereo/image_raw', self.rgb_callback, 10)
        # self.create_subscription(Image, '/oak/stereo/image_raw/compressed', self.rgb_callback, 10)
        self.create_subscription(Image, '/oak/depth/image_raw', self.depth_callback, 10)
        self.create_subscription(CameraInfo, '/color/camera_info', self.camera_info_callback, 10)

        # Publishers
        self.rover_pose_pub = self.create_publisher(ObjectPoseArray, '/detected_rovers', 10)
        self.viz_pub = self.create_publisher(Image, '/rover_detection/visualization', 10)

        # Timers - dual frequency
        low_res_period = 1.0 / self.low_res_fps
        high_res_period = 1.0 / self.high_res_fps
        
        self.low_res_timer = self.create_timer(low_res_period, self.process_low_res)
        self.high_res_timer = self.create_timer(high_res_period, self.process_high_res)
        
        # GUI setup
        if not self.no_gui:
            cv2.namedWindow("Rover Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Rover Detection", 960, 540)
            # Display timer at 30 FPS
            self.display_timer = self.create_timer(0.033, self.display_frame)
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("🤖 OPTIMIZED Rover Detector with Multi-Scale Detection")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Low-res tracking: {self.low_res_fps} Hz @ 320px")
        self.get_logger().info(f"High-res pose: {self.high_res_fps} Hz @ 640px")
        self.get_logger().info(f"Max distance: {self.max_distance}m")
        self.get_logger().info(f"Buffering: {'Enabled' if self.enable_buffer else 'Disabled'}")
        self.get_logger().info(f"GUI: {'Disabled' if self.no_gui else 'Enabled'}")
        self.get_logger().info("=" * 80)

    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            K = np.array(msg.k).reshape(3, 3)
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]
            self.camera_info_received = True
            self.get_logger().info(f"Camera intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}")

    def rgb_callback(self, msg):
        self.current_rgb_msg = msg

    def depth_callback(self, msg):
        try:
            self.current_depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width)
        except Exception as e:
            self.get_logger().error(f"Depth error: {e}")

    def pixel_to_3d_point(self, u, v, depth_mm):
        """Convert pixel + depth to 3D point"""
        if not self.camera_info_received or depth_mm == 0:
            return None
        
        z = depth_mm / 1000.0
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return (x, y, z)

    def calculate_angles(self, u, v):
        """Calculate horizontal/vertical angles from center"""
        if not self.camera_info_received:
            return (0.0, 0.0)
        
        angle_h = math.atan((u - self.cx) / self.fx)
        angle_v = math.atan((v - self.cy) / self.fy)
        
        return (angle_h, -angle_v)

    def assign_rover_id(self, center, distance):
        """Assign persistent ID using nearest-neighbor tracking"""
        best_id = None
        best_dist = float('inf')
        
        for rover_id, buffer_data in self.rover_buffer.items():
            last_center = (buffer_data['rover'].center_x, buffer_data['rover'].center_y)
            dx = center[0] - last_center[0]
            dy = center[1] - last_center[1]
            pixel_dist = math.sqrt(dx**2 + dy**2)
            
            if pixel_dist < 50 and pixel_dist < best_dist:
                best_dist = pixel_dist
                best_id = rover_id
        
        if best_id is not None:
            return best_id
        else:
            new_id = self.next_rover_id
            self.next_rover_id += 1
            return new_id

    def detect_at_resolution(self, target_width):
        """Run rover detection at specified resolution"""
        if self.current_rgb_msg is None or self.current_depth is None:
            return []
        
        if not self.camera_info_received:
            return []
        
        try:
            rgb_full = self.bridge.imgmsg_to_cv2(self.current_rgb_msg, 'bgr8')
            depth_full = self.current_depth
            
            scale_factor = target_width / rgb_full.shape[1]
            new_height = int(rgb_full.shape[0] * scale_factor)
            
            rgb_scaled = cv2.resize(rgb_full, (target_width, new_height), 
                                   interpolation=cv2.INTER_LINEAR)
            depth_scaled = cv2.resize(depth_full, (target_width, new_height), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Black detection
            hsv = cv2.cvtColor(rgb_scaled, cv2.COLOR_BGR2HSV)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, self.black_threshold])
            mask = cv2.inRange(hsv, lower_black, upper_black)
            
            # Morphology
            kernel_size = max(3, int(5 * scale_factor))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_rover_area * (scale_factor ** 2):
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
                
                # Get depth
                roi_size = int(self.depth_roi_size * scale_factor)
                roi_y1 = max(0, cy - roi_size)
                roi_y2 = min(depth_scaled.shape[0], cy + roi_size)
                roi_x1 = max(0, cx - roi_size)
                roi_x2 = min(depth_scaled.shape[1], cx + roi_size)
                depth_roi = depth_scaled[roi_y1:roi_y2, roi_x1:roi_x2]
                valid_depths = depth_roi[depth_roi > 0]
                
                if len(valid_depths) == 0:
                    continue
                
                avg_depth_mm = np.median(valid_depths)
                
                # Scale back to full resolution
                inv_scale = 1.0 / scale_factor
                cx_full = int(cx * inv_scale)
                cy_full = int(cy * inv_scale)
                x_full = int(x * inv_scale)
                y_full = int(y * inv_scale)
                w_full = int(w * inv_scale)
                h_full = int(h * inv_scale)
                
                # Calculate 3D position
                point_3d = self.pixel_to_3d_point(cx_full, cy_full, avg_depth_mm)
                
                if point_3d is None:
                    continue
                
                x_3d, y_3d, z_3d = point_3d
                distance = math.sqrt(x_3d**2 + y_3d**2 + z_3d**2)
                
                # Distance filter
                if distance > self.max_distance:
                    continue
                
                angle_h, angle_v = self.calculate_angles(cx_full, cy_full)
                
                detections.append({
                    'bbox': (x_full, y_full, w_full, h_full),
                    'center': (cx_full, cy_full),
                    'point_3d': point_3d,
                    'distance': distance,
                    'angle_h': angle_h,
                    'angle_v': angle_v,
                    'area': area * (inv_scale ** 2),
                    'resolution': target_width
                })
            
            return detections
            
        except Exception as e:
            self.get_logger().error(f"Detection error: {e}")
            return []

    def process_low_res(self):
        """Fast low-res detection for tracking"""
        detections = self.detect_at_resolution(320)
        
        for det in detections:
            pos_hash = (int(det['center'][0] / 50), int(det['center'][1] / 50))
            
            if pos_hash not in self.current_detections:
                self.current_detections[pos_hash] = {
                    'detection': det,
                    'source': 'low_res'
                }

    def process_high_res(self):
        """High-res detection with pose calculation and publishing"""
        detections = self.detect_at_resolution(640)
        
        # Override with high-res detections
        for det in detections:
            pos_hash = (int(det['center'][0] / 50), int(det['center'][1] / 50))
            self.current_detections[pos_hash] = {
                'detection': det,
                'source': 'high_res'
            }
        
        # Process and publish
        self._calculate_and_publish_poses()
        
        # Clear for next cycle
        self.current_detections.clear()

    def _calculate_and_publish_poses(self):
        """Calculate poses, apply buffering, and publish"""
        now = self.get_clock().now().nanoseconds / 1e9
        current_frame_ids = set()
        
        # Process detections
        for pos_hash, data in self.current_detections.items():
            det = data['detection']
            
            rover_id = self.assign_rover_id(det['center'], det['distance'])
            
            rover_msg = ObjectPose()
            rover_msg.id = rover_id
            
            pose = Pose()
            pose.position.x = float(det['point_3d'][0])
            pose.position.y = float(det['point_3d'][1])
            pose.position.z = float(det['point_3d'][2])
            pose.orientation.w = 1.0
            rover_msg.pose = pose
            
            rover_msg.distance = float(det['distance'])
            rover_msg.angle_horizontal = float(det['angle_h'])
            rover_msg.angle_vertical = float(det['angle_v'])
            
            x, y, w, h = det['bbox']
            rover_msg.center_x = float(det['center'][0])
            rover_msg.center_y = float(det['center'][1])
            rover_msg.bbox_x_min = float(x)
            rover_msg.bbox_y_min = float(y)
            rover_msg.bbox_x_max = float(x + w)
            rover_msg.bbox_y_max = float(y + h)
            rover_msg.area = float(det['area'])
            rover_msg.confidence = min(1.0, det['area'] / 5000.0)
            
            # Buffering
            if self.enable_buffer and rover_id in self.rover_buffer:
                prev_rover = self.rover_buffer[rover_id]['rover']
                
                rover_msg.distance = (
                    self.buffer_alpha * rover_msg.distance +
                    (1.0 - self.buffer_alpha) * prev_rover.distance
                )
                
                rover_msg.pose.position.x = (
                    self.buffer_alpha * rover_msg.pose.position.x +
                    (1.0 - self.buffer_alpha) * prev_rover.pose.position.x
                )
                rover_msg.pose.position.y = (
                    self.buffer_alpha * rover_msg.pose.position.y +
                    (1.0 - self.buffer_alpha) * prev_rover.pose.position.y
                )
                rover_msg.pose.position.z = (
                    self.buffer_alpha * rover_msg.pose.position.z +
                    (1.0 - self.buffer_alpha) * prev_rover.pose.position.z
                )
            
            self.rover_buffer[rover_id] = {
                'rover': copy.deepcopy(rover_msg),
                'last_seen': now
            }
            
            current_frame_ids.add(rover_id)
        
        # Remove stale rovers
        ids_to_remove = [rid for rid, data in self.rover_buffer.items() 
                        if (now - data['last_seen']) > self.buffer_max_age]
        for rid in ids_to_remove:
            del self.rover_buffer[rid]
            self.get_logger().info(f"Rover {rid} timed out")
        
        # Build message
        rover_array_msg = ObjectPoseArray()
        rover_array_msg.header.stamp = self.get_clock().now().to_msg()
        rover_array_msg.header.frame_id = "oak_rgb_camera_optical_frame"
        
        if self.enable_buffer:
            for rover_id, buffer_data in self.rover_buffer.items():
                rover_array_msg.rovers.append(buffer_data['rover'])
        else:
            for rover_id in current_frame_ids:
                if rover_id in self.rover_buffer:
                    rover_array_msg.rovers.append(self.rover_buffer[rover_id]['rover'])
        
        self.rover_pose_pub.publish(rover_array_msg)
        
        # Log
        for rover in rover_array_msg.rovers:
            angle_deg = math.degrees(rover.angle_horizontal)
            self.get_logger().info(
                f"Rover {rover.id}: {rover.distance:.2f}m @ {angle_deg:+.1f}° "
                f"[{rover.pose.position.x:+.2f}, {rover.pose.position.y:+.2f}, {rover.pose.position.z:.2f}]",
                throttle_duration_sec=1.0
            )
        
        # Create and cache visualization
        self._create_visualization(rover_array_msg)

    def _create_visualization(self, rover_array_msg):
        """Create visualization and cache for GUI/publishing"""
        if self.current_rgb_msg is None:
            return
        
        try:
            rgb_full = self.bridge.imgmsg_to_cv2(self.current_rgb_msg, 'bgr8')
            viz = rgb_full.copy()
            
            for rover in rover_array_msg.rovers:
                x_min = int(rover.bbox_x_min)
                y_min = int(rover.bbox_y_min)
                x_max = int(rover.bbox_x_max)
                y_max = int(rover.bbox_y_max)
                cx = int(rover.center_x)
                cy = int(rover.center_y)
                
                # Bounding box (thick red)
                cv2.rectangle(viz, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                
                # Crosshair (cyan)
                cv2.line(viz, (cx - 15, cy), (cx + 15, cy), (255, 255, 0), 2)
                cv2.line(viz, (cx, cy - 15), (cx, cy + 15), (255, 255, 0), 2)
                cv2.circle(viz, (cx, cy), 5, (255, 255, 0), -1)
                
                # Label with background
                angle_deg = math.degrees(rover.angle_horizontal)
                label = f"Rover {rover.id}: {rover.distance:.2f}m @ {angle_deg:+.1f} deg"
                
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                cv2.rectangle(viz, (x_min, y_min - 30), (x_min + text_w + 10, y_min), (0, 0, 0), -1)
                cv2.putText(viz, label, (x_min + 5, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Info overlay
            cv2.putText(viz, f"Rovers: {len(rover_array_msg.rovers)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(viz, f"Max dist: {self.max_distance}m", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Cache for GUI
            self.cached_visualization = viz
            
            # Publish
            viz_msg = self.bridge.cv2_to_imgmsg(viz, encoding='bgr8')
            viz_msg.header = rover_array_msg.header
            self.viz_pub.publish(viz_msg)
            
        except Exception as e:
            self.get_logger().error(f"Viz error: {e}")

    def display_frame(self):
        """Display cached visualization in GUI window"""
        if self.cached_visualization is None:
            # Show waiting message
            blank = np.zeros((540, 960, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera data...", 
                       (200, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Rover Detection", blank)
        else:
            cv2.imshow("Rover Detection", self.cached_visualization)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--low_res_fps", type=float, default=15.0)
    parser.add_argument("--high_res_fps", type=float, default=5.0)
    parser.add_argument("--max_distance", type=float, default=4.0)
    parser.add_argument("--enable_buffer", action="store_true")
    parser.add_argument("--buffer_alpha", type=float, default=0.7)
    parser.add_argument("--no_gui", action="store_true", help="Disable GUI window")
    parsed, _ = parser.parse_known_args()
    
    node = RoverDetectorWithPose()
    
    from rclpy.parameter import Parameter
    params = [
        Parameter('low_res_fps', Parameter.Type.DOUBLE, parsed.low_res_fps),
        Parameter('high_res_fps', Parameter.Type.DOUBLE, parsed.high_res_fps),
        Parameter('max_distance', Parameter.Type.DOUBLE, parsed.max_distance),
        Parameter('enable_buffer', Parameter.Type.BOOL, parsed.enable_buffer),
        Parameter('buffer_alpha', Parameter.Type.DOUBLE, parsed.buffer_alpha),
        Parameter('no_gui', Parameter.Type.BOOL, parsed.no_gui),
    ]
    node.set_parameters(params)
    
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