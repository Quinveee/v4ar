#!/usr/bin/env python3
"""
AprilTag Detection with PnP Pose - DUAL FREQUENCY OPTIMIZATION + BUFFERING
Low-res detection runs fast, high-res detection runs slower
Includes temporal buffering for stability with proper timeout handling
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import copy
from apriltag import apriltag
from perception_msgs.msg import MarkerPoseArray, MarkerPose


def rotation_matrix_to_quaternion(R: np.ndarray):
    q = np.zeros(4, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        idx = np.argmax(np.diag(R))
        if idx == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s

    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


class AprilTagVisualizationNode(Node):
    def __init__(self,
                 no_gui: bool = True,
                 tag_family: str = 'tagStandard41h12',
                 tag_size: float = 0.162,
                 low_res_fps: float = 10.0,
                 high_res_fps: float = 3.0,
                 enable_buffer: bool = True,
                 buffer_max_age_sec: float = 0.5,
                 buffer_alpha: float = 0.7):

        super().__init__('apriltag_visualization')

        self.bridge = CvBridge()
        self.no_gui = no_gui
        self.tag_size = tag_size

        self.current_color_msg = None
        self.K = None
        self.D = None
        self.mapx = None
        self.mapy = None
        self.camera_info_received = False
        
        self.low_res_fps = low_res_fps
        self.high_res_fps = high_res_fps
        
        self.enable_buffer = enable_buffer
        self.buffer_max_age_sec = buffer_max_age_sec
        self.buffer_alpha = buffer_alpha
        
        # Detection state: stores currently detected tags (cleared each high-res cycle)
        self.current_detections = {}
        
        # Buffer state: stores smoothed markers with timestamps
        # marker_id -> {"marker": MarkerPose, "last_seen": float (timestamp)}
        self.marker_buffer = {}
        
        self.color_sub = self.create_subscription(Image, '/color/image', self.color_callback, 10)
        self.color_info_sub = self.create_subscription(CameraInfo, '/color/camera_info', self.camera_info_callback, 1)

        self.pub = self.create_publisher(MarkerPoseArray, '/oak/detected_markers', 10)

        low_res_period = 1.0 / self.low_res_fps
        high_res_period = 1.0 / self.high_res_fps
        
        self.low_res_timer = self.create_timer(low_res_period, self.process_low_res)
        self.high_res_timer = self.create_timer(high_res_period, self.process_high_res)

        self.declare_parameter('tag_family', tag_family)
        self.declare_parameter('no_gui', no_gui)
        self.declare_parameter('tag_size', tag_size)
        self.declare_parameter('enable_buffer', enable_buffer)
        self.declare_parameter('buffer_max_age_sec', buffer_max_age_sec)
        self.declare_parameter('buffer_alpha', buffer_alpha)

        self.tag_family = self.get_parameter('tag_family').value
        self.no_gui = self.get_parameter('no_gui').value
        self.tag_size = self.get_parameter('tag_size').value
        self.enable_buffer = self.get_parameter('enable_buffer').value
        self.buffer_max_age_sec = self.get_parameter('buffer_max_age_sec').value
        self.buffer_alpha = self.get_parameter('buffer_alpha').value

        self.detector = apriltag(
            self.tag_family, threads=2, maxhamming=2, decimate=0.25,
            blur=0.0, refine_edges=True, debug=False
        )
        self.get_logger().info(f"AprilTag detector initialized: {self.tag_family}")
        self.get_logger().info(f"Tag size: {self.tag_size}m")

        if not self.no_gui:
            cv2.namedWindow("Detected AprilTags", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detected AprilTags", 640, 480) 

        self.get_logger().info(f"Dual-frequency detection:")
        self.get_logger().info(f"  Low-res (160px): {self.low_res_fps} Hz")
        self.get_logger().info(f"  High-res (320px): {self.high_res_fps} Hz")
        
        if self.enable_buffer:
            self.get_logger().info(f"Buffering ENABLED:")
            self.get_logger().info(f"  Max age: {self.buffer_max_age_sec}s")
            self.get_logger().info(f"  Smoothing alpha: {self.buffer_alpha}")
        else:
            self.get_logger().info("Buffering DISABLED")


    def color_callback(self, msg: Image):
        self.current_color_msg = msg

    def camera_info_callback(self, msg: CameraInfo):
        if not self.camera_info_received:
            self.K = np.array(msg.k).reshape((3, 3))
            self.D = np.array(msg.d)
            P_matrix = np.array(msg.p).reshape((3, 4))
            K_new = P_matrix[:, 0:3] 
            h, w = msg.height, msg.width
            
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.K, self.D, None, K_new, (w, h), cv2.CV_32FC1
            )
            
            self.K = K_new
            self.camera_info_received = True
            self.get_logger().info(f"Camera calibration loaded: {w}x{h}")
            self.destroy_subscription(self.color_info_sub)


    def process_low_res(self):
        """Fast low-res detection - updates current_detections"""
        if not self._check_data_ready():
            return
        
        detections = self._detect_at_resolution(160)
        
        # Add to current detections (will be cleared after high-res processing)
        for d in detections:
            tag_id = d['id']
            if tag_id not in self.current_detections:
                self.current_detections[tag_id] = {
                    'detection': d,
                    'scale_factor': 160 / self.current_color_msg.width,
                    'source': 'low_res'
                }


    def process_high_res(self):
        """Slow high-res detection - calculates poses and publishes"""
        if not self._check_data_ready():
            return
        
        detections = self._detect_at_resolution(320)
        
        # Override with high-res detections (more accurate)
        for d in detections:
            tag_id = d['id']
            self.current_detections[tag_id] = {
                'detection': d,
                'scale_factor': 320 / self.current_color_msg.width,
                'source': 'high_res'
            }
        
        # Calculate poses and publish
        self._calculate_and_publish_poses()
        
        # CRITICAL: Clear current detections after publishing
        # This ensures we only publish what we actually see in this frame
        self.current_detections.clear()


    def _check_data_ready(self):
        if self.current_color_msg is None or not self.camera_info_received:
            self.get_logger().warn("Waiting for data...", throttle_duration_sec=2.0)
            return False
        return True


    def _detect_at_resolution(self, target_width):
        """Run AprilTag detection at specified resolution"""
        try:
            color_image = self.bridge.imgmsg_to_cv2(self.current_color_msg, 'bgr8')
            color_image_rectified = cv2.remap(color_image, self.mapx, self.mapy, cv2.INTER_LINEAR)
            
            scale_factor = target_width / color_image_rectified.shape[1]
            downscaled = cv2.resize(
                color_image_rectified, 
                (target_width, int(color_image_rectified.shape[0] * scale_factor)), 
                interpolation=cv2.INTER_LINEAR
            )
            
            gray = cv2.cvtColor(downscaled, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            return self.detector.detect(gray)
            
        except Exception as e:
            self.get_logger().error(f"Detection error: {e}")
            return []


    def _calculate_and_publish_poses(self):
        """Calculate 3D poses using PnP and publish for all detected tags"""
        try:
            color_image = self.bridge.imgmsg_to_cv2(self.current_color_msg, 'bgr8')
            color_image_rectified = cv2.remap(color_image, self.mapx, self.mapy, cv2.INTER_LINEAR)
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return
        
        detected_vis = color_image_rectified.copy()
        
        # Get current timestamp
        now = self.get_clock().now().nanoseconds / 1e9  # Convert to seconds

        s = self.tag_size / 2.0
        obj_pts = np.array([
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0]
        ], dtype=np.float32)

        # Process current detections and update buffer
        current_frame_ids = set()
        
        for tag_id, data in self.current_detections.items():
            d = data['detection']
            scale_factor = data['scale_factor']
            
            try:
                if 'lb' in d:
                    corners_d = np.array([d['lb'], d['rb'], d['rt'], d['lt']])
                elif 'lb-rb-rt-lt' in d:
                    corners_d = d['lb-rb-rt-lt']
                else: 
                    continue
                
                corners_full_res = (corners_d / scale_factor).astype(np.float32)
                center_full_res = (d['center'] / scale_factor).astype(int)
                
                lt, rt, rb, lb = tuple(corners_full_res[3].astype(int)), tuple(corners_full_res[2].astype(int)), \
                                 tuple(corners_full_res[1].astype(int)), tuple(corners_full_res[0].astype(int))
                center = tuple(center_full_res)

                pose_msg = Pose()
                distance_value = -1.0
                
                img_pts = np.array([lt, rt, rb, lb], dtype=np.float32)
                dist_coeffs = np.zeros((4, 1), dtype=np.float32)
                
                success, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, self.K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                if not success:
                    continue
                
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.flatten()
                
                distance_value = float(np.linalg.norm(t))
                
                pose_msg.position.x = float(t[0])
                pose_msg.position.y = float(t[1])
                pose_msg.position.z = float(t[2])
                
                qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
                pose_msg.orientation.x = qx
                pose_msg.orientation.y = qy
                pose_msg.orientation.z = qz
                pose_msg.orientation.w = qw

                # Create marker message
                marker_msg = MarkerPose()
                marker_msg.id = int(tag_id)
                marker_msg.pose = pose_msg
                marker_msg.distance = float(distance_value)
                marker_msg.center_x = float(center[0])
                marker_msg.center_y = float(center[1])
                xs = [lt[0], rt[0], rb[0], lb[0]]
                ys = [lt[1], rt[1], rb[1], lb[1]]
                marker_msg.bbox_x_min = float(min(xs))
                marker_msg.bbox_y_min = float(min(ys))
                marker_msg.bbox_x_max = float(max(xs))
                marker_msg.bbox_y_max = float(max(ys))
                
                # Update buffer with smoothing if enabled
                if self.enable_buffer and tag_id in self.marker_buffer:
                    prev_marker = self.marker_buffer[tag_id]['marker']
                    
                    # Exponential moving average for distance
                    smoothed_distance = (
                        self.buffer_alpha * marker_msg.distance +
                        (1.0 - self.buffer_alpha) * prev_marker.distance
                    )
                    marker_msg.distance = smoothed_distance
                    
                    # Smooth position (optional, comment out if too slow)
                    marker_msg.pose.position.x = (
                        self.buffer_alpha * marker_msg.pose.position.x +
                        (1.0 - self.buffer_alpha) * prev_marker.pose.position.x
                    )
                    marker_msg.pose.position.y = (
                        self.buffer_alpha * marker_msg.pose.position.y +
                        (1.0 - self.buffer_alpha) * prev_marker.pose.position.y
                    )
                    marker_msg.pose.position.z = (
                        self.buffer_alpha * marker_msg.pose.position.z +
                        (1.0 - self.buffer_alpha) * prev_marker.pose.position.z
                    )
                
                # Update buffer
                self.marker_buffer[tag_id] = {
                    'marker': copy.deepcopy(marker_msg),
                    'last_seen': now
                }
                
                current_frame_ids.add(tag_id)

            except Exception as e:
                self.get_logger().error(f"Pose calculation error for tag {tag_id}: {e}")
                continue

        # Remove stale markers from buffer
        ids_to_remove = []
        for marker_id, buffer_data in self.marker_buffer.items():
            age = now - buffer_data['last_seen']
            if age > self.buffer_max_age_sec:
                ids_to_remove.append(marker_id)
        
        for marker_id in ids_to_remove:
            del self.marker_buffer[marker_id]
            self.get_logger().info(f"Tag {marker_id} timed out (not seen for {self.buffer_max_age_sec}s)")

        # Build message to publish (only from buffer if enabled, otherwise from current detections)
        marker_array_msg = MarkerPoseArray()
        marker_array_msg.header.stamp = self.current_color_msg.header.stamp
        marker_array_msg.header.frame_id = self.current_color_msg.header.frame_id

        if self.enable_buffer:
            # Publish all markers in buffer that are fresh
            for marker_id, buffer_data in self.marker_buffer.items():
                marker_array_msg.markers.append(buffer_data['marker'])
        else:
            # Publish only current frame detections
            for marker_id in current_frame_ids:
                if marker_id in self.marker_buffer:
                    marker_array_msg.markers.append(self.marker_buffer[marker_id]['marker'])

        # Visualize only markers we're publishing
        for marker_msg in marker_array_msg.markers:
            # Find corners from marker (reconstruct from bbox for visualization)
            cx = int(marker_msg.center_x)
            cy = int(marker_msg.center_y)
            
            # Simple visualization
            cv2.circle(detected_vis, (cx, cy), 5, (0, 0, 255), -1)
            
            # Draw bounding box
            x_min = int(marker_msg.bbox_x_min)
            y_min = int(marker_msg.bbox_y_min)
            x_max = int(marker_msg.bbox_x_max)
            y_max = int(marker_msg.bbox_y_max)
            
            cv2.rectangle(detected_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Label
            source = "buffered" if self.enable_buffer and marker_msg.id not in current_frame_ids else "current"
            cv2.putText(detected_vis, f"id={marker_msg.id} [{source}]",
                       (x_min, y_min - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(detected_vis, f"D={marker_msg.distance:.2f}m",
                       (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        self.pub.publish(marker_array_msg)
        
        if not self.no_gui:
            display_det = cv2.resize(detected_vis, (640, 480))
            cv2.imshow("Detected AprilTags", display_det)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and not self.no_gui:
            raise KeyboardInterrupt


def main(args=None):
    rclpy.init(args=args)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag_family", type=str, default="tagStandard41h12")
    parser.add_argument("--tag_size", type=float, default=0.162, help="Tag size in meters")
    parser.add_argument("--no_gui", action="store_true")
    parser.add_argument("--low_res_fps", type=float, default=10.0, help="Low-res detection frequency")
    parser.add_argument("--high_res_fps", type=float, default=3.0, help="High-res detection frequency")
    parser.add_argument("--enable_buffer", action="store_true", help="Enable temporal buffering")
    parser.add_argument("--buffer_max_age", type=float, default=0.5, help="Max age for buffered markers (seconds)")
    parser.add_argument("--buffer_alpha", type=float, default=0.7, help="Smoothing factor (0-1, higher = more responsive)")
    parsed, _ = parser.parse_known_args()

    node = AprilTagVisualizationNode(
        no_gui=parsed.no_gui,
        tag_family=parsed.tag_family,
        tag_size=parsed.tag_size,
        low_res_fps=parsed.low_res_fps,
        high_res_fps=parsed.high_res_fps,
        enable_buffer=parsed.enable_buffer,
        buffer_max_age_sec=parsed.buffer_max_age,
        buffer_alpha=parsed.buffer_alpha
    )

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