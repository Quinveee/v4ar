#!/usr/bin/env python3
"""
AprilTag Detection - FIXED PRIORITY (High-res always overrides low-res)
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
        
        self.current_detections = {}
        self.marker_buffer = {}
        
        # Track detection difficulty
        self.detection_history = []
        self.use_adaptive_scales = True
        
        # Track start time for delayed warnings
        self._start_time = None
        
        self.color_sub = self.create_subscription(
            Image, '/oak/rgb/image_raw', self.color_callback, 10
        )
        self.color_info_sub = self.create_subscription(
            CameraInfo, '/oak/rgb/camera_info', self.camera_info_callback, 1
        )
        
        self.get_logger().info(
            f"Subscribed to /oak/rgb/image_raw and /oak/rgb/camera_info"
        )

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
            self.tag_family, 
            threads=2, 
            maxhamming=2,
            decimate=0.2,
            blur=0.0,
            refine_edges=True,
            debug=False
        )
        
        self.get_logger().info(f"AprilTag detector initialized: {self.tag_family}")
        self.get_logger().info(f"Tag size: {self.tag_size}m")
        
        # Initialize start time after node is fully set up
        self._start_time = self.get_clock().now()

        if not self.no_gui:
            cv2.namedWindow("Detected AprilTags", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detected AprilTags", 640, 480) 

        self.get_logger().info(f"Adaptive dual-frequency detection:")
        self.get_logger().info(f"  Low-res (120px): {self.low_res_fps} Hz - fast tracking")
        self.get_logger().info(f"  High-res (240-400px): {self.high_res_fps} Hz - accurate pose")
        
        if self.enable_buffer:
            self.get_logger().info(f"Buffering ENABLED (age={self.buffer_max_age_sec}s, alpha={self.buffer_alpha})")


    def color_callback(self, msg: Image):
        if self.current_color_msg is None:
            self.get_logger().info(
                f"Received first image message: {msg.width}x{msg.height}, "
                f"encoding={msg.encoding}, frame_id={msg.header.frame_id}"
            )
        self.current_color_msg = msg

    def camera_info_callback(self, msg: CameraInfo):
        self.get_logger().info(
            f"Received camera_info message: {msg.width}x{msg.height}, "
            f"frame_id={msg.header.frame_id}"
        )
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
            self.get_logger().info(f"Publisher created for /oak/detected_markers")
            self.destroy_subscription(self.color_info_sub)


    def _should_use_extra_scales(self):
        """Decide if we need extra scales based on recent detection difficulty"""
        if not self.use_adaptive_scales or len(self.detection_history) < 3:
            return False
        
        recent_detections = self.detection_history[-5:]
        avg_detections = sum(recent_detections) / len(recent_detections)
        
        return avg_detections < 1.5


    def process_low_res(self):
        """
        Fast low-res detection for TRACKING ONLY.
        Used to discover new tags quickly, but will be overridden by high-res.
        """
        if not self._check_data_ready():
            # Don't log here - high_res will log if needed
            return
        
        detections = self._detect_at_resolution(120)
        
        for d in detections:
            tag_id = d['id']
            # FIXED: Only add if NOT exists OR if existing is also low-res
            if tag_id not in self.current_detections:
                self.current_detections[tag_id] = {
                    'detection': d,
                    'scale_factor': 120 / self.current_color_msg.width,
                    'resolution': 120,  # NEW: Track resolution
                    'source': 'low_res'
                }


    def process_high_res(self):
        """
        Accurate high-res detection for POSE ESTIMATION.
        ALWAYS overrides low-res detections.
        """
        if not self._check_data_ready():
            # Only warn after 2 seconds to give callbacks time to receive messages
            if self._start_time is not None:
                elapsed = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
                if elapsed > 2.0 and not hasattr(self, '_logged_data_not_ready'):
                    self.get_logger().warn(
                        f"Data not ready after {elapsed:.1f}s: "
                        f"color_msg={self.current_color_msg is not None}, "
                        f"camera_info={self.camera_info_received}. "
                        f"Check if topics /oak/rgb/image_raw and /oak/rgb/camera_info are publishing."
                    )
                    self._logged_data_not_ready = True
            return
        
        # Adaptive scaling
        if self._should_use_extra_scales():
            scales = [240, 320, 400]
            self.get_logger().info("Using extra scales", throttle_duration_sec=5.0)
        else:
            scales = [240, 320]
        
        # Detect at each scale, keeping highest resolution
        for scale_width in scales:
            detections = self._detect_at_resolution(scale_width)
            
            for d in detections:
                tag_id = d['id']
                
                # CRITICAL FIX: Always override if higher resolution
                should_update = False
                
                if tag_id not in self.current_detections:
                    # New detection
                    should_update = True
                else:
                    # Compare resolutions - higher res ALWAYS wins
                    existing_res = self.current_detections[tag_id].get('resolution', 0)
                    if scale_width > existing_res:
                        should_update = True
                
                if should_update:
                    self.current_detections[tag_id] = {
                        'detection': d,
                        'scale_factor': scale_width / self.current_color_msg.width,
                        'resolution': scale_width,  # Track resolution
                        'source': f'high_res_{scale_width}'
                    }
        
        # Track detection count
        self.detection_history.append(len(self.current_detections))
        if len(self.detection_history) > 10:
            self.detection_history.pop(0)
        
        # Calculate poses and publish
        self._calculate_and_publish_poses()
        
        # Clear for next cycle
        self.current_detections.clear()


    def _check_data_ready(self):
        if self.current_color_msg is None or not self.camera_info_received:
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
            
            # Preprocessing
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Only sharpen for small scales
            if target_width <= 160:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32) / 1.5
                gray = cv2.filter2D(gray, -1, kernel)
            
            return self.detector.detect(gray)
            
        except Exception as e:
            self.get_logger().error(f"Detection error: {e}")
            return []


    def _calculate_and_publish_poses(self):
        """Calculate 3D poses using PnP - now uses highest resolution available"""
        try:
            color_image = self.bridge.imgmsg_to_cv2(self.current_color_msg, 'bgr8')
            color_image_rectified = cv2.remap(color_image, self.mapx, self.mapy, cv2.INTER_LINEAR)
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return
        
        detected_vis = color_image_rectified.copy()
        now = self.get_clock().now().nanoseconds / 1e9

        s = self.tag_size / 2.0
        obj_pts = np.array([
            [-s,  s, 0], [ s,  s, 0], [ s, -s, 0], [-s, -s, 0]
        ], dtype=np.float32)

        current_frame_ids = set()
        
        for tag_id, data in self.current_detections.items():
            d = data['detection']
            scale_factor = data['scale_factor']
            resolution = data['resolution']
            
            try:
                if 'lb' in d:
                    corners_d = np.array([d['lb'], d['rb'], d['rt'], d['lt']])
                elif 'lb-rb-rt-lt' in d:
                    corners_d = d['lb-rb-rt-lt']
                else: 
                    continue
                
                # Scale corners back to full resolution
                corners_full_res = (corners_d / scale_factor).astype(np.float32)
                center_full_res = (d['center'] / scale_factor).astype(int)
                
                lt, rt, rb, lb = tuple(corners_full_res[3].astype(int)), tuple(corners_full_res[2].astype(int)), \
                                 tuple(corners_full_res[1].astype(int)), tuple(corners_full_res[0].astype(int))
                center = tuple(center_full_res)

                pose_msg = Pose()
                distance_value = -1.0
                
                # PnP with high-res corners
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
                
                # Buffering with smoothing
                if self.enable_buffer and tag_id in self.marker_buffer:
                    prev_marker = self.marker_buffer[tag_id]['marker']
                    marker_msg.distance = (
                        self.buffer_alpha * marker_msg.distance +
                        (1.0 - self.buffer_alpha) * prev_marker.distance
                    )
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
                
                self.marker_buffer[tag_id] = {
                    'marker': copy.deepcopy(marker_msg),
                    'last_seen': now,
                    'resolution': resolution  # Track what resolution was used
                }
                
                current_frame_ids.add(tag_id)

            except Exception as e:
                self.get_logger().error(f"Pose error for tag {tag_id}: {e}")
                continue

        # Remove stale markers
        ids_to_remove = [mid for mid, data in self.marker_buffer.items() 
                        if (now - data['last_seen']) > self.buffer_max_age_sec]
        for mid in ids_to_remove:
            del self.marker_buffer[mid]

        # Publish
        marker_array_msg = MarkerPoseArray()
        marker_array_msg.header.stamp = self.current_color_msg.header.stamp
        marker_array_msg.header.frame_id = self.current_color_msg.header.frame_id

        if self.enable_buffer:
            for marker_id, buffer_data in self.marker_buffer.items():
                marker_array_msg.markers.append(buffer_data['marker'])
        else:
            for marker_id in current_frame_ids:
                if marker_id in self.marker_buffer:
                    marker_array_msg.markers.append(self.marker_buffer[marker_id]['marker'])

        # Always publish, even if empty (so topic appears)
        self.pub.publish(marker_array_msg)
        
        # Log periodically for debugging
        if not hasattr(self, '_publish_count'):
            self._publish_count = 0
        self._publish_count += 1
        if self._publish_count <= 3 or self._publish_count % 30 == 0:
            self.get_logger().info(
                f"Published /oak/detected_markers: {len(marker_array_msg.markers)} markers "
                f"(publish #{self._publish_count})"
            )

        # Visualize
        for marker_msg in marker_array_msg.markers:
            cx, cy = int(marker_msg.center_x), int(marker_msg.center_y)
            cv2.circle(detected_vis, (cx, cy), 5, (0, 0, 255), -1)
            
            x_min, y_min = int(marker_msg.bbox_x_min), int(marker_msg.bbox_y_min)
            x_max, y_max = int(marker_msg.bbox_x_max), int(marker_msg.bbox_y_max)
            cv2.rectangle(detected_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Show which resolution was used for pose
            res_used = self.marker_buffer.get(marker_msg.id, {}).get('resolution', '?')
            cv2.putText(detected_vis, f"id={marker_msg.id} @{res_used}px",
                       (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(detected_vis, f"D={marker_msg.distance:.2f}m",
                       (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
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
    parser.add_argument("--tag_size", type=float, default=0.162)
    parser.add_argument("--no_gui", action="store_true")
    parser.add_argument("--low_res_fps", type=float, default=10.0)
    parser.add_argument("--high_res_fps", type=float, default=3.0)
    parser.add_argument("--enable_buffer", action="store_true")
    parser.add_argument("--buffer_max_age", type=float, default=0.5)
    parser.add_argument("--buffer_alpha", type=float, default=0.7)
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

    node.get_logger().info("Node initialized, starting to spin...")
    node.get_logger().info("Waiting for messages on /oak/rgb/image_raw and /oak/rgb/camera_info")
    
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