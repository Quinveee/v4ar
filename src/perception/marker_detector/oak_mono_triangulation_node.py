#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import yaml
import numpy as np
from perception_msgs.msg import MarkerPoseArray
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory
import os
import math
from collections import defaultdict
from scipy.spatial.transform import Rotation as R


# Import solvers
from .solvers.weighted_least_squares import WeightedLeastSquaresSolver
from .solvers.new_solver import LeastSquaresSolver

SOLVER_CLASSES = {
    'least_squares': LeastSquaresSolver,
    'weighted': WeightedLeastSquaresSolver,
}


def compute_yaw_from_apriltag(tag_msg, marker_map):
    """
    Compute robot yaw based on:
    - tag orientation in camera frame (tag_msg.pose.orientation)
    - known tag orientation in world frame (marker_map[id][2])
    """

    # 1. Rotation matrix from tag → camera
    q = [
        tag_msg.pose.orientation.x,
        tag_msg.pose.orientation.y,
        tag_msg.pose.orientation.z,
        tag_msg.pose.orientation.w
    ]
    R_ct = R.from_quat(q).as_matrix()

    # 2. Tag normal direction in camera frame (tag's +Z axis)
    tag_normal_cam = R_ct @ np.array([0, 0, 1.0])

    # 3. Camera → tag direction (reverse normal)
    cam_to_tag = -tag_normal_cam

    # 4. Bearing angle to tag in camera/robot frame
    cam_yaw_to_tag = math.atan2(cam_to_tag[1], cam_to_tag[0])

    # 5. Tag yaw in world frame (from YAML config)
    _, _, tag_yaw_world = marker_map[tag_msg.id]

    # 6. Robot yaw = difference
    yaw_robot = tag_yaw_world - cam_yaw_to_tag

    # Normalize
    return math.atan2(math.sin(yaw_robot), math.cos(yaw_robot))


class TriangulationNode(Node):
    def __init__(self, oak_weight, mono_weight):
        super().__init__('triangulation_node')

        # --- Parameters ---
        self.declare_parameter('marker_config', 'config/markers.yaml')
        self.declare_parameter('solver_type', 'least_squares')
        self.declare_parameter('use_buffered_markers', False)
        self.declare_parameter('oak_weight', oak_weight)  # NEW: Weight for OAK detections
        self.declare_parameter('mono_weight', mono_weight)  # NEW: Weight for mono detections
        
        config_path = self.get_parameter('marker_config').value
        solver_type = self.get_parameter('solver_type').value
        use_buffered = self.get_parameter('use_buffered_markers').value
        self.oak_weight = self.get_parameter('oak_weight').value
        self.mono_weight = self.get_parameter('mono_weight').value

        # --- Load marker configuration ---
        if not os.path.isabs(config_path):
            package_share = get_package_share_directory('perception')
            config_path = os.path.join(package_share, config_path)
        self.marker_map = self.load_marker_config(config_path)
        self.get_logger().info(
            f"Loaded {len(self.marker_map)} markers from {config_path}")

        # --- Instantiate solver dynamically ---
        SolverClass = SOLVER_CLASSES.get(solver_type, LeastSquaresSolver)
        self.solver = SolverClass()
        self.get_logger().info(
            f"Using triangulation solver: {SolverClass.__name__}")

        # --- Detection storage for fusion ---
        # NEW: Store latest detections from both cameras with timestamps
        self.mono_detections = {}  # {marker_id: (MarkerPose, timestamp)}
        self.oak_detections = {}   # {marker_id: (MarkerPose, timestamp)}
        self.detection_timeout = 0.5  # seconds
        
        # --- Orientation buffer for smoothing ---
        self.orientation_buffer = []  # List of (yaw, distance, timestamp, is_new)
        self.max_buffer_size = 10
        self.buffer_timeout = 0.5  # seconds
        self.smoothing_alpha = 0.3  # Weight for new measurements
        self.current_smoothed_yaw = None
        
        # Load axis markers if defined
        self.axis_markers = {}
        if 'axis_markers' in self.marker_map_data:
            axis_config = self.marker_map_data['axis_markers']
            self.axis_markers = {
                'axis_0_deg': axis_config.get('axis_0_deg', []),
                'axis_90_deg': axis_config.get('axis_90_deg', []),
                'axis_180_deg': axis_config.get('axis_180_deg', []),
                'axis_270_deg': axis_config.get('axis_270_deg', []),
            }
            self.get_logger().info(f"Loaded axis markers: {self.axis_markers}")
        else:
            self.get_logger().warn("No axis_markers defined in config, using fallback orientation")
        
        # --- ROS I/O ---
        # Mono camera subscription
        mono_topic = '/detected_markers_buffered' if use_buffered else '/detected_markers'
        self.mono_sub = self.create_subscription(
            MarkerPoseArray, mono_topic, self.mono_marker_callback, 10)
        self.get_logger().info(f"Subscribed to mono markers: {mono_topic}")
        
        # NEW: OAK camera subscription
        oak_topic = '/oak/detected_markers'
        self.oak_sub = self.create_subscription(
            MarkerPoseArray, oak_topic, self.oak_marker_callback, 10)
        self.get_logger().info(f"Subscribed to OAK markers: {oak_topic}")
        
        self.pub = self.create_publisher(PoseStamped, '/robot_pose_raw', 10)
        
        self.get_logger().info(f"Weights: OAK={self.oak_weight:.1f}x, Mono={self.mono_weight:.1f}x")

    # -----------------------------------------------------

    def load_marker_config(self, path):
        """Load marker world coordinates from YAML config."""
        try:
            with open(path, 'r') as f:
                self.marker_map_data = yaml.safe_load(f)
            return {int(k): tuple(v) for k, v in self.marker_map_data['marker_map'].items()}
        except Exception as e:
            self.get_logger().error(f"Failed to load marker config: {e}")
            self.marker_map_data = {}
            return {}

    # -----------------------------------------------------

    def mono_marker_callback(self, msg: MarkerPoseArray):
        """Callback for mono camera detections"""
        now = self.get_clock().now().nanoseconds / 1e9
        for m in msg.markers:
            if m.id in self.marker_map:
                self.mono_detections[m.id] = (m, now)  # Store marker + timestamp
        
        # Trigger fusion after receiving mono data
        self._fuse_and_triangulate(msg.header)

    def oak_marker_callback(self, msg: MarkerPoseArray):
        """Callback for OAK camera detections (higher priority)"""
        now = self.get_clock().now().nanoseconds / 1e9
        for m in msg.markers:
            if m.id in self.marker_map:
                self.oak_detections[m.id] = (m, now)  # Store marker + timestamp
        
        # Trigger fusion after receiving OAK data
        self._fuse_and_triangulate(msg.header)

    def _compute_orientation_from_axes(self, tag_msg):
        """Compute orientation from axis-aligned markers (if available).
        
        Returns yaw angle in radians, or None if no axis markers found.
        """
        if not self.axis_markers:
            return None
        
        tag_id = tag_msg.id
        
        # Check which axis group this marker belongs to
        axis_angles = {}  # 'axis_0_deg', 'axis_90_deg', etc. -> [marker_ids]
        
        for axis_name, marker_ids in self.axis_markers.items():
            if tag_id in marker_ids:
                axis_angles[axis_name] = marker_ids
        
        if not axis_angles:
            return None  # Not an axis marker
        
        # Extract quaternion and convert to rotation matrix
        q = [tag_msg.pose.orientation.x,
             tag_msg.pose.orientation.y,
             tag_msg.pose.orientation.z,
             tag_msg.pose.orientation.w]
        R_ct = R.from_quat(q).as_matrix()
        
        # Camera-to-tag vector (from tag orientation)
        tag_normal_cam = R_ct @ np.array([0.0, 0.0, 1.0])
        cam_to_tag = -tag_normal_cam[:2]  # Project to XZ plane
        cam_yaw_to_tag = math.atan2(cam_to_tag[0], cam_to_tag[2])  # Using X, Z
        
        # World bearing to tag from marker configuration
        marker_config = self.marker_map.get(tag_id, (0, 0, 0))
        if len(marker_config) >= 3:
            _, _, tag_yaw_world = marker_config[:3]
        else:
            tag_yaw_world = 0.0
        
        # Robot yaw relative to tag
        yaw_robot = tag_yaw_world - cam_yaw_to_tag
        
        # Extract the axis expected angle
        axis_name = next(iter(axis_angles.keys()))
        expected_angle = {
            'axis_0_deg': 0.0,
            'axis_90_deg': math.pi / 2.0,
            'axis_180_deg': math.pi,
            'axis_270_deg': 3 * math.pi / 2.0
        }.get(axis_name, 0.0)
        
        # Normalize yaw to [-pi, pi]
        yaw_robot = math.atan2(math.sin(yaw_robot), math.cos(yaw_robot))
        
        # Check if within 15 degrees of expected angle
        angle_diff = abs(yaw_robot - expected_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        if angle_diff < math.radians(15):
            # Snap to nearest 90-degree axis
            return expected_angle
        
        return yaw_robot

    def _smooth_orientation(self, yaw, distance, is_new=False):
        """Apply temporal smoothing and weighted averaging to orientation.
        
        Uses circular mean with:
        - Distance weighting: 1/(d+0.1)
        - Time decay: exp(-age/timeout)
        - Exponential smoothing: alpha=0.3
        
        Returns smoothed yaw in radians.
        """
        now = self.get_clock().now().nanoseconds / 1e9
        
        # Add new measurement to buffer
        self.orientation_buffer.append((yaw, distance, now, is_new))
        
        # Remove expired entries
        self.orientation_buffer = [
            (y, d, t, n) for y, d, t, n in self.orientation_buffer
            if (now - t) <= self.buffer_timeout
        ]
        
        # Keep buffer size manageable
        if len(self.orientation_buffer) > self.max_buffer_size:
            self.orientation_buffer = self.orientation_buffer[-self.max_buffer_size:]
        
        if not self.orientation_buffer:
            return yaw
        
        # Compute weighted circular mean
        weighted_sin = 0.0
        weighted_cos = 0.0
        total_weight = 0.0
        
        for y, d, t, is_new_entry in self.orientation_buffer:
            age = now - t
            
            # Distance weight (closer markers have higher weight)
            dist_weight = 1.0 / (d + 0.1)
            
            # Time decay weight (recent measurements have higher weight)
            time_weight = math.exp(-age / self.buffer_timeout) if age > 0 else 1.0
            
            # New measurement boost
            new_weight = 2.0 if is_new_entry else 1.0
            
            weight = dist_weight * time_weight * new_weight
            
            weighted_sin += weight * math.sin(y)
            weighted_cos += weight * math.cos(y)
            total_weight += weight
        
        if total_weight > 0:
            avg_yaw = math.atan2(weighted_sin / total_weight, weighted_cos / total_weight)
        else:
            avg_yaw = yaw
        
        # Exponential smoothing
        if self.current_smoothed_yaw is not None:
            # Use circular mean for blending angles
            delta = math.atan2(math.sin(avg_yaw - self.current_smoothed_yaw),
                             math.cos(avg_yaw - self.current_smoothed_yaw))
            smoothed = self.current_smoothed_yaw + self.smoothing_alpha * delta
        else:
            smoothed = avg_yaw
        
        # Normalize to [-pi, pi]
        self.current_smoothed_yaw = math.atan2(math.sin(smoothed), math.cos(smoothed))
        
        return self.current_smoothed_yaw

    def _fuse_and_triangulate(self, header):
        """Fuse detections from both cameras and triangulate"""
        try:
            now = self.get_clock().now().nanoseconds / 1e9
            
            # Build fused detection list with weights
            # For triangulation, we use distances; for orientation, we use full marker objects
            fused_detections = []
            fused_markers = []  # Keep track of full marker objects for orientation
            
            # Add OAK detections (higher weight)
            for marker_id, (marker_obj, timestamp) in self.oak_detections.items():
                if (now - timestamp) <= self.detection_timeout:
                    fused_detections.append((marker_id, marker_obj.distance, self.oak_weight))
                    fused_markers.append((marker_id, marker_obj, self.oak_weight))
            
            # Add mono detections (lower weight, only if not seen by OAK recently)
            for marker_id, (marker_obj, timestamp) in self.mono_detections.items():
                if (now - timestamp) <= self.detection_timeout:
                    # Only use mono if OAK hasn't seen this marker recently
                    if marker_id not in self.oak_detections:
                        fused_detections.append((marker_id, marker_obj.distance, self.mono_weight))
                        fused_markers.append((marker_id, marker_obj, self.mono_weight))
            
            # Remove stale detections
            self.oak_detections = {
                k: v for k, v in self.oak_detections.items() 
                if (now - v[1]) <= self.detection_timeout
            }
            self.mono_detections = {
                k: v for k, v in self.mono_detections.items() 
                if (now - v[1]) <= self.detection_timeout
            }
            
            if len(fused_detections) < 2:
                self.get_logger().warn(
                    f"Need at least 2 markers. OAK: {len(self.oak_detections)}, "
                    f"Mono: {len(self.mono_detections)}", 
                    throttle_duration_sec=2.0
                )
                return
            
            # Log detection sources
            oak_ids = [m_id for m_id, _, w in fused_detections if w == self.oak_weight]
            mono_ids = [m_id for m_id, _, w in fused_detections if w == self.mono_weight]
            self.get_logger().info(
                f"Fused detections: OAK={oak_ids}, Mono={mono_ids}",
                throttle_duration_sec=1.0
            )
            
            # Check if solver supports weights
            if hasattr(self.solver, 'solve_weighted'):
                # Use weighted solver (detections with weights)
                result = self.solver.solve_weighted(fused_detections, self.marker_map)
            else:
                # Fallback: Use regular solver (just use id and distance, ignore weights)
                detections = [(m_id, dist) for m_id, dist, _ in fused_detections]
                result = self.solver.solve(detections, self.marker_map)
                self.get_logger().warn(
                    "Solver doesn't support weights, using unweighted triangulation",
                    throttle_duration_sec=10.0
                )

            # Handle both 2D (x, y) and 3D (x, y, yaw) outputs
            if len(result) == 3:
                x, y, yaw = result
            else:
                x, y = result
                
                # Compute yaw from AprilTag detections
                yaw = None
                closest_marker = None
                closest_distance = float('inf')
                
                # Find closest detection to use for orientation (prefer OAK)
                for marker_id, marker_obj, weight in fused_markers:
                    if marker_obj.distance < closest_distance:
                        closest_distance = marker_obj.distance
                        closest_marker = (marker_id, marker_obj)
                
                if closest_marker is not None:
                    marker_id, marker_obj = closest_marker
                    
                    # Try axis-based orientation first
                    axis_yaw = self._compute_orientation_from_axes(marker_obj)
                    if axis_yaw is not None:
                        yaw = axis_yaw
                    else:
                        # Fallback to compute_yaw_from_apriltag
                        yaw = compute_yaw_from_apriltag(marker_obj, self.marker_map)
                    
                    # Apply orientation smoothing
                    yaw = self._smooth_orientation(yaw, closest_distance, is_new=True)
                
                if yaw is None:
                    # Use previous smoothed yaw if available
                    if self.current_smoothed_yaw is not None:
                        yaw = self.current_smoothed_yaw
                    else:
                        yaw = 0.0

            # Debug logging
            self.get_logger().info(f"Triangulated: x={x:.4f}, y={y:.4f}, yaw={yaw:.4f}")

            pose_msg = PoseStamped()
            pose_msg.header = header
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position.x = float(x)
            pose_msg.pose.position.y = float(y)

            # Convert yaw to quaternion
            import math
            half_yaw = yaw / 2.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = math.sin(half_yaw)
            pose_msg.pose.orientation.w = math.cos(half_yaw)

            self.pub.publish(pose_msg)
            self.get_logger().info(
                f"Robot position: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}",
                throttle_duration_sec=0.5
            )
            
        except Exception as e:
            self.get_logger().error(f"Error in triangulation: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--oak_weight", type=float, default=2.0, 
                       help="Weight for OAK camera detections")
    parser.add_argument("--mono_weight", type=float, default=1.0,
                       help="Weight for mono camera detections")
    parsed, _ = parser.parse_known_args()
    
    node = TriangulationNode(parsed.oak_weight, parsed.mono_weight)
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()