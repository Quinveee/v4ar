#!/usr/bin/env python3
import argparse
import rclpy
from rclpy.node import Node
import yaml
import numpy as np
from perception_msgs.msg import MarkerPoseArray
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory
import os
import math
from .utils import compute_yaw_from_apriltag, compute_average_yaw_from_markers


from .solvers.weighted_least_squares import WeightedLeastSquaresSolver
from .solvers.new_solver import LeastSquaresSolver

SOLVER_CLASSES = {
    'least_squares': LeastSquaresSolver,
    'weighted': LeastSquaresSolver
}

# -------------------------------------------------------
#  NODE
# -------------------------------------------------------
class TriangulationNode(Node):
    def __init__(self, topic_name='/detected_markers'):
        super().__init__('triangulation_node')

        # --- Parameters ---
        self.declare_parameter('marker_config', 'config/markers.yaml')
        self.declare_parameter('solver_type', 'least_squares')
        self.declare_parameter('use_buffered_markers', False)
        self.declare_parameter('input_topic', topic_name)

        config_path = self.get_parameter('marker_config').value
        solver_type = self.get_parameter('solver_type').value
        use_buffered = self.get_parameter('use_buffered_markers').value
        input_topic = self.get_parameter('input_topic').value
        # --- Load marker configuration ---
        if not os.path.isabs(config_path):
            package_share = get_package_share_directory('perception')
            config_path = os.path.join(package_share, config_path)

        self.marker_map = self.load_marker_config(config_path)
        self.get_logger().info(
            f"Loaded {len(self.marker_map)} markers from {config_path}"
        )

        # --- Instantiate solver dynamically ---
        SolverClass = SOLVER_CLASSES.get(solver_type, LeastSquaresSolver)
        self.solver = SolverClass()

        self.get_logger().info(
            f"Using triangulation solver: {SolverClass.__name__}"
        )

        # --- Orientation buffer for smoothing ---
        self.orientation_buffer = []  # List of (yaw, distance, timestamp, is_new)
        self.max_buffer_size = 10
        self.buffer_timeout = 0.5  # seconds
        self.smoothing_alpha = 0.3  # Weight for new measurements (0-1, higher = more responsive)
        self.current_smoothed_yaw = None

        # --- ROS I/O ---
        # input_topic = '/detected_markers_buffered' if use_buffered else '/detected_markers'
        # input_topic = '/detected_markers'
        # input_topic = '/oak/detected_markers'
        self.sub = self.create_subscription(
            MarkerPoseArray, "/oak/detected_markers", self.marker_callback, 10
        )
        self.get_logger().info(f"Subscribed to marker topic: {input_topic}")

        self.pub = self.create_publisher(PoseStamped, '/robot_pose_raw', 10)

        self.yaw = 0


    # -----------------------------------------------------

    def load_marker_config(self, path):
        """Load marker world coordinates and axis definitions from YAML config."""
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            marker_map = {int(k): tuple(v) for k, v in data['marker_map'].items()}
            
            # Load axis markers if defined
            self.axis_markers = {}
            if 'axis_markers' in data:
                axis_config = data['axis_markers']
                self.axis_markers = {
                    'axis_0_deg': axis_config.get('axis_0_deg', []),
                    'axis_90_deg': axis_config.get('axis_90_deg', []),
                    'axis_180_deg': axis_config.get('axis_180_deg', []),
                    'axis_270_deg': axis_config.get('axis_270_deg', []),
                }
                self.get_logger().info(f"Loaded axis markers: {self.axis_markers}")
            else:
                self.get_logger().warn("No axis_markers defined in config, using fallback orientation")
            
            return marker_map
        except Exception as e:
            self.get_logger().error(f"Failed to load marker config: {e}")
            return {}


    # -----------------------------------------------------

    def marker_callback(self, msg: MarkerPoseArray):
        try:
            detections = [(m.id, m.distance) for m in msg.markers if m.id in self.marker_map]
            
            if len(detections) < 2:
                self.get_logger().warn("Need at least 2 known markers for triangulation.")
                return

            # Handle both 2D and 3D solver outputs
            result = self.solver.solve(detections, self.marker_map)
            if len(result) == 3:
                x, y, _ = result  # Ignore solver's yaw estimate
            else:
                x, y = result

            # Compute orientation from average of all visible markers
            yaw = compute_average_yaw_from_markers(x, y, msg.markers, self.marker_map)
            self.get_logger().info(f"Yaw from markers: {yaw}")
            if yaw is None:
                self.get_logger().warn("No yaw computed, using previous yaw")
                yaw = self.yaw
            self.yaw = yaw
            #yaw -= math.pi / 4
            # Smooth orientation using buffer
            #yaw = self._smooth_orientation(yaw, msg)

            # Construct message
            pose_msg = PoseStamped()
            pose_msg.header = msg.header
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position.x = float(x)
            pose_msg.pose.position.y = float(y)

            # Convert yaw → quaternion
            half_yaw = yaw / 2.0
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = math.sin(half_yaw)
            pose_msg.pose.orientation.w = math.cos(half_yaw)

            self.pub.publish(pose_msg)

            self.get_logger().info(
                f"Robot pose: x={x:.2f}, y={y:.2f}, yaw rad={yaw:.2f}, yaw={math.degrees(yaw):.1f} deg"
            )

        except Exception as e:
            self.get_logger().error(f"Error in marker_callback: {e.with_traceback(None)}")

    def _compute_orientation_from_axes(self, msg: MarkerPoseArray, robot_x: float, robot_y: float):
        """Compute robot orientation from axis markers.
        
        When robot faces an axis directly, orientation is rounded to nearest 90°.
        Prioritizes closest markers on each axis.
        """
        if not self.axis_markers or not any(self.axis_markers.values()):
            # Fallback to original method
            closest_tag = min(
                [m for m in msg.markers if m.id in self.marker_map],
                key=lambda m: m.distance
            )
            return compute_yaw_from_apriltag(closest_tag, self.marker_map)
        
        now = self.get_clock().now().nanoseconds / 1e9
        orientation_estimates = []
        
        # Check each axis
        axis_angles = {
            'axis_0_deg': 0.0,
            'axis_90_deg': math.pi / 2,
            'axis_180_deg': math.pi,
            'axis_270_deg': -math.pi / 2,
        }
        
        for axis_name, target_angle in axis_angles.items():
            marker_ids = self.axis_markers.get(axis_name, [])
            if not marker_ids:
                continue
            
            # Find closest marker on this axis
            axis_markers = [m for m in msg.markers if m.id in marker_ids and m.id in self.marker_map]
            if not axis_markers:
                continue
            
            closest_axis_marker = min(axis_markers, key=lambda m: m.distance)
            
            # Compute angle from robot to marker in world frame
            marker_world_x, marker_world_y = self.marker_map[closest_axis_marker.id][:2]
            dx_world = marker_world_x - robot_x
            dy_world = marker_world_y - robot_y
            angle_to_marker_world = math.atan2(dy_world, dx_world)
            
            # Get marker position in camera frame (from solvePnP - tvec)
            marker_x_cam = closest_axis_marker.pose.position.x
            marker_y_cam = closest_axis_marker.pose.position.y
            
            # Angle from camera to marker in camera frame
            # Note: atan2 handles negative x correctly (gives angle in correct quadrant)
            angle_to_marker_cam = math.atan2(marker_y_cam, marker_x_cam)
            
            # Compute robot orientation
            # When marker is in front (x > 0): robot_yaw = world_angle - camera_angle
            # When marker is behind (x < 0): need to flip the camera angle by 180°
            if marker_x_cam > 0:
                # Marker is in front of camera
                robot_yaw = angle_to_marker_world - angle_to_marker_cam
            else:
                # Marker is behind camera - flip camera angle
                robot_yaw = angle_to_marker_world - (angle_to_marker_cam + math.pi)
            
            # Normalize
            robot_yaw = math.atan2(math.sin(robot_yaw), math.cos(robot_yaw))
            
            # If robot is facing this axis directly (within threshold), round to axis angle
            angle_diff = abs(robot_yaw - target_angle)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff)  # Handle wraparound
            
            threshold = math.radians(15)  # 15 degree threshold
            if angle_diff < threshold:
                robot_yaw = target_angle
            
            orientation_estimates.append((robot_yaw, closest_axis_marker.distance, True))
        
        if not orientation_estimates:
            # Fallback to original method
            closest_tag = min(
                [m for m in msg.markers if m.id in self.marker_map],
                key=lambda m: m.distance
            )
            return compute_yaw_from_apriltag(closest_tag, self.marker_map)
        
        # Sort by distance (closest first) and prioritize new detections
        orientation_estimates.sort(key=lambda x: x[1])  # Sort by distance
        
        # Use closest estimate
        return orientation_estimates[0][0]

    def _smooth_orientation(self, new_yaw: float, msg: MarkerPoseArray):
        """Smooth orientation using buffer, prioritizing closest and new markers."""
        now = self.get_clock().now().nanoseconds / 1e9
        
        # Get closest marker distance for weighting
        if msg.markers:
            closest_distance = min([m.distance for m in msg.markers if m.id in self.marker_map], default=1.0)
        else:
            closest_distance = 1.0
        
        # Add new measurement to buffer (prioritize new detections)
        self.orientation_buffer.append((new_yaw, closest_distance, now, True))
        
        # Remove old measurements
        self.orientation_buffer = [
            (yaw, dist, ts, is_new) for yaw, dist, ts, is_new in self.orientation_buffer
            if (now - ts) <= self.buffer_timeout
        ]
        
        # Limit buffer size (keep most recent)
        if len(self.orientation_buffer) > self.max_buffer_size:
            self.orientation_buffer = self.orientation_buffer[-self.max_buffer_size:]
        
        # Mark old entries as not new
        self.orientation_buffer = [
            (yaw, dist, ts, False) for yaw, dist, ts, _ in self.orientation_buffer
        ]
        
        if not self.orientation_buffer:
            return new_yaw
        
        # Weighted average: prioritize closest markers and new detections
        weights = []
        yaws = []
        
        for yaw, distance, timestamp, is_new in self.orientation_buffer:
            # Weight inversely proportional to distance (closer = higher weight)
            distance_weight = 1.0 / (distance + 0.1)
            
            # Boost weight for new detections
            new_weight = 2.0 if is_new else 1.0
            
            # Time decay (more recent = higher weight)
            age = now - timestamp
            time_weight = math.exp(-age / self.buffer_timeout)
            
            total_weight = distance_weight * new_weight * time_weight
            weights.append(total_weight)
            yaws.append(yaw)
        
        # Circular mean for angles
        sin_sum = sum(w * math.sin(y) for w, y in zip(weights, yaws))
        cos_sum = sum(w * math.cos(y) for w, y in zip(weights, yaws))
        total_weight = sum(weights)
        
        if total_weight > 0:
            smoothed_yaw = math.atan2(sin_sum / total_weight, cos_sum / total_weight)
        else:
            smoothed_yaw = new_yaw
        
        # Exponential smoothing with previous value
        if self.current_smoothed_yaw is not None:
            # Handle angle wrapping
            diff = smoothed_yaw - self.current_smoothed_yaw
            diff = math.atan2(math.sin(diff), math.cos(diff))
            smoothed_yaw = self.current_smoothed_yaw + self.smoothing_alpha * diff
            smoothed_yaw = math.atan2(math.sin(smoothed_yaw), math.cos(smoothed_yaw))
        
        self.current_smoothed_yaw = smoothed_yaw
        return smoothed_yaw


def main(args=None):
    parser = argparse.ArgumentParser(description='Triangulation Node')
    parser.add_argument('--topic', type=str, default='/detected_markers',
                        help='Input topic for detected markers')
    parsed_args = parser.parse_args(args)
    
    rclpy.init(args=args)
    node = TriangulationNode(topic_name=parsed_args.topic)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
