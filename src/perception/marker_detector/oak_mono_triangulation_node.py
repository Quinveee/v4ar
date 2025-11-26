#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import yaml
import numpy as np
from perception_msgs.msg import MarkerPoseArray
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory
import os
from collections import defaultdict


# Import solvers
from solvers.weighted_least_squares import WeightedLeastSquaresSolver
from solvers.new_solver import LeastSquaresSolver

SOLVER_CLASSES = {
    'least_squares': LeastSquaresSolver,
    'weighted': WeightedLeastSquaresSolver,
}


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
        # NEW: Store latest detections from both cameras
        self.mono_detections = {}  # {marker_id: (distance, timestamp)}
        self.oak_detections = {}   # {marker_id: (distance, timestamp)}
        self.detection_timeout = 0.5  # seconds
        
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
                data = yaml.safe_load(f)
            return {int(k): tuple(v) for k, v in data['marker_map'].items()}
        except Exception as e:
            self.get_logger().error(f"Failed to load marker config: {e}")
            return {}

    # -----------------------------------------------------

    def mono_marker_callback(self, msg: MarkerPoseArray):
        """Callback for mono camera detections"""
        now = self.get_clock().now().nanoseconds / 1e9
        
        for m in msg.markers:
            if m.id in self.marker_map:
                self.mono_detections[m.id] = (m.distance, now)
        
        # Trigger fusion after receiving mono data
        self._fuse_and_triangulate(msg.header)

    def oak_marker_callback(self, msg: MarkerPoseArray):
        """Callback for OAK camera detections (higher priority)"""
        now = self.get_clock().now().nanoseconds / 1e9
        
        for m in msg.markers:
            if m.id in self.marker_map:
                self.oak_detections[m.id] = (m.distance, now)
        
        # Trigger fusion after receiving OAK data
        self._fuse_and_triangulate(msg.header)

    def _fuse_and_triangulate(self, header):
        """Fuse detections from both cameras and triangulate"""
        try:
            now = self.get_clock().now().nanoseconds / 1e9
            
            # Build fused detection list with weights
            fused_detections = []
            
            # Add OAK detections (higher weight)
            for marker_id, (distance, timestamp) in self.oak_detections.items():
                if (now - timestamp) <= self.detection_timeout:
                    fused_detections.append((marker_id, distance, self.oak_weight))
            
            # Add mono detections (lower weight, only if not seen by OAK recently)
            for marker_id, (distance, timestamp) in self.mono_detections.items():
                if (now - timestamp) <= self.detection_timeout:
                    # Only use mono if OAK hasn't seen this marker recently
                    if marker_id not in self.oak_detections or \
                       (now - self.oak_detections[marker_id][1]) > self.detection_timeout:
                        fused_detections.append((marker_id, distance, self.mono_weight))
            
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
            oak_ids = [m_id for m_id, _, _ in fused_detections if _ == self.oak_weight]
            mono_ids = [m_id for m_id, _, _ in fused_detections if _ == self.mono_weight]
            self.get_logger().info(
                f"Fused detections: OAK={oak_ids}, Mono={mono_ids}",
                throttle_duration_sec=1.0
            )
            
            # Check if solver supports weights
            if hasattr(self.solver, 'solve_weighted'):
                # Use weighted solver (detections with weights)
                xy = self.solver.solve_weighted(fused_detections, self.marker_map)
            else:
                # Fallback: Use regular solver (just use id and distance, ignore weights)
                detections = [(m_id, dist) for m_id, dist, _ in fused_detections]
                xy = self.solver.solve(detections, self.marker_map)
                self.get_logger().warn(
                    "Solver doesn't support weights, using unweighted triangulation",
                    throttle_duration_sec=10.0
                )
            
            x, y = xy
            
            # Debug logging
            self.get_logger().info(f"Triangulated: x={x:.4f}, y={y:.4f}")
            
            pose_msg = PoseStamped()
            pose_msg.header = header
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position.x = float(x)
            pose_msg.pose.position.y = float(y)
            pose_msg.pose.orientation.w = 1.0
            
            self.pub.publish(pose_msg)
            self.get_logger().info(
                f"Robot position: x={x:.2f}, y={y:.2f}",
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
    
    # Set weights from CLI
    from rclpy.parameter import Parameter
    node.set_parameters([
        Parameter('oak_weight', Parameter.Type.DOUBLE, parsed.oak_weight),
        Parameter('mono_weight', Parameter.Type.DOUBLE, parsed.mono_weight),
    ])
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()