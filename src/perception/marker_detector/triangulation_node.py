#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import yaml
import numpy as np
from perception_msgs.msg import MarkerPoseArray
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory
import os


# Import solvers
# from .solvers.least_squares_solver import LeastSquaresSolver
from solvers.weighted_least_squares import WeightedLeastSquaresSolver
from solvers.new_solver import LeastSquaresSolver

SOLVER_CLASSES = {
    'least_squares': LeastSquaresSolver,
    'weighted': WeightedLeastSquaresSolver,
    # 'two_marker_circle': TwoMarkerCircleSolver,
}


class TriangulationNode(Node):
    def __init__(self):
        super().__init__('triangulation_node')

        # --- Parameters ---
        self.declare_parameter('marker_config', 'config/markers.yaml')
        self.declare_parameter('solver_type', 'least_squares')
        self.declare_parameter('use_buffered_markers', False)
        config_path = self.get_parameter('marker_config').value
        solver_type = self.get_parameter('solver_type').value
        use_buffered = self.get_parameter('use_buffered_markers').value

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

        # --- ROS I/O ---
        input_topic = '/detected_markers_buffered' if use_buffered else '/detected_markers'
        self.sub = self.create_subscription(
            MarkerPoseArray, input_topic, self.marker_callback, 10)
        self.get_logger().info(f"Subscribed to marker topic: {input_topic}")
        self.pub = self.create_publisher(PoseStamped, '/robot_pose', 10)

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

    def marker_callback(self, msg: MarkerPoseArray):
        try:
            detections = [(m.id, m.distance)
                          for m in msg.markers if m.id in self.marker_map]

            self.get_logger().info(f"Marker map: {self.marker_map}")

            if len(detections) < 2:
                self.get_logger().warn("Need at least 2 known markers for triangulation.")
                return

            xy = self.solver.solve(detections, self.marker_map)
            x, y = xy

            # Debug logging - log solver output
            self.get_logger().info(f"Solver output: x={x:.4f}, y={y:.4f}")
            self.get_logger().info(f"Markers: {msg.markers}")

            pose_msg = PoseStamped()
            pose_msg.header = msg.header
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position.x = float(x)
            pose_msg.pose.position.y = float(y)
            pose_msg.pose.orientation.w = 1.0

            self.pub.publish(pose_msg)
            self.get_logger().info(
                f"Triangulated robot position: x={x:.2f}, y={y:.2f}")
        except Exception as e:
            self.get_logger().error(f"Error in marker_callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TriangulationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
