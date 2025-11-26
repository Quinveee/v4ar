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


# -------------------------------------------------------
#  NODE
# -------------------------------------------------------
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
            f"Loaded {len(self.marker_map)} markers from {config_path}"
        )

        # --- Instantiate solver dynamically ---
        SolverClass = SOLVER_CLASSES.get(solver_type, LeastSquaresSolver)
        self.solver = SolverClass()

        self.get_logger().info(
            f"Using triangulation solver: {SolverClass.__name__}"
        )

        # --- ROS I/O ---
        input_topic = '/detected_markers_buffered' if use_buffered else '/detected_markers'
        self.sub = self.create_subscription(
            MarkerPoseArray, input_topic, self.marker_callback, 10
        )
        self.get_logger().info(f"Subscribed to marker topic: {input_topic}")

        self.pub = self.create_publisher(PoseStamped, '/robot_pose_raw', 10)


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

            # Pick closest tag for yaw computation
            closest_tag = min(
                [m for m in msg.markers if m.id in self.marker_map],
                key=lambda m: m.distance
            )

            yaw = compute_yaw_from_apriltag(closest_tag, self.marker_map)

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
                f"Robot pose: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f} deg"
            )

        except Exception as e:
            self.get_logger().error(f"Error in marker_callback: {e.with_traceback(None)}")


def main(args=None):
    rclpy.init(args=args)
    node = TriangulationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
