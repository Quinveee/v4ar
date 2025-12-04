#!/usr/bin/env python3
"""
Unified triangulation node that fuses markers from OAK and mono cameras.
Uses the same triangulation logic as triangulation_node.py.
"""
import rclpy
from rclpy.node import Node
import yaml
from perception_msgs.msg import MarkerPoseArray
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory
import os
import math
import time

from .solvers.new_solver import LeastSquaresSolver
from .solvers.weighted_least_squares import WeightedLeastSquaresSolver
from .utils import compute_average_yaw_from_markers


def load_marker_config(config_path):
    """Load marker world coordinates from YAML config."""
    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        return {int(k): tuple(v) for k, v in data['marker_map'].items()}
    except Exception as e:
        print(f"Failed to load marker config: {e}")
        return {}


class TriangulationUniNode(Node):
    def __init__(self):
        super().__init__('triangulation_uni_node')

        # --- Parameters ---
        self.declare_parameter('marker_config', 'config/markers.yaml')
        self.declare_parameter('oak_weight', 1.0)
        self.declare_parameter('mono_weight', 1.0)
        self.declare_parameter('detection_timeout', 0.5)
        self.declare_parameter('solver_type', 'least_squares')
        self.declare_parameter('weight_power', 2.0)

        config_path = self.get_parameter('marker_config').value
        self.oak_weight = self.get_parameter('oak_weight').value
        self.mono_weight = self.get_parameter('mono_weight').value
        self.detection_timeout = self.get_parameter('detection_timeout').value
        solver_type = self.get_parameter('solver_type').value
        weight_power = self.get_parameter('weight_power').value

        # --- Load config ---
        if not os.path.isabs(config_path):
            package_share = get_package_share_directory('perception')
            config_path = os.path.join(package_share, config_path)

        self.marker_map = load_marker_config(config_path)
        self.get_logger().info(f"Loaded {len(self.marker_map)} markers from {config_path}")

        # --- Initialize solver based on solver_type parameter ---
        if solver_type == 'weighted_least_squares':
            self.solver = WeightedLeastSquaresSolver(weight_power=weight_power)
            self.get_logger().info(f"Using WeightedLeastSquaresSolver (weight_power={weight_power})")
        else:
            self.solver = LeastSquaresSolver()
            self.get_logger().info("Using LeastSquaresSolver (geometric + residual selection)")

        # --- Detection storage ---
        self.oak_detections = {}   # marker_id -> (marker_obj, timestamp)
        self.mono_detections = {}  # marker_id -> (marker_obj, timestamp)

        # --- Yaw state ---
        self.yaw = 0.0

        # --- ROS setup ---
        self.sub_mono = self.create_subscription(
            MarkerPoseArray, '/detected_markers', self._mono_cb, 10)
        self.sub_oak = self.create_subscription(
            MarkerPoseArray, '/oak/detected_markers', self._oak_cb, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/robot_pose_raw', 10)

        self.get_logger().info(f"Weights: OAK={self.oak_weight:.1f}x, Mono={self.mono_weight:.1f}x")

    def _mono_cb(self, msg):
        if self.mono_weight <= 0:
            return
        now = time.time()
        for m in msg.markers:
            if m.id in self.marker_map:
                self.mono_detections[m.id] = (m, now)
        self._process(msg.header)

    def _oak_cb(self, msg):
        if self.oak_weight <= 0:
            return
        now = time.time()
        for m in msg.markers:
            if m.id in self.marker_map:
                self.oak_detections[m.id] = (m, now)
        self._process(msg.header)

    def _fuse_detections(self):
        """Fuse detections from both sources, OAK takes priority for same marker."""
        now = time.time()
        fused = []

        # Collect valid OAK detections
        oak_ids = set()
        for m_id, (m, t) in list(self.oak_detections.items()):
            if now - t <= self.detection_timeout:
                fused.append((m_id, m.distance))
                oak_ids.add(m_id)
            else:
                del self.oak_detections[m_id]

        # Add mono detections not covered by OAK
        mono_ids = set()
        for m_id, (m, t) in list(self.mono_detections.items()):
            if now - t <= self.detection_timeout:
                if m_id not in oak_ids:
                    fused.append((m_id, m.distance))
                mono_ids.add(m_id)
            else:
                del self.mono_detections[m_id]

        return fused, oak_ids, mono_ids

    def _get_all_markers_for_yaw(self, oak_ids, mono_ids):
        """Get all marker objects for yaw computation with weights."""
        markers_with_weights = []

        for m_id in oak_ids:
            if m_id in self.oak_detections:
                m, _ = self.oak_detections[m_id]
                markers_with_weights.append((m_id, m.distance, self.oak_weight))

        for m_id in mono_ids:
            if m_id not in oak_ids and m_id in self.mono_detections:
                m, _ = self.mono_detections[m_id]
                markers_with_weights.append((m_id, m.distance, self.mono_weight))

        return markers_with_weights

    def _process(self, header):
        try:
            fused, oak_ids, mono_ids = self._fuse_detections()

            if len(fused) < 2:
                return

            # Use the same solver logic as triangulation_node
            result = self.solver.solve(fused, self.marker_map)

            if len(result) == 3:
                x, y, _ = result  # Ignore solver's yaw estimate
            else:
                x, y = result

            # Compute yaw using weighted average from all markers
            markers_for_yaw = self._get_all_markers_for_yaw(oak_ids, mono_ids)
            yaw = compute_average_yaw_from_markers(
                x, y, markers_for_yaw, self.marker_map,
                oak_weight=self.oak_weight,
                mono_weight=self.mono_weight
            )
            self.yaw = yaw

            # Publish pose
            self._publish_pose(header, x, y, yaw)

            self.get_logger().info(
                f"Robot pose: x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}° "
                f"(oak={len(oak_ids)}, mono={len(mono_ids) - len(oak_ids & mono_ids)})",
                throttle_duration_sec=0.5
            )

        except Exception as e:
            self.get_logger().error(f"Error in _process: {e}")

    def _publish_pose(self, header, x, y, yaw):
        """Publish the estimated robot pose."""
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = 0.0

        half_yaw = yaw / 2.0
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = math.sin(half_yaw)
        pose_msg.pose.orientation.w = math.cos(half_yaw)

        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TriangulationUniNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
