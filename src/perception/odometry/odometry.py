#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.time import Time
import math

try:
    # Try relative import (works when installed as package via ros2 run)
    from .odometry_strategies import (
        CmdVelStrategy,
        RF2OStrategy
    )
except ImportError:
    # Fallback to absolute import (works when running directly)
    import sys
    import os
    # Add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from odometry.odometry_strategies import (
        CmdVelStrategy,
        RF2OStrategy
    )

STRATEGIES = {
    "cmd_vel": CmdVelStrategy,
    "rf2o": RF2OStrategy,
}


def yaw_to_quaternion(yaw):
    """Convert yaw angle to quaternion."""
    half = yaw / 2
    return (0.0, 0.0, math.sin(half), math.cos(half))


def quaternion_to_yaw(qx, qy, qz, qw):
    """Convert quaternion to yaw angle."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                     1.0 - 2.0 * (qy * qy + qz * qz))


class OdometryNode(Node):
    def __init__(self):
        super().__init__("odometry_node")

        # Parameters
        self.declare_parameter("strategy_type", "cmd_vel")
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("triangulation_topic", "/robot_pose_raw")

        strategy_type = self.get_parameter("strategy_type").value
        Strategy = STRATEGIES.get(strategy_type, CmdVelStrategy)
        self.strategy = Strategy()
        self.strategy_type = strategy_type
        self.get_logger().info(f"Using odometry strategy: {Strategy.__name__}")

        # Internal vars
        self.last_update_time = None
        self.initialized_from_triangulation = False

        # ROS setup - subscribe to triangulation for initial pose
        triangulation_topic = self.get_parameter("triangulation_topic").value
        self.triangulation_sub = self.create_subscription(
            PoseStamped, triangulation_topic, self.triangulation_callback, 10
        )

        # Strategy-specific subscriptions
        if strategy_type == "cmd_vel":
            self.cmd_vel_sub = self.create_subscription(
                Twist, "/cmd_vel", self.cmd_vel_callback, 10
            )
        elif strategy_type == "rf2o":
            self.rf2o_sub = self.create_subscription(
                Odometry, "/rf2o_laser_odometry", self.rf2o_callback, 10
            )

        # Publisher - consistent topic name
        topic_name = "/odom_pose_processed"
        self.pose_pub = self.create_publisher(PoseStamped, topic_name, 10)
        self.get_logger().info(f"Publishing odometry to: {topic_name}")

        # Timer for periodic publishing
        rate = self.get_parameter("publish_rate").value
        self.create_timer(1.0 / rate, self.timer_callback)

    def triangulation_callback(self, msg: PoseStamped):
        """Initialize odometry with first triangulation measurement."""
        if not self.initialized_from_triangulation:
            x = msg.pose.position.x
            y = msg.pose.position.y
            
            # Extract yaw from quaternion
            qx = msg.pose.orientation.x
            qy = msg.pose.orientation.y
            qz = msg.pose.orientation.z
            qw = msg.pose.orientation.w
            theta = quaternion_to_yaw(qx, qy, qz, qw)

            self.strategy.initialize(x, y, theta)
            self.initialized_from_triangulation = True
            self.get_logger().info(
                f"Initialized odometry from triangulation: x={x:.2f}, y={y:.2f}, theta={theta:.2f}"
            )

    def cmd_vel_callback(self, msg: Twist):
        """Handle velocity commands."""
        if self.strategy_type == "cmd_vel":
            if self.strategy.is_initialized():
                # Store the latest cmd_vel values in the strategy
                self.strategy.v = msg.linear.x
                self.strategy.w = msg.angular.z
                # Update with current time
                now = self.get_clock().now()
                self.strategy.update(msg, now)
                self.get_logger().debug(
                    f"Received cmd_vel: v={msg.linear.x:.2f}, w={msg.angular.z:.2f}"
                )

    def rf2o_callback(self, msg: Odometry):
        """Handle rf2o laser odometry."""
        if self.strategy.is_initialized():
            self.strategy.update(msg)

    def timer_callback(self):
        """Periodically publish current odometry pose."""
        if not self.strategy.is_initialized():
            # Log once that we're waiting for initialization
            if not hasattr(self, '_logged_waiting'):
                self.get_logger().warn(
                    "Odometry not initialized yet. Waiting for triangulation message on "
                    f"{self.get_parameter('triangulation_topic').value}"
                )
                self._logged_waiting = True
            return

        now = self.get_clock().now()
        
        # For cmd_vel strategy, continuously update using last known velocities
        # This ensures odometry updates even if cmd_vel messages are sparse
        if self.strategy_type == "cmd_vel" and self.strategy.is_initialized():
            # Use the last known cmd_vel values stored in the strategy
            from geometry_msgs.msg import Twist
            dummy_msg = Twist()
            dummy_msg.linear.x = getattr(self.strategy, 'v', 0.0)
            dummy_msg.angular.z = getattr(self.strategy, 'w', 0.0)
            self.strategy.update(dummy_msg, now)
        
        x, y, theta = self.strategy.get_pose()

        out = PoseStamped()
        out.header.stamp = now.to_msg()
        out.header.frame_id = "world"
        out.pose.position.x = x
        out.pose.position.y = y
        qx, qy, qz, qw = yaw_to_quaternion(theta)
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        self.pose_pub.publish(out)
        
        # Log periodically for debugging (every 50 publishes = ~2.5 seconds at 20Hz)
        if not hasattr(self, '_publish_count'):
            self._publish_count = 0
        self._publish_count += 1
        if self._publish_count <= 5 or self._publish_count % 50 == 0:
            self.get_logger().info(
                f"Publishing odometry pose: x={x:.2f}, y={y:.2f}, theta={theta:.2f} "
                f"(publish #{self._publish_count})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

