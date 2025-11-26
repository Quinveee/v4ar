#!/usr/bin/env python3
"""
Navigation with Avoidance Node using Strategy Pattern.

This node provides a flexible navigation system where different control strategies
can be plugged in without modifying the core ROS node logic.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from perception_msgs.msg import ObjectPoseArray
import math

from .navigation_strategies import (
    BaseNavigationStrategy,
    PotentialFieldStrategy,
    DirectGoalStrategy,
    DWAStrategy
)

# Registry of available strategies
AVAILABLE_STRATEGIES = {
    "potential_field": PotentialFieldStrategy,
    "direct_goal": DirectGoalStrategy,
    "dwa": DWAStrategy,
}


class NavigationWithAvoidanceNode(Node):
    """
    ROS2 node for navigation with obstacle avoidance using pluggable strategies.

    This node handles:
    - ROS communication (subscriptions, publishers, timers)
    - Coordinate transformations (camera frame to world frame)
    - State management (robot pose, obstacles)

    The actual control logic is delegated to a navigation strategy that implements
    the BaseNavigationStrategy interface.
    """

    def __init__(self, strategy: BaseNavigationStrategy = None):
        """
        Initialize the navigation node.

        Args:
            strategy: Optional pre-configured strategy instance. If None, creates
                     strategy based on ROS parameters.

        ROS Parameters:
            strategy_type (str): Type of navigation strategy ('potential_field')
            target_x (float): Target x coordinate in meters
            target_y (float): Target y coordinate in meters
            safe_distance (float): Safe distance from obstacles in meters
            repulse_strength (float): Repulsion strength from obstacles
            goal_tolerance (float): Distance threshold to consider goal reached
            max_linear_velocity (float): Maximum forward velocity in m/s
            angular_gain (float): Proportional gain for angular control
        """
        super().__init__('navigation_with_avoidance_node')

        # Declare ROS parameters with defaults
        self.declare_parameter("strategy_type", "dwa")
        self.declare_parameter("target_x", 0.0)
        self.declare_parameter("target_y", 3.0)
        self.declare_parameter("safe_distance", 1.2)
        self.declare_parameter("repulse_strength", 1.5)
        self.declare_parameter("goal_tolerance", 0.1)
        self.declare_parameter("max_linear_velocity", 0.1)
        self.declare_parameter("angular_gain", 0.5)

        # Get parameter values
        strategy_type = self.get_parameter("strategy_type").value

        # Initialize or create strategy
        if strategy is None:
            if strategy_type not in AVAILABLE_STRATEGIES:
                self.get_logger().warn(
                    f"Unknown strategy '{strategy_type}', defaulting to 'potential_field'"
                )
                strategy_type = "potential_field"

            # Create strategy with parameters
            strategy_class = AVAILABLE_STRATEGIES[strategy_type]
            self.strategy = strategy_class(
                safe_distance=self.get_parameter("safe_distance").value,
                repulse_strength=self.get_parameter("repulse_strength").value,
                goal_tolerance=self.get_parameter("goal_tolerance").value,
                max_linear_velocity=self.get_parameter("max_linear_velocity").value,
                angular_gain=self.get_parameter("angular_gain").value
            )
        else:
            self.strategy = strategy

        # Robot state
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        # Obstacles in world frame
        self.obstacles = []  # List of (x, y) tuples

        # ROS Subscriptions
        self.sub_pose = self.create_subscription(
            PoseStamped, "/robot_pose", self.pose_callback, 10
        )
        self.sub_rovers = self.create_subscription(
            ObjectPoseArray, "/detected_rovers", self.rover_callback, 10
        )

        # ROS Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.heading_pub = self.create_publisher(Vector3, "/control/heading_vector", 10)

        # Control loop timer (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)

        # Log initialization info
        target_x = self.get_parameter("target_x").value
        target_y = self.get_parameter("target_y").value

        self.get_logger().info(
            f"NavigationWithAvoidanceNode initialized with strategy: {strategy_type}"
        )
        self.get_logger().info(
            f"Target: ({target_x:.2f}, {target_y:.2f})"
        )

    # -------------------------------------------------------------------
    # ROS Callbacks
    # -------------------------------------------------------------------

    def pose_callback(self, msg: PoseStamped):
        """Update robot pose from incoming PoseStamped message."""
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y

        q = msg.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def rover_callback(self, msg: ObjectPoseArray):
        """
        Update obstacle list from detected rovers.

        Converts rover positions from camera frame to world frame.
        """
        self.obstacles = []

        for obj in msg.rovers:
            # TODO: What happens when we move the camera???
            # Rover is relative to the camera — assume camera aligned with robot
            dx = obj.pose.position.z     # forward
            dy = -obj.pose.position.x    # left/right

            # Convert to world frame
            world_x = self.robot_x + dx * \
                math.cos(self.robot_yaw) - dy * math.sin(self.robot_yaw)
            world_y = self.robot_y + dx * \
                math.sin(self.robot_yaw) + dy * math.cos(self.robot_yaw)

            self.obstacles.append((world_x, world_y))

    # -------------------------------------------------------------------
    # Control Loop
    # -------------------------------------------------------------------

    def control_loop(self):
        """
        Main control loop - delegates to the navigation strategy.

        This method is called at a fixed rate (20 Hz) and:
        1. Checks if robot pose is available
        2. Calls the strategy to compute control commands
        3. Publishes the commands and optional visualization data
        """
        # Wait for robot pose to be available
        if self.robot_x is None:
            return

        # Read target parameters dynamically (allows live updates)
        target_x = self.get_parameter("target_x").value
        target_y = self.get_parameter("target_y").value

        # Debug logging (throttled)
        self.get_logger().debug(
            f"Robot at ({self.robot_x:.2f}, {self.robot_y:.2f}), "
            f"Target: ({target_x:.2f}, {target_y:.2f})",
            throttle_duration_sec=1.0
        )

        # Delegate control computation to the strategy
        cmd, heading_vec, goal_reached = self.strategy.compute_control(
            robot_x=self.robot_x,
            robot_y=self.robot_y,
            robot_yaw=self.robot_yaw,
            target_x=target_x,
            target_y=target_y,
            obstacles=self.obstacles
        )

        # Publish velocity command
        self.cmd_pub.publish(cmd)

        # Publish heading vector for visualization (if provided)
        if heading_vec is not None:
            self.heading_pub.publish(heading_vec)

        # Log when goal is reached
        if goal_reached:
            self.get_logger().info("Target reached.", throttle_duration_sec=1.0)


def main(args=None):
    """Main entry point for the navigation node."""
    rclpy.init(args=args)
    node = NavigationWithAvoidanceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
