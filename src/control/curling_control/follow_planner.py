#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Vector3
from std_msgs.msg import Bool
from perception_msgs.msg import ObjectPoseArray
import math

from .navigation_strategies import (
    BaseNavigationStrategy,
    PotentialFieldStrategy,
    DirectGoalStrategy,
    DWAStrategy,
    IntegratedYawDirectGoalStrategy,
    GridDirectGoalStrategy
)

# Available strategies
AVAILABLE_STRATEGIES = {
    "potential_field": PotentialFieldStrategy,
    "direct_goal": DirectGoalStrategy,
    "integrated_yaw_direct_goal": IntegratedYawDirectGoalStrategy,
    "direct_goal_integrated_yaw": IntegratedYawDirectGoalStrategy,
    "dwa": DWAStrategy,
    "discretized": GridDirectGoalStrategy
}


class NavigationWithAvoidanceNode(Node):
    """Navigation node compatible with hierarchical (planner → navigator) setup."""

    def __init__(self, strategy: BaseNavigationStrategy = None):
        super().__init__("navigation_with_avoidance_node")

        # Parameters (defaults are mostly ignored if using /local_goal)
        self.declare_parameter("strategy_type", "direct_goal")
        self.declare_parameter("safe_distance", 1.0)
        self.declare_parameter("repulse_strength", 1.5)
        self.declare_parameter("goal_tolerance", 0.1)
        self.declare_parameter("max_linear_velocity", 0.2)
        self.declare_parameter("angular_gain", 0.8)

        strategy_type = self.get_parameter("strategy_type").value

        # Strategy selection
        if strategy is None:
            if strategy_type not in AVAILABLE_STRATEGIES:
                self.get_logger().warn(
                    f"Unknown strategy '{strategy_type}', defaulting to 'direct_goal'"
                )
                strategy_type = "direct_goal"

            strategy_class = AVAILABLE_STRATEGIES[strategy_type]
            self.strategy = strategy_class(
                safe_distance=self.get_parameter("safe_distance").value,
                repulse_strength=self.get_parameter("repulse_strength").value,
                goal_tolerance=self.get_parameter("goal_tolerance").value,
                max_linear_velocity=self.get_parameter("max_linear_velocity").value,
                angular_gain=self.get_parameter("angular_gain").value,
            )
        else:
            self.strategy = strategy

        # Internal robot state
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None
        self.obstacles = []
        self.local_goal = None
        self.global_goal_reached = False

        # Subscribers
        self.create_subscription(PoseStamped, "/robot_pose", self.pose_callback, 10)
        self.create_subscription(ObjectPoseArray, "/detected_rovers", self.rover_callback, 10)
        self.create_subscription(PoseStamped, "/local_goal", self.local_goal_callback, 10)
        self.create_subscription(Bool, "/goal_reached", self.goal_reached_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.heading_pub = self.create_publisher(Vector3, "/control/heading_vector", 10)

        # Timer (20 Hz)
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info(f"Navigation node initialized with strategy: {strategy_type}")

    # -------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------

    def pose_callback(self, msg: PoseStamped):
        """Update robot pose."""
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        q = msg.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def rover_callback(self, msg: ObjectPoseArray):
        """Convert obstacle positions into world frame coordinates."""
        if self.robot_x is None or self.robot_y is None or self.robot_yaw is None:
            return

        self.obstacles.clear()
        for obj in msg.rovers:
            dx = obj.pose.position.z
            dy = -obj.pose.position.x
            world_x = self.robot_x + dx * math.cos(self.robot_yaw) - dy * math.sin(self.robot_yaw)
            world_y = self.robot_y + dx * math.sin(self.robot_yaw) + dy * math.cos(self.robot_yaw)
            self.obstacles.append((world_x, world_y))

    def local_goal_callback(self, msg: PoseStamped):
        """Receive intermediate target (from planner)."""
        self.local_goal = (msg.pose.position.x, msg.pose.position.y)
        self.global_goal_reached = False
        self.get_logger().info(f"🎯 New local goal received: ({self.local_goal[0]:.2f}, {self.local_goal[1]:.2f})")

    def goal_reached_callback(self, msg: Bool):
        """Stop motion if the global goal is reached."""
        if msg.data:
            self.global_goal_reached = True
            self.get_logger().info(" Global goal reached! Stopping robot.")
            stop_cmd = Twist()
            self.cmd_pub.publish(stop_cmd)

    # -------------------------------------------------------------------
    # Control Loop
    # -------------------------------------------------------------------

    def control_loop(self):
        """Delegate control to navigation strategy."""
        if self.global_goal_reached:
            return  # stop everything

        if self.robot_x is None or self.local_goal is None:
            return  # wait for both pose and local goal

        target_x, target_y = self.local_goal

        cmd, heading_vec, goal_reached = self.strategy.compute_control(
            robot_x=self.robot_x,
            robot_y=self.robot_y,
            robot_yaw=self.robot_yaw,
            target_x=target_x,
            target_y=target_y,
            obstacles=self.obstacles
        )

        self.cmd_pub.publish(cmd)

        if heading_vec is not None:
            self.heading_pub.publish(heading_vec)

        if goal_reached:
            self.get_logger().info("✅ Local goal reached.", throttle_duration_sec=1.0)

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------

    def destroy_node(self):
        stop = Twist()
        self.cmd_pub.publish(stop)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = NavigationWithAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()