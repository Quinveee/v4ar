#!/usr/bin/env python3
"""
Nav2 Navigation Strategy Node.

Integrates with Nav2 stack to navigate to goals while avoiding obstacles detected
by the obstacle detector. Publishes obstacles to Nav2 costmap and sends navigation
goals via Nav2's action interface.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from perception_msgs.msg import ObjectPoseArray
from std_srvs.srv import Empty
import math
import tf2_ros
from tf2_ros import TransformException


class Nav2NavigationStrategy(Node):
    """
    Nav2-based navigation strategy that:
    1. Subscribes to detected obstacles from obstacle detector
    2. Transforms obstacles to world frame
    3. Publishes obstacles to Nav2 costmap (via obstacle layer)
    4. Sends navigation goals to Nav2
    5. Monitors Nav2 navigation status
    """

    def __init__(self):
        super().__init__('nav2_navigation_strategy')

        # Parameters
        self.declare_parameter("goal_tolerance", 0.2)
        self.declare_parameter("obstacle_radius", 0.3)  # Radius around each obstacle
        self.declare_parameter("robot_pose_topic", "/odom_pose_processed")
        self.declare_parameter("obstacle_topic", "/detected_rovers")
        self.declare_parameter("publish_obstacle_costmap", True)
        self.declare_parameter("target_x", 0.0)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_yaw", 0.0)  # Optional yaw angle (default: 0.0)
        self.declare_parameter("auto_navigate", False)  # Auto-navigate on startup

        self.goal_tolerance = self.get_parameter("goal_tolerance").value
        self.obstacle_radius = self.get_parameter("obstacle_radius").value
        robot_pose_topic = self.get_parameter("robot_pose_topic").value
        obstacle_topic = self.get_parameter("obstacle_topic").value

        # Current robot pose
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None
        self.robot_pose_received = False

        # Obstacles in world frame
        self.obstacles_world = []  # List of (x, y) tuples

        # Nav2 Action Client
        self.nav2_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.current_goal_handle = None
        self.navigation_active = False

        # TF Buffer for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.robot_pose_sub = self.create_subscription(
            PoseStamped, robot_pose_topic, self.robot_pose_callback, 10
        )
        self.obstacle_sub = self.create_subscription(
            ObjectPoseArray, obstacle_topic, self.obstacle_callback, 10
        )

        # Publishers
        # Publish obstacles for visualization
        self.obstacle_viz_pub = self.create_publisher(
            ObjectPoseArray, "/nav2/obstacles_world", 10
        )

        # Publish current goal for visualization
        self.goal_viz_pub = self.create_publisher(
            PoseStamped, "/nav2/current_goal", 10
        )

        # Services
        self.navigate_service = self.create_service(
            Empty, "/nav2/navigate_to_goal", self.navigate_service_callback
        )

        self.get_logger().info("Nav2 Navigation Strategy initialized")
        self.get_logger().info(f"Waiting for Nav2 action server...")

        # Wait for Nav2 action server
        if not self.nav2_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn(
                "Nav2 action server not available. Make sure Nav2 is running."
            )
        else:
            self.get_logger().info("Nav2 action server connected!")

        # Auto-navigate if enabled
        auto_nav = self.get_parameter("auto_navigate").value
        if auto_nav:
            target_x = self.get_parameter("target_x").value
            target_y = self.get_parameter("target_y").value
            target_yaw = self.get_parameter("target_yaw").value
            if target_yaw is None or target_yaw == 0.0:
                target_yaw = None
            # Wait a bit for pose to be received
            self.create_timer(2.0, lambda: self.navigate_to_goal(target_x, target_y, target_yaw))

    def robot_pose_callback(self, msg: PoseStamped):
        """Update current robot pose from odometry."""
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y

        # Extract yaw from quaternion
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        self.robot_yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)
        )

        self.robot_pose_received = True

    def obstacle_callback(self, msg: ObjectPoseArray):
        """
        Process detected obstacles and transform to world frame.
        
        Obstacles are published in camera frame, need to transform to world frame.
        """
        if not self.robot_pose_received or self.robot_x is None:
            return

        self.obstacles_world = []

        # Transform obstacles from camera frame to world frame
        try:
            # Try to get transform from camera to base_link, then to world
            # For now, assume camera is aligned with robot (similar to follow.py)
            for obj in msg.rovers:
                # Camera frame: z is forward, x is right, y is down
                # Robot frame: x is forward, y is left
                dx_camera = obj.pose.position.z  # forward in camera frame
                dy_camera = -obj.pose.position.x  # left/right in camera frame

                # Transform to world frame using robot pose
                world_x = self.robot_x + dx_camera * math.cos(self.robot_yaw) - \
                         dy_camera * math.sin(self.robot_yaw)
                world_y = self.robot_y + dx_camera * math.sin(self.robot_yaw) + \
                         dy_camera * math.cos(self.robot_yaw)

                self.obstacles_world.append((world_x, world_y))

        except Exception as e:
            self.get_logger().error(f"Error transforming obstacles: {e}")

        # Publish obstacles in world frame for visualization
        obstacle_msg = ObjectPoseArray()
        obstacle_msg.header.stamp = self.get_clock().now().to_msg()
        obstacle_msg.header.frame_id = "world"
        # Copy obstacles with transformed positions
        for i, (wx, wy) in enumerate(self.obstacles_world):
            if i < len(msg.rovers):
                obj = msg.rovers[i]
                obstacle_msg.rovers.append(obj)
                # Update pose to world frame
                obstacle_msg.rovers[-1].pose.position.x = wx
                obstacle_msg.rovers[-1].pose.position.y = wy
                obstacle_msg.rovers[-1].pose.position.z = 0.0

        self.obstacle_viz_pub.publish(obstacle_msg)

        self.get_logger().debug(
            f"Processed {len(self.obstacles_world)} obstacles in world frame",
            throttle_duration_sec=1.0
        )

    def navigate_to_goal(self, target_x: float, target_y: float, target_yaw: float = None):
        """
        Send navigation goal to Nav2.
        
        Args:
            target_x: Target x position in world frame (meters)
            target_y: Target y position in world frame (meters)
            target_yaw: Target orientation (radians, optional)
        
        Returns:
            True if goal was sent successfully, False otherwise
        """
        if not self.nav2_action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Nav2 action server not available")
            return False

        # Cancel any existing goal
        if self.current_goal_handle is not None:
            self.nav2_action_client.cancel_goal_async(self.current_goal_handle)

        # Create goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.header.frame_id = "map"  # Nav2 typically uses "map" frame
        goal_msg.pose.pose.position.x = float(target_x)
        goal_msg.pose.pose.position.y = float(target_y)
        goal_msg.pose.pose.position.z = 0.0

        # Set orientation
        if target_yaw is not None:
            half_yaw = target_yaw / 2.0
            goal_msg.pose.pose.orientation.x = 0.0
            goal_msg.pose.pose.orientation.y = 0.0
            goal_msg.pose.pose.orientation.z = math.sin(half_yaw)
            goal_msg.pose.pose.orientation.w = math.cos(half_yaw)
        else:
            # Point toward goal if no yaw specified
            if self.robot_x is not None:
                dx = target_x - self.robot_x
                dy = target_y - self.robot_y
                yaw = math.atan2(dy, dx)
                half_yaw = yaw / 2.0
                goal_msg.pose.pose.orientation.z = math.sin(half_yaw)
                goal_msg.pose.pose.orientation.w = math.cos(half_yaw)
            else:
                goal_msg.pose.pose.orientation.w = 1.0

        # Send goal
        self.get_logger().info(
            f"Sending Nav2 goal: ({target_x:.2f}, {target_y:.2f})"
        )

        send_goal_future = self.nav2_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav2_feedback_callback
        )

        # Store future to check later
        self.send_goal_future = send_goal_future

        # Publish goal for visualization
        self.goal_viz_pub.publish(goal_msg.pose)

        # Check if goal was accepted (non-blocking)
        self.create_timer(0.1, self._check_goal_status)

        return True

    def _check_goal_status(self):
        """Check if goal was accepted (called by timer)."""
        if hasattr(self, 'send_goal_future') and self.send_goal_future.done():
            self.current_goal_handle = self.send_goal_future.result()
            if self.current_goal_handle is not None:
                self.navigation_active = True
                self.get_logger().info("Nav2 goal accepted")
            else:
                self.get_logger().error("Nav2 goal rejected")
            # Don't need to cancel timer, it will just return early next time

    def navigate_service_callback(self, request, response):
        """Service callback to trigger navigation to parameter-specified goal."""
        target_x = self.get_parameter("target_x").value
        target_y = self.get_parameter("target_y").value
        target_yaw_param = self.get_parameter("target_yaw").value
        target_yaw = None if (target_yaw_param is None or target_yaw_param == 0.0) else target_yaw_param
        
        success = self.navigate_to_goal(target_x, target_y, target_yaw)
        return response

    def nav2_feedback_callback(self, feedback_msg):
        """Handle Nav2 navigation feedback."""
        feedback = feedback_msg.feedback
        self.get_logger().debug(
            f"Nav2 feedback: distance remaining = {feedback.distance_remaining:.2f}m",
            throttle_duration_sec=1.0
        )

    def is_navigation_active(self) -> bool:
        """Check if navigation is currently active."""
        return self.navigation_active

    def get_navigation_status(self) -> str:
        """Get current navigation status."""
        if not self.navigation_active:
            return "idle"
        if self.current_goal_handle is None:
            return "sending_goal"
        # Check goal status
        if self.current_goal_handle.status == 2:  # ACCEPTED
            return "navigating"
        elif self.current_goal_handle.status == 3:  # EXECUTING
            return "navigating"
        elif self.current_goal_handle.status == 4:  # CANCELED
            return "cancelled"
        elif self.current_goal_handle.status == 5:  # SUCCEEDED
            self.navigation_active = False
            return "succeeded"
        elif self.current_goal_handle.status == 6:  # ABORTED
            self.navigation_active = False
            return "aborted"
        return "unknown"

    def cancel_navigation(self):
        """Cancel current navigation goal."""
        if self.current_goal_handle is not None:
            self.nav2_action_client.cancel_goal_async(self.current_goal_handle)
            self.current_goal_handle = None
            self.navigation_active = False
            self.get_logger().info("Navigation cancelled")

    def is_goal_reached(self, robot_x: float, robot_y: float,
                       target_x: float, target_y: float) -> bool:
        """Check if goal is reached."""
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = math.sqrt(dx * dx + dy * dy)
        return distance < self.goal_tolerance

    def get_obstacles_world(self):
        """Get obstacles in world frame."""
        return self.obstacles_world.copy()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = Nav2NavigationStrategy()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

