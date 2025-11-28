#!/usr/bin/env python3
"""Control node for robot navigation from start to target position."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.time import Time
import math

from control_strategies import GoToTargetStrategy


def yaw_to_quaternion(yaw):
    """Convert yaw angle to quaternion."""
    half = yaw / 2
    return (0.0, 0.0, math.sin(half), math.cos(half))


def quaternion_to_yaw(qx, qy, qz, qw):
    """Convert quaternion to yaw angle."""
    return math.atan2(2.0 * (qw * qz + qx * qy),
                     1.0 - 2.0 * (qy * qy + qz * qz))


class ControlNode(Node):
    """Control node that navigates robot from start to target position."""

    def __init__(self):
        super().__init__("control_node")

        # Parameters
        self.declare_parameter("start_x", 0.0)
        self.declare_parameter("start_y", 0.0)
        self.declare_parameter("start_theta", 0.0)
        self.declare_parameter("target_x", 3.0)
        self.declare_parameter("target_y", 4.5)
        self.declare_parameter("use_odometry_update", True)
        self.declare_parameter("update_rate", 20.0)
        self.declare_parameter("pose_topic", "/odom_pose_processed")
        self.declare_parameter("max_linear_velocity_driving", 0.2)  # m/s - speed after orientation
        self.declare_parameter("max_angular_velocity", 1.0)  # rad/s

        # Get parameters
        start_x = self.get_parameter("start_x").value
        start_y = self.get_parameter("start_y").value
        start_theta = self.get_parameter("start_theta").value
        target_x = self.get_parameter("target_x").value
        target_y = self.get_parameter("target_y").value
        use_odometry = self.get_parameter("use_odometry_update").value
        update_rate = self.get_parameter("update_rate").value
        pose_topic = self.get_parameter("pose_topic").value
        max_linear_velocity_driving = self.get_parameter("max_linear_velocity_driving").value
        max_angular_velocity = self.get_parameter("max_angular_velocity").value

        # Initialize control strategy
        self.strategy = GoToTargetStrategy(
            use_odometry_update=use_odometry,
            max_linear_velocity_driving=max_linear_velocity_driving,
            max_angular_velocity=max_angular_velocity
        )
        self.strategy.initialize(start_x, start_y, start_theta, target_x, target_y)

        self.get_logger().info(
            f"Control initialized: start=({start_x:.2f}, {start_y:.2f}, {math.degrees(start_theta):.1f}°), "
            f"target=({target_x:.2f}, {target_y:.2f}), "
            f"driving_speed={max_linear_velocity_driving:.2f}m/s"
        )

        # Current pose from odometry (if available)
        self.current_x = start_x
        self.current_y = start_y
        self.current_theta = start_theta
        self.pose_initialized = False

        # Subscriptions
        self.pose_sub = self.create_subscription(
            PoseStamped, pose_topic, self.pose_callback, 10
        )
        self.get_logger().info(f"Subscribed to pose: {pose_topic}")

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # Temporary topics for visualization
        self.control_pose_pub = self.create_publisher(
            PoseStamped, "/control/temp_pose", 10
        )
        self.control_trajectory_pub = self.create_publisher(
            PoseStamped, "/control/temp_trajectory", 10
        )

        # Trajectory storage for visualization
        self.trajectory = []

        # Timer for control loop
        self.create_timer(1.0 / update_rate, self.control_loop)
        self.last_update_time = None

        self.get_logger().info("Control node started")

    def pose_callback(self, msg: PoseStamped):
        """Update current pose from odometry."""
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        self.current_theta = quaternion_to_yaw(qx, qy, qz, qw)
        
        self.pose_initialized = True

    def control_loop(self):
        """Main control loop - computes and publishes control commands."""
        if not self.pose_initialized:
            # Use strategy's internal pose estimate if odometry not available
            self.current_x, self.current_y, self.current_theta = self.strategy.get_current_pose()
        
        # Compute control command
        linear_vel, angular_vel = self.strategy.compute_control(
            self.current_x, self.current_y, self.current_theta
        )

        # Update strategy's internal pose estimate using cmd_vel (if enabled)
        now = self.get_clock().now()
        if self.last_update_time is not None:
            dt = (now - self.last_update_time).nanoseconds / 1e9
            self.strategy.update_from_cmd_vel(linear_vel, angular_vel, dt)
        self.last_update_time = now

        # Publish control command
        cmd_msg = Twist()
        cmd_msg.linear.x = float(linear_vel)
        cmd_msg.angular.z = float(angular_vel)
        self.cmd_vel_pub.publish(cmd_msg)

        # Publish temporary pose for visualization
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now.to_msg()
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = self.current_x
        pose_msg.pose.position.y = self.current_y
        qx, qy, qz, qw = yaw_to_quaternion(self.current_theta)
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        self.control_pose_pub.publish(pose_msg)

        # Add to trajectory and publish
        self.trajectory.append((self.current_x, self.current_y))
        if len(self.trajectory) > 1000:  # Limit trajectory size
            self.trajectory = self.trajectory[-1000:]
        
        # Publish trajectory point
        traj_msg = PoseStamped()
        traj_msg.header.stamp = now.to_msg()
        traj_msg.header.frame_id = "world"
        traj_msg.pose.position.x = self.current_x
        traj_msg.pose.position.y = self.current_y
        self.control_trajectory_pub.publish(traj_msg)

        # Log status
        phase = self.strategy.get_phase()
        distance = math.sqrt(
            (self.strategy.target_x - self.current_x)**2 +
            (self.strategy.target_y - self.current_y)**2
        )
        
        if self.strategy.is_target_reached(self.current_x, self.current_y):
            self.get_logger().info(
                f"Target reached! Final position: ({self.current_x:.2f}, {self.current_y:.2f})"
            )
        else:
            self.get_logger().debug(
                f"Control: phase={phase}, distance={distance:.2f}m, "
                f"v={linear_vel:.2f}m/s, w={math.degrees(angular_vel):.1f}°/s"
            )


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

