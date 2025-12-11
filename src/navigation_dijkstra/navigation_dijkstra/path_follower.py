#!/usr/bin/env python3
"""
Path Follower Node for UGV Rover

This node implements a simple proportional controller that follows a planned path.
It subscribes to a path (nav_msgs/Path) and odometry, then publishes velocity commands
to drive the rover along the path by following waypoints sequentially.

Topics:
    Subscribes:
        /planned_path (nav_msgs/Path): Path to follow
        /odom (nav_msgs/Odometry): Robot odometry for current pose
    
    Publishes:
        /cmd_vel (geometry_msgs/Twist): Velocity commands

Parameters:
    k_linear (float): Proportional gain for linear velocity (default: 0.5)
    k_angular (float): Proportional gain for angular velocity (default: 2.0)
    max_linear_speed (float): Maximum linear velocity in m/s (default: 0.3)
    max_angular_speed (float): Maximum angular velocity in rad/s (default: 1.0)
    waypoint_threshold (float): Distance threshold to advance to next waypoint in meters (default: 0.15)
    goal_threshold (float): Distance threshold to consider goal reached in meters (default: 0.1)
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class Pose2D:
    """Simple 2D pose representation."""
    x: float
    y: float
    theta: float  # yaw angle in radians


class PathFollower(Node):
    """Simple waypoint-following controller using proportional control."""
    
    def __init__(self):
        super().__init__('path_follower')
        
        # Declare parameters
        self.declare_parameter('k_linear', 0.5)
        self.declare_parameter('k_angular', 2.0)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('waypoint_threshold', 0.15)
        self.declare_parameter('goal_threshold', 0.1)
        
        # Get parameters
        self.k_linear = self.get_parameter('k_linear').value
        self.k_angular = self.get_parameter('k_angular').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.waypoint_threshold = self.get_parameter('waypoint_threshold').value
        self.goal_threshold = self.get_parameter('goal_threshold').value
        
        # State
        self.current_pose: Optional[Pose2D] = None
        self.path: Optional[Path] = None
        self.current_waypoint_idx: int = 0
        self.goal_reached: bool = False
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribers
        self.path_sub = self.create_subscription(
            Path, '/planned_path', self.path_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Control timer (50 Hz)
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info('Path follower initialized')
        self.get_logger().info(f'Control gains: k_linear={self.k_linear}, k_angular={self.k_angular}')
        self.get_logger().info(f'Speed limits: linear={self.max_linear_speed} m/s, '
                               f'angular={self.max_angular_speed} rad/s')
    
    def quaternion_to_yaw(self, x: float, y: float, z: float, w: float) -> float:
        """
        Convert quaternion to yaw angle (rotation around Z-axis).
        
        Uses the formula:
            yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        
        Returns yaw in radians [-pi, pi].
        """
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def odom_callback(self, msg: Odometry):
        """
        Update current robot pose from odometry.
        
        Extracts (x, y) position and yaw angle from the odometry message.
        """
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        
        yaw = self.quaternion_to_yaw(ori.x, ori.y, ori.z, ori.w)
        
        self.current_pose = Pose2D(pos.x, pos.y, yaw)
    
    def path_callback(self, msg: Path):
        """
        Receive a new path to follow.
        
        Resets the waypoint index and goal reached flag.
        """
        if len(msg.poses) == 0:
            self.get_logger().warn('Received empty path!')
            self.path = None
            return
        
        self.path = msg
        self.current_waypoint_idx = 0
        self.goal_reached = False
        
        self.get_logger().info(f'Received new path with {len(msg.poses)} waypoints')
    
    def normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-pi, pi] range.
        
        This ensures that heading errors are computed correctly across
        the -pi/pi boundary.
        """
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def compute_control(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """
        Compute velocity commands to drive toward a target waypoint.
        
        Uses a simple proportional controller:
            1. Calculate distance and heading to target
            2. Compute heading error (difference between current and desired heading)
            3. Apply proportional control with speed limits:
                - linear_vel = k_linear * distance (clamped to max_linear_speed)
                - angular_vel = k_angular * heading_error (clamped to max_angular_speed)
        
        Args:
            target_x: Target x coordinate in world frame (meters)
            target_y: Target y coordinate in world frame (meters)
        
        Returns:
            (linear_velocity, angular_velocity) tuple
        """
        if self.current_pose is None:
            return (0.0, 0.0)
        
        # Calculate vector from robot to target
        dx = target_x - self.current_pose.x
        dy = target_y - self.current_pose.y
        
        # Calculate distance to target
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Calculate desired heading (angle from robot to target)
        desired_heading = math.atan2(dy, dx)
        
        # Calculate heading error (how much we need to turn)
        heading_error = self.normalize_angle(desired_heading - self.current_pose.theta)
        
        # Proportional control for linear velocity
        # Scale by distance, but limit to max speed
        linear_vel = self.k_linear * distance
        linear_vel = max(-self.max_linear_speed, min(self.max_linear_speed, linear_vel))
        
        # Reduce linear speed when turning sharply
        # This helps prevent the robot from overshooting waypoints during turns
        if abs(heading_error) > math.pi / 4:  # More than 45 degrees off
            linear_vel *= 0.5
        
        # Proportional control for angular velocity
        angular_vel = self.k_angular * heading_error
        angular_vel = max(-self.max_angular_speed, min(self.max_angular_speed, angular_vel))
        
        return (linear_vel, angular_vel)
    
    def control_loop(self):
        """
        Main control loop (called at 50 Hz).
        
        Logic:
            1. Check if we have a valid path and current pose
            2. Get current target waypoint
            3. Compute distance to waypoint
            4. If close enough, advance to next waypoint
            5. If last waypoint reached, stop
            6. Otherwise, compute and publish velocity commands
        """
        # Don't do anything if goal already reached
        if self.goal_reached:
            return
        
        # Check if we have necessary data
        if self.path is None or self.current_pose is None:
            return
        
        if len(self.path.poses) == 0:
            return
        
        # Check if we've gone through all waypoints
        if self.current_waypoint_idx >= len(self.path.poses):
            if not self.goal_reached:
                self.get_logger().info('Goal reached! Stopping.')
                self.goal_reached = True
                
                # Publish zero velocity
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)
            return
        
        # Get current target waypoint
        target_pose = self.path.poses[self.current_waypoint_idx].pose
        target_x = target_pose.position.x
        target_y = target_pose.position.y
        
        # Calculate distance to current waypoint
        dx = target_x - self.current_pose.x
        dy = target_y - self.current_pose.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Check if we're close enough to advance to next waypoint
        # Use different thresholds for intermediate waypoints vs final goal
        is_last_waypoint = (self.current_waypoint_idx == len(self.path.poses) - 1)
        threshold = self.goal_threshold if is_last_waypoint else self.waypoint_threshold
        
        if distance < threshold:
            self.current_waypoint_idx += 1
            
            if self.current_waypoint_idx < len(self.path.poses):
                self.get_logger().info(f'Reached waypoint {self.current_waypoint_idx}/{len(self.path.poses)}')
            else:
                self.get_logger().info('Goal reached! Stopping.')
                self.goal_reached = True
                
                # Publish zero velocity
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd)
            return
        
        # Compute velocity commands
        linear_vel, angular_vel = self.compute_control(target_x, target_y)
        
        # Publish velocity command
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Make sure to stop the robot on shutdown
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        node.cmd_vel_pub.publish(cmd)
        
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
