#!/usr/bin/env python3
"""
Simple map-based navigator - no Nav2 required!

Just give it a goal, and it drives there avoiding obstacles.
Uses your existing localization (/robot_pose) and obstacle detection (/detected_rovers).
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from perception_msgs.msg import ObjectPoseArray
import math


class SimpleNavigator(Node):
    def __init__(self):
        super().__init__('simple_navigator')
        
        # Parameters
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_tolerance', 0.15)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('obstacle_distance', 0.5)
        
        # State
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None
        self.obstacles = []
        self.goal_reached = False
        
        # ROS interfaces
        self.sub_pose = self.create_subscription(
            PoseStamped, '/robot_pose', self.pose_callback, 10)
        self.sub_obstacles = self.create_subscription(
            ObjectPoseArray, '/detected_rovers', self.obstacle_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info(f"Simple Navigator started. Goal: ({self.get_parameter('goal_x').value}, {self.get_parameter('goal_y').value})")
    
    def pose_callback(self, msg: PoseStamped):
        """Update robot pose."""
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        
        # Extract yaw from quaternion
        q = msg.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
    
    def obstacle_callback(self, msg: ObjectPoseArray):
        """Update obstacle positions."""
        if self.robot_x is None:
            return
        
        self.obstacles = []
        for obj in msg.rovers:
            # Convert from camera frame to world frame
            dx = obj.pose.position.z
            dy = -obj.pose.position.x
            
            world_x = self.robot_x + dx * math.cos(self.robot_yaw) - dy * math.sin(self.robot_yaw)
            world_y = self.robot_y + dx * math.sin(self.robot_yaw) + dy * math.cos(self.robot_yaw)
            
            self.obstacles.append((world_x, world_y))
    
    def control_loop(self):
        """Main navigation control."""
        if self.robot_x is None or self.goal_reached:
            return
        
        goal_x = self.get_parameter('goal_x').value
        goal_y = self.get_parameter('goal_y').value
        goal_tol = self.get_parameter('goal_tolerance').value
        
        # Distance to goal
        dx = goal_x - self.robot_x
        dy = goal_y - self.robot_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if reached
        if distance < goal_tol:
            self.get_logger().info("Goal reached!")
            self.goal_reached = True
            self.pub_cmd.publish(Twist())  # Stop
            self.timer.cancel()
            rclpy.shutdown()
            return
        
        # Desired heading to goal
        goal_heading = math.atan2(dy, dx)
        heading_error = self.angle_diff(goal_heading, self.robot_yaw)
        
        # Check for obstacles in path
        obstacle_detected = False
        obs_dist = self.get_parameter('obstacle_distance').value
        
        for ox, oy in self.obstacles:
            obs_dx = ox - self.robot_x
            obs_dy = oy - self.robot_y
            obs_distance = math.sqrt(obs_dx*obs_dx + obs_dy*obs_dy)
            
            if obs_distance < obs_dist:
                # Obstacle too close
                obstacle_detected = True
                self.get_logger().warn(f"Obstacle detected at {obs_distance:.2f}m!")
                break
        
        # Generate velocity commands
        cmd = Twist()
        
        if obstacle_detected:
            # Stop and turn away
            cmd.linear.x = 0.0
            cmd.angular.z = 0.5  # Turn to avoid
        elif abs(heading_error) > 0.3:
            # Turn in place
            cmd.linear.x = 0.0
            cmd.angular.z = self.get_parameter('max_angular_speed').value * math.copysign(1, heading_error)
        else:
            # Drive toward goal
            max_speed = self.get_parameter('max_linear_speed').value
            cmd.linear.x = min(max_speed, distance * 0.5)  # Slow down near goal
            cmd.angular.z = 2.0 * heading_error  # Proportional turning
        
        self.pub_cmd.publish(cmd)
        
        self.get_logger().debug(
            f"Distance: {distance:.2f}m, Heading error: {math.degrees(heading_error):.1f}°"
        )
    
    @staticmethod
    def angle_diff(a, b):
        """Compute shortest angular difference."""
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))


def main(args=None):
    rclpy.init(args=args)
    node = SimpleNavigator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
