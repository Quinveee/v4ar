#!/usr/bin/env python3
"""
Online Navigator Node (Laptop Side) - Using Nav2

This node runs on the laptop and performs:
1. Receives maps from RTAB-Map bridge
2. Uses Nav2's planner server to plan paths
3. Generates movement commands from planned paths
4. Publishes commands for rover to execute

Architecture:
    Laptop (this node):
        - Subscribes to /map (from RTAB-Map bridge)
        - Subscribes to /odom (robot position from rover)
        - Subscribes to /goal (goal pose)
        - Uses Nav2 planner server to plan paths
        - Generates velocity commands from path
        - Publishes /navigation_commands (geometry_msgs/Twist) for rover
    
    Rover:
        - Runs command_executor node
        - Subscribes to /navigation_commands
        - Executes commands (publishes to /cmd_vel or similar)

Topics:
    Subscribes:
        /map (nav_msgs/OccupancyGrid): Map from RTAB-Map bridge
        /odom (nav_msgs/Odometry): Robot odometry from rover
        /goal (geometry_msgs/PoseStamped): Goal pose
    
    Publishes:
        /navigation_commands (geometry_msgs/Twist): Movement commands for rover
        /planned_path (nav_msgs/Path): Planned path for visualization in RViz
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.srv import ComputePathToPose
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class Pose2D:
    """Simple 2D pose representation."""
    x: float
    y: float
    theta: float


class OnlineNavigator(Node):
    """
    Online navigator that uses Nav2 planner and generates commands.
    
    Runs on laptop - uses Nav2's planner server for path planning,
    then generates velocity commands for rover to execute.
    """
    
    def __init__(self):
        super().__init__('online_navigator')
        
        # Parameters
        self.declare_parameter('planner_server', 'planner_server')
        self.declare_parameter('k_linear', 0.5)
        self.declare_parameter('k_angular', 2.0)
        self.declare_parameter('max_linear_speed', 0.3)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('waypoint_threshold', 0.15)
        self.declare_parameter('goal_threshold', 0.1)
        self.declare_parameter('control_frequency', 10.0)  # Hz
        
        # Get parameters
        planner_server_name = self.get_parameter('planner_server').value
        self.k_linear = self.get_parameter('k_linear').value
        self.k_angular = self.get_parameter('k_angular').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.waypoint_threshold = self.get_parameter('waypoint_threshold').value
        self.goal_threshold = self.get_parameter('goal_threshold').value
        control_freq = self.get_parameter('control_frequency').value
        
        # State
        self.current_map: Optional[OccupancyGrid] = None
        self.current_pose: Optional[Pose2D] = None
        self.current_path: Optional[Path] = None
        self.current_waypoint_idx: int = 0
        self.goal_pose: Optional[PoseStamped] = None
        self.goal_reached: bool = False
        
        # Nav2 Planner Service Client
        planner_service = f'/{planner_server_name}/compute_path_to_pose'
        self.planner_client = self.create_client(ComputePathToPose, planner_service)
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/navigation_commands', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal', self.goal_callback, 10)
        
        # Control timer
        control_period = 1.0 / control_freq
        self.control_timer = self.create_timer(control_period, self.control_loop)
        
        # Wait for planner server
        self.get_logger().info(f'Waiting for Nav2 planner server at {planner_service}...')
        while not self.planner_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Planner server not available, waiting...')
        
        self.get_logger().info('Online Navigator initialized (laptop side)')
        self.get_logger().info('Using Nav2 planner server for path planning')
        self.get_logger().info('Waiting for map, odometry, and goal...')
    
    def map_callback(self, msg: OccupancyGrid):
        """Receive map from RTAB-Map bridge."""
        self.get_logger().info(
            f'Received map: {msg.info.width}x{msg.info.height}, '
            f'resolution={msg.info.resolution}m'
        )
        self.current_map = msg
        
        # Replan if we have a goal
        if self.goal_pose is not None and self.current_pose is not None:
            self.get_logger().info('Map updated, replanning...')
            self.plan_path()
    
    def odom_callback(self, msg: Odometry):
        """Update current robot position from odometry."""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        
        yaw = self.quaternion_to_yaw(ori.x, ori.y, ori.z, ori.w)
        self.current_pose = Pose2D(pos.x, pos.y, yaw)
    
    def goal_callback(self, msg: PoseStamped):
        """Handle new goal pose."""
        self.get_logger().info(
            f'Received goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})'
        )
        self.goal_pose = msg
        self.goal_reached = False
        self.current_waypoint_idx = 0
        
        # Plan path if we have map and current pose
        if self.current_map is not None and self.current_pose is not None:
            self.plan_path()
        else:
            self.get_logger().warn('Cannot plan: missing map or current pose')
    
    def quaternion_to_yaw(self, x: float, y: float, z: float, w: float) -> float:
        """Convert quaternion to yaw angle."""
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def plan_path(self):
        """Plan path from current position to goal using Nav2 planner."""
        if self.current_map is None or self.current_pose is None or self.goal_pose is None:
            return
        
        # Create request for Nav2 planner
        request = ComputePathToPose.Request()
        
        # Set start pose (current robot position)
        request.start.header.stamp = self.get_clock().now().to_msg()
        request.start.header.frame_id = 'map'
        request.start.pose.position.x = self.current_pose.x
        request.start.pose.position.y = self.current_pose.y
        request.start.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        yaw = self.current_pose.theta
        request.start.pose.orientation.x = 0.0
        request.start.pose.orientation.y = 0.0
        request.start.pose.orientation.z = math.sin(yaw / 2.0)
        request.start.pose.orientation.w = math.cos(yaw / 2.0)
        
        # Set goal pose
        request.goal = self.goal_pose
        request.goal.header.stamp = self.get_clock().now().to_msg()
        request.goal.header.frame_id = 'map'
        
        # Set tolerance
        request.tolerance = 0.2
        
        # Call Nav2 planner service
        self.get_logger().info('Calling Nav2 planner service...')
        future = self.planner_client.call_async(request)
        future.add_done_callback(self.planner_response_callback)
    
    def planner_response_callback(self, future):
        """Handle response from Nav2 planner."""
        try:
            response = future.result()
            
            if response.path.poses:
                self.current_path = response.path
                self.current_waypoint_idx = 0
                
                # Publish path for visualization in RViz
                self.path_pub.publish(self.current_path)
                
                self.get_logger().info(
                    f'Nav2 planned path with {len(self.current_path.poses)} waypoints'
                )
            else:
                self.get_logger().error('Nav2 planner returned empty path!')
                self.current_path = None
                
        except Exception as e:
            self.get_logger().error(f'Failed to get path from Nav2 planner: {e}')
            self.current_path = None
    
    def compute_control(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """Compute velocity commands to drive toward target waypoint."""
        if self.current_pose is None:
            return (0.0, 0.0)
        
        dx = target_x - self.current_pose.x
        dy = target_y - self.current_pose.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        desired_heading = math.atan2(dy, dx)
        heading_error = self.normalize_angle(desired_heading - self.current_pose.theta)
        
        # Proportional control
        linear_vel = self.k_linear * distance
        linear_vel = max(-self.max_linear_speed, min(self.max_linear_speed, linear_vel))
        
        # Reduce speed when turning sharply
        if abs(heading_error) > math.pi / 4:
            linear_vel *= 0.5
        
        angular_vel = self.k_angular * heading_error
        angular_vel = max(-self.max_angular_speed, min(self.max_angular_speed, angular_vel))
        
        return (linear_vel, angular_vel)
    
    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] range."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle
    
    def control_loop(self):
        """Main control loop - generates and publishes commands."""
        if self.goal_reached or self.current_path is None or self.current_pose is None:
            return
        
        if self.current_waypoint_idx >= len(self.current_path.poses):
            if not self.goal_reached:
                self.get_logger().info('Goal reached!')
                self.goal_reached = True
                cmd = Twist()
                self.cmd_pub.publish(cmd)
            return
        
        # Get current target waypoint
        target_pose = self.current_path.poses[self.current_waypoint_idx].pose
        target_x = target_pose.position.x
        target_y = target_pose.position.y
        
        # Calculate distance to waypoint
        dx = target_x - self.current_pose.x
        dy = target_y - self.current_pose.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Check if we should advance to next waypoint
        is_last_waypoint = (self.current_waypoint_idx == len(self.current_path.poses) - 1)
        threshold = self.goal_threshold if is_last_waypoint else self.waypoint_threshold
        
        if distance < threshold:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx < len(self.current_path.poses):
                self.get_logger().info(
                    f'Reached waypoint {self.current_waypoint_idx}/{len(self.current_path.poses)}'
                )
            return
        
        # Compute and publish velocity command
        linear_vel, angular_vel = self.compute_control(target_x, target_y)
        
        cmd = Twist()
        cmd.linear.x = linear_vel
        cmd.angular.z = angular_vel
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = OnlineNavigator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot on shutdown
        cmd = Twist()
        node.cmd_pub.publish(cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
