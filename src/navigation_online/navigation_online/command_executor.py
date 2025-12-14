#!/usr/bin/env python3
"""
Command Executor Node (Rover Side)

This node runs on the rover and executes movement commands received from the laptop.

Architecture:
    Laptop:
        - Computes mapping, planning, and generates commands
        - Publishes /navigation_commands (geometry_msgs/Twist)
    
    Rover (this node):
        - Subscribes to /navigation_commands
        - Executes commands by publishing to /cmd_vel (or rover's command topic)

Topics:
    Subscribes:
        /navigation_commands (geometry_msgs/Twist): Movement commands from laptop
    
    Publishes:
        /cmd_vel (geometry_msgs/Twist): Velocity commands to rover's base controller
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CommandExecutor(Node):
    """
    Command executor that receives commands from laptop and executes them on rover.
    
    This is a simple relay node that forwards navigation commands to the rover's
    base controller. It can also add safety checks, rate limiting, or other
    rover-specific logic.
    """
    
    def __init__(self):
        super().__init__('command_executor')
        
        # Parameters
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('max_linear_speed', 0.5)  # Safety limit
        self.declare_parameter('max_angular_speed', 1.5)  # Safety limit
        self.declare_parameter('enable_safety_limits', True)
        
        cmd_topic = self.get_parameter('cmd_vel_topic').value
        self.max_linear = self.get_parameter('max_linear_speed').value
        self.max_angular = self.get_parameter('max_angular_speed').value
        self.enable_safety = self.get_parameter('enable_safety_limits').value
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_topic, 10)
        
        # Subscribers
        self.cmd_sub = self.create_subscription(
            Twist, '/navigation_commands', self.command_callback, 10)
        
        self.get_logger().info('Command Executor initialized (rover side)')
        self.get_logger().info(f'Subscribed to /navigation_commands')
        self.get_logger().info(f'Publishing to {cmd_topic}')
        
        if self.enable_safety:
            self.get_logger().info(
                f'Safety limits enabled: linear={self.max_linear} m/s, '
                f'angular={self.max_angular} rad/s'
            )
    
    def command_callback(self, msg: Twist):
        """
        Receive command from laptop and execute it on rover.
        
        Applies safety limits if enabled, then publishes to rover's command topic.
        """
        cmd = Twist()
        
        if self.enable_safety:
            # Apply safety limits
            cmd.linear.x = max(-self.max_linear, min(self.max_linear, msg.linear.x))
            cmd.linear.y = max(-self.max_linear, min(self.max_linear, msg.linear.y))
            cmd.linear.z = 0.0
            cmd.angular.x = 0.0
            cmd.angular.y = 0.0
            cmd.angular.z = max(-self.max_angular, min(self.max_angular, msg.angular.z))
        else:
            # Forward command as-is
            cmd = msg
        
        # Publish to rover's command topic
        self.cmd_vel_pub.publish(cmd)
        
        self.get_logger().debug(
            f'Executing command: linear=({cmd.linear.x:.2f}, {cmd.linear.y:.2f}), '
            f'angular=({cmd.angular.z:.2f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = CommandExecutor()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop rover on shutdown
        cmd = Twist()
        node.cmd_vel_pub.publish(cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

