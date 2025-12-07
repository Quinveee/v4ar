#!/usr/bin/env python3
"""Node to republish odom->base_footprint transform from /odom messages."""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomToTf(Node):
    """Republishes odom->base_footprint transform from /odom topic."""

    def __init__(self):
        super().__init__('odom_to_tf')
        
        # Declare parameters
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        
        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Subscribe to odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10
        )
        
        self.get_logger().info(
            f'odom_to_tf: Subscribing to /odom, publishing {self.odom_frame} -> {self.base_frame}'
        )

    def odom_callback(self, msg: Odometry):
        """Callback to republish transform from odometry message."""
        # Create transform from odom message
        t = TransformStamped()
        
        # Set header
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id  # Should be 'odom'
        t.child_frame_id = msg.child_frame_id    # Should be 'base_footprint'
        
        # Copy pose transform
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation.x = msg.pose.pose.orientation.x
        t.transform.rotation.y = msg.pose.pose.orientation.y
        t.transform.rotation.z = msg.pose.pose.orientation.z
        t.transform.rotation.w = msg.pose.pose.orientation.w
        
        # Broadcast transform
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomToTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()