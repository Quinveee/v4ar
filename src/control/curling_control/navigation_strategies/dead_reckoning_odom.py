#!/usr/bin/env python3
"""
Dead Reckoning Odometry Node for Nav2.

Publishes odometry from commanded velocity (cmd_vel) for Nav2.
This provides odometry when laser-based odometry (RF2O) is not available.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import math
import tf2_ros


class DeadReckoningOdom(Node):
    """
    Dead reckoning odometry from cmd_vel.
    
    Subscribes to /cmd_vel and publishes:
    - nav_msgs/Odometry to /odom topic
    - TF transform: odom → base_footprint
    """

    def __init__(self):
        super().__init__('dead_reckoning_odom')

        # Parameters
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("base_frame_id", "base_footprint")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("publish_rate", 50.0)  # Hz

        odom_topic = self.get_parameter("odom_topic").value
        cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.base_frame_id = self.get_parameter("base_frame_id").value
        self.odom_frame_id = self.get_parameter("odom_frame_id").value
        self.publish_tf = self.get_parameter("publish_tf").value
        publish_rate = self.get_parameter("publish_rate").value

        # State
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0  # Linear velocity
        self.w = 0.0  # Angular velocity
        self.last_update_time = None

        # Subscriptions
        self.cmd_vel_sub = self.create_subscription(
            Twist, cmd_vel_topic, self.cmd_vel_callback, 10
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
        
        # TF broadcaster
        if self.publish_tf:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Timer for periodic publishing
        self.create_timer(1.0 / publish_rate, self.publish_odom)

        self.get_logger().info(
            f"Dead reckoning odometry initialized: "
            f"publishing to {odom_topic}, frame: {self.odom_frame_id} → {self.base_frame_id}"
        )

    def cmd_vel_callback(self, msg: Twist):
        """Update velocity commands."""
        self.v = msg.linear.x
        self.w = msg.angular.z

    def publish_odom(self):
        """Publish odometry and update pose using dead reckoning."""
        now = self.get_clock().now()

        # Calculate time delta
        if self.last_update_time is not None:
            dt = (now - self.last_update_time).nanoseconds / 1e9

            if dt > 0 and dt < 1.0:  # Sanity check
                # Dead reckoning motion model
                if abs(self.w) < 1e-6:
                    # Straight line motion
                    dx = self.v * math.cos(self.theta) * dt
                    dy = self.v * math.sin(self.theta) * dt
                    dtheta = 0.0
                else:
                    # Arc motion
                    radius = self.v / self.w if abs(self.w) > 1e-6 else 0.0
                    dtheta = self.w * dt
                    dx = radius * (math.sin(self.theta + dtheta) - math.sin(self.theta))
                    dy = radius * (-math.cos(self.theta + dtheta) + math.cos(self.theta))

                # Update pose
                self.x += dx
                self.y += dy
                self.theta += dtheta
                self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        self.last_update_time = now

        # Create and publish Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = now.to_msg()
        odom_msg.header.frame_id = self.odom_frame_id
        odom_msg.child_frame_id = self.base_frame_id

        # Position
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.position.z = 0.0

        # Orientation (quaternion from yaw)
        half_yaw = self.theta / 2.0
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = math.sin(half_yaw)
        odom_msg.pose.pose.orientation.w = math.cos(half_yaw)

        # Velocity
        odom_msg.twist.twist.linear.x = self.v
        odom_msg.twist.twist.linear.y = 0.0
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = self.w

        # Covariance (simple model - can be tuned)
        odom_msg.pose.covariance[0] = 0.1  # x
        odom_msg.pose.covariance[7] = 0.1  # y
        odom_msg.pose.covariance[35] = 0.1  # yaw
        odom_msg.twist.covariance[0] = 0.1  # vx
        odom_msg.twist.covariance[35] = 0.1  # vyaw

        self.odom_pub.publish(odom_msg)

        # Publish TF transform
        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = now.to_msg()
            t.header.frame_id = self.odom_frame_id
            t.child_frame_id = self.base_frame_id

            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0

            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = math.sin(half_yaw)
            t.transform.rotation.w = math.cos(half_yaw)

            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = DeadReckoningOdom()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

