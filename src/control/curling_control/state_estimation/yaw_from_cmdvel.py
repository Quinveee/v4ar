#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32


class YawFromCmdVel(Node):
    """
    Integrate yaw from the commanded angular velocity on /cmd_vel.

    y[k+1] = y[k] + wz * dt

    Parameters:
      initial_yaw (float, rad): starting yaw in world frame
    Publishes:
      /estimated_yaw (std_msgs/Float32): current yaw estimate
    """

    def __init__(self):
        super().__init__('yaw_from_cmdvel')

        self.declare_parameter('initial_yaw', 0.0)
        self.yaw = float(self.get_parameter('initial_yaw').value)

        self.last_time = self.get_clock().now()

        self.sub_cmd = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_callback, 10
        )
        self.pub_yaw = self.create_publisher(Float32, '/estimated_yaw', 10)

        # Optional timer to keep publishing even if commands stop
        self.timer = self.create_timer(0.05, self.publish_yaw)

        self.get_logger().info(
            f"YawFromCmdVel started with initial_yaw={self.yaw:.3f} rad"
        )

    def cmd_callback(self, msg: Twist):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        # integrate angular velocity around z
        self.yaw += msg.angular.z * dt

        # normalize to [-pi, pi]
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

    def publish_yaw(self):
        msg = Float32()
        msg.data = self.yaw
        self.pub_yaw.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = YawFromCmdVel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
