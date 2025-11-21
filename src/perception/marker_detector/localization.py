#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from rclpy.time import Time

from .localization_strategies import *

STRATEGIES = {
    "kalman": KalmanLocalization,
    "complementary": ComplementaryLocalization,
    "robust": RobustLocalization,
    "identity": IdentityLocalization
}


def yaw_to_quaternion(yaw):
    import math
    half = yaw / 2
    return (0.0, 0.0, math.sin(half), math.cos(half))

class LocalizationNode(Node):
    def __init__(self):
        super().__init__("localization_node")

        self.declare_parameter("strategy_type", "identity")
        self.declare_parameter("publish_rate", 20.0)

        strategy_type = self.get_parameter("strategy_type").value
        Strategy = STRATEGIES.get(strategy_type, KalmanLocalization)
        self.localizer = Strategy()
        self.get_logger().info(f"Using localization strategy: {Strategy.__name__}")

        # Internal vars
        self.last_update_time = None
        self.v = 0.0
        self.w = 0.0

        # ROS setup
        self.pose_pub = self.create_publisher(PoseStamped, "/robot_pose", 10)
        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(PoseStamped, "/robot_pose_raw", self.pose_callback, 10)

        rate = self.get_parameter("publish_rate").value
        self.create_timer(1.0 / rate, self.timer_callback)

    def cmd_callback(self, msg):
        self.v = msg.linear.x
        self.w = msg.angular.z

    def pose_callback(self, msg):
        self.localizer.update(msg)

    def timer_callback(self):
        now = self.get_clock().now()
        if self.last_update_time is None:
            self.last_update_time = now
            return
        dt = (now - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = now
        markers = {}
        self.localizer.predict(self.v, self.w, dt)
        x, y, theta = self.localizer.get_pose()

        out = PoseStamped()
        out.header.stamp = now.to_msg()
        out.header.frame_id = "world"
        out.pose.position.x = x
        out.pose.position.y = y
        qx, qy, qz, qw = yaw_to_quaternion(theta)
        out.pose.orientation.x = qx
        out.pose.orientation.y = qy
        out.pose.orientation.z = qz
        out.pose.orientation.w = qw
        self.pose_pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
