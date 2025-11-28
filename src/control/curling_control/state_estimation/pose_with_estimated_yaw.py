#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from geometry_msgs.msg import Quaternion


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    # roll = pitch = 0
    q.w = math.cos(yaw / 2.0)
    q.z = math.sin(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    return q


class PoseWithEstimatedYaw(Node):
    """
    Take x,y from /robot_pose (or /robot_pose_raw) and yaw from /estimated_yaw,
    publish PoseStamped on /robot_pose_fused.
    """

    def __init__(self):
        super().__init__('pose_with_estimated_yaw')

        # Parameters so you can choose where x,y come from
        self.declare_parameter('input_pose_topic', '/robot_pose')
        in_topic = self.get_parameter('input_pose_topic').value

        self.pose_sub = self.create_subscription(
            PoseStamped, in_topic, self.pose_callback, 10
        )
        self.yaw_sub = self.create_subscription(
            Float32, '/estimated_yaw', self.yaw_callback, 10
        )
        self.pose_pub = self.create_publisher(PoseStamped, '/robot_pose_fused', 10)

        self.last_x = None
        self.last_y = None
        self.last_z = 0.0
        self.last_yaw = 0.0

        self.get_logger().info(
            f"PoseWithEstimatedYaw listening to pose on {in_topic}, "
            f"yaw on /estimated_yaw, publishing /robot_pose_fused"
        )

    def pose_callback(self, msg: PoseStamped):
        self.last_x = msg.pose.position.x
        self.last_y = msg.pose.position.y
        self.last_z = msg.pose.position.z
        # you could also read initial yaw from here once, if you want
        self.publish_if_ready(frame_id=msg.header.frame_id)

    def yaw_callback(self, msg: Float32):
        self.last_yaw = msg.data
        self.publish_if_ready()

    def publish_if_ready(self, frame_id: str = "world"):
        if self.last_x is None or self.last_y is None:
            return

        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = frame_id

        out.pose.position.x = self.last_x
        out.pose.position.y = self.last_y
        out.pose.position.z = self.last_z

        out.pose.orientation = yaw_to_quaternion(self.last_yaw)

        self.pose_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PoseWithEstimatedYaw()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
