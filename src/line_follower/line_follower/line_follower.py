import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from line_msgs.msg import DetectedLine


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        #  Publisher to robot velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber to detected line
        self.sub = self.create_subscription(
            DetectedLine,
            '/detected_line',
            self.line_callback,
            10
        )

        # Control parameters
        self.forward_speed = 0.2  # constant forward velocity
        self.k_offset = 0.005     # proportional steering gain
        self.k_angle = 0.01       # proportional angle correction

        self.get_logger().info("✅ Line Follower Node started — listening to /detected_line")

    def line_callback(self, msg: DetectedLine):
        """
        Called whenever a DetectedLine message is received.
        Compute steering command based on line position and orientation.
        """
        twist = Twist()

        # Always move forward
        twist.linear.x = self.forward_speed

        # Simple P-control for steering:
        # Turn rate is proportional to how far the line is from the center
        steering = -(self.k_offset * msg.offset + self.k_angle * msg.angle)
        twist.angular.z = steering

        # Publish the command
        self.cmd_pub.publish(twist)
        self.get_logger().info(f"Forward speed: {self.forward_speed}")
        self.get_logger().info("Steering: {steering}")
        self.get_logger().info(
            f"Line detected | offset={msg.offset:.2f}, angle={msg.angle:.2f} -> steering={steering:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
