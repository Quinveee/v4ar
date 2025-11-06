import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from line_msgs.msg import DetectedLine  # custom message


class LineFollower(Node):
    """
    Node that subscribes to DetectedLine and publishes Twist commands.
    """

    def __init__(self):
        super().__init__('line_follower')

        # --- Parameters ---
        self.declare_parameter('base_speed', 0.15)
        self.declare_parameter('Kp_offset', 0.002)
        self.declare_parameter('Kp_angle', 1.0)

        self.base_speed = self.get_parameter('base_speed').get_parameter_value().double_value
        self.Kp_offset = self.get_parameter('Kp_offset').get_parameter_value().double_value
        self.Kp_angle = self.get_parameter('Kp_angle').get_parameter_value().double_value

        # --- Subscriber ---
        self.subscription = self.create_subscription(
            DetectedLine,
            'detected_line',
            self.listener_callback,
            10
        )

        # --- Publisher ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("✅ Line Follower Node started — listening to /detected_line")

    def listener_callback(self, msg: DetectedLine):
        """Callback for detected line messages."""
        # Compute control from offset and angle
        angular_z = -self.Kp_offset * msg.offset_x - self.Kp_angle * msg.angle

        cmd = Twist()
        cmd.linear.x = self.base_speed
        cmd.angular.z = angular_z

        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f"Line offset={msg.offset_x:.1f}, angle={msg.angle:.3f} → angular.z={angular_z:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
