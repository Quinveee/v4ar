import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from line_msgs.msg import DetectedLine, DetectedLines
from std_msgs.msg import Float32
import argparse


class LineFollower(Node):
    def __init__(self, smoothing_factor=0.3, speed_control='gradual'):
        super().__init__('line_follower')

        # Publisher for velocity commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Publisher for smoothed angle (for visualization/logging)
        self.smoothed_angle_pub = self.create_publisher(Float32, '/line_follower/smoothed_angle', 10)
        self.selected_line_pub = self.create_publisher(DetectedLine, '/selected_line', 10)

        # Subscriber to multiple detected lines
        self.sub = self.create_subscription(
            DetectedLines,
            '/detected_lines',
            self.lines_callback,
            10
        )

        # Control parameters
        self.forward_speed = 0.2   # constant forward velocity
        self.k_offset = 0.005      # proportional gain for horizontal offset
        self.k_angle = 0.01        # proportional gain for line angle

        # Speed control mode
        self.speed_control = speed_control  # 'gradual', 'threshold', or 'none'

        # EMA smoothing parameters
        self.alpha = smoothing_factor  # EMA smoothing factor (0-1, higher = more weight to recent)
        self.smoothed_angle = 0.0
        self.first_measurement = True
        
        # History tracking for logging/plotting (stored indefinitely)
        self.raw_angle_history = []
        self.smoothed_angle_history = []

        self.get_logger().info(
            f"Line Follower Node started — listening to /detected_lines"
        )
        self.get_logger().info(
            f"EMA smoothing: alpha={smoothing_factor:.2f}, speed_control={speed_control}"
        )

    # -------------------------------------------------------------------------
    # Main callback
    # -------------------------------------------------------------------------
    def lines_callback(self, msg: DetectedLines):
        """
        Called when multiple DetectedLine messages are received.
        Select the most relevant line (e.g., closest to image center) and compute control.
        """
        if not msg.lines:
            self.get_logger().warn(" No lines detected — stopping.")
            self.publish_stop()
            return

        # Pick the line we want to follow
        best_line = self.select_line(msg.lines)

        if best_line is None:
            self.get_logger().warn(" No valid line found — stopping.")
            self.publish_stop()
            return
        self.selected_line_pub.publish(best_line)
        # Apply EMA smoothing to the angle
        smoothed_angle = self.apply_ema_smoothing(best_line.angle)

        # Publish smoothed angle for visualization
        smoothed_msg = Float32()
        smoothed_msg.data = smoothed_angle
        self.smoothed_angle_pub.publish(smoothed_msg)

        # Adjust speed based on angle using selected control mode
        current_speed = self.calculate_speed(smoothed_angle)

        # Compute steering using smoothed angle
        steering = -(self.k_offset * best_line.offset_x + self.k_angle * smoothed_angle)

        # Publish movement
        twist = Twist()
        twist.linear.x = current_speed
        twist.angular.z = steering
        self.cmd_pub.publish(twist)

        self.get_logger().info(
            f"Following line | offset={best_line.offset_x:.2f}, "
            f"raw_angle={best_line.angle:.2f}, smoothed_angle={smoothed_angle:.2f} → "
            f"speed={current_speed:.3f}, steering={steering:.3f}"
        )

    # -------------------------------------------------------------------------
    # Speed control
    # -------------------------------------------------------------------------
    def calculate_speed(self, smoothed_angle):
        """
        Calculate forward speed based on line angle and selected control mode.
        
        Modes:
        - 'gradual': Smooth inverse relationship (0° = 100%, 90° = 50%)
        - 'threshold': Half speed if angle > 30°, else full speed
        - 'none': Always full speed
        """
        angle_magnitude = abs(smoothed_angle)
        
        if self.speed_control == 'gradual':
            # Gradual slowdown: at 0° go full speed, at 90° go ~50% speed
            speed_factor = max(0.5, 1.0 - (angle_magnitude / 180.0))
            return self.forward_speed * speed_factor
            
        elif self.speed_control == 'threshold':
            # Simple threshold: half speed if angle > 30°
            if angle_magnitude > 30.0:
                return self.forward_speed * 0.5
            else:
                return self.forward_speed
                
        else:  # 'none' or any other value
            # No speed adjustment - always full speed
            return self.forward_speed

    # -------------------------------------------------------------------------
    # EMA smoothing
    # -------------------------------------------------------------------------
    def apply_ema_smoothing(self, raw_angle):
        """
        Apply exponential moving average to smooth angle measurements.
        EMA formula: smoothed = alpha * raw + (1 - alpha) * previous_smoothed
        
        Also stores raw and smoothed angles in history for later plotting/analysis.
        """
        # Store raw angle
        self.raw_angle_history.append(raw_angle)
        
        # Calculate EMA
        if self.first_measurement:
            # First measurement - initialize with raw value
            self.smoothed_angle = raw_angle
            self.first_measurement = False
        else:
            # EMA: more weight to recent measurements
            self.smoothed_angle = self.alpha * raw_angle + (1 - self.alpha) * self.smoothed_angle
        
        # Store smoothed angle
        self.smoothed_angle_history.append(self.smoothed_angle)
        
        return self.smoothed_angle

    # -------------------------------------------------------------------------
    # Line selection strategy
    # -------------------------------------------------------------------------
    def select_line(self, lines):
        """
        Selects which line to follow.
        Currently picks the one with smallest |offset_x| (closest to center).
        """
        best_line = None
        min_offset = float('inf')

        for line in lines:
            offset = abs(line.offset_x)
            if offset < min_offset:
                min_offset = offset
                best_line = line

        return best_line

    # -------------------------------------------------------------------------
    # Safety helper
    # -------------------------------------------------------------------------
    def publish_stop(self):
        """Stop the robot safely if no valid line is found."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)


def main(args=None):
    parser = argparse.ArgumentParser(description="ROS2 Line Follower with EMA Smoothing")
    parser.add_argument(
        "--smoothing_factor",
        type=float,
        default=0.3,
        help="EMA smoothing factor alpha (0-1). Higher = more weight to recent values (default: 0.3)"
    )
    parser.add_argument(
        "--speed_control",
        type=str,
        default='gradual',
        choices=['gradual', 'threshold', 'none'],
        help="Speed control mode: 'gradual' (smooth slowdown), 'threshold' (half speed if >30°), 'none' (constant speed) (default: gradual)"
    )
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    
    node = LineFollower(
        smoothing_factor=parsed_args.smoothing_factor,
        speed_control=parsed_args.speed_control
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # On shutdown, log history info
        node.get_logger().info(
            f"Shutting down. Collected {len(node.raw_angle_history)} angle measurements."
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
