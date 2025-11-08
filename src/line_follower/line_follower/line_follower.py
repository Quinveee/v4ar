import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from line_msgs.msg import DetectedLine, DetectedLines
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse
import math
from .closest_line_selector import ClosestLineSelector
from .confidence_line_selector import ConfidenceLineSelector
from .mean_line_selector import MeanLineSelector

# Available selector classes
SELECTOR_CLASSES = {
    'closest': ClosestLineSelector,
    'confidence': ConfidenceLineSelector,
    'mean': MeanLineSelector,
}


class LineFollower(Node):
    def __init__(self, smoothing_factor=0.3, speed_control='gradual', selector_type='closest', extend_lines=False):
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

        # Subscribe to image to get dimensions (for line extension)
        self.bridge = CvBridge()
        self.image_width = None
        self.image_height = None
        self.extend_lines = extend_lines

        if self.extend_lines:
            self.image_sub = self.create_subscription(
                Image,
                '/processed_image',
                self.image_callback,
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

        # Line selector strategy
        self.selector = SELECTOR_CLASSES.get(selector_type, ClosestLineSelector)()
        self.selector_type = selector_type

        self.get_logger().info(
            f"Line Follower Node started — listening to /detected_lines"
        )
        self.get_logger().info(
            f"EMA smoothing: alpha={smoothing_factor:.2f}, speed_control={speed_control}"
        )
        self.get_logger().info(
            f"Line selector: {selector_type} ({type(self.selector).__name__})"
        )
        self.get_logger().info(
            f"Line extension: {'ENABLED' if extend_lines else 'DISABLED'}"
        )

    # -------------------------------------------------------------------------
    # Image callback (for getting dimensions)
    # -------------------------------------------------------------------------
    def image_callback(self, msg: Image):
        """Get image dimensions from the image topic."""
        if self.image_width is None or self.image_height is None:
            # Only need to get dimensions once
            image = self.bridge.imgmsg_to_cv2(msg)
            self.image_height, self.image_width = image.shape[:2]
            self.get_logger().info(
                f"Image dimensions detected: {self.image_width}x{self.image_height}"
            )

    # -------------------------------------------------------------------------
    # Line extension helper
    # -------------------------------------------------------------------------
    def extend_line_to_screen(self, line: DetectedLine) -> DetectedLine:
        """
        Extend a line segment to cover the full screen width or height.
        Uses the line equation to extrapolate to screen boundaries.
        """
        if self.image_width is None or self.image_height is None:
            # If dimensions not available yet, return original line
            return line

        x1, y1, x2, y2 = float(line.x1), float(line.y1), float(line.x2), float(line.y2)

        # Calculate line direction
        dx = x2 - x1
        dy = y2 - y1

        # Avoid division by zero
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return line

        # Find intersections with screen boundaries
        # We'll extend the line to the edges of the screen
        t_values = []

        # Check intersection with left edge (x = 0)
        if abs(dx) > 1e-6:
            t = -x1 / dx
            t_values.append(t)

        # Check intersection with right edge (x = width)
        if abs(dx) > 1e-6:
            t = (self.image_width - x1) / dx
            t_values.append(t)

        # Check intersection with top edge (y = 0)
        if abs(dy) > 1e-6:
            t = -y1 / dy
            t_values.append(t)

        # Check intersection with bottom edge (y = height)
        if abs(dy) > 1e-6:
            t = (self.image_height - y1) / dy
            t_values.append(t)

        # Calculate all intersection points
        points = []
        for t in t_values:
            px = x1 + t * dx
            py = y1 + t * dy
            # Check if point is within screen bounds
            if 0 <= px <= self.image_width and 0 <= py <= self.image_height:
                points.append((px, py))

        # If we have at least 2 valid intersection points, use them
        if len(points) >= 2:
            # Use the two most distant points
            new_x1, new_y1 = points[0]
            new_x2, new_y2 = points[-1]

            # Create extended line
            extended_line = DetectedLine()
            extended_line.header = line.header
            extended_line.x1 = int(new_x1)
            extended_line.y1 = int(new_y1)
            extended_line.x2 = int(new_x2)
            extended_line.y2 = int(new_y2)

            # Recalculate offset_x and angle for extended line
            line_center_x = (new_x1 + new_x2) / 2
            extended_line.offset_x = float(line_center_x - self.image_width / 2)
            dx_new, dy_new = new_x2 - new_x1, new_y2 - new_y1
            extended_line.angle = math.atan2(dy_new, dx_new)

            return extended_line

        # If extension failed, return original line
        return line

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

        # Extend lines to full screen if enabled
        if self.extend_lines and self.image_width is not None:
            extended_lines = [self.extend_line_to_screen(line) for line in msg.lines]
            # Create new message with extended lines
            extended_msg = DetectedLines()
            extended_msg.header = msg.header
            extended_msg.lines = extended_lines
            msg = extended_msg

        # Pick the line we want to follow using the selected strategy
        best_line = self.selector.select_line(msg.lines)

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
    parser.add_argument(
        "--selector",
        type=str,
        default='closest',
        choices=['closest', 'confidence', 'mean'],
        help="Line selection strategy: 'closest' (nearest to center), 'confidence' (confidence-based tracking), 'mean' (average of all lines) (default: closest)"
    )
    parser.add_argument(
        "--extend_lines",
        action='store_true',
        help="Extend detected lines to cover full screen before processing (default: False)"
    )
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)

    node = LineFollower(
        smoothing_factor=parsed_args.smoothing_factor,
        speed_control=parsed_args.speed_control,
        selector_type=parsed_args.selector,
        extend_lines=parsed_args.extend_lines
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
