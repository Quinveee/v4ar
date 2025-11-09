import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from line_msgs.msg import DetectedLine, DetectedLines
import matplotlib.pyplot as plt
from collections import deque
import numpy as np


class AnglePlotNode(Node):
    def __init__(self, history_length=100):
        super().__init__('angle_plot_visualization')
        
        # Subscriber to raw angle data (from line detection)
        self.lines_sub = self.create_subscription(
            DetectedLines,
            '/detected_lines',
            self.lines_callback,
            10
        )
        
        # Subscriber to smoothed angle data (from line follower)
        self.smoothed_sub = self.create_subscription(
            Float32,
            '/line_follower/smoothed_angle',
            self.smoothed_callback,
            10
        )
        
        # Data storage
        self.max_points = history_length
        self.timestamps = deque(maxlen=self.max_points)
        self.raw_angles = deque(maxlen=self.max_points)
        self.smoothed_angles = deque(maxlen=self.max_points)
        self.time_start = self.get_clock().now().nanoseconds / 1e9
        
        # Track latest smoothed angle to pair with raw measurements
        self.latest_smoothed = None
        
        # Setup matplotlib for live plotting
        plt.ion()  # Enable interactive mode
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line_raw, = self.ax.plot([], [], 'b-', alpha=0.5, linewidth=1, label='Raw Angle')
        self.line_smoothed, = self.ax.plot([], [], 'r-', linewidth=2, label='Smoothed Angle (EMA)')
        
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Angle (degrees)')
        self.ax.set_title('Line Following: Raw vs Smoothed Angle')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        
        self.get_logger().info("  AnglePlotNode started")
        self.get_logger().info("   Listening to /detected_lines (raw angle)")
        self.get_logger().info("   Listening to /line_follower/smoothed_angle")
        self.get_logger().info(f"   Displaying last {history_length} measurements")
    
    def lines_callback(self, msg):
        """Receive raw angle from detected lines."""
        if not msg.lines:
            return
        
        # Use the same line selection strategy as line_follower (closest to center)
        best_line = self.select_line(msg.lines)
        if best_line is None:
            return
        
        current_time = self.get_clock().now().nanoseconds / 1e9 - self.time_start
        raw_angle = best_line.angle
        
        # Store raw angle data
        self.timestamps.append(current_time)
        self.raw_angles.append(raw_angle)
        
        # Add corresponding smoothed angle (use latest if available)
        if self.latest_smoothed is not None:
            self.smoothed_angles.append(self.latest_smoothed)
        else:
            # If no smoothed data yet, use raw angle as placeholder
            self.smoothed_angles.append(raw_angle)
        
        # Update plot
        self.update_plot()
    
    def smoothed_callback(self, msg):
        """Receive smoothed angle from line follower."""
        self.latest_smoothed = msg.data
    
    def select_line(self, lines):
        """Select the line closest to image center (same as line_follower)."""
        best_line = None
        min_offset = float('inf')
        
        for line in lines:
            offset = abs(line.offset_x)
            if offset < min_offset:
                min_offset = offset
                best_line = line
        
        return best_line
    
    def update_plot(self):
        """Update the matplotlib plot with latest data."""
        if len(self.timestamps) < 2:
            return
        
        # Convert deques to lists for plotting
        times = list(self.timestamps)
        raw = list(self.raw_angles)
        smoothed = list(self.smoothed_angles)
        
        # Update line data
        self.line_raw.set_data(times, raw)
        self.line_smoothed.set_data(times, smoothed)
        
        # Auto-scale axes
        self.ax.relim()
        self.ax.autoscale_view()
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # Small pause to allow GUI updates


def main(args=None):
    import argparse
    
    parser = argparse.ArgumentParser(description="ROS2 Angle Plot Visualization")
    parser.add_argument(
        "--history_length",
        type=int,
        default=100,
        help="Number of data points to display in the plot (default: 100)"
    )
    parsed_args, ros_args = parser.parse_known_args()
    
    rclpy.init(args=ros_args)
    
    node = AnglePlotNode(history_length=parsed_args.history_length)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down angle plot visualization.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')


if __name__ == '__main__':
    main()
