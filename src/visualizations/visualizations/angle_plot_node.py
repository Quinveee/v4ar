import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from line_msgs.msg import DetectedLine, DetectedLines
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import cv2
import os


class AnglePlotNode(Node):
    def __init__(self, history_length=100, record_video=None):
        super().__init__('angle_plot_visualization')
        
        self.record_path = record_video
        
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
        
        # Video recording setup
        self.video_writer = None
        self.video_fps = 20  # estimated capture rate
        
        if self.record_path:
            os.makedirs(os.path.dirname(self.record_path) or ".", exist_ok=True)
            self.get_logger().info(f"Current working directory: {os.getcwd()}")
            self.get_logger().info(f"Recording will be saved to: {self.record_path}")
        
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
        
        self.get_logger().info("✅ AnglePlotNode started")
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
        
        # --- Record if requested
        if self.record_path:
            # Convert matplotlib figure to image array
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            if self.video_writer is None:
                h, w = img_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(self.record_path, fourcc, self.video_fps, (w, h))
                self.get_logger().info("Recording started.")
            
            self.video_writer.write(img_bgr)
    
    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f"Saved recorded video to {self.record_path}")
        plt.close('all')
        super().destroy_node()


def main(args=None):
    import argparse
    
    parser = argparse.ArgumentParser(description="ROS2 Angle Plot Visualization")
    parser.add_argument(
        "--history_length",
        type=int,
        default=100,
        help="Number of data points to display in the plot (default: 100)"
    )
    parser.add_argument(
        "--record_path",
        type=str,
        default=None,
        help="Path to save the recorded visualization video (e.g., './angle_plot.avi')."
    )
    parsed_args, ros_args = parser.parse_known_args()
    
    rclpy.init(args=ros_args)
    
    node = AnglePlotNode(
        history_length=parsed_args.history_length,
        record_video=parsed_args.record_path
    )
    
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
