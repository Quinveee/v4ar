#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class LineFollowerNode(Node):
    def __init__(self):
        super().__init__('line_follower_node')
        
        # Bridge to convert ROS images to OpenCV
        self.bridge = CvBridge()
        
        # Publisher to send velocity commands to robot
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Publisher to control camera angle
        self.joint_pub = self.create_publisher(JointState, '/ugv/joint_states', 10)
        
        # Subscriber to receive camera images
        self.subscription = self.create_subscription(
            Image,
            '/video_frames',
            self.image_callback,
            10)
        
        # State machine: tracks what the robot is doing
        self.state = 'INIT'  # Can be: INIT, ALIGNING, FOLLOWING
        self.camera_set = False
        
        # Control parameters (tune these!)
        self.linear_speed = 0.15        # How fast to drive forward (m/s)
        self.alignment_tolerance = 5.0  # How close to aligned (degrees)
        self.target_angle = 90.0        # Lines should be vertical (90°)
        self.kp = 0.01                  # Control gain
        
        self.get_logger().info('Line Follower Started!')
        
        # Wait 2 seconds then point camera at ceiling
        self.create_timer(2.0, self.setup_camera)
    
    def setup_camera(self):
        """
        STEP 1: Point camera straight up at ceiling
        Called once at startup after 2 second delay
        """
        if not self.camera_set:
            # Create message to control pan-tilt camera
            joint_msg = JointState()
            joint_msg.name = ['pt_base_link_to_pt_link1', 'pt_link1_to_pt_link2']
            joint_msg.position = [0.0, 1.57]  # pan=0 (center), tilt=1.57rad (90° up)
            
            # Publish the command
            self.joint_pub.publish(joint_msg)
            
            self.camera_set = True
            self.state = 'ALIGNING'
            self.get_logger().info('Camera pointing at ceiling - Starting alignment')
    
    def image_callback(self, msg):
        """
        MAIN LOOP: Called every time a new camera image arrives
        Runs ~30 times per second
        """

        # DEBUG: Confirm we're receiving images
        self.get_logger().info(f'Image received! State: {self.state}')
        
        # Don't process until camera is set up
        if self.state == 'INIT':
            self.get_logger().info('Still in INIT state, waiting...')
            return
        
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info(f'Image converted, shape: {cv_image.shape}')
            
            # STEP 2: Detect bright lines in image
            lines, debug_image = self.detect_ceiling_lines(cv_image)
            self.get_logger().info(f'Lines detected: {len(lines) if lines is not None else 0}')
            
            # STEP 3: Calculate average angle of detected lines
            avg_angle = self.calculate_average_angle(lines)
            
            # STEP 4: Decide what to do based on current state
            if self.state == 'ALIGNING':
                # Rotate to align with lines
                cmd_vel = self.align_with_lines(avg_angle)
            elif self.state == 'FOLLOWING':
                # Drive forward while staying aligned
                cmd_vel = self.follow_lines(avg_angle)
            
            # STEP 5: Send velocity command to robot
            self.cmd_pub.publish(cmd_vel)
            
            # STEP 6: Show visualization for debugging
            self.visualize(debug_image, lines, avg_angle)
            
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')
    
    def detect_ceiling_lines(self, image):
        """
        STEP 2: Find bright lines (ceiling lights) in image
        
        Process:
        - Convert to grayscale
        - Threshold to find bright areas (>200 brightness)
        - Edge detection with Canny
        - Line detection with Hough Transform
        
        Returns: (list_of_lines, debug_image)
        """
        # Convert color image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Keep only VERY bright pixels (ceiling lights are bright!)
        # Pixels with value > 200 become white (255)
        # Pixels with value <= 200 become black (0)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Apply Canny edge detection on bright areas
        edges = cv2.Canny(bright_mask, 50, 150)
        
        # Detect lines using Hough Transform
        # Returns list of lines as [x1, y1, x2, y2] coordinates
        lines = cv2.HoughLinesP(
            edges,              # Input edge image
            1,                  # Distance resolution (pixels)
            np.pi/180,         # Angle resolution (radians)
            50,                # Threshold: minimum votes to be a line
            minLineLength=80,  # Minimum length of line (pixels)
            maxLineGap=30      # Maximum gap between line segments (pixels)
        )
        
        return lines, image.copy()
    
    def calculate_average_angle(self, lines):
        """
        STEP 3: Calculate the average angle of all detected lines
        
        For each line:
        - Calculate angle using arctan2
        - Convert to degrees
        - Normalize to 0-180° range
        
        Returns: median angle (or None if no lines)
        """
        if lines is None or len(lines) == 0:
            return None
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle of line
            # arctan2(dy, dx) gives angle in radians
            angle_rad = np.arctan2(y2 - y1, x2 - x1)
            
            # Convert to degrees
            angle_deg = np.degrees(angle_rad)
            
            # Normalize to 0-180 range
            # (lines don't have direction, so 10° = 190°)
            if angle_deg < 0:
                angle_deg += 180
            
            angles.append(angle_deg)
        
        # Use median (more robust than mean - ignores outliers)
        avg_angle = np.median(angles)
        
        return avg_angle
    
    def align_with_lines(self, avg_angle):
        """
        STEP 4A: ALIGNING STATE
        Rotate robot until ceiling lines are vertical (90°)
        
        Logic:
        - If no lines detected: rotate slowly to search
        - Calculate error = current_angle - target_angle (90°)
        - If error small enough: switch to FOLLOWING
        - Otherwise: rotate to reduce error
        
        Returns: Twist message with velocities
        """
        cmd_vel = Twist()
        
        # Case 1: No lines detected
        if avg_angle is None:
            cmd_vel.linear.x = 0.0      # Don't move forward
            cmd_vel.angular.z = 0.2     # Rotate slowly to search
            self.get_logger().warn('No lines - searching...')
            return cmd_vel
        
        # Case 2: Lines detected - calculate error
        error = avg_angle - self.target_angle
        
        self.get_logger().info(f'ALIGNING - Angle: {avg_angle:.1f}°, Error: {error:.1f}°')
        
        # Case 3: Error small enough - we're aligned!
        if abs(error) < self.alignment_tolerance:
            self.state = 'FOLLOWING'
            self.get_logger().info('✓ ALIGNED! Now following')
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
        
        # Case 4: Need to rotate more
        else:
            cmd_vel.linear.x = 0.0                    # Don't move forward yet
            cmd_vel.angular.z = -self.kp * error      # Rotate proportional to error
            
            # Limit rotation speed to safe range
            cmd_vel.angular.z = np.clip(cmd_vel.angular.z, -0.5, 0.5)
        
        return cmd_vel
    
    def follow_lines(self, avg_angle):
        """
        STEP 4B: FOLLOWING STATE
        Drive forward while keeping lines vertical
        
        Logic:
        - If no lines: STOP and go back to ALIGNING
        - If error too large: go back to ALIGNING
        - Otherwise: drive forward with small corrections
        
        Returns: Twist message with velocities
        """
        cmd_vel = Twist()
        
        # Case 1: Lost the lines
        if avg_angle is None:
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.get_logger().warn('Lost lines - stopping')
            self.state = 'ALIGNING'  # Go back to alignment mode
            return cmd_vel
        
        # Case 2: Calculate error
        error = avg_angle - self.target_angle
        
        # Case 3: Error too large - lost alignment
        if abs(error) > self.alignment_tolerance * 3:
            self.get_logger().warn('Lost alignment - realigning')
            self.state = 'ALIGNING'
            return self.align_with_lines(avg_angle)
        
        # Case 4: Following with small corrections
        cmd_vel.linear.x = self.linear_speed           # Drive forward
        cmd_vel.angular.z = -self.kp * error * 0.5     # Gentle correction (50% gain)
        
        # Limit turn rate while moving
        cmd_vel.angular.z = np.clip(cmd_vel.angular.z, -0.3, 0.3)
        
        self.get_logger().info(f'FOLLOWING - Angle: {avg_angle:.1f}°, Speed: {cmd_vel.linear.x:.2f}')
        
        return cmd_vel
    
    def visualize(self, image, lines, avg_angle):
        """
        STEP 6: Draw visualization for debugging
        Shows detected lines, angles, and state
        """
        height, width = image.shape[:2]
        
        # Draw all detected lines in GREEN
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw vertical reference line (target orientation)
        cv2.line(image, (width//2, 0), (width//2, height), (255, 0, 0), 2)
        
        # Draw text info
        state_text = f'State: {self.state}'
        angle_text = f'Angle: {avg_angle:.1f}' if avg_angle else 'Angle: N/A'
        
        cv2.putText(image, state_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, angle_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Color border based on state
        color = (0, 255, 0) if self.state == 'FOLLOWING' else (0, 165, 255)
        cv2.rectangle(image, (5, 5), (width-5, 80), color, 3)
        
        cv2.imshow('Line Following', image)
        cv2.waitKey(1)

def main(args=None):
    """
    Main entry point
    """
    rclpy.init(args=args)
    node = LineFollowerNode()
    
    try:
        rclpy.spin(node)  # Keep running until Ctrl+C
    except KeyboardInterrupt:
        # Emergency stop when Ctrl+C pressed
        stop_cmd = Twist()
        node.cmd_pub.publish(stop_cmd)
        node.get_logger().info('Stopped!')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()