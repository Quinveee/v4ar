#!/usr/bin/env python3
"""
AprilTag Detection Visualization Node (Baseline)
Shows 2 windows:
1. Original image
2. Detected AprilTags with annotations

Press 's' to save screenshots
Press 'q' to quit
"""

import time
import rclpy
from rclpy.time import Time as RclTime
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime

# Try to import apriltag library
try:
    import pupil_apriltags as apriltag
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False
    print("WARNING: apriltag library not found. Install with: pip install apriltag")


class AprilTagVisualizationNode(Node):
    """
    Minimal AprilTag detection baseline for Session 1
    
    Detects AprilTags in camera images and visualizes:
    - Bounding boxes around detected tags
    - Tag family and ID
    - Center point
    
    Keyboard controls:
    - Press 's' to save screenshots
    - Press 'q' to quit
    """
    
    def __init__(self, output_dir='apriltag_screenshots', total_frames=None, 
                 num_screenshots=5, no_gui=False, tag_family='tagStandard41h12'):
        super().__init__('apriltag_visualization')
        
        if not APRILTAG_AVAILABLE:
            self.get_logger().error("AprilTag library not available! Please install: pip install apriltag")
            raise ImportError("apriltag library required")
        
        self.bridge = CvBridge()
        self.no_gui = no_gui
        
        # Performance tracking
        self.proc_ema = 0.0
        self.last_msg_stamp = None
        self.last_log_t = time.perf_counter()
        
        # Screenshot tracking
        self.screenshot_count = 0
        self.output_dir = output_dir
        
        # Frame counting and auto-screenshot
        self.frame_count = 0
        self.total_frames = total_frames
        self.num_screenshots = num_screenshots
        self.screenshots_taken = 0
        self.screenshot_frames = []
        
        # Calculate which frames to capture if total_frames is provided
        if self.total_frames is not None:
            interval = self.total_frames / (self.num_screenshots + 1)
            self.screenshot_frames = [int(interval * (i + 1)) for i in range(self.num_screenshots)]
            self.get_logger().info(f"Auto-screenshot enabled at frames: {self.screenshot_frames}")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store current frames for screenshot capture
        self.current_original = None
        self.current_detected = None
        
        # Parameters (tunable)
        self.declare_parameter('tag_family', tag_family)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('output_dir', output_dir)
        self.declare_parameter('total_frames', total_frames if total_frames is not None else 0)
        self.declare_parameter('num_screenshots', num_screenshots)
        self.declare_parameter('no_gui', no_gui)
        
        self.tag_family = self.get_parameter('tag_family').value
        image_topic = self.get_parameter('image_topic').value
        self.output_dir = self.get_parameter('output_dir').value
        self.no_gui = self.get_parameter('no_gui').value
        
        # Update total_frames from parameter if provided
        param_total_frames = self.get_parameter('total_frames').value
        if param_total_frames > 0:
            self.total_frames = param_total_frames
            interval = self.total_frames / (self.num_screenshots + 1)
            self.screenshot_frames = [int(interval * (i + 1)) for i in range(self.num_screenshots)]
        
        # Initialize AprilTag detector with optimized parameters (from working notebook)
        self.detector = apriltag.Detector(
            families=self.tag_family,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25
        )
        
        self.get_logger().info(
            f"Detector initialized with optimized parameters:\n"
            f"  nthreads=4, quad_decimate=1.0, refine_edges=True"
        )
        
        # Subscription
        self.sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            50  # Increased queue size for reliability
        )
        
        # Window setup (only if GUI enabled)
        if not self.no_gui:
            cv2.namedWindow("1. Original Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("2. Detected AprilTags", cv2.WINDOW_NORMAL)
            
            # Arrange windows
            cv2.resizeWindow("1. Original Image", 640, 480)
            cv2.resizeWindow("2. Detected AprilTags", 640, 480)
            
            cv2.moveWindow("1. Original Image", 0, 0)
            cv2.moveWindow("2. Detected AprilTags", 670, 0)
        
        self.get_logger().info(
            f"AprilTagVisualizationNode started with parameters:\n"
            f"  Tag family: {self.tag_family}\n"
            f"  Image topic: {image_topic}\n"
            f"  Output directory: {self.output_dir}\n"
            f"  Total frames: {self.total_frames if self.total_frames else 'Auto-detect'}\n"
            f"  Num screenshots: {self.num_screenshots}\n"
            f"  Auto-screenshot frames: {self.screenshot_frames if self.screenshot_frames else 'Will calculate'}\n"
            f"  GUI mode: {'DISABLED - Fast mode' if self.no_gui else 'ENABLED'}\n"
            f"  \n"
            f"  CONTROLS:\n"
            f"  {'Manual screenshot disabled (use --no-gui false to enable)' if self.no_gui else 'Press s to save screenshot'}\n"
            f"  {'Press Ctrl+C to quit' if self.no_gui else 'Press q to quit'}"
        )

    def save_screenshots(self):
        """Save all windows as screenshots"""
        if self.current_original is None:
            self.get_logger().warn("No frames to save yet!")
            return
        
        self.screenshot_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filenames
        filename_base = f"screenshot_{self.screenshot_count:03d}_{timestamp}"
        
        file1 = os.path.join(self.output_dir, f"{filename_base}_1_original.png")
        file2 = os.path.join(self.output_dir, f"{filename_base}_2_detected.png")
        
        # Save images
        cv2.imwrite(file1, self.current_original)
        cv2.imwrite(file2, self.current_detected)
        
        self.get_logger().info(
            f"\n{'='*70}\n"
            f"📸 SCREENSHOT #{self.screenshot_count} SAVED!\n"
            f"{'='*70}\n"
            f"  1. {file1}\n"
            f"  2. {file2}\n"
            f"{'='*70}"
        )
        
        print(f"\n✅ Screenshot #{self.screenshot_count} saved to {self.output_dir}/")

    def image_callback(self, msg: Image):
        """Process and visualize AprilTag detection"""
        # Increment frame counter
        self.frame_count += 1
        
        t0 = time.perf_counter()
        
        # --- lag (message age) ---
        now = self.get_clock().now()
        msg_t = RclTime.from_msg(msg.header.stamp)
        lag = (now - msg_t).nanoseconds / 1e9  # seconds
        
        # --- inter-message spacing (skips) ---
        skipped = 0
        if self.last_msg_stamp is not None:
            dt = (msg_t - self.last_msg_stamp).nanoseconds / 1e9
            expected = 1.0 / 30.0  # Expected camera rate
            if dt > 1.5 * expected:
                skipped = int(round(dt / expected) - 1)
        self.last_msg_stamp = msg_t
        
        # Convert ROS image to OpenCV
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = image.shape[:2]
        
        # Convert to grayscale for AprilTag detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement (helps with dull images)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Detect AprilTags
        results = self.detector.detect(gray)
        
        # ===== WINDOW 1: Original Image =====
        self.current_original = image.copy()
        if not self.no_gui:
            cv2.imshow("1. Original Image", self.current_original)
        
        # ===== WINDOW 2: Detected AprilTags =====
        detected_vis = image.copy()
        
        tag_count = len(results)
        
        # Loop over AprilTag detection results
        for r in results:
            # Extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            
            # Draw the bounding box of the AprilTag detection
            cv2.line(detected_vis, ptA, ptB, (0, 255, 0), 2)
            cv2.line(detected_vis, ptB, ptC, (0, 255, 0), 2)
            cv2.line(detected_vis, ptC, ptD, (0, 255, 0), 2)
            cv2.line(detected_vis, ptD, ptA, (0, 255, 0), 2)
            
            # Draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(detected_vis, (cX, cY), 5, (0, 0, 255), -1)
            
            # Draw the tag family and ID on the image
            tagFamily = r.tag_family.decode("utf-8")
            tagID = r.tag_id
            cv2.putText(detected_vis, f"{tagFamily} (ID: {tagID})", 
                       (ptA[0], ptA[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add info text (only if GUI enabled)
        if not self.no_gui:
            cv2.putText(detected_vis, f"Detected: {tag_count} tags", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(detected_vis, f"Family: {self.tag_family}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(detected_vis, "Press 's' to save screenshot", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        self.current_detected = detected_vis.copy()
        if not self.no_gui:
            cv2.imshow("2. Detected AprilTags", self.current_detected)
        
        # Auto-screenshot at designated frames
        if self.screenshot_frames and self.frame_count in self.screenshot_frames:
            self.save_screenshots()
            self.screenshots_taken += 1
            self.get_logger().info(
                f"🎯 Auto-screenshot {self.screenshots_taken}/{self.num_screenshots} "
                f"at frame {self.frame_count}/{self.total_frames}"
            )
        
        # Handle keyboard input (only if GUI enabled)
        if not self.no_gui:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_screenshots()
            elif key == ord('q'):
                self.get_logger().info("Quit key pressed, shutting down...")
                raise KeyboardInterrupt
        
        # Improved logging
        if self.total_frames:
            progress_pct = (self.frame_count / self.total_frames) * 100
            self.get_logger().info(
                f"Frame {self.frame_count}/{self.total_frames} ({progress_pct:.1f}%) - {tag_count} tags detected"
            )
        elif tag_count > 0:
            self.get_logger().info(f"Frame {self.frame_count} - Detected {tag_count} tags")
        
        # --- processing time + EMA ---
        t_proc = time.perf_counter() - t0
        alpha = 0.2
        self.proc_ema = alpha * t_proc + (1 - alpha) * self.proc_ema
        
        # --- occasional log ---
        if time.perf_counter() - self.last_log_t > 2.0:
            self.last_log_t = time.perf_counter()
            self.get_logger().info(
                f"proc={t_proc*1000:.1f}ms (ema {self.proc_ema*1000:.1f}ms) | lag={lag:.2f}s | skipped≈{skipped}"
            )


def main(args=None):
    rclpy.init(args=args)
    
    import argparse
    parser = argparse.ArgumentParser(description="AprilTag Visualization Node (Baseline)")
    parser.add_argument("--tag_family", type=str, default="tagStandard41h12",
                       help="AprilTag family (tag36h11, tagStandard41h12, tag25h9, tag16h5, etc.)")
    parser.add_argument("--output_dir", type=str, default="apriltag_screenshots",
                       help="Directory to save screenshots")
    parser.add_argument("--total_frames", type=int, default=None,
                       help="Total frames in rosbag (for auto-screenshot)")
    parser.add_argument("--num_screenshots", type=int, default=5,
                       help="Number of screenshots to take automatically")
    parser.add_argument("--no-gui", action="store_true",
                       help="Disable GUI (faster, auto-screenshot only)")
    
    parsed_args, ros_args = parser.parse_known_args()
    
    node = AprilTagVisualizationNode(
        output_dir=parsed_args.output_dir,
        total_frames=parsed_args.total_frames,
        num_screenshots=parsed_args.num_screenshots,
        no_gui=parsed_args.no_gui,
        tag_family=parsed_args.tag_family
    )
    
    print("\n" + "="*70)
    print("APRILTAG DETECTION - BASELINE VISUALIZATION")
    print("="*70)
    print(f"Tag family: {parsed_args.tag_family}")
    print(f"Output directory: {parsed_args.output_dir}")
    print(f"GUI mode: {'DISABLED (Fast mode)' if parsed_args.no_gui else 'ENABLED'}")
    if parsed_args.total_frames:
        print(f"Auto-screenshot: {parsed_args.num_screenshots} screenshots over {parsed_args.total_frames} frames")
    else:
        print(f"Manual screenshot mode (press 's' to capture)")
    print("="*70)
    if not parsed_args.no_gui:
        print("\nShowing 2 windows:")
        print("  1. Original Image")
        print("  2. Detected AprilTags")
    else:
        print("\nGUI DISABLED - Running in fast mode")
        print("Screenshots will be saved automatically")
    print("\n🎮 CONTROLS:")
    if not parsed_args.no_gui:
        print("  Press 's' to save screenshot")
        print("  Press 'q' to quit")
    print("  Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()