#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class LineDetectorNode(Node):
    def __init__(self):
        super().__init__('line_detector_node')
        
        self.bridge = CvBridge()
        
        # Subscribe to camera images
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # or /image_rect for rectified
            self.image_callback,
            10)
        
        # Counter for saving images
        self.frame_count = 0
        
        self.get_logger().info('Line Detector Node started!')
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Detect lines with Canny (baseline)
            canny_result = self.detect_lines_canny(cv_image)
            
            # Display results
            cv2.imshow('Original Image', cv_image)
            cv2.imshow('Canny Edge Detection', canny_result)
            
            # Press 's' to save current frame
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_images(cv_image, canny_result)
            
            self.frame_count += 1
            
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')
    
    def detect_lines_canny(self, image):
        """
        TASK 2: Basic Canny edge detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        return edges
    
    def save_images(self, original, processed):
        """Save images for your report"""
        cv2.imwrite(f'/root/v4ar/docs/results/original_{self.frame_count}.png', original)
        cv2.imwrite(f'/root/v4ar/docs/results/canny_{self.frame_count}.png', processed)
        self.get_logger().info(f'Saved frame {self.frame_count}')

def main(args=None):
    rclpy.init(args=args)
    node = LineDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()