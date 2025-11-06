# Basic ROS 2 program to subscribe to real-time streaming
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com

# Import the necessary libraries
from email.mime import image
import rclpy  # Python library for ROS 2
from rclpy.node import Node  # Handles the creation of nodes
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library
import numpy as np


class ImageSubscriber(Node):
    """
    Create an ImageSubscriber class, which is a subclass of the Node class.
    """

    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('image_subscriber')

        # Create the subscriber. This subscriber will receive an Image
        # from the video_frames topic. The queue size is 10 messages.
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

    def detect_lines_canny(self, image):
        """
        TASK 2: Basic Canny edge detection
        """
        # Convert to grayscale
        original = image.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

        edges = cv2.Canny(blurred, 50, 150)

        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(edges,
                                rho=1,              # distance resolution in pixels
                                theta=np.pi/180,    # angle resolution in radians
                                threshold=80,       # min number of votes
                                minLineLength=50,   # shortest line to detect
                                maxLineGap=10)      # max allowed gap between line segments

        # Draw lines on a copy of the original image
        line_img = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # return cv2.addWeighted(original, 0.1, line_img, 1, 0)

        return line_img

    def detect_ceiling_light_brightness(self, image):
        """
        CEILING LIGHT FOLLOWER: Brightness-based line detection with overexposure filtering
        Simple algorithm for following bright ceiling lights - no edge detection needed!
        """
        height, width = image.shape[:2]
        line_img = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to smooth out noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # STEP 1: Remove overexposed/saturated pixels (sunlight/glare removal)
        # Check if >30% of pixels are at maximum brightness (overexposed)
        max_brightness = 255
        overexposed_pixels = np.sum(blurred >= max_brightness)
        total_pixels = height * width
        overexposure_ratio = overexposed_pixels / total_pixels
        
        # Create mask to exclude overexposed regions
        if overexposure_ratio > 0.30:  # If >30% pixels are maxed out
            # Remove pixels that are too bright (likely sun glare/overexposure)
            overexposure_threshold = 250  # Remove pixels above this value
            clean_mask = blurred < overexposure_threshold
            cleaned_image = blurred.copy()
            cleaned_image[~clean_mask] = 0  # Set overexposed pixels to black
            
            # Debug info
            print(f"Overexposure detected: {overexposure_ratio:.1%} pixels saturated")
            print(f"Removing pixels above {overexposure_threshold}")
            
            # Use cleaned image for further processing
            processing_image = cleaned_image
        else:
            # No significant overexposure, use original
            processing_image = blurred
            print(f"Normal lighting: {overexposure_ratio:.1%} pixels saturated")
        
        # STEP 2: Find bright regions from the cleaned image (ceiling lights)
        # Use adaptive threshold on the cleaned image
        brightness_threshold = np.mean(processing_image) + 0.5 * np.std(processing_image)
        _, bright_mask = cv2.threshold(processing_image, brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Debug info
        # bright_pixels = np.sum(bright_mask > 0)
        # bright_ratio = bright_pixels / total_pixels
        # print(f"Bright threshold: {brightness_threshold:.1f}")
        # print(f"Bright pixels found: {bright_ratio:.1%}")
        
        # Morphological operations to clean up and connect light segments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        
        # Use Hough Line Transform on bright regions (not edges!)
        lines = cv2.HoughLinesP(bright_mask,
                                rho=1,              
                                theta=np.pi/180,    
                                threshold=50,       # Lower threshold since we have clean bright regions
                                minLineLength=30,   
                                maxLineGap=20)     # Allow larger gaps for light fixtures
        
        if lines is not None:
            print(f"Found {len(lines)} total lines")
            
            # Filter for mostly vertical lines (since camera points up)
            vertical_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line angle
                if x2 != x1:  # Avoid division by zero
                    angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
                else:
                    angle = 90  # Perfectly vertical
                
                # Keep lines that are mostly vertical (60-90 degrees)
                if angle > 60:
                    vertical_lines.append(line[0])
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            print(f"Found {len(vertical_lines)} vertical lines")
            
            # Find the best line to follow (closest to center or longest)
            if vertical_lines:
                best_line = self.select_line_to_follow(vertical_lines, width, height)
                if best_line is not None:
                    x1, y1, x2, y2 = best_line
                    
                    # Calculate and display angle
                    if x2 != x1:
                        angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
                    else:
                        angle = 90
                    
                    # Draw the selected line in red
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    
                    # Draw navigation info
                    line_center_x = (x1 + x2) // 2
                    image_center_x = width // 2
                    offset = line_center_x - image_center_x
                    
                    # Draw center line and offset
                    cv2.line(line_img, (image_center_x, 0), (image_center_x, height), (255, 0, 0), 2)
                    cv2.line(line_img, (line_center_x, 0), (line_center_x, height), (0, 0, 255), 2)
                    
                    # Display navigation text
                    if abs(offset) < 10:
                        nav_text = "GO STRAIGHT"
                        color = (0, 255, 0)
                    elif offset > 0:
                        nav_text = f"TURN RIGHT ({offset}px)"
                        color = (0, 100, 255)
                    else:
                        nav_text = f"TURN LEFT ({abs(offset)}px)"
                        color = (255, 100, 0)
                    
                    cv2.putText(line_img, nav_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Display angle and overexposure info
                    angle_text = f"Angle: {angle:.1f}deg"
                    cv2.putText(line_img, angle_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    overexp_text = f"Overexp: {overexposure_ratio:.1%}"
                    color_overexp = (0, 0, 255) if overexposure_ratio > 0.30 else (255, 255, 255)
                    cv2.putText(line_img, overexp_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_overexp, 2)
        else:
            print("No lines found after filtering!")
            # Optional: show the bright mask for debugging
            cv2.imshow("bright_mask_debug", bright_mask)
        
        return line_img
    
    def select_line_to_follow(self, lines, width, height):
        """
        Select the best line to follow based on:
        1. Proximity to image center
        2. Line length
        3. Vertical orientation
        """
        if not lines:
            return None
        
        image_center_x = width // 2
        best_line = None
        best_score = -1
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Calculate line properties
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            line_center_x = (x1 + x2) // 2
            distance_from_center = abs(line_center_x - image_center_x)
            
            # Score: prioritize longer lines closer to center
            # Normalize distance (0-1) and length (0-1)
            distance_score = 1 - (distance_from_center / (width // 2))
            length_score = min(line_length / 100, 1.0)  # Normalize to max 100px
            
            # Combined score (weight center proximity more for navigation)
            score = 0.7 * distance_score + 0.3 * length_score
            
            if score > best_score:
                best_score = score
                best_line = line
        
        return best_line

    # def detect_edges_dog(image, sigma1=1, sigma2=2):
    #     """Simple Difference-of-Gaussians edge detector."""
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     # Blur with two different Gaussian sigmas
    #     blur1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    #     blur2 = cv2.GaussianBlur(gray, (0, 0), sigma2)

    #     # Difference of Gaussians
    #     dog = blur1 - blur2

    #     # Normalize for display
    #     dog = cv2.normalize(dog, None, 0, 255,
    #                         cv2.NORM_MINMAX).astype(np.uint8)

    #     # Optional: threshold to get binary edges
    #     _, edges = cv2.threshold(dog, 20, 255, cv2.THRESH_BINARY)

    #     return edges

    def listener_callback(self, data):
        """
        Callback function.
        """
        # Display the message on the console
        self.get_logger().info('Receiving video frame')

        # Convert ROS Image message to OpenCV image
        current_frame = self.br.imgmsg_to_cv2(data)

        # Use brightness-based ceiling light detection (no Canny!)
        current_frame = self.detect_ceiling_light_brightness(current_frame)
        
        # Alternative: Use traditional Canny detection
        # current_frame = self.detect_lines_canny(current_frame)
        # current_frame = self.detect_edges_dog(current_frame)

        # Display image
        cv2.imshow("camera", current_frame)

        cv2.waitKey(1)


def main(args=None):

    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    image_subscriber = ImageSubscriber()

    # Spin the node so the callback function is called.
    rclpy.spin(image_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    image_subscriber.destroy_node()

    # Shutdown the ROS client library for Python
    rclpy.shutdown()


if __name__ == '__main__':
    main()
