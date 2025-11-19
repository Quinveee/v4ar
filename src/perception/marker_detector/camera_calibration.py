#!/usr/bin/env python3
"""
Camera Calibration Script for UGV Rover
Captures images from ROS camera topic and performs chessboard calibration
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime


class CameraCalibrationNode(Node):
    def __init__(self, chessboard_size=(7, 6), square_size=1.0, 
                 num_images=20, output_dir='calibration_images'):
        super().__init__('camera_calibration')
        
        self.bridge = CvBridge()
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.num_images = num_images
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calibration data
        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane
        self.captured_count = 0
        self.current_frame = None
        
        # Prepare object points (0,0,0), (1,0,0), ..., (6,5,0)
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 
                                     0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Subscribe to camera topic
        self.declare_parameter('image_topic', '/image_raw')
        image_topic = self.get_parameter('image_topic').value
        
        self.sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )
        
        cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Calibration", 800, 600)
        
        self.get_logger().info(f"Camera Calibration Node Started")
        self.get_logger().info(f"Chessboard size: {chessboard_size}")
        self.get_logger().info(f"Need {num_images} images with detected corners")
        self.get_logger().info(f"Press 'c' to capture, 'q' to quit and calibrate")
        
    def image_callback(self, msg):
        """Process incoming images"""
        self.current_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, self.chessboard_size, None
        )
        
        # Draw visualization
        display_frame = self.current_frame.copy()
        
        if ret:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self.criteria
            )
            
            # Draw corners
            cv2.drawChessboardCorners(
                display_frame, self.chessboard_size, corners2, ret
            )
            
            # Add text
            cv2.putText(
                display_frame, 
                f"Chessboard detected! Press 'c' to capture",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                display_frame,
                "No chessboard detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        
        # Show capture count
        cv2.putText(
            display_frame,
            f"Captured: {self.captured_count}/{self.num_images}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        cv2.imshow("Camera Calibration", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and ret:
            # Capture this frame
            self.objpoints.append(self.objp)
            self.imgpoints.append(corners2)
            self.captured_count += 1
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                self.output_dir, 
                f"calib_{self.captured_count:03d}_{timestamp}.jpg"
            )
            cv2.imwrite(filename, self.current_frame)
            
            self.get_logger().info(
                f"Captured image {self.captured_count}/{self.num_images}: {filename}"
            )
            
            if self.captured_count >= self.num_images:
                self.get_logger().info("Enough images captured! Press 'q' to calibrate")
        
        elif key == ord('q'):
            if self.captured_count >= 3:
                self.perform_calibration()
            else:
                self.get_logger().warn(
                    f"Need at least 3 images, only have {self.captured_count}"
                )
            raise KeyboardInterrupt
    
    def perform_calibration(self):
        """Perform camera calibration with captured images"""
        self.get_logger().info(f"Performing calibration with {self.captured_count} images...")
        
        if len(self.imgpoints) == 0:
            self.get_logger().error("No valid images for calibration!")
            return
        
        # Get image size from first frame
        img_shape = self.current_frame.shape[:2][::-1]  # (width, height)
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_shape, None, None
        )
        
        if ret:
            self.get_logger().info("=" * 60)
            self.get_logger().info("CALIBRATION SUCCESSFUL!")
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"\nCamera Matrix:\n{camera_matrix}")
            self.get_logger().info(f"\nDistortion Coefficients:\n{dist_coeffs}")
            
            # Calculate reprojection error
            mean_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(
                    self.objpoints[i], rvecs[i], tvecs[i], 
                    camera_matrix, dist_coeffs
                )
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            
            mean_error /= len(self.objpoints)
            self.get_logger().info(f"\nMean Reprojection Error: {mean_error:.4f} pixels")
            
            # Save calibration to file
            calib_file = os.path.join(self.output_dir, 'camera_calibration.npz')
            np.savez(
                calib_file,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                rvecs=rvecs,
                tvecs=tvecs,
                mean_error=mean_error,
                image_shape=img_shape
            )
            self.get_logger().info(f"\nCalibration saved to: {calib_file}")
            
            # Save as YAML for ROS
            yaml_file = os.path.join(self.output_dir, 'camera_calibration.yaml')
            with open(yaml_file, 'w') as f:
                f.write("image_width: {}\n".format(img_shape[0]))
                f.write("image_height: {}\n".format(img_shape[1]))
                f.write("camera_name: ugv_camera\n")
                f.write("camera_matrix:\n")
                f.write("  rows: 3\n")
                f.write("  cols: 3\n")
                f.write("  data: {}\n".format(camera_matrix.flatten().tolist()))
                f.write("distortion_model: plumb_bob\n")
                f.write("distortion_coefficients:\n")
                f.write("  rows: 1\n")
                f.write("  cols: 5\n")
                f.write("  data: {}\n".format(dist_coeffs.flatten().tolist()))
            
            self.get_logger().info(f"YAML calibration saved to: {yaml_file}")
            self.get_logger().info("=" * 60)
        else:
            self.get_logger().error("Calibration failed!")


def main(args=None):
    rclpy.init(args=args)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cols", type=int, default=7, 
                       help="Number of inner corners in chessboard columns")
    parser.add_argument("--rows", type=int, default=6,
                       help="Number of inner corners in chessboard rows")
    parser.add_argument("--square-size", type=float, default=1.0,
                       help="Size of chessboard squares in your units (e.g., cm)")
    parser.add_argument("--num-images", type=int, default=20,
                       help="Number of images to capture")
    parser.add_argument("--output-dir", type=str, default="calibration_images",
                       help="Directory to save calibration images")
    
    parsed, _ = parser.parse_known_args()
    
    node = CameraCalibrationNode(
        chessboard_size=(parsed.cols, parsed.rows),
        square_size=parsed.square_size,
        num_images=parsed.num_images,
        output_dir=parsed.output_dir
    )
    
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
