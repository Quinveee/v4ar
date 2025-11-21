#!/usr/bin/env python3
"""
AprilTag Detection with OAK-D Depth - OPTIMIZED FOR LOW-COMPUTE
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime
from message_filters import ApproximateTimeSynchronizer, Subscriber

from apriltag import apriltag
from perception_msgs.msg import MarkerPoseArray, MarkerPose


# --- Utility Function ---
def rotation_matrix_to_quaternion(R: np.ndarray):
    # (Implementation remains the same)
    q = np.zeros(4, dtype=np.float64)
    trace = np.trace(R)

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[3] = 0.25 / s
        q[0] = (R[2, 1] - R[1, 2]) * s
        q[1] = (R[0, 2] - R[2, 0]) * s
        q[2] = (R[1, 0] - R[0, 1]) * s
    else:
        idx = np.argmax(np.diag(R))
        if idx == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s

    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


class AprilTagVisualizationNode(Node):
    def __init__(self,
                 output_dir: str = 'apriltag_screenshots',
                 no_gui: bool = True,
                 tag_family: str = 'tagStandard41h12'):

        super().__init__('apriltag_visualization')

        self.bridge = CvBridge()
        self.no_gui = no_gui

        self.screenshot_count = 0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.current_original = None
        self.current_detected = None
        self.frame_count = 0
        self.first_detection = True
        self.tag_size = 0.162

        # ------------------------------------------------------------------
        # --- NEW: Camera Matrix and Distortion Coefficients initialized ---
        # ------------------------------------------------------------------
        self.K = None
        self.D = None
        self.P = None # Projection matrix for undistort
        self.mapx = None
        self.mapy = None
        self.get_logger().info("Awaiting CameraInfo to initialize K and D matrices.")

        self.color_info_sub = self.create_subscription(
            CameraInfo,
            '/oak/rgb/camera_info',
            self.camera_info_callback,
            1
        )
        
        # ------------------------------------------------------------------
        # --- TOPICS: Use raw color and the converted depth stream ---
        # ------------------------------------------------------------------
        self.color_sub = Subscriber(self, Image, '/oak/rgb/image_raw')
        self.depth_sub = Subscriber(self, Image, '/oak/stereo/converted_depth')

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.sync_callback)

        self.pub = self.create_publisher(MarkerPoseArray, '/detected_markers', 10)

        # ------------------------------------------------------------------
        # --- AprilTag Detector Setup (tuned for lower res) ---
        # ------------------------------------------------------------------
        self.declare_parameter('tag_family', tag_family)
        self.declare_parameter('output_dir', output_dir)
        self.declare_parameter('no_gui', no_gui)

        self.tag_family = self.get_parameter('tag_family').value
        self.no_gui = self.get_parameter('no_gui').value

        # Tuned parameters for better performance on lower res/compute
        self.detector = apriltag(
            self.tag_family,
            threads=2,             # Lower threads for less CPU usage
            maxhamming=2,
            decimate=0.25,
            blur=0.0,              # Disable blur if image is already clean
            refine_edges=True,
            debug=False
        )
        self.get_logger().info(f"AprilTag detector initialized: {self.tag_family}")

        if not self.no_gui:
            cv2.namedWindow("1. Original Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow("2. Detected AprilTags", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("1. Original Image", 640, 480)
            cv2.resizeWindow("2. Detected AprilTags", 640, 480)

        self.get_logger().info("Node initialized, waiting for synchronized color + depth")
        
    def camera_info_callback(self, msg: CameraInfo):
        """Extract K and D matrices from CameraInfo message."""
        if self.K is None:
            # K is 3x3 (row-major)
            self.K = np.array(msg.k).reshape((3, 3))
            
            # D is 1xn array
            self.D = np.array(msg.d)
            
            # P is 3x4 projection matrix (often the rectified K matrix)
            P_matrix = np.array(msg.p).reshape((3, 4))
            # We'll use the 3x3 part of P as the New Camera Matrix (K_new) for undistortion
            K_new = P_matrix[:, 0:3] 
            
            h, w = msg.height, msg.width
            
            # Calculate the undistortion maps only once
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                self.K, 
                self.D, 
                None, # Identity for R since we are using rectified image space
                K_new, 
                (w, h), 
                cv2.CV_32FC1
            )
            
            # Update the K matrix to the new rectified K for pose estimation
            self.K = K_new
            self.get_logger().info(f"Camera parameters loaded and rectification map created. Image size: {w}x{h}")
            self.destroy_subscription(self.color_info_sub) # Stop subscribing to info

    def save_screenshots(self):
        # (Implementation remains the same)
        if self.current_original is None or self.current_detected is None:
            self.get_logger().warn("No frames yet to save!")
            return

        self.screenshot_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        f1 = os.path.join(self.output_dir, f"shot_{self.screenshot_count:03d}_{ts}_orig.png")
        f2 = os.path.join(self.output_dir, f"shot_{self.screenshot_count:03d}_{ts}_det.png")

        cv2.imwrite(f1, self.current_original)
        cv2.imwrite(f2, self.current_detected)

        self.get_logger().info(f"Saved screenshots:\n  {f1}\n  {f2}")

    def sync_callback(self, color_msg: Image, depth_msg: Image):
        if self.K is None:
            # Skip if camera parameters haven't been loaded yet
            return 
            
        self.frame_count += 1

        color_image = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

        # ------------------------------------------------------------------
        # --- OPTIMIZATION 1: Undistort the image (Equivalent to Rectify)
        # ------------------------------------------------------------------
        color_image_undistorted = cv2.remap(
            color_image, 
            self.mapx, 
            self.mapy, 
            cv2.INTER_LINEAR
        )
        
        # ------------------------------------------------------------------
        # --- OPTIMIZATION 2: Downscale for faster AprilTag detection ---
        # ------------------------------------------------------------------
        # Target width for detection (e.g., 640px)
        target_width = 640 
        scale_factor = target_width / color_image_undistorted.shape[1]
        
        downscaled_image = cv2.resize(
            color_image_undistorted, 
            (target_width, int(color_image_undistorted.shape[0] * scale_factor)), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Update the scale factor for later use in pose estimation
        self.scale_factor = scale_factor
        
        self.current_original = color_image_undistorted.copy() # Store the full-res rectified image for visualization

        # ------------------------------------------------------------------
        # --- Detection on Downscaled Grayscale Image ---
        # ------------------------------------------------------------------
        gray = cv2.cvtColor(downscaled_image, cv2.COLOR_BGR2GRAY)

        # CLAHE (Contrast Limiting Adaptive Histogram Equalization) is kept
        # as a useful pre-processing step for detection robustness.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray) 

        detections = self.detector.detect(gray)
        detected_vis = color_image_undistorted.copy() # Draw on full-res image

        marker_array_msg = MarkerPoseArray()
        marker_array_msg.header = color_msg.header

        for d in detections:
            # ... (Corner extraction logic remains the same) ...
            try:
                if 'lb' in d:
                    corners_d = np.array([d['lb'], d['rb'], d['rt'], d['lt']])
                elif 'lb-rb-rt-lt' in d:
                    corners_d = d['lb-rb-rt-lt']
                else:
                    self.get_logger().error(f"Unknown corner format. Keys: {d.keys()}")
                    continue
                
                # Scale corners back to original full-resolution image coordinates
                corners_full_res = (corners_d / self.scale_factor).astype(int)
                
                lt, rt, rb, lb = tuple(corners_full_res[3]), tuple(corners_full_res[2]), \
                                 tuple(corners_full_res[1]), tuple(corners_full_res[0])
                
                center_d = d['center']
                center_full_res = (center_d / self.scale_factor).astype(int)
                center = tuple(center_full_res)


                # ------------------------------------------------------------------
                # --- Drawing and Pose Message Initialization ---
                # ------------------------------------------------------------------
                # Drawing on the full-resolution image
                cv2.line(detected_vis, lt, rt, (0, 255, 0), 2)
                cv2.line(detected_vis, rt, rb, (0, 255, 0), 2)
                cv2.line(detected_vis, rb, lb, (0, 255, 0), 2)
                cv2.line(detected_vis, lb, lt, (0, 255, 0), 2)

                cv2.circle(detected_vis, center, 5, (0, 0, 255), -1)

                tag_id = d['id']
                cv2.putText(
                    detected_vis,
                    f"{self.tag_family} id={tag_id}",
                    (lt[0] - 70, lt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2
                )

                pose_msg = Pose()
                pose_msg.orientation.w = 1.0
                distance_value = -1.0

                # ------------------------------------------------------------------
                # --- 3D Position Calculation (using full-res center) ---
                # ------------------------------------------------------------------
                try:
                    cx, cy = int(center[0]), int(center[1])
                    
                    if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                        # Depth image may have a different resolution than color image. 
                        # This assumes the depth image is *aligned* or *reprojected* to 
                        # the same frame as the color image, which is common with OAK-D drivers.
                        depth_at_center = depth_image[cy, cx]
                        
                        if depth_at_center > 0 and not np.isnan(depth_at_center):
                            # Convert depth from mm (typical OAK-D output) to meters
                            z_depth = float(depth_at_center) / 1000.0
                            
                            # Use the rectified K matrix (self.K) for 3D projection
                            fx = self.K[0, 0]
                            fy = self.K[1, 1]
                            cx_cam = self.K[0, 2]
                            cy_cam = self.K[1, 2]
                            
                            # 3D coordinates (X is right, Y is down, Z is forward)
                            x_3d = (cx - cx_cam) * z_depth / fx
                            y_3d = (cy - cy_cam) * z_depth / fy
                            
                            distance_value = np.sqrt(x_3d**2 + y_3d**2 + z_depth**2)
                            
                            pose_msg.position.x = float(x_3d)
                            pose_msg.position.y = float(y_3d)
                            pose_msg.position.z = float(z_depth)
                            
                            cv2.putText(
                                detected_vis,
                                f"Z={z_depth:.2f}m Dist={distance_value:.2f}m",
                                (lt[0] - 55, lt[1] - 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 2
                            )
                        else:
                            self.get_logger().warn(f"Invalid depth at center: {depth_at_center}")
                    else:
                        self.get_logger().warn(f"Center out of bounds: ({cx}, {cy})")

                except Exception as e:
                    self.get_logger().warn(f"Depth lookup failed: {e}")

            except Exception as e:
                self.get_logger().error(f"Error processing detection: {e}")
                continue

            # ... (MarkerPose message creation remains the same) ...
            marker_msg = MarkerPose()
            marker_msg.id = int(tag_id)
            marker_msg.pose = pose_msg
            marker_msg.distance = float(distance_value)
            marker_msg.center_x = float(center[0])
            marker_msg.center_y = float(center[1])

            xs = [lt[0], rt[0], rb[0], lb[0]]
            ys = [lt[1], rt[1], rb[1], lb[1]]
            marker_msg.bbox_x_min = float(min(xs))
            marker_msg.bbox_y_min = float(min(ys))
            marker_msg.bbox_x_max = float(max(xs))
            marker_msg.bbox_y_max = float(max(ys))
            marker_msg.corners = [
                float(lt[0]), float(lt[1]),
                float(rt[0]), float(rt[1]),
                float(rb[0]), float(rb[1]),
                float(lb[0]), float(lb[1]),
            ]

            marker_array_msg.markers.append(marker_msg)

        self.current_detected = detected_vis.copy()

        if not self.no_gui:
            # Downscale for display to save CPU/GPU if the original is very large
            display_res = cv2.resize(self.current_original, (640, 480))
            display_det = cv2.resize(self.current_detected, (640, 480))
            
            cv2.imshow("1. Original Image", display_res)
            cv2.imshow("2. Detected AprilTags", display_det)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_screenshots()
            if key == ord('q'):
                raise KeyboardInterrupt

        self.pub.publish(marker_array_msg)


def main(args=None):
    rclpy.init(args=args)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag_family", type=str, default="tagStandard41h12")
    parser.add_argument("--output_dir", type=str, default="apriltag_screenshots")
    parser.add_argument("--no_gui", action="store_true")
    parsed, _ = parser.parse_known_args()

    node = AprilTagVisualizationNode(
        output_dir=parsed.output_dir,
        no_gui=parsed.no_gui,
        tag_family=parsed.tag_family
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()