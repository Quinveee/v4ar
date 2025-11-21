#!/usr/bin/env python3
"""
AprilTag Detection Visualization Node (manual rectification using given K/D/P)
"""

import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime
import yaml

# your rover's apriltag library
from apriltag import apriltag
from perception_msgs.msg import MarkerPoseArray, MarkerPose


def rotation_matrix_to_quaternion(R: np.ndarray):
    """
    Convert a 3x3 rotation matrix to quaternion (x, y, z, w).
    """
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
                 tag_family: str = 'tagStandard41h12',
                 calibration_file: str = None):

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

        # --- Calibration parameters (K, D, P) ---

        # Updated calibration parameters
        self.K = np.array([
            [286.9896545092208, 0.0, 311.7114840273407],
            [0.0, 290.8395992360502, 249.9287049631703],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.D = np.array([[
            -0.17995587161461585, 0.020688274841999105, -
            0.005297672531455161, 0.003378882156848116, 0.0
        ]], dtype=np.float64)

        # Projection matrix from calibration (3x4)
        self.P = np.array([
            [190.74984649646845, 0.0, 318.0141593176815, 0.0],
            [0.0, 247.35103262891005, 248.37293105876694, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float64)

        self.get_logger().info("Updated calibration parameters with new matrices.")

        if calibration_file:
            self.load_calibration(calibration_file)

        # rectification maps
        self.map1 = None
        self.map2 = None
        self.new_K = None

        # Parameters
        self.declare_parameter('tag_family', tag_family)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('output_dir', output_dir)
        self.declare_parameter('no_gui', no_gui)
        self.declare_parameter('calibration_file', calibration_file or '')
        self.declare_parameter('use_raw', False)
        use_raw = self.get_parameter('use_raw').value

        if use_raw:
            self.K = np.array([
                [289.11451, 0.0, 347.23664],
                [0.0, 289.75319, 235.67429],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

            self.D = np.array([[
                -0.208848, 0.028006, -0.000705, -0.00082, 0.0
            ]], dtype=np.float64)

            self.P = np.array([
                [196.89772, 0.0, 342.88724, 0.0],
                [0.0, 234.53159, 231.54267, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ], dtype=np.float64)

            self.get_logger().info("Using image_rect calibration parameters.")
        else:
            self.get_logger().info("Using our calibration parameters.")

        self.tag_family = self.get_parameter('tag_family').value
        image_topic = self.get_parameter('image_topic').value
        self.no_gui = self.get_parameter('no_gui').value
        calib_param = self.get_parameter('calibration_file').value

        if calib_param and not calibration_file:
            self.load_calibration(calib_param)

        # AprilTag detector
        self.detector = apriltag(
            self.tag_family,
            threads=4,
            maxhamming=2,
            decimate=0.25,
            blur=0.5,
            refine_edges=True,
            debug=False
        )
        self.get_logger().info(
            f"AprilTag detector initialized: {self.tag_family}")

        # Single subscription: raw image only
        self.sub = self.create_subscription(
            Image,
            '/processed_image',
            self.image_callback,
            10
        )

        self.pub = self.create_publisher(
            MarkerPoseArray, '/detected_markers', 10)

        self.display_rectified_image = None

        if self.no_gui:
            cv2.namedWindow("1. Original Image", cv2.WINDOW_NORMAL)
            cv2.namedWindow(
                "2. Detected AprilTags (Postprocessed)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("1. Original Image", 640, 480)
            cv2.resizeWindow("2. Detected AprilTags (Postprocessed)", 640, 480)

        # add buffer
        self.marker_buffer = {}

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def save_screenshots(self):
        if self.current_original is None or self.current_detected is None:
            self.get_logger().warn("No frames yet to save!")
            return

        self.screenshot_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        f1 = os.path.join(
            self.output_dir, f"shot_{self.screenshot_count:03d}_{ts}_orig.png")
        f2 = os.path.join(
            self.output_dir, f"shot_{self.screenshot_count:03d}_{ts}_det.png")

        cv2.imwrite(f1, self.current_original)
        cv2.imwrite(f2, self.current_detected)

        self.get_logger().info(f"Saved screenshots:\n  {f1}\n  {f2}")

    # ------------------------------------------------------------------ #
    # Main callback
    # ------------------------------------------------------------------ #
    def image_callback(self, msg: Image):
        self.frame_count += 1

        # Convert raw image
        rectified = self.bridge.imgmsg_to_cv2(msg)
        self.current_original = rectified.copy()

        if self.new_K is None and self.P is not None:
            self.new_K = self.P[:, :3]

        # This is the image we run AprilTag on
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

        # Optional preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # kernel = np.array([[-1, -1, -1],
        #                    [-1,  9, -1],
        #                    [-1, -1, -1]], dtype=np.float32)
        # postprocessed = cv2.filter2D(gray, -1, kernel)

        detections = self.detector.detect(gray)
        detected_vis = rectified.copy()

        marker_array_msg = MarkerPoseArray()
        marker_array_msg.header = msg.header

        for d in detections:
            if self.first_detection:
                self.get_logger().info(f"Detection keys: {d.keys()}")
                self.get_logger().info(f"Detection structure: {d}")
                self.first_detection = False

            try:
                # Extract corners
                if 'lb' in d:
                    lb = tuple(d['lb'].astype(int))
                    rb = tuple(d['rb'].astype(int))
                    rt = tuple(d['rt'].astype(int))
                    lt = tuple(d['lt'].astype(int))
                elif 'lb-rb-rt-lt' in d:
                    corners = d['lb-rb-rt-lt']
                    lb = tuple(corners[0].astype(int))
                    rb = tuple(corners[1].astype(int))
                    rt = tuple(corners[2].astype(int))
                    lt = tuple(corners[3].astype(int))
                else:
                    self.get_logger().error(
                        f"Unknown corner format. Keys: {d.keys()}")
                    continue

                # Draw box
                cv2.line(detected_vis, lt, rt, (0, 255, 0), 2)
                cv2.line(detected_vis, rt, rb, (0, 255, 0), 2)
                cv2.line(detected_vis, rb, lb, (0, 255, 0), 2)
                cv2.line(detected_vis, lb, lt, (0, 255, 0), 2)

                # Center
                center = tuple(d['center'].astype(int))
                cv2.circle(detected_vis, center, 5, (0, 0, 255), -1)

                # Label
                tag_id = d['id']
                cv2.putText(
                    detected_vis,
                    f"{self.tag_family} id={tag_id}",
                    (lt[0] - 70, lt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2
                )

                # ----------- DEPTH / POSE ESTIMATION ----------------
                pose_msg = Pose()
                pose_msg.orientation.w = 1.0
                distance_value = -1.0

                try:
                    # Build image corner array in correct order:
                    # AprilTag standard order:  lt, rt, rb, lb
                    img_pts = np.array([lt, rt, rb, lb], dtype=np.float32)

                    # Camera intrinsics
                    # fx = self.K[0, 0]
                    # fy = self.K[1, 1]
                    # cx = self.K[0, 2]
                    # cy = self.K[1, 2]
                    fx = self.new_K[0, 0]
                    fy = self.new_K[1, 1]
                    cx = self.new_K[0, 2]
                    cy = self.new_K[1, 2]

                    camera_matrix = self.new_K
                    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
                    # camera_matrix = self.K
                    # dist_coeffs = self.D

                    # 3D tag model (square, lying flat on z=0)
                    s = self.tag_size / 2.0
                    obj_pts = np.array([
                        [-s,  s, 0],
                        [s,  s, 0],
                        [s, -s, 0],
                        [-s, -s, 0]
                    ], dtype=np.float32)

                    # Solve for pose
                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, camera_matrix, dist_coeffs
                    )

                    if not success:
                        raise RuntimeError("solvePnP returned False")

                    # Convert rotation vector -> matrix
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.flatten()

                    distance = np.linalg.norm(t)
                    distance_value = float(distance)

                    pose_msg.position.x = float(t[0])
                    pose_msg.position.y = float(t[1])
                    pose_msg.position.z = float(t[2])

                    qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
                    pose_msg.orientation.x = qx
                    pose_msg.orientation.y = qy
                    pose_msg.orientation.z = qz
                    pose_msg.orientation.w = qw

                    # Display depth + distance
                    cv2.putText(
                        detected_vis,
                        f"Z={t[2]:.2f}m Dist={distance:.2f}m",
                        (lt[0] - 55, lt[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2
                    )

                except Exception as e:
                    self.get_logger().warn(f"Pose failed (PnP): {e}")

            except Exception as e:
                self.get_logger().error(f"Error processing detection: {e}")
                continue

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

        # Update the detected AprilTags window to show the rectified image with detections
        self.current_detected = detected_vis.copy()

        if self.no_gui:
            cv2.imshow("1. Original Image", self.current_original)
            cv2.imshow("2. Detected AprilTags (Postprocessed)",
                       self.current_detected)

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
    parser.add_argument("--output_dir", type=str,
                        default="apriltag_screenshots")
    parser.add_argument("--no_gui", action="store_false")
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration YAML (optional)"
    )
    parser.add_argument(
        "--use_raw",
        action="store_true",
        help="Use experimental calibration parameters instead of the default ones."
    )
    parsed, _ = parser.parse_known_args()

    node = AprilTagVisualizationNode(
        output_dir=parsed.output_dir,
        no_gui=parsed.no_gui,
        tag_family=parsed.tag_family,
        calibration_file=parsed.calibration
    )

    if parsed.use_raw:
        node.set_parameters([rclpy.parameter.Parameter(
            'use_raw', rclpy.Parameter.Type.BOOL, True)])

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

# !/usr/bin/env python3

# """
# AprilTag Detection Visualization Node (manual rectification using given K/D/P)
# """

# import time
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import os
# from datetime import datetime
# import yaml

# # your rover's apriltag library
# from apriltag import apriltag


# class AprilTagVisualizationNode(Node):
#     def __init__(self,
#                  output_dir: str = 'apriltag_screenshots',
#                  no_gui: bool = False,
#                  tag_family: str = 'tagStandard41h12',
#                  calibration_file: str = None):

#         super().__init__('apriltag_visualization')

#         self.bridge = CvBridge()
#         self.no_gui = no_gui

#         self.screenshot_count = 0
#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)

#         self.current_original = None
#         self.current_detected = None
#         self.frame_count = 0
#         self.first_detection = True

#         # --- Calibration parameters (K, D, P) ---

#         # Updated calibration parameters
#         self.K = np.array([
#             [286.9896545092208, 0.0, 311.7114840273407],
#             [0.0, 290.8395992360502, 249.9287049631703],
#             [0.0, 0.0, 1.0]
#         ], dtype=np.float64)

#         self.D = np.array([[
#             -0.17995587161461585, 0.020688274841999105, -0.005297672531455161, 0.003378882156848116, 0.0
#         ]], dtype=np.float64)

#         # Projection matrix from calibration (3x4)
#         self.P = np.array([
#             [190.74984649646845, 0.0, 318.0141593176815, 0.0],
#             [0.0, 247.35103262891005, 248.37293105876694, 0.0],
#             [0.0, 0.0, 1.0, 0.0]
#         ], dtype=np.float64)

#         self.get_logger().info("Updated calibration parameters with new matrices.")

#         if calibration_file:
#             self.load_calibration(calibration_file)

#         # rectification maps
#         self.map1 = None
#         self.map2 = None
#         self.new_K = None

#         # Tag size and camera params for distance measurement
#         self.tag_size = 0.162  # 16.2 cm in meters

#         # Extract camera params from K matrix for apriltag
#         self.camera_params = [
#             self.K[0, 0],  # fx = 286.99
#             self.K[1, 1],  # fy = 290.84
#             self.K[0, 2],  # cx = 311.71
#             self.K[1, 2]   # cy = 249.93
#         ]

#         # Parameters
#         self.declare_parameter('tag_family', tag_family)
#         self.declare_parameter('image_topic', '/image_raw')
#         self.declare_parameter('output_dir', output_dir)
#         self.declare_parameter('no_gui', no_gui)
#         self.declare_parameter('calibration_file', calibration_file or '')

#         self.tag_family = self.get_parameter('tag_family').value
#         image_topic = self.get_parameter('image_topic').value
#         self.no_gui = self.get_parameter('no_gui').value
#         calib_param = self.get_parameter('calibration_file').value

#         if calib_param and not calibration_file:
#             self.load_calibration(calib_param)

#         # AprilTag detector
#         self.detector = apriltag(
#             self.tag_family,
#             threads=4,
#             maxhamming=2,
#             decimate=0.25,
#             blur=0.8,
#             refine_edges=True,
#             debug=False
#         )
#         self.get_logger().info(f"AprilTag detector initialized: {self.tag_family}")

#         # Single subscription: raw image only
#         self.sub = self.create_subscription(
#             Image, image_topic, self.image_callback, 10
#         )

#         self.display_rectified_image = None

#         if not self.no_gui:
#             cv2.namedWindow("1. Original Image", cv2.WINDOW_NORMAL)
#             cv2.namedWindow("2. Detected AprilTags (Postprocessed)", cv2.WINDOW_NORMAL)
#             cv2.resizeWindow("1. Original Image", 640, 480)
#             cv2.resizeWindow("2. Detected AprilTags (Postprocessed)", 640, 480)

#     # ------------------------------------------------------------------ #
#     # Calibration loader
#     # ------------------------------------------------------------------ #
#     def load_calibration(self, yaml_file: str):
#         """
#         Load calibration from YAML.
#         Supports:
#           - nav2 / camera_calibration style:
#               camera_matrix: {data: [...]}
#               distortion_coefficients: {data: [...]}
#               projection_matrix: {data: [...]}
#           - oST style:
#               K: [...]
#               D: [...]
#               P: [...]
#         """
#         try:
#             with open(yaml_file, 'r') as f:
#                 data = yaml.safe_load(f)

#             K = None
#             D = None
#             P = None

#             if 'camera_matrix' in data:
#                 K = np.array(data['camera_matrix']['data'],
#                              dtype=np.float64).reshape(3, 3)
#                 D = np.array(data['distortion_coefficients']['data'],
#                              dtype=np.float64).reshape(1, -1)
#                 if 'projection_matrix' in data:
#                     P = np.array(data['projection_matrix']['data'],
#                                  dtype=np.float64).reshape(3, 4)
#             elif 'K' in data:
#                 K = np.array(data['K'], dtype=np.float64).reshape(3, 3)
#                 D = np.array(data['D'], dtype=np.float64).reshape(1, -1)
#                 if 'P' in data:
#                     P = np.array(data['P'], dtype=np.float64).reshape(3, 4)

#             if K is not None:
#                 self.K = K
#             if D is not None:
#                 self.D = D
#             if P is not None:
#                 self.P = P

#             self.get_logger().info(f"Loaded calibration from {yaml_file}")
#             self.get_logger().info(f"K:\n{self.K}")
#             self.get_logger().info(f"D: {self.D}")
#             if self.P is not None:
#                 self.get_logger().info(f"P:\n{self.P}")
#         except Exception as e:
#             self.get_logger().error(f"Failed to load calibration from {yaml_file}: {e}")

#     # ------------------------------------------------------------------ #
#     # Utilities
#     # ------------------------------------------------------------------ #
#     def save_screenshots(self):
#         if self.current_original is None or self.current_detected is None:
#             self.get_logger().warn("No frames yet to save!")
#             return

#         self.screenshot_count += 1
#         ts = datetime.now().strftime("%Y%m%d_%H%M%S")

#         f1 = os.path.join(self.output_dir, f"shot_{self.screenshot_count:03d}_{ts}_orig.png")
#         f2 = os.path.join(self.output_dir, f"shot_{self.screenshot_count:03d}_{ts}_det.png")

#         cv2.imwrite(f1, self.current_original)
#         cv2.imwrite(f2, self.current_detected)

#         self.get_logger().info(f"Saved screenshots:\n  {f1}\n  {f2}")

#     # ------------------------------------------------------------------ #
#     # Main callback
#     # ------------------------------------------------------------------ #
#     def image_callback(self, msg: Image):
#         self.frame_count += 1

#         # Convert raw image
#         raw = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         self.current_original = raw.copy()

#         h, w = raw.shape[:2]

#         # Init rectification maps once
#         if self.map1 is None or self.map2 is None:
#             if self.P is not None:
#                 # Use projection matrix from calibration as new K
#                 self.new_K = self.P[:, :3]
#                 self.get_logger().info("Using P[:,:3] as new camera matrix for rectification.")
#             else:
#                 # Fallback: compute new_K with alpha=0 (crop to valid region)
#                 self.new_K, _ = cv2.getOptimalNewCameraMatrix(
#                     self.K, self.D, (w, h), 0.0, (w, h)
#                 )
#                 self.get_logger().warn("No P found in calibration, using getOptimalNewCameraMatrix(alpha=0).")

#             self.map1, self.map2 = cv2.initUndistortRectifyMap(
#                 self.K, self.D, None, self.new_K, (w, h), cv2.CV_32FC1
#             )
#             self.get_logger().info("Initialized undistortion maps.")

#         # Rectify using our K/D/P
#         rectified = cv2.remap(raw, self.map1, self.map2, cv2.INTER_LINEAR)

#         # This is the image we run AprilTag on
#         gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

#         # Optional preprocessing
#         # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         # gray = clahe.apply(gray)

#         # kernel = np.array([[-1, -1, -1],
#         #                    [-1,  9, -1],
#         #                    [-1, -1, -1]], dtype=np.float32)
#         # postprocessed = cv2.filter2D(gray, -1, kernel)

#         detections = self.detector.detect(gray)
#         detected_vis = rectified.copy()

#         for d in detections:
#             if self.first_detection:
#                 self.get_logger().info(f"Detection keys: {d.keys()}")
#                 self.get_logger().info(f"Detection structure: {d}")
#                 self.first_detection = False

#             try:
#                 # Extract corners
#                 if 'lb' in d:
#                     lb = tuple(d['lb'].astype(int))
#                     rb = tuple(d['rb'].astype(int))
#                     rt = tuple(d['rt'].astype(int))
#                     lt = tuple(d['lt'].astype(int))
#                 elif 'lb-rb-rt-lt' in d:
#                     corners = d['lb-rb-rt-lt']
#                     lb = tuple(corners[0].astype(int))
#                     rb = tuple(corners[1].astype(int))
#                     rt = tuple(corners[2].astype(int))
#                     lt = tuple(corners[3].astype(int))
#                 else:
#                     self.get_logger().error(f"Unknown corner format. Keys: {d.keys()}")
#                     continue

#                 # Calculate pose and distance
#                 pose, e0, e1 = self.detector.detection_pose(d, self.camera_params, self.tag_size)
#                 distance = np.linalg.norm(pose[:3, 3])

#                 # Draw box
#                 cv2.line(detected_vis, lt, rt, (0, 255, 0), 2)
#                 cv2.line(detected_vis, rt, rb, (0, 255, 0), 2)
#                 cv2.line(detected_vis, rb, lb, (0, 255, 0), 2)
#                 cv2.line(detected_vis, lb, lt, (0, 255, 0), 2)

#                 # Center
#                 center = tuple(d['center'].astype(int))
#                 cv2.circle(detected_vis, center, 5, (0, 0, 255), -1)

#                 # Label with distance
#                 tag_id = d['id']
#                 cv2.putText(
#                     detected_vis,
#                     f"{self.tag_family} id={tag_id} dist={distance:.2f}m",
#                     (lt[0], lt[1] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                     (0, 255, 0), 2
#                 )
#             except Exception as e:
#                 self.get_logger().error(f"Error processing detection: {e}")
#                 continue

#         # Update the detected AprilTags window to show the rectified image with detections
#         self.current_detected = detected_vis.copy()

#         if not self.no_gui:
#             cv2.imshow("1. Original Image", self.current_original)
#             cv2.imshow("2. Detected AprilTags (Postprocessed)", self.current_detected)

#             key = cv2.waitKey(1) & 0xFF
#             if key == ord('s'):
#                 self.save_screenshots()
#             if key == ord('q'):
#                 raise KeyboardInterrupt


# def main(args=None):
#     rclpy.init(args=args)

#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--tag_family", type=str, default="tagStandard41h12")
#     parser.add_argument("--output_dir", type=str, default="apriltag_screenshots")
#     parser.add_argument("--no-gui", action="store_true")
#     parser.add_argument(
#         "--calibration",
#         type=str,
#         default=None,
#         help="Path to calibration YAML (optional)"
#     )
#     parsed, _ = parser.parse_known_args()

#     node = AprilTagVisualizationNode(
#         output_dir=parsed.output_dir,
#         no_gui=parsed.no_gui,
#         tag_family=parsed.tag_family,
#         calibration_file=parsed.calibration
#     )

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
