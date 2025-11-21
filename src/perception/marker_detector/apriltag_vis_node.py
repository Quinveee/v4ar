#!/usr/bin/env python3

import copy
from typing import Dict

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime

# your rover's apriltag library
from apriltag import apriltag
from perception_msgs.msg import MarkerPoseArray, MarkerPose
from rcl_interfaces.msg import SetParametersResult


def rotation_matrix_to_quaternion(R: np.ndarray):
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
                 tag_family: str = 'tagStandard41h12'):

        super().__init__('apriltag_visualization')

        self.bridge = CvBridge()

        self.screenshot_count = 0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.current_original = None
        self.current_detected = None
        self.frame_count = 0
        self.first_detection = True
        self.tag_size = 0.162  # 16.2 cm in meters

        # --- Default calibration parameters (K, D, P) ---
        self.K = np.array([
            [286.9896545092208, 0.0, 311.7114840273407],
            [0.0, 290.8395992360502, 249.9287049631703],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.D = np.array([[
            -0.17995587161461585, 0.020688274841999105,
            -0.005297672531455161, 0.003378882156848116, 0.0
        ]], dtype=np.float64)

        self.P = np.array([
            [190.74984649646845, 0.0, 318.0141593176815, 0.0],
            [0.0, 247.35103262891005, 248.37293105876694, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float64)

        self.get_logger().info("Using hardcoded default calibration parameters.")

        # Parameters
        self.declare_parameter('tag_family', tag_family)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('output_dir', output_dir)
        self.declare_parameter('use_raw', False)

        # NEW: buffer-related parameters
        self.declare_parameter('enable_buffer', False)
        self.declare_parameter('buffer_max_age_sec', 0.3)
        self.declare_parameter('buffer_distance_alpha', 0.6)

        # NEW: multiscale-related parameters
        self.declare_parameter('enable_multiscale', False)
        # Comma-separated list of scales, e.g. "0.7,1.0,1.4"
        self.declare_parameter('multiscale_scales', '1.0')

        # NEW: visualization-related parameters
        self.declare_parameter('enable_gui', False)
        self.declare_parameter('publish_debug_image', False)

        use_raw = self.get_parameter('use_raw').value

        if use_raw:
            # Experimental "raw" calibration
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

            self.get_logger().info("Using hardcoded experimental RAW calibration parameters.")
        else:
            self.get_logger().info("Using hardcoded default calibration parameters (non-RAW).")

        self.tag_family = self.get_parameter('tag_family').value
        image_topic = self.get_parameter('image_topic').value

        # new_K used for PnP
        self.new_K = None

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
            f"AprilTag detector initialized: {self.tag_family}"
        )

        # Subscription: processed image
        self.sub = self.create_subscription(
            Image,
            '/processed_image',   # or image_topic if you want
            self.image_callback,
            10
        )

        # Publisher for marker poses
        self.pub = self.create_publisher(
            MarkerPoseArray, '/detected_markers', 10
        )

        # --- NEW: internal state for buffering ---
        self.enable_buffer = self.get_parameter('enable_buffer').value
        self.buffer_max_age_sec = self.get_parameter('buffer_max_age_sec').value
        self.buffer_distance_alpha = self.get_parameter('buffer_distance_alpha').value

        # --- NEW: multiscale config ---
        self.enable_multiscale = self.get_parameter('enable_multiscale').value
        multiscale_str = self.get_parameter('multiscale_scales').value
        self.multiscale_scales = self._parse_multiscale_scales(multiscale_str)

        # --- NEW: visualization flags ---
        self.enable_gui = self.get_parameter('enable_gui').value
        self.publish_debug_image = self.get_parameter('publish_debug_image').value

        # Optional debug image publisher
        self.debug_pub = None
        if self.publish_debug_image:
            self.debug_pub = self.create_publisher(
                Image,
                '/apriltag/detected_image',
                10
            )

        # Setup GUI windows if requested
        if self.enable_gui:
            cv2.namedWindow("AprilTag Original", cv2.WINDOW_NORMAL)
            cv2.namedWindow("AprilTag Detected", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AprilTag Original", 640, 480)
            cv2.resizeWindow("AprilTag Detected", 640, 480)

        # id -> {"marker": MarkerPose, "last_seen": Time}
        self.marker_state: Dict[int, Dict[str, object]] = {}

        if self.enable_buffer:
            self.get_logger().info(
                f"Marker buffering ENABLED: max_age_sec={self.buffer_max_age_sec:.2f}, "
                f"distance_alpha={self.buffer_distance_alpha:.2f}"
            )
        else:
            self.get_logger().info("Marker buffering DISABLED (publishing raw detections).")

        if self.enable_multiscale:
            self.get_logger().info(
                f"Multiscale detection ENABLED. Scales={self.multiscale_scales}"
            )
        else:
            self.get_logger().info("Multiscale detection DISABLED (using single scale).")

        if self.enable_gui:
            self.get_logger().info("OpenCV GUI visualization ENABLED.")
        else:
            self.get_logger().info("OpenCV GUI visualization DISABLED.")

        if self.publish_debug_image:
            self.get_logger().info("Debug detected image topic PUBLISHED on /apriltag/detected_image.")
        else:
            self.get_logger().info("Debug detected image topic DISABLED.")

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def save_screenshots(self):
        """Available if you ever want to dump frames to disk."""
        if self.current_original is None or self.current_detected is None:
            self.get_logger().warn("No frames yet to save!")
            return

        self.screenshot_count += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        f1 = os.path.join(
            self.output_dir, f"shot_{self.screenshot_count:03d}_{ts}_orig.png"
        )
        f2 = os.path.join(
            self.output_dir, f"shot_{self.screenshot_count:03d}_{ts}_det.png"
        )

        cv2.imwrite(f1, self.current_original)
        cv2.imwrite(f2, self.current_detected)

        self.get_logger().info(f"Saved screenshots:\n  {f1}\n  {f2}")

    def _parse_multiscale_scales(self, s: str):
        """
        Parse a comma-separated list of scales into a list of floats.

        Example:
            "0.7,1.0,1.4" -> [0.7, 1.0, 1.4]

        Guarantees at least [1.0] if parsing fails.
        """
        try:
            parts = [p.strip() for p in s.split(',') if p.strip()]
            scales = [float(p) for p in parts]
            if not scales:
                return [1.0]
            # Remove any non-positive or absurd scales
            scales = [sc for sc in scales if sc > 0.05]
            return scales or [1.0]
        except Exception as e:
            self.get_logger().warn(f"Failed to parse multiscale_scales '{s}': {e}")
            return [1.0]

    def _apply_visualization_settings(self):
        """
        Create/destroy OpenCV windows and debug publisher according to
        the current parameters. Safe to call multiple times.
        """
        # Read current parameter values (use get_parameter to reflect runtime changes)
        try:
            enable_gui = self.get_parameter('enable_gui').value
        except Exception:
            enable_gui = bool(getattr(self, 'enable_gui', False))

        try:
            publish_debug = self.get_parameter('publish_debug_image').value
        except Exception:
            publish_debug = bool(getattr(self, 'publish_debug_image', False))

        # Handle debug image publisher
        if publish_debug and self.debug_pub is None:
            self.debug_pub = self.create_publisher(Image, '/apriltag/detected_image', 10)
            self.get_logger().info("Debug detected image topic PUBLISHED on /apriltag/detected_image.")
        elif not publish_debug and self.debug_pub is not None:
            try:
                self.destroy_publisher(self.debug_pub)
            except Exception:
                pass
            self.debug_pub = None
            self.get_logger().info("Debug detected image topic DISABLED.")

        # Handle GUI windows
        gui_setup = getattr(self, '_gui_setup', False)
        if enable_gui and not gui_setup:
            try:
                cv2.namedWindow("AprilTag Original", cv2.WINDOW_NORMAL)
                cv2.namedWindow("AprilTag Detected", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("AprilTag Original", 640, 480)
                cv2.resizeWindow("AprilTag Detected", 640, 480)
                self._gui_setup = True
                self.get_logger().info("OpenCV GUI visualization ENABLED.")
            except cv2.error as e:
                self.get_logger().warn(f"OpenCV GUI error when enabling windows: {e}")
                self._gui_setup = False
        elif not enable_gui and gui_setup:
            try:
                cv2.destroyWindow("AprilTag Original")
                cv2.destroyWindow("AprilTag Detected")
            except cv2.error:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
            self._gui_setup = False
            self.get_logger().info("OpenCV GUI visualization DISABLED.")

        # Update cached flags
        self.enable_gui = enable_gui
        self.publish_debug_image = publish_debug

    def _on_set_parameters(self, params):
        """rclpy parameter callback to react to runtime parameter changes."""
        changed = False
        for p in params:
            if p.name == 'enable_gui':
                changed = True
            elif p.name == 'publish_debug_image':
                changed = True
            elif p.name == 'multiscale_scales':
                # reparse scales if changed
                try:
                    self.multiscale_scales = self._parse_multiscale_scales(p.value)
                except Exception:
                    pass
                changed = True
            elif p.name == 'enable_multiscale':
                try:
                    self.enable_multiscale = bool(p.value)
                except Exception:
                    pass
                changed = True

        if changed:
            # Apply visualization state changes (create/destroy windows/publishers)
            try:
                self._apply_visualization_settings()
            except Exception as e:
                self.get_logger().warn(f"Error applying visualization settings: {e}")

        # Always return successful so parameter setting isn't blocked
        result = SetParametersResult()
        result.successful = True
        result.reason = ''
        return result

    def _detect_multiscale(self, gray):
        """
        Run AprilTag detection on multiple image scales and merge results.

        For each scale:
          - resize the gray image
          - run detector
          - rescale all corner / center coordinates back to original image coordinates
        Then:
          - deduplicate by tag id, keeping the detection with largest image area.
        """
        h, w = gray.shape[:2]
        all_dets = []

        for scale in self.multiscale_scales:
            # Avoid ridiculous sizes
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w < 32 or new_h < 32:
                continue

            if abs(scale - 1.0) < 1e-3:
                img_scaled = gray
            else:
                img_scaled = cv2.resize(
                    gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )

            dets = self.detector.detect(img_scaled)
            for d in dets:
                # Shallow copy so we don't mutate original dict from detector
                d2 = dict(d)

                # Rescale all pixel coordinates back to original image space
                for key in ['lb', 'rb', 'rt', 'lt', 'center']:
                    if key in d2:
                        d2[key] = d2[key] / scale

                if 'lb-rb-rt-lt' in d2:
                    d2['lb-rb-rt-lt'] = d2['lb-rb-rt-lt'] / scale

                d2['_scale'] = scale  # optional debug info
                all_dets.append(d2)

        if not all_dets:
            return []

        # Deduplicate by tag id: keep the one with largest apparent area
        dedup = {}
        for d in all_dets:
            tag_id = d['id']

            # Estimate area from corners
            if 'lb-rb-rt-lt' in d:
                corners = d['lb-rb-rt-lt']
            else:
                corners = np.array([d['lb'], d['rb'], d['rt'], d['lt']])
            xs = corners[:, 0]
            ys = corners[:, 1]
            area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))

            if tag_id not in dedup or area > dedup[tag_id]['area']:
                dedup[tag_id] = {'det': d, 'area': area}

        return [v['det'] for v in dedup.values()]

    # ------------------------------------------------------------------ #
    # Main callback
    # ------------------------------------------------------------------ #
    def image_callback(self, msg: Image):
        self.frame_count += 1

        # Convert incoming image
        rectified = self.bridge.imgmsg_to_cv2(msg)
        self.current_original = rectified.copy()

        if self.new_K is None and self.P is not None:
            self.new_K = self.P[:, :3]

        # Gray for detection
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

        # Optional preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]], dtype=np.float32)
        postprocessed = cv2.filter2D(gray, -1, kernel)

        # --- Detection: single scale or multiscale ---
        if self.enable_multiscale:
            detections = self._detect_multiscale(gray)
        else:
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
                        f"Unknown corner format. Keys: {d.keys()}"
                    )
                    continue

                # Draw box (only affects detected_vis, for screenshots / debug)
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
                    # Image points: lt, rt, rb, lb
                    img_pts = np.array([lt, rt, rb, lb], dtype=np.float32)

                    # Camera intrinsics
                    camera_matrix = self.new_K
                    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

                    # 3D tag model
                    s = self.tag_size / 2.0
                    obj_pts = np.array([
                        [-s,  s, 0],
                        [ s,  s, 0],
                        [ s, -s, 0],
                        [-s, -s, 0]
                    ], dtype=np.float32)

                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, camera_matrix, dist_coeffs
                    )

                    if not success:
                        raise RuntimeError("solvePnP returned False")

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

                    # Optional text overlay in detected_vis only
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

        # Store last annotated image
        self.current_detected = detected_vis.copy()

        # --- NEW: visualization (OpenCV windows) ---
        if self.enable_gui:
            try:
                cv2.imshow("AprilTag Original", self.current_original)
                cv2.imshow("AprilTag Detected", self.current_detected)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    self.save_screenshots()
                # if key == ord('q'):
                #     rclpy.shutdown()  # optional, usually just Ctrl+C
            except cv2.error as e:
                self.get_logger().warn(f"OpenCV GUI error: {e}")

        # --- NEW: publish debug image if requested ---
        if self.publish_debug_image and self.debug_pub is not None:
            dbg_msg = self.bridge.cv2_to_imgmsg(self.current_detected, encoding='bgr8')
            dbg_msg.header = msg.header
            self.debug_pub.publish(dbg_msg)

        # ----------- inline buffering logic -----------
        if not self.enable_buffer:
            # Old behaviour: publish raw detections
            self.pub.publish(marker_array_msg)
            return

        # With buffering enabled, update internal state and publish "stable" markers
        try:
            now = Time.from_msg(marker_array_msg.header.stamp)
        except Exception:
            # Fallback to node clock if header.stamp is empty
            now = self.get_clock().now()

        # 1. Update markers we actually see in this frame
        for m in marker_array_msg.markers:
            if m.id in self.marker_state:
                prev_marker: MarkerPose = self.marker_state[m.id]["marker"]
                # low-pass filter on distance
                filtered_distance = (
                    self.buffer_distance_alpha * m.distance
                    + (1.0 - self.buffer_distance_alpha) * prev_marker.distance
                )
                new_marker = copy.deepcopy(m)
                new_marker.distance = float(filtered_distance)
                # we keep the current pose; only distance is smoothed
            else:
                # First time we see this marker
                new_marker = copy.deepcopy(m)

            self.marker_state[m.id] = {
                "marker": new_marker,
                "last_seen": now,
            }

        # 2. Build output array with all markers that are still "fresh" enough
        out = MarkerPoseArray()
        out.header = marker_array_msg.header

        ids_to_delete = []
        for marker_id, state in self.marker_state.items():
            age = (now - state["last_seen"]).nanoseconds / 1e9

            if age <= self.buffer_max_age_sec:
                # Still fresh enough → keep
                out.markers.append(state["marker"])
            else:
                # Too old → forget this marker
                ids_to_delete.append(marker_id)

        for marker_id in ids_to_delete:
            del self.marker_state[marker_id]

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag_family", type=str, default="tagStandard41h12")
    parser.add_argument("--output_dir", type=str,
                        default="apriltag_screenshots")
    parser.add_argument(
        "--use_raw",
        action="store_true",
        help="Use experimental hardcoded RAW calibration instead of the default one."
    )
    # NEW: CLI flag to enable buffer easily
    parser.add_argument(
        "--enable_buffer",
        action="store_true",
        help="Enable temporal buffering of detected markers."
    )
    parser.add_argument(
        "--enable_multiscale",
        action="store_true",
        help="Enable multiscale AprilTag detection."
    )
    parser.add_argument(
        "--multiscale_scales",
        type=str,
        default="1.0",
        help="Comma-separated list of scales, e.g. '0.7,1.0,1.4'."
    )
    # NEW: visualization flags
    parser.add_argument(
        "--enable_gui",
        action="store_true",
        help="Show OpenCV windows (requires X/xlaunch)."
    )
    parser.add_argument(
        "--publish_debug_image",
        action="store_true",
        help="Publish annotated detected image on /apriltag/detected_image."
    )

    parsed, _ = parser.parse_known_args()

    node = AprilTagVisualizationNode(
        output_dir=parsed.output_dir,
        tag_family=parsed.tag_family,
    )

    params = []
    if parsed.use_raw:
        params.append(Parameter('use_raw', Parameter.Type.BOOL, True))
    if parsed.enable_buffer:
        params.append(Parameter('enable_buffer', Parameter.Type.BOOL, True))
    if parsed.enable_multiscale:
        params.append(Parameter('enable_multiscale', Parameter.Type.BOOL, True))
    # Always push the string, so it matches the param declaration
    params.append(Parameter('multiscale_scales', Parameter.Type.STRING, parsed.multiscale_scales))
    if parsed.enable_gui:
        params.append(Parameter('enable_gui', Parameter.Type.BOOL, True))
    if parsed.publish_debug_image:
        params.append(Parameter('publish_debug_image', Parameter.Type.BOOL, True))

    if params:
        node.set_parameters(params)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
