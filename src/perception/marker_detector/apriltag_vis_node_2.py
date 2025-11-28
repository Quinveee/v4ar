#!/usr/bin/env python3

import copy
from typing import Dict, List

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

        # Parameters (same interface as original node)
        self.declare_parameter('tag_family', tag_family)
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('output_dir', output_dir)
        self.declare_parameter('use_raw', False)

        # Buffer-related parameters (unchanged API)
        self.declare_parameter('enable_buffer', False)
        self.declare_parameter('buffer_max_age_sec', 0.3)
        self.declare_parameter('buffer_distance_alpha', 0.6)

        # Multiscale-related parameters
        self.declare_parameter('enable_multiscale', False)
        self.declare_parameter('multiscale_scales', '1.0')

        # ROI / temporal smoothing parameters (new, but optional)
        self.declare_parameter('enable_roi_tracking', True)
        self.declare_parameter('roi_half_size', 80)
        self.declare_parameter('roi_upscale', 3.0)
        self.declare_parameter('max_history', 5)
        # 0.0 = no filtering, anything goes. You can raise this later.
        self.declare_parameter('decision_margin_threshold', 0.0)

        # Visualization-related parameters
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

        # === Stronger detector config for tiny / far tags ===
        # Keep maxhamming=2 on rover to avoid memory issues.
        self.detector = apriltag(
            self.tag_family,
            threads=4,
            maxhamming=2,
            decimate=0.25,    # full-res
            blur=0.5,
            refine_edges=True,
            debug=False
        )
        self.get_logger().info(
            f"AprilTag detector initialized: {self.tag_family}"
        )
        self.get_logger().info(
            "Detector params: decimate=1.0 (full-res), blur=0.0, "
            "refine_edges=True, maxhamming=2"
        )
        self.get_logger().info(
            "Image preprocessing: CLAHE (clipLimit=3.0) + Subtle Unsharp Mask"
        )
        self.get_logger().info(
            "Extended range strategy: Multiscale upscaling + ROI tracking + temporal smoothing."
        )

        # Subscription: processed image (same as original)
        self.sub = self.create_subscription(
            Image,
            '/processed_image',   # or image_topic if you want
            self.image_callback,
            10
        )

        # Publisher for marker poses (same topic as original)
        self.pub = self.create_publisher(
            MarkerPoseArray, '/detected_markers', 10
        )

        # --- Buffer config (unchanged behaviour) ---
        self.enable_buffer = self.get_parameter('enable_buffer').value
        self.buffer_max_age_sec = self.get_parameter('buffer_max_age_sec').value
        self.buffer_distance_alpha = self.get_parameter('buffer_distance_alpha').value

        # --- Multiscale config ---
        self.enable_multiscale = self.get_parameter('enable_multiscale').value
        multiscale_str = self.get_parameter('multiscale_scales').value
        self.multiscale_scales = self._parse_multiscale_scales(multiscale_str)

        # --- ROI / temporal config ---
        self.enable_roi_tracking = self.get_parameter('enable_roi_tracking').value
        self.roi_half_size = int(self.get_parameter('roi_half_size').value)
        self.roi_upscale = float(self.get_parameter('roi_upscale').value)
        self.max_history = int(self.get_parameter('max_history').value)
        self.decision_margin_threshold = float(
            self.get_parameter('decision_margin_threshold').value
        )

        # id -> list of recent detections (for smoothing)
        self.tag_histories: Dict[int, List[dict]] = {}
        # id -> {"center": (cx, cy), "distance": d} from last frame (for ROI)
        self.last_smoothed_tags: Dict[int, dict] = {}

        # --- Visualization flags ---
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
            try:
                cv2.namedWindow("AprilTag Original", cv2.WINDOW_NORMAL)
                cv2.namedWindow("AprilTag Detected", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("AprilTag Original", 640, 480)
                cv2.resizeWindow("AprilTag Detected", 640, 480)
                self.get_logger().info("OpenCV GUI visualization ENABLED.")
            except cv2.error as e:
                self.get_logger().warn(f"OpenCV GUI error when creating windows: {e}")
                self.enable_gui = False
        else:
            self.get_logger().info("OpenCV GUI visualization DISABLED.")

        # id -> {"marker": MarkerPose, "last_seen": Time} for buffer
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

        if self.enable_roi_tracking:
            self.get_logger().info(
                f"ROI tracking ENABLED: half_size={self.roi_half_size}, "
                f"upscale={self.roi_upscale}"
            )
        else:
            self.get_logger().info("ROI tracking DISABLED.")

        if self.publish_debug_image:
            self.get_logger().info("Debug detected image topic PUBLISHED on /apriltag/detected_image.")
        else:
            self.get_logger().info("Debug detected image topic DISABLED.")

        # Register parameter callback so CLI params update GUI / multiscale without restart
        try:
            self.add_on_set_parameters_callback(self._on_set_parameters)
        except Exception:
            pass

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

        Guarantees at least [1.0] if parsing fails.
        """
        try:
            parts = [p.strip() for p in s.split(',') if p.strip()]
            scales = [float(p) for p in parts]
            if not scales:
                return [1.0]
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
            if p.name in ('enable_gui', 'publish_debug_image'):
                changed = True
            elif p.name == 'multiscale_scales':
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
            elif p.name == 'enable_roi_tracking':
                try:
                    self.enable_roi_tracking = bool(p.value)
                except Exception:
                    pass
                changed = True

        if changed:
            try:
                self._apply_visualization_settings()
            except Exception as e:
                self.get_logger().warn(f"Error applying visualization settings: {e}")

        result = SetParametersResult()
        result.successful = True
        result.reason = ''
        return result

    # ---------- detection helpers (from apriltag_vis_node_2) ---------- #

    def _detect_multiscale(self, gray):
        """
        Run AprilTag detection on multiple image scales and merge results.
        """
        h, w = gray.shape[:2]
        all_dets = []

        for scale in self.multiscale_scales:
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w < 32 or new_h < 32:
                continue

            if abs(scale - 1.0) < 1e-3:
                img_scaled = gray
            else:
                img_scaled = cv2.resize(
                    gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC
                )

            dets = self.detector.detect(img_scaled)
            for d in dets:
                d2 = dict(d)  # copy
                for key in ['lb', 'rb', 'rt', 'lt', 'center']:
                    if key in d2:
                        d2[key] = d2[key] / scale
                if 'lb-rb-rt-lt' in d2:
                    d2['lb-rb-rt-lt'] = d2['lb-rb-rt-lt'] / scale
                d2['_scale'] = scale
                all_dets.append(d2)

        return self._deduplicate_detections(all_dets)

    def _detect_single_with_fallback(self, gray):
        """Single-scale detect with simple downscale fallback."""
        detections = self.detector.detect(gray)
        if detections:
            return detections

        if gray.shape[0] <= 64 or gray.shape[1] <= 64:
            return []

        all_dets = []
        for scale_factor in [0.7, 0.5]:
            gray_scaled = cv2.resize(
                gray, None, fx=scale_factor, fy=scale_factor,
                interpolation=cv2.INTER_LINEAR
            )
            dets = self.detector.detect(gray_scaled)
            for d in dets:
                d2 = dict(d)
                inv_scale = 1.0 / scale_factor
                for key in ['lb', 'rb', 'rt', 'lt', 'center']:
                    if key in d2:
                        d2[key] = d2[key] * inv_scale
                if 'lb-rb-rt-lt' in d2:
                    d2['lb-rb-rt-lt'] = d2['lb-rb-rt-lt'] * inv_scale
                all_dets.append(d2)

        return self._deduplicate_detections(all_dets)

    def _detect_in_rois(self, gray):
        """
        ROI-first detection: for each tracked tag from previous frames,
        crop a window around its center, upscale, and run detection there.
        """
        if not self.enable_roi_tracking or not self.last_smoothed_tags:
            return []

        h, w = gray.shape[:2]
        all_dets = []

        for tag_id, info in self.last_smoothed_tags.items():
            cx, cy = info['center']
            cx = int(cx)
            cy = int(cy)

            half = self.roi_half_size
            x0 = max(0, cx - half)
            x1 = min(w, cx + half)
            y0 = max(0, cy - half)
            y1 = min(h, cy + half)

            if x1 - x0 < 16 or y1 - y0 < 16:
                continue

            roi = gray[y0:y1, x0:x1]
            roi_big = cv2.resize(
                roi, None,
                fx=self.roi_upscale,
                fy=self.roi_upscale,
                interpolation=cv2.INTER_CUBIC
            )

            dets = self.detector.detect(roi_big)
            for d in dets:
                d2 = dict(d)
                # map back to original image coordinates
                for key in ['lb', 'rb', 'rt', 'lt', 'center']:
                    if key in d2:
                        d2[key] = d2[key] / self.roi_upscale
                        d2[key][0] += x0
                        d2[key][1] += y0
                if 'lb-rb-rt-lt' in d2:
                    corners = d2['lb-rb-rt-lt'] / self.roi_upscale
                    corners[:, 0] += x0
                    corners[:, 1] += y0
                    d2['lb-rb-rt-lt'] = corners

                d2['_source'] = 'roi'
                all_dets.append(d2)

        return self._deduplicate_detections(all_dets)

    def _deduplicate_detections(self, all_dets):
        """Merge detections by id, preferring larger area & higher margin."""
        if not all_dets:
            return []

        dedup = {}
        for d in all_dets:
            tag_id = d['id']
            if 'lb-rb-rt-lt' in d:
                corners = d['lb-rb-rt-lt']
            else:
                corners = np.array([d['lb'], d['rb'], d['rt'], d['lt']])
            xs = corners[:, 0]
            ys = corners[:, 1]
            area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
            # Your apriltag lib uses 'margin', not 'decision_margin'
            dec_margin = float(d.get('decision_margin', d.get('margin', 0.0)))
            score = area + 5.0 * dec_margin  # heuristic

            if tag_id not in dedup or score > dedup[tag_id]['score']:
                dedup[tag_id] = {'det': d, 'score': score}

        return [v['det'] for v in dedup.values()]

    def _smooth_detections(self, raw_infos):
        """
        Temporal smoothing / majority vote for center & distance.
        Only keep tags with sufficient average decision_margin.
        """
        smoothed = []

        for info in raw_infos:
            tag_id = info['id']
            history = self.tag_histories.get(tag_id, [])
            history.append(info)
            if len(history) > self.max_history:
                history = history[-self.max_history:]
            self.tag_histories[tag_id] = history

            xs = [h['center'][0] for h in history]
            ys = [h['center'][1] for h in history]
            dists = [h['distance'] for h in history if h['distance'] is not None]
            margins = [h['decision_margin'] for h in history if h['decision_margin'] is not None]

            med_center = (float(np.median(xs)), float(np.median(ys)))
            med_dist = float(np.median(dists)) if dists else info['distance']
            avg_margin = float(np.mean(margins)) if margins else 0.0

            if avg_margin < self.decision_margin_threshold:
                # reject consistently low-quality detections
                continue

            info_sm = info.copy()
            info_sm['center'] = med_center
            info_sm['distance'] = med_dist
            info_sm['avg_decision_margin'] = avg_margin
            smoothed.append(info_sm)

        # update last_smoothed_tags for ROI in next frame
        self.last_smoothed_tags = {
            i['id']: {
                'center': i['center'],
                'distance': i['distance']
            } for i in smoothed
        }

        return smoothed

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

        # Preprocessing to enhance small markers at distance
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.addWeighted(gray, 1.2, blurred, -0.2, 0)
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # --- ROI-first detection, then full-frame fallback --- #
        roi_dets = self._detect_in_rois(gray) if self.enable_roi_tracking else []

        if self.enable_multiscale:
            full_dets = self._detect_multiscale(gray)
        else:
            full_dets = self._detect_single_with_fallback(gray)

        detections = self._deduplicate_detections(roi_dets + full_dets)

        detected_vis = rectified.copy()
        marker_array_msg = MarkerPoseArray()
        marker_array_msg.header = msg.header

        raw_infos = []

        # First pass: build raw_infos for temporal smoothing
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

                center = tuple(d['center'].astype(int))
                tag_id = int(d['id'])
                decision_margin = float(d.get('decision_margin', d.get('margin', 0.0)))

                # Optional PnP just for distance estimate to smooth ROI (pose recomputed later)
                distance = None
                try:
                    img_pts = np.array([lt, rt, rb, lb], dtype=np.float32)
                    camera_matrix = self.new_K
                    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

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

                    t = tvec.flatten()
                    distance = float(np.linalg.norm(t))

                except Exception as e:
                    self.get_logger().warn(f"Pose (for distance) failed (PnP): {e}")

                raw_infos.append({
                    'id': tag_id,
                    'lt': lt, 'rt': rt, 'rb': rb, 'lb': lb,
                    'center': center,
                    'distance': distance,
                    'decision_margin': decision_margin
                })

            except Exception as e:
                self.get_logger().error(f"Error processing detection: {e}")
                continue

        # Temporal smoothing / majority vote (center & distance)
        smoothed_infos = self._smooth_detections(raw_infos)

        # Second pass: build MarkerPose messages from smoothed infos
        for info in smoothed_infos:
            lt = info['lt']
            rt = info['rt']
            rb = info['rb']
            lb = info['lb']
            cx, cy = info['center']
            center = (int(cx), int(cy))
            tag_id = info['id']
            avg_margin = info.get('avg_decision_margin', 0.0)

            pose_msg = Pose()
            pose_msg.orientation.w = 1.0
            distance_value = -1.0

            # Full PnP pose estimation for publishing pose
            try:
                img_pts = np.array([lt, rt, rb, lb], dtype=np.float32)
                camera_matrix = self.new_K
                dist_coeffs = np.zeros((4, 1), dtype=np.float32)

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

            except Exception as e:
                self.get_logger().warn(f"Pose failed (PnP): {e}")

            # Draw on visualization image
            cv2.line(detected_vis, lt, rt, (0, 255, 0), 2)
            cv2.line(detected_vis, rt, rb, (0, 255, 0), 2)
            cv2.line(detected_vis, rb, lb, (0, 255, 0), 2)
            cv2.line(detected_vis, lb, lt, (0, 255, 0), 2)

            cv2.circle(detected_vis, center, 5, (0, 0, 255), -1)

            cv2.putText(
                detected_vis,
                f"{self.tag_family} id={tag_id}",
                (lt[0] - 70, lt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2
            )

            if distance_value >= 0.0:
                cv2.putText(
                    detected_vis,
                    f"Dist={distance_value:.2f}m M={avg_margin:.1f}",
                    (lt[0] - 55, lt[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2
                )

            # Build MarkerPose message (same format as original node)
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

        # Visualization (OpenCV windows)
        if self.enable_gui:
            try:
                cv2.imshow("AprilTag Original", self.current_original)
                cv2.imshow("AprilTag Detected", self.current_detected)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    self.save_screenshots()
            except cv2.error as e:
                self.get_logger().warn(f"OpenCV GUI error: {e}")

        # Publish debug image if requested
        if self.publish_debug_image and self.debug_pub is not None:
            dbg_msg = self.bridge.cv2_to_imgmsg(self.current_detected, encoding='bgr8')
            dbg_msg.header = msg.header
            self.debug_pub.publish(dbg_msg)

        # ----------- Buffering logic (unchanged behaviour) -----------
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
                # pose stays current frame
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
    # CLI flag to enable buffer
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
        help="Comma-separated list of scales, e.g. '2.0,1.5,1.0'."
    )
    # Visualization flags
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

    # Push CLI flags into ROS params so the node reacts via param callback
    params = []
    if parsed.use_raw:
        params.append(Parameter('use_raw', Parameter.Type.BOOL, True))
    if parsed.enable_buffer:
        params.append(Parameter('enable_buffer', Parameter.Type.BOOL, True))
    if parsed.enable_multiscale:
        params.append(Parameter('enable_multiscale', Parameter.Type.BOOL, True))
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
