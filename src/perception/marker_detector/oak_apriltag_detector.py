"""
AprilTag Detection on OAK camera
Single-timer, multi-scale detector with optional simple mode (no multiscale / ROI / buffer).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import copy
from typing import Dict
from apriltag import apriltag
from perception_msgs.msg import MarkerPoseArray, MarkerPose


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
    def __init__(
        self,
        tag_family: str = 'tagStandard41h12',
        tag_size: float = 0.162,
        # Single detection FPS (used for all scales)
        detection_fps: float = 5.0,
        enable_buffer: bool = True,
        buffer_max_age_sec: float = 2.0,
        buffer_alpha: float = 0.7,
        simple_detection: bool = False,
        simple_scale_width: int = 320,
    ):
        super().__init__('apriltag_visualization')

        self.bridge = CvBridge()
        self.tag_size = tag_size

        # Camera intrinsics / rectification
        self.K = None
        self.D = None
        self.mapx = None
        self.mapy = None
        self.camera_info_received = False

        # Detection scheduling
        self.detection_fps = detection_fps

        # Simple temporal buffer on top of raw detections
        self.enable_buffer = enable_buffer
        self.buffer_max_age_sec = buffer_max_age_sec
        self.buffer_alpha = buffer_alpha
        self.marker_buffer: Dict[int, dict] = {}

        # Per-cycle detections (id → detection dict)
        self.current_detections: Dict[int, dict] = {}

        # ROI-based detection config (uses buffered markers)
        self.enable_roi_tracking = True
        self.roi_half_size = 80      # half-size in pixels at full resolution
        self.roi_upscale = 2.0       # how much to blow up each ROI before detection

        # Simple detection mode (single pass, no multiscale/ROI/buffer)
        self.simple_detection = simple_detection
        self.simple_scale_width = simple_scale_width

        # OAK topics
        self.color_sub = self.create_subscription(
            Image, '/oak/rgb/image_rect', self.color_callback, 10
        )
        self.color_info_sub = self.create_subscription(
            CameraInfo, '/oak/rgb/camera_info', self.camera_info_callback, 1
        )

        self.pub = self.create_publisher(
            MarkerPoseArray, '/oak/detected_markers', 10
        )

        # Detection timer (single pipeline, multi-scale or simple inside)
        period = 1.0 / max(self.detection_fps, 0.1)
        self.detect_timer = self.create_timer(period, self.process_frame)

        # Parameters (so you can override from ROS2)
        self.declare_parameter('tag_family', tag_family)
        self.declare_parameter('tag_size', tag_size)
        self.declare_parameter('enable_buffer', enable_buffer)
        self.declare_parameter('buffer_max_age_sec', buffer_max_age_sec)
        self.declare_parameter('buffer_alpha', buffer_alpha)
        self.declare_parameter('simple_detection', simple_detection)
        self.declare_parameter('simple_scale_width', simple_scale_width)

        self.tag_family = self.get_parameter('tag_family').value
        self.tag_size = float(self.get_parameter('tag_size').value)
        self.enable_buffer = bool(self.get_parameter('enable_buffer').value)
        self.buffer_max_age_sec = float(
            self.get_parameter('buffer_max_age_sec').value
        )
        self.buffer_alpha = float(self.get_parameter('buffer_alpha').value)
        self.simple_detection = bool(
            self.get_parameter('simple_detection').value
        )
        self.simple_scale_width = int(
            self.get_parameter('simple_scale_width').value
        )

        # AprilTag detector config
        self.detector = apriltag(
            self.tag_family,
            threads=4,
            maxhamming=2,
            decimate=1.0,
            blur=0.0,
            refine_edges=True,
            debug=False,
        )

        self.get_logger().info(f"AprilTag detector initialized: {self.tag_family}")
        self.get_logger().info(f"Tag size: {self.tag_size} m")

        if self.simple_detection:
            # Simple mode: override the fancier stuff
            self.enable_buffer = False
            self.enable_roi_tracking = False
            self.get_logger().info(
                f"Simple detection mode ENABLED "
                f"(width={self.simple_scale_width if self.simple_scale_width > 0 else 'full-res'}). "
                "Multi-scale, ROI tracking and temporal buffer are disabled."
            )
        else:
            self.get_logger().info(
                f"Detection timer: {self.detection_fps:.1f} Hz, "
                "scales=[240, 320, 400]"
            )
            if self.enable_buffer:
                self.get_logger().info(
                    f"Buffering ENABLED (age={self.buffer_max_age_sec:.2f}s, "
                    f"alpha={self.buffer_alpha:.2f})"
                )
            else:
                self.get_logger().info(
                    "Buffering DISABLED (publishing raw detections)."
                )

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def color_callback(self, msg: Image):
        # Just store the latest image; detection happens in the timer.
        self.current_color_msg = msg

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info_received:
            return

        K = np.array(msg.k, dtype=np.float64).reshape((3, 3))
        D = np.array(msg.d, dtype=np.float64)
        P_matrix = np.array(msg.p, dtype=np.float64).reshape((3, 4))
        K_new = P_matrix[:, 0:3]
        h, w = msg.height, msg.width

        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            K, D, None, K_new, (w, h), cv2.CV_32FC1
        )

        self.K = K_new
        self.D = D
        self.camera_info_received = True
        self.get_logger().info(f"Camera calibration loaded: {w}x{h}")
        self.destroy_subscription(self.color_info_sub)

    # ------------------------------------------------------------------ #
    # Detection scheduling
    # ------------------------------------------------------------------ #

    def _check_data_ready(self):
        return (
            getattr(self, 'current_color_msg', None) is not None
            and self.camera_info_received
        )

    def process_frame(self):
        """
        Main detection entrypoint, called at a fixed rate.
        Either runs:
        - simple single-scale detection, or
        - multi-scale detection (240/320/400px width).
        """
        if not self._check_data_ready():
            return

        # Reset per-cycle detections
        self.current_detections.clear()

        if self.simple_detection:
            # Simple one-pass detection
            self._simple_detect_once()
        else:
            # Try several image widths; keep the highest-resolution
            # successful detection per tag id.
            for scale_width in (240, 320, 400):
                detections = self._detect_at_resolution(scale_width)
                for d in detections:
                    tag_id = d['id']
                    should_update = False

                    if tag_id not in self.current_detections:
                        should_update = True
                    else:
                        existing_res = self.current_detections[tag_id].get(
                            'resolution', 0
                        )
                        if scale_width > existing_res:
                            should_update = True

                    if should_update:
                        self.current_detections[tag_id] = {
                            'detection': d,
                            # Scale factor from full-res → this downscaled width
                            'scale_factor': scale_width / float(
                                self.current_color_msg.width
                            ),
                            'resolution': scale_width,
                            'source': f'scale_{scale_width}',
                        }

        # Compute PnP poses and publish
        self._calculate_and_publish_poses()

    # ------------------------------------------------------------------ #
    # ROI-based detection helpers
    # ------------------------------------------------------------------ #

    def _detect_in_rois(self, gray, target_width, full_w, full_h):
        """
        ROI-first detection: for each buffered marker, crop a window around its
        last-seen center, upscale, and run detection there.
        """
        if not self.enable_roi_tracking or not self.marker_buffer:
            return []

        h, w = gray.shape[:2]
        all_dets = []

        # scale factor from full-res to this downscaled image
        sf = float(target_width) / float(full_w)

        # ROI half-size in this scale
        half = max(8, int(self.roi_half_size * sf))

        for marker_id, data in self.marker_buffer.items():
            marker = data['marker']
            cx_full = marker.center_x
            cy_full = marker.center_y

            # map full-res center to current gray resolution
            cx = int(cx_full * sf)
            cy = int(cy_full * sf)

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
                interpolation=cv2.INTER_CUBIC,
            )

            dets = self.detector.detect(roi_big)
            for d in dets:
                d2 = dict(d)
                # map back to current downscaled image coordinates
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

        return all_dets

    def _deduplicate_detections(self, all_dets):
        """Merge detections by id, preferring larger area & higher decision_margin."""
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
            dec_margin = float(d.get('decision_margin', d.get('margin', 0.0)))
            score = area + 5.0 * dec_margin

            if tag_id not in dedup or score > dedup[tag_id]['score']:
                dedup[tag_id] = {'det': d, 'score': score}

        return [v['det'] for v in dedup.values()]

    def _detect_at_resolution(self, target_width):
        """
        Run AprilTag detection at the given image width, using ROI-first +
        full-frame detection on the latest OAK frame.
        """
        try:
            color_image = self.bridge.imgmsg_to_cv2(
                self.current_color_msg, desired_encoding='bgr8'
            )
            color_image_rectified = cv2.remap(
                color_image, self.mapx, self.mapy, cv2.INTER_LINEAR
            )

            full_h, full_w = color_image_rectified.shape[:2]
            scale_factor = target_width / float(full_w)
            downscaled = cv2.resize(
                color_image_rectified,
                (target_width, int(full_h * scale_factor)),
                interpolation=cv2.INTER_LINEAR,
            )

            gray = cv2.cvtColor(downscaled, cv2.COLOR_BGR2GRAY)

            # Contrast-limited adaptive hist eq
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

            # Only sharpen at the smallest scale to avoid noise amplification
            if target_width <= 160:
                kernel = np.array(
                    [[-1, -1, -1],
                     [-1,  9, -1],
                     [-1, -1, -1]],
                    dtype=np.float32,
                ) / 1.5
                gray = cv2.filter2D(gray, -1, kernel)

            # ROI-first detection using buffered markers
            roi_dets = self._detect_in_rois(gray, target_width, full_w, full_h)

            # Full-frame detection
            full_dets = list(self.detector.detect(gray))

            # Merge + deduplicate
            detections = self._deduplicate_detections(roi_dets + full_dets)
            return detections

        except Exception as e:
            self.get_logger().error(f"Detection error at width {target_width}: {e}")
            return []

    def _simple_detect_once(self):
        """
        Very simple AprilTag detection:
        - Rectify once
        - Optional single downscale
        - Gray conversion
        - Single detector.detect call
        - No ROI, no CLAHE, no sharpening
        """
        try:
            color_image = self.bridge.imgmsg_to_cv2(
                self.current_color_msg, desired_encoding='bgr8'
            )
            color_image_rectified = cv2.remap(
                color_image, self.mapx, self.mapy, cv2.INTER_LINEAR
            )
        except Exception as e:
            self.get_logger().error(f"Simple detection image error: {e}")
            return

        full_h, full_w = color_image_rectified.shape[:2]

        # Decide whether to downscale or use full-res
        if (
            self.simple_scale_width is not None
            and self.simple_scale_width > 0
            and self.simple_scale_width < full_w
        ):
            target_width = self.simple_scale_width
            sf = target_width / float(full_w)
            downscaled = cv2.resize(
                color_image_rectified,
                (target_width, int(full_h * sf)),
                interpolation=cv2.INTER_LINEAR,
            )
            gray = cv2.cvtColor(downscaled, cv2.COLOR_BGR2GRAY)
            resolution = target_width
        else:
            gray = cv2.cvtColor(color_image_rectified, cv2.COLOR_BGR2GRAY)
            sf = 1.0
            resolution = full_w

        try:
            dets = self.detector.detect(gray)
        except Exception as e:
            self.get_logger().error(f"Simple detection error: {e}")
            return

        for d in dets:
            tag_id = d['id']
            self.current_detections[tag_id] = {
                'detection': d,
                'scale_factor': sf,
                'resolution': resolution,
                'source': 'simple',
            }

    # ------------------------------------------------------------------ #
    # Pose calculation + publishing
    # ------------------------------------------------------------------ #

    def _calculate_and_publish_poses(self):
        """Calculate 3D poses using PnP with the highest-resolution detections."""
        try:
            color_image = self.bridge.imgmsg_to_cv2(
                self.current_color_msg, desired_encoding='bgr8'
            )
            color_image_rectified = cv2.remap(
                color_image, self.mapx, self.mapy, cv2.INTER_LINEAR
            )
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        # We keep this around in case you want to visualize later
        _detected_vis = color_image_rectified.copy()

        now = self.get_clock().now().nanoseconds / 1e9

        s = self.tag_size / 2.0
        obj_pts = np.array(
            [
                [-s,  s, 0],
                [ s,  s, 0],
                [ s, -s, 0],
                [-s, -s, 0],
            ],
            dtype=np.float32,
        )

        current_frame_ids = set()

        for tag_id, data in self.current_detections.items():
            d = data['detection']
            scale_factor = data['scale_factor']
            resolution = data['resolution']

            try:
                if 'lb' in d:
                    corners_d = np.array([d['lb'], d['rb'], d['rt'], d['lt']])
                elif 'lb-rb-rt-lt' in d:
                    corners_d = d['lb-rb-rt-lt']
                else:
                    continue

                # Scale corners back to full resolution
                corners_full_res = (corners_d / scale_factor).astype(np.float32)
                center_full_res = (d['center'] / scale_factor).astype(int)

                lt, rt, rb, lb = (
                    tuple(corners_full_res[3].astype(int)),
                    tuple(corners_full_res[2].astype(int)),
                    tuple(corners_full_res[1].astype(int)),
                    tuple(corners_full_res[0].astype(int)),
                )
                center = tuple(center_full_res)

                pose_msg = Pose()
                distance_value = -1.0

                img_pts = np.array([lt, rt, rb, lb], dtype=np.float32)
                dist_coeffs = np.zeros((4, 1), dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    self.K,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )

                if not success:
                    continue

                R, _ = cv2.Rodrigues(rvec)
                t = tvec.flatten()
                distance_value = float(np.linalg.norm(t))

                pose_msg.position.x = float(t[0])
                pose_msg.position.y = float(t[1])
                pose_msg.position.z = float(t[2])

                qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
                pose_msg.orientation.x = qx
                pose_msg.orientation.y = qy
                pose_msg.orientation.z = qz
                pose_msg.orientation.w = qw

                marker_msg = MarkerPose()
                marker_msg.id = int(tag_id)
                marker_msg.pose = pose_msg
                marker_msg.distance = distance_value
                marker_msg.center_x = float(center[0])
                marker_msg.center_y = float(center[1])

                xs = [lt[0], rt[0], rb[0], lb[0]]
                ys = [lt[1], rt[1], rb[1], lb[1]]
                marker_msg.bbox_x_min = float(min(xs))
                marker_msg.bbox_y_min = float(min(ys))
                marker_msg.bbox_x_max = float(max(xs))
                marker_msg.bbox_y_max = float(max(ys))

                # Buffering with smoothing of position + distance
                if self.enable_buffer and tag_id in self.marker_buffer:
                    prev_marker = self.marker_buffer[tag_id]['marker']
                    marker_msg.distance = (
                        self.buffer_alpha * marker_msg.distance
                        + (1.0 - self.buffer_alpha) * prev_marker.distance
                    )
                    marker_msg.pose.position.x = (
                        self.buffer_alpha * marker_msg.pose.position.x
                        + (1.0 - self.buffer_alpha) * prev_marker.pose.position.x
                    )
                    marker_msg.pose.position.y = (
                        self.buffer_alpha * marker_msg.pose.position.y
                        + (1.0 - self.buffer_alpha) * prev_marker.pose.position.y
                    )
                    marker_msg.pose.position.z = (
                        self.buffer_alpha * marker_msg.pose.position.z
                        + (1.0 - self.buffer_alpha) * prev_marker.pose.position.z
                    )

                # Update buffer entry
                self.marker_buffer[tag_id] = {
                    'marker': copy.deepcopy(marker_msg),
                    'last_seen': now,
                    'resolution': resolution,
                }

                current_frame_ids.add(tag_id)

            except Exception as e:
                self.get_logger().error(f"Pose error for tag {tag_id}: {e}")
                continue

        # Remove stale markers based on age
        ids_to_remove = [
            mid
            for mid, data in self.marker_buffer.items()
            if (now - data['last_seen']) > self.buffer_max_age_sec
        ]
        for mid in ids_to_remove:
            del self.marker_buffer[mid]

        # Build outgoing message
        marker_array_msg = MarkerPoseArray()
        marker_array_msg.header.stamp = self.current_color_msg.header.stamp
        marker_array_msg.header.frame_id = self.current_color_msg.header.frame_id

        if self.enable_buffer:
            for marker_id, buffer_data in self.marker_buffer.items():
                marker_array_msg.markers.append(buffer_data['marker'])
        else:
            for marker_id in current_frame_ids:
                if marker_id in self.marker_buffer:
                    marker_array_msg.markers.append(
                        self.marker_buffer[marker_id]['marker']
                    )

        self.pub.publish(marker_array_msg)


def main(args=None):
    rclpy.init(args=args)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag_family", type=str, default="tagStandard41h12")
    parser.add_argument("--tag_size", type=float, default=0.162)
    parser.add_argument(
        "--detection_fps",
        type=float,
        default=5.0,
        help="How often to run the detector.",
    )
    parser.add_argument("--enable_buffer", action="store_true")
    parser.add_argument(
        "--buffer_max_age",
        type=float,
        default=2.0,
        help="Maximum age [s] to keep a marker in the buffer.",
    )
    parser.add_argument(
        "--buffer_alpha",
        type=float,
        default=0.7,
        help="Exponential smoothing factor for positions/distances.",
    )
    parser.add_argument(
        "--simple_detection",
        action="store_true",
        help=(
            "Use a single-scale, simple AprilTag detector "
            "(no multiscale, no ROI tracking, no temporal buffer)."
        ),
    )
    parser.add_argument(
        "--simple_scale_width",
        type=int,
        default=320,
        help=(
            "Target width in pixels for simple detection. "
            "Set to 0 or a value >= image width to use full resolution."
        ),
    )

    parsed, _ = parser.parse_known_args()

    node = AprilTagVisualizationNode(
        tag_family=parsed.tag_family,
        tag_size=parsed.tag_size,
        detection_fps=parsed.detection_fps,
        enable_buffer=parsed.enable_buffer,
        buffer_max_age_sec=parsed.buffer_max_age,
        buffer_alpha=parsed.buffer_alpha,
        simple_detection=parsed.simple_detection,
        simple_scale_width=parsed.simple_scale_width,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
