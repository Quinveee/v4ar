#!/usr/bin/env python3

import os
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3, PoseArray
from nav_msgs.msg import Path
from perception_msgs.msg import MarkerPoseArray
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np
import yaml


def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion to yaw angle in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw

# --- Field drawing constants and helper ---
# Dimensions in mm (soccer field spec used for drawing)
A = 9000  # field length
B = 6000  # field width
C = 50    # line width
D = 100   # penalty mark size (diameter)
E = 600   # goal area length
F = 2200  # goal area width
G = 1650  # penalty area length
H = 4000  # penalty area width
I = 1300  # penalty mark distance from goal line
J = 1500  # center circle diameter


def draw_field(scale_px_per_mm: float = 0.1):
    """
    Returns a BGR OpenCV image of the field drawn to scale.

    scale_px_per_mm: how many pixels per mm (0.1 → 900x600 px image)
    """
    field_w_px = int(A * scale_px_per_mm)
    field_h_px = int(B * scale_px_per_mm)

    # green background
    img = np.full((field_h_px, field_w_px, 3), (0, 128, 0), dtype=np.uint8)

    def to_px(x_mm, y_mm):
        """Convert (x,y) in mm (origin bottom-left) -> (u,v) pixels (origin top-left)."""
        u = int(x_mm * scale_px_per_mm)
        v = field_h_px - int(y_mm * scale_px_per_mm)
        return u, v

    line_thickness = max(1, int(C * scale_px_per_mm))

    # --- Outer field rectangle ---
    bl = to_px(0, 0)
    tr = to_px(A, B)
    cv2.rectangle(img, bl, tr, (255, 255, 255), line_thickness)

    # --- Halfway line ---
    mid_x = A / 2
    p1 = to_px(mid_x, 0)
    p2 = to_px(mid_x, B)
    cv2.line(img, p1, p2, (255, 255, 255), line_thickness)

    # --- Center circle ---
    center = to_px(mid_x, B / 2)
    radius_px = int((J / 2) * scale_px_per_mm)
    cv2.circle(img, center, radius_px, (255, 255, 255), line_thickness)

    # --- Left penalty area ---
    goal_center_y = B / 2
    pa_half_w = H / 2
    pa_left_x = 0
    pa_right_x = G

    pa_bottom_y = goal_center_y - pa_half_w
    pa_top_y = goal_center_y + pa_half_w

    p_bl = to_px(pa_left_x, pa_bottom_y)
    p_tr = to_px(pa_right_x, pa_top_y)
    cv2.rectangle(img, p_bl, p_tr, (255, 255, 255), line_thickness)

    # --- Left goal area ---
    ga_half_w = F / 2
    ga_left_x = 0
    ga_right_x = E
    ga_bottom_y = goal_center_y - ga_half_w
    ga_top_y = goal_center_y + ga_half_w

    p_bl = to_px(ga_left_x, ga_bottom_y)
    p_tr = to_px(ga_right_x, ga_top_y)
    cv2.rectangle(img, p_bl, p_tr, (255, 255, 255), line_thickness)

    # --- Left penalty mark ---
    pm_x = I
    pm_y = goal_center_y
    pm_center = to_px(pm_x, pm_y)
    pm_radius_px = max(1, int((D / 2) * scale_px_per_mm))
    cv2.circle(img, pm_center, pm_radius_px, (255, 255, 255), -1)

    # --- Right side: mirror everything around center line ---
    pa_left_x_r = A - G
    pa_right_x_r = A
    p_bl = to_px(pa_left_x_r, pa_bottom_y)
    p_tr = to_px(pa_right_x_r, pa_top_y)
    cv2.rectangle(img, p_bl, p_tr, (255, 255, 255), line_thickness)

    ga_left_x_r = A - E
    ga_right_x_r = A
    p_bl = to_px(ga_left_x_r, ga_bottom_y)
    p_tr = to_px(ga_right_x_r, ga_top_y)
    cv2.rectangle(img, p_bl, p_tr, (255, 255, 255), line_thickness)

    pm_x_r = A - I
    pm_center_r = to_px(pm_x_r, pm_y)
    cv2.circle(img, pm_center_r, pm_radius_px, -1)

    return img


class FieldVisualizationNode(Node):
    """
    Visualize the rover's global pose, triangulation geometry, and
    (optionally) the discrete navigation grid/path on a soccer field.

    Subscribes:
        - /robot_pose                 (geometry_msgs/PoseStamped)
        - /detected_markers           (perception_msgs/MarkerPoseArray)
        - /oak/detected_markers       (perception_msgs/MarkerPoseArray)
        - /control/heading_vector     (geometry_msgs/Vector3)
        - /control/grid_path          (nav_msgs/Path)          [optional]
        - /control/grid_target_cell   (geometry_msgs/PoseStamped) [optional]
        - /control/grid_blocked_cells (geometry_msgs/PoseArray)   [optional]
    """

    def __init__(self):
        super().__init__("field_visualization_node")

        # --- Parameters ---
        default_field_img = os.path.join(
            get_package_share_directory("visualizations"),
            "data",
            "field.png",
        )

        self.declare_parameter("field_image", default_field_img)
        self.declare_parameter("draw_field", True)
        self.declare_parameter("scale_px_per_mm", 0.1)
        self.declare_parameter("field_length_mm", 9000.0)
        self.declare_parameter("field_width_mm", 6000.0)
        self.declare_parameter("origin_at_center", False)

        # Markers
        self.declare_parameter("use_buffered_markers", False)
        self.declare_parameter("marker_topic", "")
        self.declare_parameter("pose_topic", "/robot_pose")
        self.declare_parameter("marker_config", "config/markers.yaml")

        # Viewpoints (optional)
        self.declare_parameter("show_viewpoints", False)
        self.declare_parameter("viewpoint_side", "both")          # 'left'/'right'/'both'
        self.declare_parameter("viewpoint_half_offset_m", 1.0)    # before/after half-line

        # Grid overlay
        self.declare_parameter("show_grid", False)
        self.declare_parameter("grid_resolution_m", 0.5)

        field_image_path = self.get_parameter("field_image").value
        self.draw_field_flag = bool(self.get_parameter("draw_field").value)
        self.scale_px_per_mm = float(self.get_parameter("scale_px_per_mm").value)
        self.field_length_mm = float(self.get_parameter("field_length_mm").value)
        self.field_width_mm = float(self.get_parameter("field_width_mm").value)
        self.origin_at_center = bool(self.get_parameter("origin_at_center").value)
        self.use_buffered_markers = bool(self.get_parameter("use_buffered_markers").value)
        self.marker_topic_param = str(self.get_parameter("marker_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)

        # Viewpoint params
        self.show_viewpoints = bool(self.get_parameter("show_viewpoints").value)
        self.viewpoint_side = str(self.get_parameter("viewpoint_side").value).lower()
        if self.viewpoint_side not in ("left", "right", "both"):
            self.viewpoint_side = "both"
        self.viewpoint_half_offset_m = float(
            self.get_parameter("viewpoint_half_offset_m").value
        )

        # Grid overlay params
        self.show_grid = bool(self.get_parameter("show_grid").value)
        self.grid_resolution_m = float(self.get_parameter("grid_resolution_m").value)

        # --- Load marker_map from YAML (in meters) ---
        config_path = self.get_parameter("marker_config").value
        if not os.path.isabs(config_path):
            try:
                perception_share = get_package_share_directory("perception")
                config_path = os.path.join(perception_share, config_path)
            except Exception as e:
                self.get_logger().warn(
                    f"Could not resolve perception share dir: {e}. "
                    f"Using marker_config path as-is: {config_path}"
                )

        self.marker_map = self.load_marker_config(config_path)
        self.get_logger().info(f"Loaded {len(self.marker_map)} markers from {config_path}")

        # --- Create / Load field image ---
        if self.draw_field_flag:
            try:
                self.field_img = draw_field(scale_px_per_mm=self.scale_px_per_mm)
                self.get_logger().info(
                    f"Using programmatically drawn field (scale={self.scale_px_per_mm} px/mm)."
                )
            except Exception as e:
                self.get_logger().warn(
                    f"draw_field failed: {e}. Falling back to image file."
                )
                self.field_img = cv2.imread(field_image_path, cv2.IMREAD_COLOR)
        else:
            self.field_img = cv2.imread(field_image_path, cv2.IMREAD_COLOR)

        if self.field_img is None:
            self.get_logger().warn(
                f"Could not load field image at '{field_image_path}'. Using a blank field instead."
            )
            self.field_img = np.full((600, 900, 3), (0, 128, 0), dtype=np.uint8)

        self.field_h, self.field_w = self.field_img.shape[:2]

        # Scale factors: pixels per mm
        self.px_per_mm_x = self.field_w / self.field_length_mm
        self.px_per_mm_y = self.field_h / self.field_width_mm

        self.get_logger().info(
            f"Field image size: {self.field_w}x{self.field_h} px "
            f"(scale: {self.px_per_mm_x:.3f} px/mm, {self.px_per_mm_y:.3f} px/mm)"
        )

        # Latest pose & detections
        self.latest_pose: PoseStamped | None = None
        self.latest_markers = []
        self.latest_oak_markers = []
        self.latest_vector = None

        # Latest grid/path info
        self.latest_grid_path: Path | None = None
        self.latest_grid_target: PoseStamped | None = None
        self.latest_grid_blocked: PoseArray | None = None

        # --- Subscriptions ---
        if self.marker_topic_param:
            marker_topic = self.marker_topic_param
        else:
            marker_topic = (
                "/detected_markers_buffered"
                if self.use_buffered_markers
                else "/detected_markers"
            )

        self.pose_sub = self.create_subscription(
            PoseStamped, self.pose_topic, self.pose_callback, 10
        )
        self.markers_sub = self.create_subscription(
            MarkerPoseArray, marker_topic, self.markers_callback, 10
        )
        self.oak_markers_sub = self.create_subscription(
            MarkerPoseArray, "/oak/detected_markers", self.oak_markers_callback, 10
        )

        self.vector_sub = self.create_subscription(
            Vector3,
            "/control/heading_vector",
            self.vector_callback,
            10,
        )

        # Grid path visualization subscriptions
        self.grid_path_sub = self.create_subscription(
            Path, "/control/grid_path", self.grid_path_callback, 10
        )
        self.grid_target_sub = self.create_subscription(
            PoseStamped, "/control/grid_target_cell", self.grid_target_callback, 10
        )
        self.grid_blocked_sub = self.create_subscription(
            PoseArray, "/control/grid_blocked_cells", self.grid_blocked_callback, 10
        )

        self.get_logger().info(
            f"FieldVisualization listening to pose: {self.pose_topic}, markers: {marker_topic}, "
            f"oak_markers: /oak/detected_markers, grid_path: /control/grid_path"
        )

        # --- Timer for rendering ---
        self.timer = self.create_timer(0.1, self.render)  # 10 Hz

        # Window
        cv2.namedWindow("Rover on Field", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Rover on Field", 900, 600)

        self.get_logger().info("FieldVisualizationNode started.")

    # ------------------------------------------------------------------ #
    # Config loader
    # ------------------------------------------------------------------ #

    def load_marker_config(self, path):
        """Load marker world coordinates from YAML. Returns {id: (x_m, y_m)}."""
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            marker_map = {int(k): tuple(v) for k, v in data["marker_map"].items()}
            return marker_map
        except Exception as e:
            self.get_logger().error(
                f"Failed to load marker config from {path}: {e}"
            )
            return {}

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    def pose_callback(self, msg: PoseStamped):
        if msg.header.frame_id and msg.header.frame_id != "world":
            self.get_logger().warn_once(
                f"Pose frame_id is '{msg.header.frame_id}', expected 'world'."
            )
        self.latest_pose = msg

    def markers_callback(self, msg: MarkerPoseArray):
        self.latest_markers = msg.markers

    def oak_markers_callback(self, msg: MarkerPoseArray):
        self.latest_oak_markers = msg.markers

    def vector_callback(self, msg: Vector3):
        self.latest_vector = (msg.x, msg.y)

    def grid_path_callback(self, msg: Path):
        self.latest_grid_path = msg

    def grid_target_callback(self, msg: PoseStamped):
        self.latest_grid_target = msg

    def grid_blocked_callback(self, msg: PoseArray):
        self.latest_grid_blocked = msg

    # ------------------------------------------------------------------ #
    # Coordinate mapping
    # ------------------------------------------------------------------ #

    def world_to_pixel(self, x_world_mm: float, y_world_mm: float):
        """
        Map world coordinates [mm] -> image pixel (u, v).

        YOUR convention:
          - Origin (0,0) at BOTTOM-RIGHT corner of the field.
          - +X: up along field width (0 -> 6000 mm).
          - +Y: left along field length (0 -> 9000 mm).

        Drawing convention (field_img):
          - Origin at bottom-left.
          - x_std: along field length, left -> right, 0..field_length_mm.
          - y_std: along field width, bottom -> top, 0..field_width_mm.
        """
        A_mm = self.field_length_mm   # 9000
        B_mm = self.field_width_mm    # 6000

        x_world_mm = float(np.clip(x_world_mm, 0.0, B_mm))
        y_world_mm = float(np.clip(y_world_mm, 0.0, A_mm))

        # x_std_mm = A - Y_world   (right-origin -> left-origin)
        # y_std_mm = X_world      (bottom-origin -> bottom-origin)
        x_std_mm = A_mm - y_world_mm
        y_std_mm = x_world_mm

        u = int(x_std_mm * self.px_per_mm_x)
        v = int(self.field_h - y_std_mm * self.px_per_mm_y)

        return u, v

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #

    def draw_static_markers(self, img):
        """Draw all markers from marker_map (global positions)."""
        for mid, v in self.marker_map.items():
            mx_m, my_m, _ = v
            x_mm = mx_m * 1000.0
            y_mm = my_m * 1000.0
            u, v_px = self.world_to_pixel(x_mm, y_mm)

            cv2.circle(img, (u, v_px), 6, (255, 0, 0), -1)
            cv2.putText(
                img,
                f"ID {mid} ({mx_m:.2f},{my_m:.2f}) m",
                (u + 8, v_px - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    def draw_viewpoints(self, img):
        """Optionally draw the 'viewpoint' locations."""
        if not self.show_viewpoints:
            return

        field_length_m = self.field_length_mm / 1000.0
        field_width_m = self.field_width_mm / 1000.0
        penalty_length_m = G / 1000.0
        penalty_width_m = H / 1000.0

        side_offset_m = (field_width_m - penalty_width_m) / 2.0
        half_line_y_m = field_length_m / 2.0
        half_offset_m = self.viewpoint_half_offset_m

        penalty_near_y = penalty_length_m
        penalty_far_y = field_length_m - penalty_length_m
        pre_half_y = half_line_y_m - half_offset_m
        post_half_y = half_line_y_m + half_offset_m

        def lane_x(side: str) -> float:
            if side == "right":
                return side_offset_m
            else:
                return field_width_m - side_offset_m

        sides_to_draw = (
            ["right", "left"] if self.viewpoint_side == "both" else [self.viewpoint_side]
        )

        for side in sides_to_draw:
            x_lane = lane_x(side)
            points_m = [
                (x_lane, penalty_near_y),
                (x_lane, pre_half_y),
                (x_lane, post_half_y),
                (x_lane, penalty_far_y),
            ]

            color = (255, 255, 0) if side == "right" else (255, 0, 255)

            prev_uv = None
            for idx, (x_m, y_m) in enumerate(points_m):
                x_mm = x_m * 1000.0
                y_mm = y_m * 1000.0
                u, v_px = self.world_to_pixel(x_mm, y_mm)

                cv2.circle(img, (u, v_px), 6, color, -1)

                label = f"{'R' if side == 'right' else 'L'}{idx + 1}"
                cv2.putText(
                    img,
                    label,
                    (u + 8, v_px - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                if prev_uv is not None:
                    cv2.line(img, prev_uv, (u, v_px), color, 2)
                prev_uv = (u, v_px)

    def draw_grid_lines(self, img):
        """
        Optionally draw the planner grid as fixed lines over the field.
        """
        if not self.show_grid or self.grid_resolution_m <= 0.0:
            return

        A_mm = self.field_length_mm
        B_mm = self.field_width_mm
        res_mm = self.grid_resolution_m * 1000.0
        if res_mm <= 0.0:
            return

        # Vertical grid lines (constant x, varying y)
        num_cols = int(B_mm / res_mm) + 1
        for i in range(num_cols + 1):
            x_mm = min(i * res_mm, B_mm)
            u1, v1 = self.world_to_pixel(x_mm, 0.0)
            u2, v2 = self.world_to_pixel(x_mm, A_mm)
            cv2.line(img, (u1, v1), (u2, v2), (80, 160, 80), 1)

        # Horizontal grid lines (constant y, varying x)
        num_rows = int(A_mm / res_mm) + 1
        for j in range(num_rows + 1):
            y_mm = min(j * res_mm, A_mm)
            u1, v1 = self.world_to_pixel(0.0, y_mm)
            u2, v2 = self.world_to_pixel(B_mm, y_mm)
            cv2.line(img, (u1, v1), (u2, v2), (80, 160, 80), 1)

    def draw_marker_ranges_from_rover(self, img, rover_u, rover_v):
        """Draw lines from rover to each currently detected marker with distance text."""
        if self.latest_markers is None:
            return

        for m in self.latest_markers:
            if m.id not in self.marker_map:
                continue

            mx_m, my_m, _ = self.marker_map[m.id]
            x_mm = mx_m * 1000.0
            y_mm = my_m * 1000.0
            mu, mv = self.world_to_pixel(x_mm, y_mm)

            cv2.line(img, (rover_u, rover_v), (mu, mv), (0, 255, 255), 2)

            mid_u = int((rover_u + mu) / 2)
            mid_v = int((rover_v + mv) / 2)
            txt = f"ID {m.id} : {m.distance:.2f} m"
            cv2.putText(
                img,
                txt,
                (mid_u + 5, mid_v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                img,
                txt,
                (mid_u + 5, mid_v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            avg_px_per_mm = 0.5 * (self.px_per_mm_x + self.px_per_mm_y)
            radius_px = int(m.distance * 1000.0 * avg_px_per_mm)
            if 0 < radius_px < 5000:
                cv2.circle(img, (mu, mv), radius_px, (128, 128, 255), 1)

    def draw_oak_marker_ranges_from_rover(self, img, rover_u, rover_v):
        """Draw lines from rover to each OAK detected marker in oak brown color."""
        if not self.latest_oak_markers:
            return

        oak_brown = (19, 69, 139)  # BGR

        for m in self.latest_oak_markers:
            if m.id not in self.marker_map:
                continue

            mx_m, my_m, _ = self.marker_map[m.id]
            x_mm = mx_m * 1000.0
            y_mm = my_m * 1000.0
            mu, mv = self.world_to_pixel(x_mm, y_mm)

            cv2.line(img, (rover_u, rover_v), (mu, mv), oak_brown, 2)

            mid_u = int((rover_u + mu) / 2)
            mid_v = int((rover_v + mv) / 2)
            txt = f"OAK {m.id} : {m.distance:.2f} m"
            cv2.putText(
                img,
                txt,
                (mid_u + 5, mid_v + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                img,
                txt,
                (mid_u + 5, mid_v + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                oak_brown,
                1,
            )

            avg_px_per_mm = 0.5 * (self.px_per_mm_x + self.px_per_mm_y)
            radius_px = int(m.distance * 1000.0 * avg_px_per_mm)
            if 0 < radius_px < 5000:
                cv2.circle(img, (mu, mv), radius_px, oak_brown, 1)

    def draw_heading_vector(self, img, rover_u, rover_v):
        if self.latest_vector is None:
            return

        vx, vy = self.latest_vector
        scale = 60  # pixels

        end_u = int(rover_u + vx * scale)
        end_v = int(rover_v - vy * scale)

        cv2.arrowedLine(
            img,
            (rover_u, rover_v),
            (end_u, end_v),
            (255, 0, 0),
            3,
            tipLength=0.1,
        )

    def draw_robot_orientation(self, img, rover_u, rover_v, yaw):
        """Draw an arrow showing the robot's orientation (yaw) and display the angle."""
        arrow_length = 80

        dx_world = math.cos(yaw)
        dy_world = math.sin(yaw)

        end_u = int(rover_u - dy_world * arrow_length)
        end_v = int(rover_v - dx_world * arrow_length)

        cv2.arrowedLine(
            img,
            (rover_u, rover_v),
            (end_u, end_v),
            (255, 255, 0),
            4,
            tipLength=0.3,
        )

        yaw_deg = math.degrees(yaw)
        txt = f"Yaw: {yaw_deg:.1f}°"
        cv2.putText(
            img,
            txt,
            (rover_u + 15, rover_v + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            img,
            txt,
            (rover_u + 15, rover_v + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    def draw_grid_path(self, img):
        """Draw the current grid-based path from /control/grid_path, if any."""
        if self.latest_grid_path is None or not self.latest_grid_path.poses:
            return

        pts = []
        for ps in self.latest_grid_path.poses:
            x_m = ps.pose.position.x
            y_m = ps.pose.position.y
            x_mm = x_m * 1000.0
            y_mm = y_m * 1000.0
            u, v = self.world_to_pixel(x_mm, y_mm)
            pts.append((u, v))

        if len(pts) == 1:
            u, v = pts[0]
            cv2.circle(img, (u, v), 6, (0, 165, 255), -1)
            return

        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], (0, 165, 255), 2)

        cv2.circle(img, pts[0], 6, (0, 255, 255), -1)   # start
        cv2.circle(img, pts[-1], 6, (0, 0, 255), -1)    # end

    def draw_grid_target_cell(self, img):
        """Draw the current target grid cell center."""
        if self.latest_grid_target is None:
            return

        x_m = self.latest_grid_target.pose.position.x
        y_m = self.latest_grid_target.pose.position.y
        x_mm = x_m * 1000.0
        y_mm = y_m * 1000.0
        u, v = self.world_to_pixel(x_mm, y_mm)

        cv2.circle(img, (u, v), 10, (0, 255, 0), 2)
        cv2.putText(
            img,
            "Target cell",
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    def draw_blocked_cells(self, img):
        """Draw blocked cells (obstacle projections)."""
        if self.latest_grid_blocked is None or not self.latest_grid_blocked.poses:
            return

        for pose in self.latest_grid_blocked.poses:
            x_m = pose.position.x
            y_m = pose.position.y
            x_mm = x_m * 1000.0
            y_mm = y_m * 1000.0
            u, v = self.world_to_pixel(x_mm, y_mm)
            cv2.rectangle(img, (u - 4, v - 4), (u + 4, v + 4), (0, 0, 0), -1)

    # ------------------------------------------------------------------ #
    # Render loop
    # ------------------------------------------------------------------ #

    def render(self):
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.get_logger().info("Closing FieldVisualizationNode window.")
            rclpy.shutdown()
            return

        field = self.field_img.copy()

        # Grid overlay and static field stuff
        self.draw_grid_lines(field)
        self.draw_static_markers(field)
        self.draw_viewpoints(field)

        # Path and blocked cells from grid planner
        self.draw_grid_path(field)
        self.draw_blocked_cells(field)
        self.draw_grid_target_cell(field)

        if self.latest_pose is None:
            cv2.putText(
                field,
                "Waiting for /robot_pose...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Rover on Field", field)
            return

        x_m = self.latest_pose.pose.position.x
        y_m = self.latest_pose.pose.position.y

        qx = self.latest_pose.pose.orientation.x
        qy = self.latest_pose.pose.orientation.y
        qz = self.latest_pose.pose.orientation.z
        qw = self.latest_pose.pose.orientation.w
        yaw = quaternion_to_yaw(qx, qy, qz, qw)

        x_mm = x_m * 1000.0
        y_mm = y_m * 1000.0

        rover_u, rover_v = self.world_to_pixel(x_mm, y_mm)

        cv2.circle(field, (rover_u, rover_v), 10, (0, 0, 255), -1)
        cv2.putText(
            field,
            f"Rover ({x_m:.2f}, {y_m:.2f}) m",
            (rover_u + 15, rover_v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        self.draw_robot_orientation(field, rover_u, rover_v, yaw)
        self.draw_heading_vector(field, rover_u, rover_v)
        self.draw_marker_ranges_from_rover(field, rover_u, rover_v)
        self.draw_oak_marker_ranges_from_rover(field, rover_u, rover_v)

        visible_ids = [m.id for m in self.latest_markers if m.id in self.marker_map]
        oak_visible_ids = [m.id for m in self.latest_oak_markers if m.id in self.marker_map]

        cv2.putText(
            field,
            f"Visible markers: {visible_ids}",
            (20, self.field_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        if oak_visible_ids:
            cv2.putText(
                field,
                f"OAK markers: {oak_visible_ids}",
                (20, self.field_h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (19, 69, 139),
                2,
            )

        cv2.imshow("Rover on Field", field)

    # ------------------------------------------------------------------ #

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Rover-on-field triangulation visualization"
    )
    parser.add_argument(
        "--field_image",
        type=str,
        default=None,
        help="Path to field image (PNG/JPG). If not given, uses the package default.",
    )
    parser.add_argument(
        "--origin_at_center",
        action="store_true",
        help="(Currently unused) Keep for compatibility.",
    )
    parser.add_argument(
        "--marker_config",
        type=str,
        default=None,
        help="Override path to markers.yaml used for visualization.",
    )
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = FieldVisualizationNode()

    params = []
    if parsed_args.field_image is not None:
        params.append(node.get_parameter("field_image").set__string(parsed_args.field_image))
    if parsed_args.origin_at_center:
        params.append(node.get_parameter("origin_at_center").set__bool(True))
    if parsed_args.marker_config is not None:
        params.append(node.get_parameter("marker_config").set__string(parsed_args.marker_config))

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
