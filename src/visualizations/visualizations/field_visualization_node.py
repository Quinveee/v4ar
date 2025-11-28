#!/usr/bin/env python3

import os
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3
from perception_msgs.msg import MarkerPoseArray
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np
import yaml


def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion to yaw angle in radians."""
    # yaw (z-axis rotation)
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
    cv2.circle(img, pm_center_r, pm_radius_px, (255, 255, 255), -1)

    return img


class FieldVisualizationNode(Node):
    """
    Visualize the rover's global pose and triangulation geometry on a soccer field.

    Subscribes:
        - /robot_pose      (geometry_msgs/PoseStamped)   [meters, world frame]
        - /detected_markers (perception_msgs/MarkerPoseArray)

    Uses marker_map from markers.yaml to:
        - draw all markers at their global coordinates,
        - draw lines from rover to each currently detected marker,
        - annotate those lines with distances.
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
        # kept for compatibility, but world_to_pixel uses your custom convention
        self.declare_parameter("origin_at_center", False)

        # Allow selecting buffered markers or custom topics
        self.declare_parameter("use_buffered_markers", False)
        # if empty, chosen based on use_buffered_markers
        self.declare_parameter("marker_topic", "")
        self.declare_parameter("pose_topic", "/robot_pose")

        # marker config (same default as triangulation_node)
        self.declare_parameter("marker_config", "config/markers.yaml")

        field_image_path = self.get_parameter("field_image").value
        self.draw_field_flag = bool(self.get_parameter("draw_field").value)
        self.scale_px_per_mm = float(
            self.get_parameter("scale_px_per_mm").value)
        self.field_length_mm = float(
            self.get_parameter("field_length_mm").value)
        self.field_width_mm = float(self.get_parameter("field_width_mm").value)
        self.origin_at_center = bool(
            self.get_parameter("origin_at_center").value)
        self.use_buffered_markers = bool(
            self.get_parameter("use_buffered_markers").value)
        self.marker_topic_param = str(self.get_parameter("marker_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)

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
        self.get_logger().info(
            f"Loaded {len(self.marker_map)} markers from {config_path}"
        )

        # --- Create / Load field image ---
        if self.draw_field_flag:
            try:
                self.field_img = draw_field(
                    scale_px_per_mm=self.scale_px_per_mm)
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
            self.field_img = np.full(
                (600, 900, 3), (0, 128, 0), dtype=np.uint8)

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

        # --- Subscriptions ---
        # Determine the marker topic: explicit `marker_topic` param wins,
        # otherwise select buffered or unbuffered topic based on `use_buffered_markers`.
        if self.marker_topic_param:
            marker_topic = self.marker_topic_param
        else:
            marker_topic = "/detected_markers_buffered" if self.use_buffered_markers else "/detected_markers"

        self.pose_sub = self.create_subscription(
            PoseStamped, self.pose_topic, self.pose_callback, 10)
        self.markers_sub = self.create_subscription(
            MarkerPoseArray, marker_topic, self.markers_callback, 10)
        self.oak_markers_sub = self.create_subscription(
            MarkerPoseArray, "/oak/detected_markers", self.oak_markers_callback, 10)
        self.latest_vector = None
        self.vector_sub = self.create_subscription(
            Vector3,
            "/control/heading_vector",
            self.vector_callback,
            10
        )

        self.get_logger().info(
            f"FieldVisualization listening to pose: {self.pose_topic}, markers: {marker_topic}, oak_markers: /oak/detected_markers")

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
            marker_map = {int(k): tuple(v)
                          for k, v in data["marker_map"].items()}
            return marker_map
        except Exception as e:
            self.get_logger().error(
                f"Failed to load marker config from {path}: {e}")
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

        # Clamp incoming values to physical extents
        x_world_mm = float(np.clip(x_world_mm, 0.0, B_mm))
        y_world_mm = float(np.clip(y_world_mm, 0.0, A_mm))

        # Transform:
        #   x_std_mm = A - Y_world   (right-origin -> left-origin)
        #   y_std_mm = X_world      (bottom-origin -> bottom-origin)
        x_std_mm = A_mm - y_world_mm
        y_std_mm = x_world_mm

        # Convert to pixel coordinates
        u = int(x_std_mm * self.px_per_mm_x)
        v = int(self.field_h - y_std_mm * self.px_per_mm_y)

        return u, v

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #

    def draw_static_markers(self, img):
        """Draw all markers from marker_map (global positions)."""
        # self.get_logger().info(f"Marker map: {self.marker_map.values()}")
        for mid, v in self.marker_map.items():
            # self.get_logger().info(f"Fuck you map: {v}")
            mx_m, my_m, _ = v
            x_mm = mx_m * 1000.0
            y_mm = my_m * 1000.0
            u, v = self.world_to_pixel(x_mm, y_mm)

            # marker dot
            cv2.circle(img, (u, v), 6, (255, 0, 0), -1)  # blue dot
            # label with id and coords in meters
            cv2.putText(
                img,
                f"ID {mid} ({mx_m:.2f},{my_m:.2f}) m",
                (u + 8, v - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

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

            # line from rover -> marker
            cv2.line(img, (rover_u, rover_v), (mu, mv), (0, 255, 255), 2)

            # distance label around the midpoint of the line
            mid_u = int((rover_u + mu) / 2)
            mid_v = int((rover_v + mv) / 2)
            cv2.putText(
                img,
                f"ID {m.id} : {m.distance:.2f} m",
                (mid_u + 5, mid_v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                img,
                f"ID {m.id} : {m.distance:.2f} m",
                (mid_u + 5, mid_v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Optional: draw distance circle around marker (triangulation geometry)
            # approximate px-per-mm isotropically
            avg_px_per_mm = 0.5 * (self.px_per_mm_x + self.px_per_mm_y)
            radius_px = int(m.distance * 1000.0 * avg_px_per_mm)
            if radius_px > 0 and radius_px < 5000:  # avoid crazy values
                cv2.circle(img, (mu, mv), radius_px, (128, 128, 255), 1)

    def draw_oak_marker_ranges_from_rover(self, img, rover_u, rover_v):
        """Draw lines from rover to each OAK detected marker in oak brown color."""
        if not self.latest_oak_markers:
            return

        # Oak brown color (BGR format): a nice brown shade
        oak_brown = (19, 69, 139)  # RGB(139, 69, 19) = Saddle Brown

        for m in self.latest_oak_markers:
            if m.id not in self.marker_map:
                continue

            mx_m, my_m, _ = self.marker_map[m.id]
            x_mm = mx_m * 1000.0
            y_mm = my_m * 1000.0
            mu, mv = self.world_to_pixel(x_mm, y_mm)

            # line from rover -> marker in oak brown
            cv2.line(img, (rover_u, rover_v), (mu, mv), oak_brown, 2)

            # distance label around the midpoint of the line
            mid_u = int((rover_u + mu) / 2)
            mid_v = int((rover_v + mv) / 2)

            # Black outline for text
            cv2.putText(
                img,
                f"OAK {m.id} : {m.distance:.2f} m",
                (mid_u + 5, mid_v + 15),  # Offset slightly down from regular markers
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                3,
            )
            # Oak brown text
            cv2.putText(
                img,
                f"OAK {m.id} : {m.distance:.2f} m",
                (mid_u + 5, mid_v + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                oak_brown,
                1,
            )

            # Optional: draw distance circle around marker in oak brown
            avg_px_per_mm = 0.5 * (self.px_per_mm_x + self.px_per_mm_y)
            radius_px = int(m.distance * 1000.0 * avg_px_per_mm)
            if radius_px > 0 and radius_px < 5000:  # avoid crazy values
                cv2.circle(img, (mu, mv), radius_px, oak_brown, 1)

    def draw_heading_vector(self, img, rover_u, rover_v):
        if self.latest_vector is None:
            return

        vx, vy = self.latest_vector

        # Scale for visibility — field is large, actual vector is unit length
        scale = 60  # pixels to draw

        # Convert (vx, vy) from world meters to pixel direction
        end_u = int(rover_u + vx * scale)
        # invert y because image pixels go downward
        end_v = int(rover_v - vy * scale)

        # Draw main arrow line
        cv2.arrowedLine(
            img,
            (rover_u, rover_v),
            (end_u, end_v),
            (255, 0, 0),  # red arrow
            3,
            tipLength=0.1,
        )

    def draw_robot_orientation(self, img, rover_u, rover_v, yaw):
        """Draw an arrow showing the robot's orientation (yaw) and display the angle."""
        # Arrow length in pixels
        arrow_length = 80

        # Calculate arrow endpoint based on yaw
        # Note: yaw is in world frame, need to account for coordinate transform
        # In world frame: yaw=0 points in +X direction (right in world coords)
        # We need to transform this to pixel coordinates

        # World coordinates: X is forward, Y is left
        # Our pixel transform: x_std = A - Y_world, y_std = X_world
        # So we need to rotate the yaw accordingly

        # Calculate direction in world frame
        dx_world = math.cos(yaw)
        dy_world = math.sin(yaw)

        # Transform to pixel direction
        # Since x_std = A - y_world and y_std = x_world
        # dx_pixel corresponds to -dy_world
        # dy_pixel corresponds to dx_world (but inverted for screen coords)
        end_u = int(rover_u - dy_world * arrow_length)
        end_v = int(rover_v - dx_world * arrow_length)

        # Draw the orientation arrow (cyan color to distinguish from heading vector)
        cv2.arrowedLine(
            img,
            (rover_u, rover_v),
            (end_u, end_v),
            (255, 255, 0),  # cyan arrow
            4,
            tipLength=0.3,
        )

        # Display the yaw angle in degrees near the robot
        yaw_deg = math.degrees(yaw)
        cv2.putText(
            img,
            f"Yaw: {yaw_deg:.1f}°",
            (rover_u + 15, rover_v + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # black outline
            3,
        )
        cv2.putText(
            img,
            f"Yaw: {yaw_deg:.1f}°",
            (rover_u + 15, rover_v + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),  # cyan text
            2,
        )

    # ------------------------------------------------------------------ #
    # Render loop
    # ------------------------------------------------------------------ #

    def render(self):
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.get_logger().info("Closing FieldVisualizationNode window.")
            rclpy.shutdown()
            return

        # Base field
        field = self.field_img.copy()

        # Draw static marker positions
        self.draw_static_markers(field)

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

        # Extract robot pose in meters
        x_m = self.latest_pose.pose.position.x
        y_m = self.latest_pose.pose.position.y

        # Extract yaw from quaternion
        qx = self.latest_pose.pose.orientation.x
        qy = self.latest_pose.pose.orientation.y
        qz = self.latest_pose.pose.orientation.z
        qw = self.latest_pose.pose.orientation.w
        yaw = quaternion_to_yaw(qx, qy, qz, qw)

        # Convert to mm for mapping
        x_mm = x_m * 1000.0
        y_mm = y_m * 1000.0

        rover_u, rover_v = self.world_to_pixel(x_mm, y_mm)

        # Draw rover
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

        # Draw robot orientation arrow (from triangulation yaw)
        self.draw_robot_orientation(field, rover_u, rover_v, yaw)

        # Draw steering vector arrow
        self.draw_heading_vector(field, rover_u, rover_v)

        # Draw lines + distance annotations to visible markers
        self.draw_marker_ranges_from_rover(field, rover_u, rover_v)

        # Draw lines + distance annotations to OAK detected markers (in oak brown)
        self.draw_oak_marker_ranges_from_rover(field, rover_u, rover_v)

        # Also show which marker IDs are currently seen
        visible_ids = [
            m.id for m in self.latest_markers if m.id in self.marker_map]
        oak_visible_ids = [
            m.id for m in self.latest_oak_markers if m.id in self.marker_map]

        cv2.putText(
            field,
            f"Visible markers: {visible_ids}",
            (20, self.field_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Show OAK markers if any
        if oak_visible_ids:
            cv2.putText(
                field,
                f"OAK markers: {oak_visible_ids}",
                (20, self.field_h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (19, 69, 139),  # Oak brown
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
        description="Rover-on-field triangulation visualization")
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

    # Optional CLI overrides
    params = []
    if parsed_args.field_image is not None:
        params.append(node.get_parameter(
            "field_image").set__string(parsed_args.field_image))
    if parsed_args.origin_at_center:
        params.append(node.get_parameter("origin_at_center").set__bool(True))
    if parsed_args.marker_config is not None:
        params.append(node.get_parameter(
            "marker_config").set__string(parsed_args.marker_config))

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
