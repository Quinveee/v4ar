#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np
import math

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


class OdometryTrajectoryVisualizationNode(Node):
    """
    Visualize odometry trajectory on a soccer field.

    Subscribes:
        - /odom_pose_* (geometry_msgs/PoseStamped) - odometry pose from odometry node
        - /robot_pose (geometry_msgs/PoseStamped) - triangulation pose for starting point

    Displays:
        - Starting point (from triangulation)
        - Current odometry position
        - Trajectory path
    """

    def __init__(self):
        super().__init__("odometry_trajectory_visualization_node")

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
        self.declare_parameter("odom_topic", "/odom_pose_processed")
        self.declare_parameter("triangulation_topic", "/robot_pose")
        self.declare_parameter("max_trajectory_points", 1000)

        field_image_path = self.get_parameter("field_image").value
        self.draw_field_flag = bool(self.get_parameter("draw_field").value)
        self.scale_px_per_mm = float(self.get_parameter("scale_px_per_mm").value)
        self.field_length_mm = float(self.get_parameter("field_length_mm").value)
        self.field_width_mm = float(self.get_parameter("field_width_mm").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.triangulation_topic = str(self.get_parameter("triangulation_topic").value)
        self.max_trajectory_points = int(self.get_parameter("max_trajectory_points").value)
        self.odom_received = False  

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

        # State
        self.start_x = None
        self.start_y = None
        self.current_x = None
        self.current_y = None
        self.current_theta = None
        self.trajectory = []  # List of (x, y) points
        self.start_initialized = False
        self.odom_received = False
        self.triangulation_received = False

        # Subscribers
        self.odom_sub = self.create_subscription(
            PoseStamped, self.odom_topic, self.odom_callback, 10
        )
        self.triangulation_sub = self.create_subscription(
            PoseStamped, self.triangulation_topic, self.triangulation_callback, 10
        )

        self.get_logger().info(
            f"Odometry trajectory visualization initialized. "
            f"Subscribed to: {self.odom_topic} and {self.triangulation_topic}"
        )

        # Timer for rendering
        self.create_timer(0.033, self.render)  # ~30 FPS

    def quaternion_to_yaw(self, qx, qy, qz, qw):
        """Convert quaternion to yaw angle."""
        return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    def world_to_pixel(self, x_m, y_m):
        """
        Convert world coordinates (meters) to pixel coordinates.
        Matches the coordinate system used by field_visualization_node:
        - Origin (0,0) at BOTTOM-RIGHT corner of the field
        - +X: up along field width (0 -> 6000 mm)
        - +Y: left along field length (0 -> 9000 mm)
        """
        # Convert meters to mm
        x_world_mm = x_m * 1000.0
        y_world_mm = y_m * 1000.0
        
        # Clamp to field bounds
        A_mm = self.field_length_mm  # 9000
        B_mm = self.field_width_mm   # 6000
        x_world_mm = float(np.clip(x_world_mm, 0.0, B_mm))
        y_world_mm = float(np.clip(y_world_mm, 0.0, A_mm))
        
        # Transform from world coordinates to standard field coordinates
        # x_std_mm = A - Y_world (right-origin -> left-origin)
        # y_std_mm = X_world (bottom-origin -> bottom-origin)
        x_std_mm = A_mm - y_world_mm
        y_std_mm = x_world_mm
        
        # Convert to pixel coordinates
        u = int(x_std_mm * self.px_per_mm_x)
        v = int(self.field_h - y_std_mm * self.px_per_mm_y)
        
        return u, v

    def triangulation_callback(self, msg: PoseStamped):
        """Initialize starting point from triangulation (only once)."""
        if not self.start_initialized:
            self.start_x = msg.pose.position.x
            self.start_y = msg.pose.position.y
            self.start_initialized = True
            self.get_logger().info(
                f"Starting point initialized from triangulation: "
                f"x={self.start_x:.2f}, y={self.start_y:.2f}"
            )

    def odom_callback(self, msg: PoseStamped):
        """Update current position and trajectory from odometry."""
        if not self.odom_received:
            self.odom_received = True
            self._odom_message_count = 0
            self.get_logger().info(
                f"Received first odometry message from {self.odom_topic}: "
                f"x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}"
            )
        
        if not hasattr(self, '_odom_message_count'):
            self._odom_message_count = 0
        self._odom_message_count += 1
        
        x = msg.pose.position.x
        y = msg.pose.position.y

        # Extract yaw from quaternion
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        theta = self.quaternion_to_yaw(qx, qy, qz, qw)

        # Check if position actually changed
        position_changed = (
            self.current_x is None or 
            self.current_y is None or
            abs(self.current_x - x) > 0.001 or 
            abs(self.current_y - y) > 0.001
        )

        self.current_x = x
        self.current_y = y
        self.current_theta = theta

        # Add to trajectory only if position changed (avoid duplicate points)
        if position_changed:
            self.trajectory.append((x, y))

            # Limit trajectory size
            if len(self.trajectory) > self.max_trajectory_points:
                self.trajectory.pop(0)
        
        # Log periodically for debugging
        if self._odom_message_count <= 5 or self._odom_message_count % 50 == 0:
            self.get_logger().info(
                f"Odometry callback #{self._odom_message_count}: x={x:.2f}, y={y:.2f}, "
                f"theta={theta:.2f}, trajectory_points={len(self.trajectory)}"
            )

    def render(self):
        """Render the visualization."""
        # Create a copy of the field image
        field = self.field_img.copy()

        # Draw starting point (from triangulation)
        if self.start_initialized and self.start_x is not None:
            start_u, start_v = self.world_to_pixel(self.start_x, self.start_y)
            cv2.circle(field, (start_u, start_v), 15, (0, 255, 0), -1)  # Green circle
            cv2.circle(field, (start_u, start_v), 15, (255, 255, 255), 2)  # White border
            cv2.putText(
                field,
                "Start",
                (start_u + 20, start_v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Draw trajectory path
        if len(self.trajectory) > 1:
            trajectory_pixels = [self.world_to_pixel(x, y) for x, y in self.trajectory]
            for i in range(len(trajectory_pixels) - 1):
                pt1 = trajectory_pixels[i]
                pt2 = trajectory_pixels[i + 1]
                cv2.line(field, pt1, pt2, (255, 0, 255), 2)  # Magenta line

        # Draw current position
        if self.current_x is not None and self.current_y is not None:
            current_u, current_v = self.world_to_pixel(self.current_x, self.current_y)
            
            # Check if coordinates are within bounds
            if 0 <= current_u < self.field_w and 0 <= current_v < self.field_h:
                cv2.circle(field, (current_u, current_v), 10, (0, 0, 255), -1)  # Red circle
                cv2.circle(field, (current_u, current_v), 10, (255, 255, 255), 2)  # White border
            else:
                # Log if coordinates are out of bounds
                if not hasattr(self, '_logged_oob'):
                    self.get_logger().warn(
                        f"Current position out of bounds: ({current_u}, {current_v}) "
                        f"from world coords ({self.current_x:.2f}, {self.current_y:.2f})"
                    )
                    self._logged_oob = True

            # Draw heading arrow
            if self.current_theta is not None:
                arrow_length = 30
                end_u = int(current_u + arrow_length * math.cos(self.current_theta))
                end_v = int(current_v - arrow_length * math.sin(self.current_theta))
                cv2.arrowedLine(
                    field, (current_u, current_v), (end_u, end_v), (255, 255, 0), 3, tipLength=0.3
                )

            # Display coordinates
            cv2.putText(
                field,
                f"Current: ({self.current_x:.2f}, {self.current_y:.2f}) m",
                (current_u + 15, current_v - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

        # Display trajectory info
        info_text = f"Trajectory points: {len(self.trajectory)}"
        if self.start_initialized:
            info_text += f" | Start: ({self.start_x:.2f}, {self.start_y:.2f})"
        else:
            info_text += " | Waiting for triangulation..."
        if not self.odom_received:
            info_text += " | Waiting for odometry..."
        if self.current_x is not None:
            info_text += f" | Current: ({self.current_x:.2f}, {self.current_y:.2f})"
        cv2.putText(
            field,
            info_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        
        # Display status for debugging
        status_text = f"Odom: {'OK' if self.odom_received else 'WAIT'} | Tri: {'OK' if self.start_initialized else 'WAIT'}"
        cv2.putText(
            field,
            status_text,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        
        debug_text = f"Topics: {self.odom_topic} | {self.triangulation_topic}"
        cv2.putText(
            field,
            debug_text,
            (20, self.field_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (150, 150, 150),
            1,
        )

        cv2.imshow("Odometry Trajectory", field)
        cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OdometryTrajectoryVisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

