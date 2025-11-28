#!/usr/bin/env python3
"""Visualization node for control system showing start, target, trajectory, and robot pose."""

import os
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from perception_msgs.msg import ObjectPoseArray
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np
import yaml

# Import field drawing from field_visualization_node
from .field_visualization_node import draw_field, quaternion_to_yaw


class ControlVisualizationNode(Node):
    """
    Visualize control system: start position, target position, trajectory, and robot pose.
    
    Subscribes:
        - /control/temp_pose (geometry_msgs/PoseStamped) - Current robot pose from control
        - /control/temp_trajectory (geometry_msgs/PoseStamped) - Trajectory points
    """

    def __init__(self):
        super().__init__("control_visualization_node")

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
        self.declare_parameter("marker_config", "config/markers.yaml")

        # Control visualization parameters
        self.declare_parameter("start_x", 0.0)
        self.declare_parameter("start_y", 0.0)
        self.declare_parameter("start_theta", 0.0)
        self.declare_parameter("target_x", 3.0)
        self.declare_parameter("target_y", 4.5)

        field_image_path = self.get_parameter("field_image").value
        self.draw_field_flag = bool(self.get_parameter("draw_field").value)
        self.scale_px_per_mm = float(self.get_parameter("scale_px_per_mm").value)
        self.field_length_mm = float(self.get_parameter("field_length_mm").value)
        self.field_width_mm = float(self.get_parameter("field_width_mm").value)

        # Get control parameters
        self.start_x = float(self.get_parameter("start_x").value)
        self.start_y = float(self.get_parameter("start_y").value)
        self.start_theta = float(self.get_parameter("start_theta").value)
        self.target_x = float(self.get_parameter("target_x").value)
        self.target_y = float(self.get_parameter("target_y").value)

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
                self.field_img = draw_field(scale_px_per_mm=self.scale_px_per_mm)
                self.get_logger().info(
                    f"Using programmatically drawn field (scale={self.scale_px_per_mm} px/mm)."
                )
            except Exception as e:
                self.get_logger().warn(f"draw_field failed: {e}. Falling back to image file.")
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

        # Latest pose & trajectory
        self.latest_control_pose: PoseStamped | None = None
        self.trajectory_points = []  # List of (x, y) tuples
        self.obstacles_world = []  # List of (x, y) tuples for obstacles

        # --- Subscriptions ---
        self.control_pose_sub = self.create_subscription(
            PoseStamped, "/control/temp_pose", self.control_pose_callback, 10
        )
        self.control_trajectory_sub = self.create_subscription(
            PoseStamped, "/control/temp_trajectory", self.trajectory_callback, 10
        )
        self.obstacle_sub = self.create_subscription(
            ObjectPoseArray, "/nav2/obstacles_world", self.obstacle_callback, 10
        )

        self.get_logger().info(
            "ControlVisualization listening to /control/temp_pose and /control/temp_trajectory"
        )

        # Window - create before timer
        cv2.namedWindow("Control Visualization", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Control Visualization", 900, 600)

        # --- Timer for rendering ---
        self.timer = self.create_timer(0.1, self.render)  # 10 Hz

        self.get_logger().info("ControlVisualizationNode started.")

    def load_marker_config(self, path):
        """Load marker world coordinates from YAML. Returns {id: (x_m, y_m)}."""
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            marker_map = {int(k): tuple(v) for k, v in data["marker_map"].items()}
            return marker_map
        except Exception as e:
            self.get_logger().error(f"Failed to load marker config from {path}: {e}")
            return {}

    def control_pose_callback(self, msg: PoseStamped):
        """Callback for control pose updates."""
        self.latest_control_pose = msg

    def trajectory_callback(self, msg: PoseStamped):
        """Callback for trajectory point updates."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.trajectory_points.append((x, y))
        
        # Limit trajectory size
        if len(self.trajectory_points) > 1000:
            self.trajectory_points = self.trajectory_points[-1000:]

    def obstacle_callback(self, msg: ObjectPoseArray):
        """Callback for obstacle updates in world frame."""
        self.obstacles_world = []
        for obj in msg.rovers:
            x = obj.pose.position.x
            y = obj.pose.position.y
            self.obstacles_world.append((x, y))

    def world_to_pixel(self, x_world_mm: float, y_world_mm: float):
        """
        Map world coordinates [mm] -> image pixel (u, v).
        Same convention as field_visualization_node.
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

    def draw_arrow(self, img, start_u, start_v, yaw, length, color, thickness):
        """Draw an arrow showing orientation."""
        dx_world = math.cos(yaw)
        dy_world = math.sin(yaw)
        
        end_u = int(start_u - dy_world * length)
        end_v = int(start_v - dx_world * length)
        
        cv2.arrowedLine(
            img,
            (start_u, start_v),
            (end_u, end_v),
            color,
            thickness,
            tipLength=0.3,
        )

    def render(self):
        """Render the visualization."""
        try:
            # Check if field image is valid
            if self.field_img is None:
                return
                
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self.get_logger().info("Closing ControlVisualizationNode window.")
                rclpy.shutdown()
                return

            # Base field
            field = self.field_img.copy()

            # Draw static marker positions (optional)
            for mid, v in self.marker_map.items():
                mx_m, my_m, _ = v
                x_mm = mx_m * 1000.0
                y_mm = my_m * 1000.0
                u, v = self.world_to_pixel(x_mm, y_mm)
                cv2.circle(field, (u, v), 4, (128, 128, 128), -1)  # Gray dots

            # Draw start position (green circle with arrow)
            start_x_mm = self.start_x * 1000.0
            start_y_mm = self.start_y * 1000.0
            start_u, start_v = self.world_to_pixel(start_x_mm, start_y_mm)
            cv2.circle(field, (start_u, start_v), 15, (0, 255, 0), 3)  # Green circle
            cv2.putText(
                field,
                "START",
                (start_u + 20, start_v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            # Draw orientation arrow at start
            self.draw_arrow(field, start_u, start_v, self.start_theta, 40, (0, 255, 0), 3)

            # Draw target position (red circle)
            target_x_mm = self.target_x * 1000.0
            target_y_mm = self.target_y * 1000.0
            target_u, target_v = self.world_to_pixel(target_x_mm, target_y_mm)
            cv2.circle(field, (target_u, target_v), 15, (0, 0, 255), 3)  # Red circle
            cv2.putText(
                field,
                "TARGET",
                (target_u + 20, target_v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            # Draw trajectory (blue line connecting all points)
            if len(self.trajectory_points) > 1:
                trajectory_pixels = []
                for x, y in self.trajectory_points:
                    x_mm = x * 1000.0
                    y_mm = y * 1000.0
                    u, v = self.world_to_pixel(x_mm, y_mm)
                    trajectory_pixels.append((u, v))
                
                # Draw trajectory as connected line
                for i in range(len(trajectory_pixels) - 1):
                    cv2.line(
                        field,
                        trajectory_pixels[i],
                        trajectory_pixels[i + 1],
                        (255, 0, 0),  # Blue
                        2,
                    )

            # Draw obstacles (red squares)
            for obs_x, obs_y in self.obstacles_world:
                obs_x_mm = obs_x * 1000.0
                obs_y_mm = obs_y * 1000.0
                obs_u, obs_v = self.world_to_pixel(obs_x_mm, obs_y_mm)
                
                # Draw square (20x20 pixels)
                size = 10
                cv2.rectangle(
                    field,
                    (obs_u - size, obs_v - size),
                    (obs_u + size, obs_v + size),
                    (0, 0, 255),  # Red
                    -1  # Filled
                )
                cv2.rectangle(
                    field,
                    (obs_u - size, obs_v - size),
                    (obs_u + size, obs_v + size),
                    (255, 255, 255),  # White outline
                    2
                )

            # Draw current robot pose (yellow circle with arrow)
            if self.latest_control_pose is not None:
                x_m = self.latest_control_pose.pose.position.x
                y_m = self.latest_control_pose.pose.position.y
                
                qx = self.latest_control_pose.pose.orientation.x
                qy = self.latest_control_pose.pose.orientation.y
                qz = self.latest_control_pose.pose.orientation.z
                qw = self.latest_control_pose.pose.orientation.w
                yaw = quaternion_to_yaw(qx, qy, qz, qw)
                
                x_mm = x_m * 1000.0
                y_mm = y_m * 1000.0
                robot_u, robot_v = self.world_to_pixel(x_mm, y_mm)
                
                # Draw robot as yellow circle
                cv2.circle(field, (robot_u, robot_v), 12, (0, 255, 255), -1)  # Yellow
                cv2.circle(field, (robot_u, robot_v), 12, (0, 0, 0), 2)  # Black outline
                
                # Draw orientation arrow
                self.draw_arrow(field, robot_u, robot_v, yaw, 50, (0, 255, 255), 4)
                
                # Display position
                cv2.putText(
                    field,
                    f"Robot ({x_m:.2f}, {y_m:.2f})",
                    (robot_u + 15, robot_v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    field,
                    f"Robot ({x_m:.2f}, {y_m:.2f})",
                    (robot_u + 15, robot_v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

            # Display status
            status_y = 30
            cv2.putText(
                field,
                f"Start: ({self.start_x:.2f}, {self.start_y:.2f})",
                (20, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                field,
                f"Target: ({self.target_x:.2f}, {self.target_y:.2f})",
                (20, status_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                field,
                f"Trajectory points: {len(self.trajectory_points)}",
                (20, status_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                field,
                f"Obstacles: {len(self.obstacles_world)}",
                (20, status_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            cv2.imshow("Control Visualization", field)
        except Exception as e:
            self.get_logger().error(f"Error in render: {e}", exc_info=True)

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControlVisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

