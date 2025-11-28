#!/usr/bin/env python3
"""
Rover Detection Visualization Node

Visualizes detected rovers on a bird's eye view field.
Displays rovers as oriented squares based on detected position and orientation.
"""

import rclpy
from rclpy.node import Node
from perception_msgs.msg import ObjectPoseArray
from geometry_msgs.msg import PoseStamped
import cv2
import numpy as np
import math


def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion to yaw angle in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


def draw_field(scale_px_per_mm: float = 0.1):
    """
    Returns a BGR OpenCV image of the field drawn to scale.
    Similar to field_visualization_node.py
    """
    # Field dimensions in mm (soccer field spec)
    A = 9000  # field length
    B = 6000  # field width
    C = 50    # line width
    
    field_w_px = int(A * scale_px_per_mm)
    field_h_px = int(B * scale_px_per_mm)
    
    # Green background
    img = np.full((field_h_px, field_w_px, 3), (0, 128, 0), dtype=np.uint8)
    
    def to_px(x_mm, y_mm):
        """Convert (x,y) in mm (origin bottom-left) -> (u,v) pixels (origin top-left)."""
        u = int(x_mm * scale_px_per_mm)
        v = field_h_px - int(y_mm * scale_px_per_mm)
        return u, v
    
    line_thickness = max(1, int(C * scale_px_per_mm))
    
    # Outer field rectangle
    bl = to_px(0, 0)
    tr = to_px(A, B)
    cv2.rectangle(img, bl, tr, (255, 255, 255), line_thickness)
    
    # Halfway line
    mid_x = A / 2
    p1 = to_px(mid_x, 0)
    p2 = to_px(mid_x, B)
    cv2.line(img, p1, p2, (255, 255, 255), line_thickness)
    
    # Center circle
    center = to_px(mid_x, B / 2)
    radius_px = int((1500 / 2) * scale_px_per_mm)  # 1500mm diameter
    cv2.circle(img, center, radius_px, (255, 255, 255), line_thickness)
    
    return img


class RoverDetectionVisualizationNode(Node):
    """
    Visualize detected rovers on a bird's eye view field.
    
    Subscribes:
        - /detected_rovers (perception_msgs/ObjectPoseArray) - Detected rovers
        - /robot_pose (geometry_msgs/PoseStamped) - Current robot pose (optional)
    """
    
    def __init__(self):
        super().__init__("rover_detection_visualization_node")
        
        # Parameters
        self.declare_parameter("scale_px_per_mm", 0.1)
        self.declare_parameter("field_length_mm", 9000.0)
        self.declare_parameter("field_width_mm", 6000.0)
        self.declare_parameter("show_robot_pose", True)
        self.declare_parameter("rover_color_bgr", [0, 0, 255])  # Red by default
        self.declare_parameter("rover_size_m", 0.3)  # Default rover size in meters
        
        self.scale_px_per_mm = float(self.get_parameter("scale_px_per_mm").value)
        self.field_length_mm = float(self.get_parameter("field_length_mm").value)
        self.field_width_mm = float(self.get_parameter("field_width_mm").value)
        self.show_robot_pose = bool(self.get_parameter("show_robot_pose").value)
        rover_color = self.get_parameter("rover_color_bgr").value
        self.rover_color = tuple(int(c) for c in rover_color)
        self.rover_size_m = float(self.get_parameter("rover_size_m").value)
        
        # Latest data
        self.latest_rovers = []
        self.latest_robot_pose = None
        
        # Create field image
        self.field_img = draw_field(scale_px_per_mm=self.scale_px_per_mm)
        self.field_h, self.field_w = self.field_img.shape[:2]
        
        # Scale factors
        self.px_per_mm_x = self.field_w / self.field_length_mm
        self.px_per_mm_y = self.field_h / self.field_width_mm
        
        self.get_logger().info(
            f"Field image size: {self.field_w}x{self.field_h} px "
            f"(scale: {self.px_per_mm_x:.3f} px/mm)"
        )
        
        # Subscriptions
        self.rovers_sub = self.create_subscription(
            ObjectPoseArray, "/detected_rovers", self.rovers_callback, 10
        )
        
        if self.show_robot_pose:
            self.robot_pose_sub = self.create_subscription(
                PoseStamped, "/robot_pose", self.robot_pose_callback, 10
            )
        
        # Timer for rendering
        self.timer = self.create_timer(0.1, self.render)  # 10 Hz
        
        # Window
        cv2.namedWindow("Detected Rovers", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detected Rovers", 900, 600)
        
        self.get_logger().info("RoverDetectionVisualizationNode started.")
    
    def rovers_callback(self, msg: ObjectPoseArray):
        """Update detected rovers list."""
        self.latest_rovers = msg.rovers
    
    def robot_pose_callback(self, msg: PoseStamped):
        """Update robot pose."""
        self.latest_robot_pose = msg
    
    def world_to_pixel(self, x_world_mm: float, y_world_mm: float):
        """
        Map world coordinates [mm] -> image pixel (u, v).
        Uses same convention as field_visualization_node.py:
        - Origin (0,0) at BOTTOM-RIGHT corner
        - +X: up along field width (0 -> 6000 mm)
        - +Y: left along field length (0 -> 9000 mm)
        """
        A_mm = self.field_length_mm   # 9000
        B_mm = self.field_width_mm    # 6000
        
        # Clamp to physical extents
        x_world_mm = float(np.clip(x_world_mm, 0.0, B_mm))
        y_world_mm = float(np.clip(y_world_mm, 0.0, A_mm))
        
        # Transform to standard field coordinates
        x_std_mm = A_mm - y_world_mm
        y_std_mm = x_world_mm
        
        # Convert to pixel coordinates
        u = int(x_std_mm * self.px_per_mm_x)
        v = int(self.field_h - y_std_mm * self.px_per_mm_y)
        
        return u, v
    
    def camera_to_world(self, camera_x, camera_y, camera_z, robot_pose=None):
        """
        Convert camera frame coordinates to world frame.
        Camera frame: X=right, Y=down, Z=forward
        Robot frame: X=forward, Y=left
        World frame: X=width (0-6000mm), Y=length (0-9000mm)
        
        If robot_pose is provided, transform relative to robot position.
        Otherwise, assume camera is at origin.
        """
        if robot_pose is None:
            # Assume camera is at field center for now
            # Camera Z (forward) -> World X (width)
            # Camera X (right) -> World Y (length, but inverted)
            world_x_mm = camera_z * 1000.0 + 3000.0  # Forward in camera = +X in world, offset to center
            world_y_mm = 4500.0 - camera_x * 1000.0  # Right in camera = -Y in world, offset to center
            return world_x_mm, world_y_mm
        
        # Extract robot pose
        robot_x_m = robot_pose.pose.position.x
        robot_y_m = robot_pose.pose.position.y
        
        # Extract robot yaw
        q = robot_pose.pose.orientation
        robot_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        
        # Camera frame: X=right, Y=down, Z=forward
        # Transform to robot frame: X=forward, Y=left
        robot_frame_x = camera_z  # Forward
        robot_frame_y = -camera_x  # Left (negative of right)
        
        # Transform to world frame (rotate by robot yaw)
        world_x_m = robot_x_m + robot_frame_x * math.cos(robot_yaw) - robot_frame_y * math.sin(robot_yaw)
        world_y_m = robot_y_m + robot_frame_x * math.sin(robot_yaw) + robot_frame_y * math.cos(robot_yaw)
        
        # Convert to mm
        return world_x_m * 1000.0, world_y_m * 1000.0
    
    def draw_oriented_rectangle(self, img, center_u, center_v, length_px, width_px, angle_rad, color, thickness=2):
        """
        Draw an oriented rectangle at the given position (bird's eye view bounding box).
        
        Args:
            img: OpenCV image
            center_u, center_v: Center pixel coordinates
            length_px: Length of rectangle in pixels (forward direction)
            width_px: Width of rectangle in pixels (side direction)
            angle_rad: Rotation angle in radians (orientation)
            color: BGR color tuple
            thickness: Line thickness
        """
        # Calculate rectangle corners (before rotation)
        # Length is along the forward direction (angle_rad), width is perpendicular
        half_length = length_px / 2.0
        half_width = width_px / 2.0
        
        corners = np.array([
            [-half_length, -half_width],  # Back-left
            [half_length, -half_width],    # Front-left
            [half_length, half_width],    # Front-right
            [-half_length, half_width]    # Back-right
        ], dtype=np.float32)
        
        # Rotate corners
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        rotated_corners = corners @ rotation_matrix.T
        
        # Translate to center
        rotated_corners[:, 0] += center_u
        rotated_corners[:, 1] += center_v
        
        # Convert to integer coordinates
        corners_int = rotated_corners.astype(np.int32)
        
        # Draw rectangle
        for i in range(4):
            pt1 = tuple(corners_int[i])
            pt2 = tuple(corners_int[(i + 1) % 4])
            cv2.line(img, pt1, pt2, color, thickness)
        
        # Draw center dot
        cv2.circle(img, (int(center_u), int(center_v)), 3, color, -1)
        
        # Draw orientation arrow (from center to front, along length direction)
        arrow_length = half_length * 0.8
        arrow_end_u = int(center_u + arrow_length * math.cos(angle_rad))
        arrow_end_v = int(center_v + arrow_length * math.sin(angle_rad))
        cv2.arrowedLine(
            img,
            (int(center_u), int(center_v)),
            (arrow_end_u, arrow_end_v),
            color,
            thickness,
            tipLength=0.3
        )
    
    def render(self):
        """Render the visualization."""
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            self.get_logger().info("Closing RoverDetectionVisualizationNode window.")
            rclpy.shutdown()
            return
        
        # Copy field image
        field = self.field_img.copy()
        
        # Draw robot pose if available
        if self.show_robot_pose and self.latest_robot_pose is not None:
            robot_x_m = self.latest_robot_pose.pose.position.x
            robot_y_m = self.latest_robot_pose.pose.position.y
            robot_x_mm = robot_x_m * 1000.0
            robot_y_mm = robot_y_m * 1000.0
            
            robot_u, robot_v = self.world_to_pixel(robot_x_mm, robot_y_mm)
            
            # Draw robot
            cv2.circle(field, (robot_u, robot_v), 8, (255, 255, 0), -1)  # Cyan
            cv2.putText(
                field,
                "Robot",
                (robot_u + 10, robot_v),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
        
        # Draw detected rovers
        for rover in self.latest_rovers:
            # Transform from camera frame to world frame
            cam_x = rover.pose.position.x
            cam_y = rover.pose.position.y
            cam_z = rover.pose.position.z
            
            # Convert to world coordinates
            world_x_mm, world_y_mm = self.camera_to_world(
                cam_x, cam_y, cam_z, self.latest_robot_pose
            )
            
            # Convert to pixel coordinates
            rover_u, rover_v = self.world_to_pixel(world_x_mm, world_y_mm)
            
            # Get rover orientation
            # The orientation in the message is already in camera frame (yaw around Z)
            q = rover.pose.orientation
            rover_yaw_camera = quaternion_to_yaw(q.x, q.y, q.z, q.w)
            
            # Transform orientation to world frame if robot pose is available
            if self.latest_robot_pose is not None:
                robot_q = self.latest_robot_pose.pose.orientation
                robot_yaw = quaternion_to_yaw(robot_q.x, robot_q.y, robot_q.z, robot_q.w)
                # Camera yaw is rotation around Z (forward axis)
                # In world frame, this needs to be added to robot yaw
                # But we also need to account for coordinate system: camera X=right, world X=forward
                # So camera yaw needs to be rotated by 90 degrees conceptually
                # Actually, camera yaw of 0 means facing forward (Z), which in world is along +X
                # So we add robot yaw to get world frame orientation
                rover_yaw = robot_yaw + rover_yaw_camera
            else:
                rover_yaw = rover_yaw_camera
            
            # Get rover size from footprint (bird's eye view dimensions)
            # Use footprint from message if available, otherwise fallback to default
            if hasattr(rover, 'footprint_length') and rover.footprint_length > 0:
                footprint_length_m = rover.footprint_length
                footprint_width_m = rover.footprint_width if hasattr(rover, 'footprint_width') else rover.footprint_length
            else:
                # Fallback to default size
                footprint_length_m = self.rover_size_m
                footprint_width_m = self.rover_size_m
            
            # Convert to pixels (use average scale for square rovers)
            avg_px_per_mm = 0.5 * (self.px_per_mm_x + self.px_per_mm_y)
            footprint_length_px = int(footprint_length_m * 1000.0 * avg_px_per_mm)
            footprint_width_px = int(footprint_width_m * 1000.0 * avg_px_per_mm)
            
            # Draw oriented rectangle using actual footprint dimensions
            self.draw_oriented_rectangle(
                field,
                rover_u,
                rover_v,
                footprint_length_px,
                footprint_width_px,
                rover_yaw,
                self.rover_color,
                thickness=3
            )
            
            # Draw label with ID and distance
            label = f"Rover {rover.id}: {rover.distance:.2f}m"
            cv2.putText(
                field,
                label,
                (rover_u + 15, rover_v - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black outline
                3,
            )
            cv2.putText(
                field,
                label,
                (rover_u + 15, rover_v - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.rover_color,
                2,
            )
        
        # Show count
        cv2.putText(
            field,
            f"Detected Rovers: {len(self.latest_rovers)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        
        cv2.imshow("Detected Rovers", field)
    
    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RoverDetectionVisualizationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

