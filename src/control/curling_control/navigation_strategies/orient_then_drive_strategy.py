"""
Orient Then Drive Navigation Strategy.

This strategy implements a two-phase navigation approach:
1. Phase 1 (ORIENT): Turn in place to face the target
2. Phase 2 (DRIVE): Drive forward toward the target

Features:
- Fixed starting position and orientation
- Fixed target position
- Optional dead-reckoning position tracking using cmd_vel
- Publishes estimated pose and start pose for visualization
"""

import math
from typing import List, Tuple, Optional
from enum import Enum
from geometry_msgs.msg import Twist, Vector3, PoseStamped
from rclpy.node import Node
from .base_strategy import BaseNavigationStrategy


class NavigationPhase(Enum):
    """Current phase of navigation."""
    ORIENT = "orient"  # Turning to face target
    DRIVE = "drive"    # Driving forward toward target
    COMPLETE = "complete"  # Goal reached


class OrientThenDriveStrategy(BaseNavigationStrategy):
    """
    Two-phase navigation strategy: orient first, then drive.
    
    This strategy:
    1. Takes a fixed starting position and orientation
    2. Takes a fixed target position
    3. First orients itself to face the target (turns in place)
    4. Then drives forward toward the target
    5. Optionally tracks position using dead-reckoning from cmd_vel
    
    Use cases:
    - Precise navigation with known start/end positions
    - Scenarios where orientation matters before movement
    - Testing and validation with fixed waypoints
    """
    
    def __init__(
        self,
        start_x: float,
        start_y: float,
        start_yaw: float,
        target_x: float,
        target_y: float,
        goal_tolerance: float = 0.1,
        orientation_tolerance: float = 0.1,  # radians (~5.7 degrees)
        max_linear_velocity: float = 0.2,
        max_angular_velocity: float = 1.0,
        angular_gain: float = 1.0,
        use_dead_reckoning: bool = False,
        control_dt: float = 0.05,  # MUST match follow.py timer period
        node: Optional[Node] = None,  # ROS node for publishing (optional)
        *args,
        **kwargs
    ):
        """
        Initialize the orient-then-drive strategy.
        
        Args:
            start_x: Starting x position in world frame (meters)
            start_y: Starting y position in world frame (meters)
            start_yaw: Starting orientation in world frame (radians)
            target_x: Target x position in world frame (meters)
            target_y: Target y position in world frame (meters)
            goal_tolerance: Distance threshold to consider goal reached (meters)
            orientation_tolerance: Angular error threshold to switch from ORIENT to DRIVE (radians)
            max_linear_velocity: Maximum forward velocity when driving (m/s)
            max_angular_velocity: Maximum angular velocity when orienting (rad/s)
            angular_gain: Proportional gain for angular velocity control
            use_dead_reckoning: If True, track position using cmd_vel (dead reckoning)
            control_dt: Time step between control calls (seconds)
            node: Optional ROS node for publishing visualization topics
        """
        self.start_x = start_x
        self.start_y = start_y
        self.start_yaw = start_yaw
        self.target_x = target_x
        self.target_y = target_y
        
        self.goal_tolerance = goal_tolerance
        self.orientation_tolerance = orientation_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.angular_gain = angular_gain
        self.use_dead_reckoning = use_dead_reckoning
        self.control_dt = control_dt
        
        # Current phase
        self.phase = NavigationPhase.ORIENT
        
        # Dead-reckoning state (if enabled)
        self.estimated_x = start_x
        self.estimated_y = start_y
        self.estimated_yaw = start_yaw
        self.last_cmd_linear_x = 0.0
        self.last_cmd_angular_z = 0.0
        
        # ROS node for publishing (optional)
        self.node = node
        if self.node is not None:
            self.start_pose_pub = self.node.create_publisher(
                PoseStamped, "/control/orient_then_drive/start_pose", 10
            )
            self.estimated_pose_pub = self.node.create_publisher(
                PoseStamped, "/control/orient_then_drive/estimated_pose", 10
            )
            self._publish_start_pose()
    
    def _publish_start_pose(self):
        """Publish the starting pose for visualization."""
        if self.node is None:
            return
        
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = self.start_x
        msg.pose.position.y = self.start_y
        msg.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        cy = math.cos(self.start_yaw * 0.5)
        sy = math.sin(self.start_yaw * 0.5)
        msg.pose.orientation.w = cy
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = sy
        
        self.start_pose_pub.publish(msg)
    
    def _publish_estimated_pose(self):
        """Publish the estimated pose for visualization."""
        if self.node is None or not self.use_dead_reckoning:
            return
        
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = self.estimated_x
        msg.pose.position.y = self.estimated_y
        msg.pose.position.z = 0.0
        
        # Convert yaw to quaternion
        cy = math.cos(self.estimated_yaw * 0.5)
        sy = math.sin(self.estimated_yaw * 0.5)
        msg.pose.orientation.w = cy
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = sy
        
        self.estimated_pose_pub.publish(msg)
    
    def _update_dead_reckoning(self, linear_x: float, angular_z: float):
        """Update estimated position using dead-reckoning from cmd_vel."""
        if not self.use_dead_reckoning:
            return
        
        # Update position based on last command
        # Simple kinematic model: x' = x + v*cos(yaw)*dt, y' = y + v*sin(yaw)*dt
        self.estimated_x += self.last_cmd_linear_x * math.cos(self.estimated_yaw) * self.control_dt
        self.estimated_y += self.last_cmd_linear_x * math.sin(self.estimated_yaw) * self.control_dt
        self.estimated_yaw += self.last_cmd_angular_z * self.control_dt
        
        # Normalize yaw to [-pi, pi]
        self.estimated_yaw = math.atan2(math.sin(self.estimated_yaw), math.cos(self.estimated_yaw))
        
        # Store current command for next iteration
        self.last_cmd_linear_x = linear_x
        self.last_cmd_angular_z = angular_z
    
    def compute_control(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        target_x: float,
        target_y: float,
        obstacles: List[Tuple[float, float]]
    ) -> Tuple[Twist, Optional[Vector3], bool]:
        """
        Compute control to navigate from start to target.
        
        Uses robot_x, robot_y, robot_yaw if dead-reckoning is disabled.
        Uses estimated_x, estimated_y, estimated_yaw if dead-reckoning is enabled.
        """
        # Choose position source
        if self.use_dead_reckoning:
            current_x = self.estimated_x
            current_y = self.estimated_y
            current_yaw = self.estimated_yaw
        else:
            current_x = robot_x
            current_y = robot_y
            current_yaw = robot_yaw
        
        # Check if goal is reached
        dx = target_x - current_x
        dy = target_y - current_y
        distance_to_goal = math.sqrt(dx*dx + dy*dy)
        
        if distance_to_goal < self.goal_tolerance:
            self.phase = NavigationPhase.COMPLETE
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self._update_dead_reckoning(0.0, 0.0)
            return (cmd, None, True)
        
        # Compute desired heading to target
        target_heading = math.atan2(dy, dx)
        
        # Compute heading error
        heading_error = self._angle_diff(target_heading, current_yaw)
        
        # Phase logic: ORIENT -> DRIVE
        if self.phase == NavigationPhase.ORIENT:
            # Check if we're oriented enough to start driving
            if abs(heading_error) < self.orientation_tolerance:
                self.phase = NavigationPhase.DRIVE
            else:
                # Turn in place
                angular_velocity = self.angular_gain * heading_error
                # Clamp to max angular velocity
                angular_velocity = max(-self.max_angular_velocity, 
                                     min(self.max_angular_velocity, angular_velocity))
                
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = angular_velocity
                
                self._update_dead_reckoning(0.0, angular_velocity)
                if self.use_dead_reckoning:
                    self._publish_estimated_pose()
                
                # Create heading vector for visualization
                heading_vec = Vector3()
                heading_vec.x = math.cos(target_heading)
                heading_vec.y = math.sin(target_heading)
                heading_vec.z = 0.0
                
                return (cmd, heading_vec, False)
        
        # Phase DRIVE: Move forward while correcting heading
        if self.phase == NavigationPhase.DRIVE:
            # Check if we've drifted too far from target heading
            if abs(heading_error) > self.orientation_tolerance * 2.0:
                # Switch back to orient phase if heading error is too large
                self.phase = NavigationPhase.ORIENT
                angular_velocity = self.angular_gain * heading_error
                angular_velocity = max(-self.max_angular_velocity, 
                                     min(self.max_angular_velocity, angular_velocity))
                
                cmd = Twist()
                cmd.linear.x = 0.0
                cmd.angular.z = angular_velocity
                
                self._update_dead_reckoning(0.0, angular_velocity)
                if self.use_dead_reckoning:
                    self._publish_estimated_pose()
                
                heading_vec = Vector3()
                heading_vec.x = math.cos(target_heading)
                heading_vec.y = math.sin(target_heading)
                heading_vec.z = 0.0
                
                return (cmd, heading_vec, False)
            
            # Drive forward with heading correction
            angular_velocity = self.angular_gain * heading_error
            angular_velocity = max(-self.max_angular_velocity, 
                                 min(self.max_angular_velocity, angular_velocity))
            
            cmd = Twist()
            cmd.linear.x = self.max_linear_velocity
            cmd.angular.z = angular_velocity
            
            self._update_dead_reckoning(self.max_linear_velocity, angular_velocity)
            if self.use_dead_reckoning:
                self._publish_estimated_pose()
            
            # Create heading vector for visualization
            heading_vec = Vector3()
            heading_vec.x = math.cos(target_heading)
            heading_vec.y = math.sin(target_heading)
            heading_vec.z = 0.0
            
            return (cmd, heading_vec, False)
        
        # Should not reach here, but return stop command
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return (cmd, None, False)
    
    def reset(self) -> None:
        """Reset strategy state to initial conditions."""
        self.phase = NavigationPhase.ORIENT
        self.estimated_x = self.start_x
        self.estimated_y = self.start_y
        self.estimated_yaw = self.start_yaw
        self.last_cmd_linear_x = 0.0
        self.last_cmd_angular_z = 0.0
        if self.node is not None:
            self._publish_start_pose()
    
    def is_goal_reached(
        self,
        robot_x: float,
        robot_y: float,
        target_x: float,
        target_y: float
    ) -> bool:
        """Check if robot is within goal tolerance."""
        if self.use_dead_reckoning:
            current_x = self.estimated_x
            current_y = self.estimated_y
        else:
            current_x = robot_x
            current_y = robot_y
        
        dx = target_x - current_x
        dy = target_y - current_y
        distance = math.sqrt(dx*dx + dy*dy)
        return distance < self.goal_tolerance
    
    def get_parameters(self) -> dict:
        """Get current strategy parameters."""
        return {
            'start_x': self.start_x,
            'start_y': self.start_y,
            'start_yaw': self.start_yaw,
            'target_x': self.target_x,
            'target_y': self.target_y,
            'goal_tolerance': self.goal_tolerance,
            'orientation_tolerance': self.orientation_tolerance,
            'max_linear_velocity': self.max_linear_velocity,
            'max_angular_velocity': self.max_angular_velocity,
            'angular_gain': self.angular_gain,
            'use_dead_reckoning': self.use_dead_reckoning,
            'phase': self.phase.value
        }
    
    def set_parameters(self, params: dict) -> None:
        """Update strategy parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Compute the shortest angular difference between two angles."""
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))

