"""
Direct Goal Navigation Strategy.

This is the simplest possible navigation strategy:
- Points directly at the goal
- Ignores all obstacles
- Useful for testing, open environments, or as a baseline
"""

import math
from typing import List, Tuple, Optional
from geometry_msgs.msg import Twist, Vector3
from .base_strategy import BaseNavigationStrategy


class DirectGoalStrategy(BaseNavigationStrategy):
    """
    Simple navigation strategy that drives directly toward the goal.
    
    This strategy:
    - Computes the direct heading to the goal
    - Turns to face the goal
    - Drives forward at constant speed
    - Completely ignores obstacles
    
    Use cases:
    - Testing and debugging
    - Open environments without obstacles
    - Baseline comparison for other strategies
    """
    
    def __init__(
        self,
        goal_tolerance: float = 0.1,
        max_linear_velocity: float = 0.2,
        angular_gain: float = 1.0,
        min_angular_error_for_forward: float = 0.3, # about 17 degrees
        *args, 
        **kwargs 
    ):
        """
        Initialize the direct goal strategy.
        
        Args:
            goal_tolerance: Distance threshold to consider goal reached (meters)
            max_linear_velocity: Forward velocity when moving (m/s)
            angular_gain: Proportional gain for angular velocity control
            min_angular_error_for_forward: Minimum heading error to stop forward motion
                                          (radians). Robot won't move forward if heading
                                          error is larger than this.
        """
        self.goal_tolerance = goal_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.angular_gain = angular_gain
        self.min_angular_error_for_forward = min_angular_error_for_forward
    
    def compute_control(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        target_x: float,
        target_y: float,
        obstacles: List[Tuple[float, float]]
    ) -> Tuple[Twist, Optional[Vector3], bool]:
        """Compute control to drive directly toward goal."""
        
        # Check if goal is reached
        if self.is_goal_reached(robot_x, robot_y, target_x, target_y):
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return (cmd, None, True)
        
        # Compute vector to goal
        dx = target_x - robot_x
        dy = target_y - robot_y
        
        # Desired heading (angle to goal)
        target_heading = math.atan2(dy, dx)
        
        # Compute heading error
        heading_error = self._angle_diff(target_heading, robot_yaw)
        
        # Angular velocity (proportional control)
        angular_velocity = self.angular_gain * heading_error
        
        # Linear velocity: only move forward if roughly facing the goal
        if abs(heading_error) < self.min_angular_error_for_forward:
            linear_velocity = self.max_linear_velocity
        else:
            # Turn in place if heading error is large
            linear_velocity = 0.0
        
        # Create command
        cmd = Twist()
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity
        
        # Create heading vector for visualization
        heading_vec = Vector3()
        heading_vec.x = math.cos(target_heading)
        heading_vec.y = math.sin(target_heading)
        heading_vec.z = 0.0
        
        return (cmd, heading_vec, False)
    
    def reset(self) -> None:
        """Reset strategy state (no internal state for this strategy)."""
        pass
    
    def is_goal_reached(
        self,
        robot_x: float,
        robot_y: float,
        target_x: float,
        target_y: float
    ) -> bool:
        """Check if robot is within goal tolerance."""
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = math.sqrt(dx*dx + dy*dy)
        return distance < self.goal_tolerance
    
    def get_parameters(self) -> dict:
        """Get current strategy parameters."""
        return {
            'goal_tolerance': self.goal_tolerance,
            'max_linear_velocity': self.max_linear_velocity,
            'angular_gain': self.angular_gain,
            'min_angular_error_for_forward': self.min_angular_error_for_forward
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

