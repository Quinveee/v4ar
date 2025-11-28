"""
Potential Field Navigation Strategy.

This strategy uses artificial potential fields for navigation:
- Attractive force pulls the robot toward the goal
- Repulsive forces push the robot away from obstacles
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from geometry_msgs.msg import Twist, Vector3
from .base_strategy import BaseNavigationStrategy


class PotentialFieldStrategy(BaseNavigationStrategy):
    """
    Navigation strategy using artificial potential fields.

    The robot is attracted to the goal and repelled by obstacles.
    The combined force field determines the desired heading direction.
    """

    def __init__(
        self,
        safe_distance: float = 1.2,
        repulse_strength: float = 1.5,
        goal_tolerance: float = 0.1,
        max_linear_velocity: float = 0.1,
        angular_gain: float = 0.5,
        min_speed_scale: float = 0.1,
        *args, 
        **kwargs
    ):
        """
        Initialize the potential field strategy.

        Args:
            safe_distance: Distance threshold for obstacle repulsion (meters)
            repulse_strength: Strength of repulsive force from obstacles
            goal_tolerance: Distance threshold to consider goal reached (meters)
            max_linear_velocity: Maximum forward velocity (m/s)
            angular_gain: Proportional gain for angular velocity control
            min_speed_scale: Minimum speed scaling factor when avoiding obstacles
        """
        self.safe_distance = safe_distance
        self.repulse_strength = repulse_strength
        self.goal_tolerance = goal_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.angular_gain = angular_gain
        self.min_speed_scale = min_speed_scale

    def compute_control(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        target_x: float,
        target_y: float,
        obstacles: List[Tuple[float, float]]
    ) -> Tuple[Twist, Optional[Vector3], bool]:
        """Compute control using potential field method."""

        # Check if goal is reached
        if self.is_goal_reached(robot_x, robot_y, target_x, target_y):
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return (cmd, None, True)

        # 1. Compute attractive force toward goal
        dx = target_x - robot_x
        dy = target_y - robot_y

        goal_vec = np.array([dx, dy])
        goal_vec_norm = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)

        # 2. Compute repulsive forces from obstacles
        repulse_sum = np.array([0.0, 0.0])

        for ox, oy in obstacles:
            vx = robot_x - ox
            vy = robot_y - oy
            distance = math.sqrt(vx*vx + vy*vy)

            if distance < self.safe_distance:
                direction_norm = np.array([vx, vy]) / (distance + 1e-6)
                strength = self.repulse_strength * (1.0 / (distance*distance + 1e-6))
                repulse_sum += direction_norm * strength

        # 3. Combine forces
        combined = goal_vec_norm + repulse_sum
        combined_norm = combined / (np.linalg.norm(combined) + 1e-6)

        # 4. Compute desired heading
        target_heading = math.atan2(combined_norm[1], combined_norm[0])

        # 5. Compute heading error
        heading_error = self._angle_diff(target_heading, robot_yaw)

        # 6. Compute angular velocity
        angular_velocity = self.angular_gain * heading_error

        # 7. Compute linear velocity with obstacle-based scaling
        repulse_mag = np.linalg.norm(repulse_sum)
        speed_scale = max(self.min_speed_scale, 1.0 - repulse_mag)
        linear_velocity = self.max_linear_velocity * speed_scale

        # 8. Create Twist message
        cmd = Twist()
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity

        # 9. Create heading vector for visualization
        heading_vec = Vector3()
        heading_vec.x = combined_norm[0]
        heading_vec.y = combined_norm[1]
        heading_vec.z = 0.0

        return (cmd, heading_vec, False)

    def reset(self) -> None:
        """Reset strategy state (no internal state to reset for this strategy)."""
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
            'safe_distance': self.safe_distance,
            'repulse_strength': self.repulse_strength,
            'goal_tolerance': self.goal_tolerance,
            'max_linear_velocity': self.max_linear_velocity,
            'angular_gain': self.angular_gain,
            'min_speed_scale': self.min_speed_scale
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

