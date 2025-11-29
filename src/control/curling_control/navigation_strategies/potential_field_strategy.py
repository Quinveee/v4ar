#!/usr/bin/env python3
"""
Enhanced Potential Field Navigation Strategy with Safe Local Minimum Detection.

This strategy:
- Uses attractive and repulsive potential fields for navigation.
- Detects local minima (robot stuck without making progress).
- Adds small random perturbations to escape local minima.
- Ignores false detections during initialization or near the goal.
"""

import math
import time
import numpy as np
from typing import List, Tuple, Optional
from geometry_msgs.msg import Twist, Vector3
from .base_strategy import BaseNavigationStrategy


class PotentialFieldStrategy(BaseNavigationStrategy):
    """Navigation using artificial potential fields + robust local minimum recovery."""

    def __init__(
        self,
        safe_distance: float = 1.2,
        repulse_strength: float = 1.5,
        goal_tolerance: float = 0.1,
        max_linear_velocity: float = 0.1,
        angular_gain: float = 0.5,
        min_speed_scale: float = 0.1,
        progress_threshold: float = 0.02,    # min distance progress (m)
        stuck_time_threshold: float = 2.0,   # seconds before considered stuck
        noise_strength: float = 0.3,         # magnitude of random perturbation
        *args,
        **kwargs
    ):
        super().__init__()

        # Core parameters
        self.safe_distance = safe_distance
        self.repulse_strength = repulse_strength
        self.goal_tolerance = goal_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.angular_gain = angular_gain
        self.min_speed_scale = min_speed_scale

        # Local minimum detection
        self.progress_threshold = progress_threshold
        self.stuck_time_threshold = stuck_time_threshold
        self.noise_strength = noise_strength

        # Internal tracking
        self.prev_goal_distance = None
        self.last_progress_time = time.time()
        self.is_stuck = False
        self.initialized = False

    # ------------------------------------------------------------------
    def compute_control(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        target_x: float,
        target_y: float,
        obstacles: List[Tuple[float, float]]
    ) -> Tuple[Twist, Optional[Vector3], bool]:
        """Compute velocity command using potential field method."""

        # 1. Goal check
        if self.is_goal_reached(robot_x, robot_y, target_x, target_y):
            cmd = Twist()
            return cmd, None, True

        # 2. Compute attractive vector toward goal
        dx = target_x - robot_x
        dy = target_y - robot_y
        goal_vec = np.array([dx, dy])
        goal_distance = np.linalg.norm(goal_vec)
        goal_vec_norm = goal_vec / (goal_distance + 1e-6)

        # 3. Compute repulsive forces from obstacles
        repulse_sum = np.zeros(2)
        for ox, oy in obstacles:
            vx = robot_x - ox
            vy = robot_y - oy
            distance = math.hypot(vx, vy)
            if distance < self.safe_distance:
                direction_norm = np.array([vx, vy]) / (distance + 1e-6)
                strength = self.repulse_strength / (distance**2 + 1e-6)
                repulse_sum += direction_norm * strength

        # 4. Combine forces
        combined = goal_vec_norm + repulse_sum

        # --- 5. LOCAL MINIMUM DETECTION ---
        current_time = time.time()

        # Initialize the first distance measurement safely
        if not self.initialized:
            self.prev_goal_distance = goal_distance
            self.last_progress_time = current_time
            self.initialized = True
        else:
            # Only detect local minima if far enough from goal
            if goal_distance > self.goal_tolerance * 2:
                delta = self.prev_goal_distance - goal_distance

                # No significant progress → possible stuck
                if delta < self.progress_threshold:
                    if current_time - self.last_progress_time > self.stuck_time_threshold:
                        if not self.is_stuck:
                            self.is_stuck = True
                            print("⚠️ [PotentialField] Local minimum detected — injecting noise.")
                        # Inject small random noise to break equilibrium
                        noise = np.random.uniform(-self.noise_strength, self.noise_strength, size=2)
                        combined += noise
                else:
                    # Reset timer if progress was made
                    self.last_progress_time = current_time
                    self.is_stuck = False

            else:
                # Near goal — don't trigger stuck behavior
                self.is_stuck = False
                self.last_progress_time = current_time

            # Update stored goal distance
            self.prev_goal_distance = goal_distance

        # 6. Normalize combined vector
        combined_norm = combined / (np.linalg.norm(combined) + 1e-6)

        # 7. Desired heading
        target_heading = math.atan2(combined_norm[1], combined_norm[0])

        # 8. Heading error and angular velocity
        heading_error = self._angle_diff(target_heading, robot_yaw)
        angular_velocity = self.angular_gain * heading_error

        # 9. Linear velocity (reduced near strong repulsion)
        repulse_mag = np.linalg.norm(repulse_sum)
        speed_scale = max(self.min_speed_scale, 1.0 - repulse_mag)
        linear_velocity = self.max_linear_velocity * speed_scale

        # 10. Compose velocity command
        cmd = Twist()
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity

        # 11. Visualization heading vector
        heading_vec = Vector3()
        heading_vec.x = combined_norm[0]
        heading_vec.y = combined_norm[1]
        heading_vec.z = 0.0

        return cmd, heading_vec, False

    # ------------------------------------------------------------------
    def reset(self):
        """Reset internal state."""
        self.prev_goal_distance = None
        self.last_progress_time = time.time()
        self.is_stuck = False
        self.initialized = False

    # ------------------------------------------------------------------
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
        return math.hypot(dx, dy) < self.goal_tolerance

    # ------------------------------------------------------------------
    def get_parameters(self) -> dict:
        """Get current strategy parameters."""
        return {
            "safe_distance": self.safe_distance,
            "repulse_strength": self.repulse_strength,
            "goal_tolerance": self.goal_tolerance,
            "max_linear_velocity": self.max_linear_velocity,
            "angular_gain": self.angular_gain,
            "min_speed_scale": self.min_speed_scale,
            "progress_threshold": self.progress_threshold,
            "stuck_time_threshold": self.stuck_time_threshold,
            "noise_strength": self.noise_strength,
        }

    # ------------------------------------------------------------------
    def set_parameters(self, params: dict):
        """Update strategy parameters dynamically."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    # ------------------------------------------------------------------
    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Compute shortest angular difference between two angles."""
        d = a - b
        return math.atan2(math.sin(d), math.cos(d))
