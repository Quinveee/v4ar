"""
Dynamic Window Approach (DWA) Navigation Strategy.

DWA is a sophisticated local planner that:
1. Samples possible velocity commands within robot's dynamic constraints
2. Simulates forward trajectories for each velocity pair
3. Scores trajectories based on: heading to goal, clearance from obstacles, velocity
4. Selects the best trajectory

Reference: Fox, D., Burgard, W., & Thrun, S. (1997). 
"The dynamic window approach to collision avoidance"
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from geometry_msgs.msg import Twist, Vector3
from .base_strategy import BaseNavigationStrategy


class DWAStrategy(BaseNavigationStrategy):
    """
    Dynamic Window Approach for local navigation.
    
    This strategy samples velocity commands, simulates trajectories,
    and selects the best one based on multiple objectives:
    - Heading: alignment with goal direction
    - Clearance: distance to nearest obstacle
    - Velocity: preference for higher speeds
    """
    
    def __init__(
        self,
        goal_tolerance: float = 0.1,
        max_linear_velocity: float = 0.3,
        min_linear_velocity: float = 0.0,
        max_angular_velocity: float = 1.0,
        linear_acceleration: float = 0.5,
        angular_acceleration: float = 2.0,
        velocity_samples: int = 10,
        angular_samples: int = 15,
        predict_time: float = 2.0,
        dt: float = 0.1,
        heading_weight: float = 0.1,
        clearance_weight: float = 0.2,
        velocity_weight: float = 0.1,
        obstacle_radius: float = 0.3,
        *args, 
        **kwargs
    ):
        """
        Initialize DWA strategy.
        
        Args:
            goal_tolerance: Distance to consider goal reached (m)
            max_linear_velocity: Maximum forward speed (m/s)
            min_linear_velocity: Minimum forward speed (m/s)
            max_angular_velocity: Maximum rotation speed (rad/s)
            linear_acceleration: Max linear acceleration (m/s²)
            angular_acceleration: Max angular acceleration (rad/s²)
            velocity_samples: Number of linear velocity samples
            angular_samples: Number of angular velocity samples
            predict_time: How far ahead to simulate (seconds)
            dt: Simulation time step (seconds)
            heading_weight: Weight for heading objective
            clearance_weight: Weight for obstacle clearance objective
            velocity_weight: Weight for velocity objective
            obstacle_radius: Safety radius around obstacles (m)
        """
        self.goal_tolerance = goal_tolerance
        self.max_linear_velocity = max_linear_velocity
        self.min_linear_velocity = min_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.linear_acceleration = linear_acceleration
        self.angular_acceleration = angular_acceleration
        self.velocity_samples = velocity_samples
        self.angular_samples = angular_samples
        self.predict_time = predict_time
        self.dt = dt
        self.heading_weight = heading_weight
        self.clearance_weight = clearance_weight
        self.velocity_weight = velocity_weight
        self.obstacle_radius = obstacle_radius
        
        # Current velocity (for dynamic window calculation)
        self.current_v = 0.0
        self.current_w = 0.0
    
    def compute_control(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        target_x: float,
        target_y: float,
        obstacles: List[Tuple[float, float]]
    ) -> Tuple[Twist, Optional[Vector3], bool]:
        """Compute control using Dynamic Window Approach."""
        
        # Check if goal is reached
        if self.is_goal_reached(robot_x, robot_y, target_x, target_y):
            self.current_v = 0.0
            self.current_w = 0.0
            cmd = Twist()
            return (cmd, None, True)
        
        # Calculate dynamic window (achievable velocities)
        dw = self._calculate_dynamic_window()
        
        # Sample velocity space and evaluate trajectories
        best_v, best_w, best_score = 0.0, 0.0, -float('inf')
        best_trajectory = None
        
        for v in np.linspace(dw[0], dw[1], self.velocity_samples):
            for w in np.linspace(dw[2], dw[3], self.angular_samples):
                # Simulate trajectory
                trajectory = self._simulate_trajectory(
                    robot_x, robot_y, robot_yaw, v, w
                )
                
                # Check collision
                if self._check_collision(trajectory, obstacles):
                    continue
                
                # Evaluate trajectory
                score = self._evaluate_trajectory(
                    trajectory, v, w, target_x, target_y, obstacles
                )
                
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w
                    best_trajectory = trajectory
        
        # Update current velocity
        self.current_v = best_v
        self.current_w = best_w
        
        # Create command
        cmd = Twist()
        cmd.linear.x = best_v
        cmd.angular.z = best_w
        
        # Create heading vector (direction of best trajectory)
        if best_trajectory is not None and len(best_trajectory) > 1:
            final_x, final_y, _ = best_trajectory[-1]
            dx = final_x - robot_x
            dy = final_y - robot_y
            norm = math.sqrt(dx*dx + dy*dy) + 1e-6
            heading_vec = Vector3()
            heading_vec.x = dx / norm
            heading_vec.y = dy / norm
            heading_vec.z = 0.0
        else:
            heading_vec = None
        
        return (cmd, heading_vec, False)

    def _calculate_dynamic_window(self) -> Tuple[float, float, float, float]:
        """
        Calculate the dynamic window of achievable velocities.

        Returns:
            (min_v, max_v, min_w, max_w) - velocity bounds
        """
        # Velocity limits based on robot constraints
        v_min = self.min_linear_velocity
        v_max = self.max_linear_velocity
        w_min = -self.max_angular_velocity
        w_max = self.max_angular_velocity

        # Dynamic window based on current velocity and acceleration
        v_min_dyn = self.current_v - self.linear_acceleration * self.dt
        v_max_dyn = self.current_v + self.linear_acceleration * self.dt
        w_min_dyn = self.current_w - self.angular_acceleration * self.dt
        w_max_dyn = self.current_w + self.angular_acceleration * self.dt

        # Intersection of constraints
        v_min = max(v_min, v_min_dyn)
        v_max = min(v_max, v_max_dyn)
        w_min = max(w_min, w_min_dyn)
        w_max = min(w_max, w_max_dyn)

        return (v_min, v_max, w_min, w_max)

    def _simulate_trajectory(
        self,
        x: float,
        y: float,
        yaw: float,
        v: float,
        w: float
    ) -> List[Tuple[float, float, float]]:
        """
        Simulate a trajectory given velocity commands.

        Args:
            x, y, yaw: Starting pose
            v: Linear velocity
            w: Angular velocity

        Returns:
            List of (x, y, yaw) poses along trajectory
        """
        trajectory = [(x, y, yaw)]
        time = 0.0

        while time < self.predict_time:
            # Update pose using simple kinematic model
            yaw += w * self.dt
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            time += self.dt

            trajectory.append((x, y, yaw))

        return trajectory

    def _check_collision(
        self,
        trajectory: List[Tuple[float, float, float]],
        obstacles: List[Tuple[float, float]]
    ) -> bool:
        """
        Check if trajectory collides with any obstacle.

        Returns:
            True if collision detected, False otherwise
        """
        for x, y, _ in trajectory:
            for ox, oy in obstacles:
                dist = math.sqrt((x - ox)**2 + (y - oy)**2)
                if dist < self.obstacle_radius:
                    return True
        return False

    def _evaluate_trajectory(
        self,
        trajectory: List[Tuple[float, float, float]],
        v: float,
        w: float,
        target_x: float,
        target_y: float,
        obstacles: List[Tuple[float, float]]
    ) -> float:
        """
        Evaluate trajectory quality based on multiple objectives.

        Returns:
            Score (higher is better)
        """
        if not trajectory:
            return -float('inf')

        # Final pose of trajectory
        final_x, final_y, final_yaw = trajectory[-1]

        # 1. Heading objective: alignment with goal
        goal_angle = math.atan2(target_y - final_y, target_x - final_x)
        heading_error = abs(self._angle_diff(goal_angle, final_yaw))
        heading_score = math.pi - heading_error  # Higher when aligned

        # 2. Clearance objective: distance to nearest obstacle
        if obstacles:
            min_dist = float('inf')
            for x, y, _ in trajectory:
                for ox, oy in obstacles:
                    dist = math.sqrt((x - ox)**2 + (y - oy)**2)
                    min_dist = min(min_dist, dist)
            clearance_score = min(min_dist, 2.0)  # Cap at 2m
        else:
            clearance_score = 2.0

        # 3. Velocity objective: prefer higher speeds
        velocity_score = v / self.max_linear_velocity

        # Weighted sum
        total_score = (
            self.heading_weight * heading_score +
            self.clearance_weight * clearance_score +
            self.velocity_weight * velocity_score
        )

        return total_score

    def reset(self) -> None:
        """Reset current velocity."""
        self.current_v = 0.0
        self.current_w = 0.0

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
            'min_linear_velocity': self.min_linear_velocity,
            'max_angular_velocity': self.max_angular_velocity,
            'linear_acceleration': self.linear_acceleration,
            'angular_acceleration': self.angular_acceleration,
            'velocity_samples': self.velocity_samples,
            'angular_samples': self.angular_samples,
            'predict_time': self.predict_time,
            'heading_weight': self.heading_weight,
            'clearance_weight': self.clearance_weight,
            'velocity_weight': self.velocity_weight,
            'obstacle_radius': self.obstacle_radius
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
