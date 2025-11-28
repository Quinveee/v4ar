"""
Base strategy interface for navigation control.

This module defines the abstract interface that all navigation strategies must implement.
Uses ROS message types directly to avoid unnecessary data conversion.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from geometry_msgs.msg import Twist, Vector3


class BaseNavigationStrategy(ABC):
    """
    Abstract base class for all navigation control strategies.

    A navigation strategy computes control commands (Twist messages)
    based on the robot's current state, target position, and detected obstacles.

    This interface allows for easy swapping of different control algorithms
    (e.g., potential fields, pure pursuit, DWA, MPC, etc.) without changing
    the ROS node structure.

    The strategy works with simple primitives (floats and tuples) rather than
    full ROS messages to keep the interface clean and testable.
    """

    @abstractmethod
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
        Compute control commands based on current state and environment.

        Args:
            robot_x: Robot's x position in world frame (meters)
            robot_y: Robot's y position in world frame (meters)
            robot_yaw: Robot's orientation in world frame (radians)
            target_x: Target x position in world frame (meters)
            target_y: Target y position in world frame (meters)
            obstacles: List of obstacle positions as (x, y) tuples in world frame

        Returns:
            Tuple containing:
                - Twist: Velocity command (linear.x and angular.z)
                - Vector3 or None: Optional heading vector for visualization
                - bool: True if goal is reached, False otherwise
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the strategy's internal state.

        This should be called when starting a new navigation task or when
        the strategy needs to be reinitialized.
        """
        pass

    @abstractmethod
    def is_goal_reached(
        self,
        robot_x: float,
        robot_y: float,
        target_x: float,
        target_y: float
    ) -> bool:
        """
        Determine if the robot has reached the goal.

        Args:
            robot_x: Robot's x position in world frame (meters)
            robot_y: Robot's y position in world frame (meters)
            target_x: Target x position in world frame (meters)
            target_y: Target y position in world frame (meters)

        Returns:
            True if goal is reached, False otherwise
        """
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """
        Get the current strategy parameters.

        Returns:
            Dictionary of parameter names and values
        """
        pass

    @abstractmethod
    def set_parameters(self, params: dict) -> None:
        """
        Update strategy parameters.

        Args:
            params: Dictionary of parameter names and values to update
        """
        pass

