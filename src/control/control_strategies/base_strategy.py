"""Base class for control strategies."""


class BaseControlStrategy:
    """Base interface for all control strategies."""

    def __init__(self):
        self.initialized = False

    def initialize(self, start_x: float, start_y: float, start_theta: float,
                   target_x: float, target_y: float):
        """Initialize control with start and target positions.
        
        Args:
            start_x: Starting x position in world frame (meters)
            start_y: Starting y position in world frame (meters)
            start_theta: Starting orientation in world frame (radians)
            target_x: Target x position in world frame (meters)
            target_y: Target y position in world frame (meters)
        """
        self.start_x = start_x
        self.start_y = start_y
        self.start_theta = start_theta
        self.target_x = target_x
        self.target_y = target_y
        self.initialized = True

    def compute_control(self, current_x: float, current_y: float, 
                        current_theta: float) -> tuple:
        """Compute control command (linear_velocity, angular_velocity).
        
        Args:
            current_x: Current x position in world frame (meters)
            current_y: Current y position in world frame (meters)
            current_theta: Current orientation in world frame (radians)
            
        Returns:
            Tuple of (linear_velocity, angular_velocity) in (m/s, rad/s)
            Returns (0.0, 0.0) when target is reached
        """
        raise NotImplementedError

    def is_target_reached(self, current_x: float, current_y: float, 
                          tolerance: float = 0.1) -> bool:
        """Check if target position has been reached.
        
        Args:
            current_x: Current x position
            current_y: Current y position
            tolerance: Distance tolerance in meters
            
        Returns:
            True if within tolerance of target
        """
        dx = self.target_x - current_x
        dy = self.target_y - current_y
        distance = (dx**2 + dy**2)**0.5
        return distance <= tolerance

    def get_current_pose(self) -> tuple:
        """Get current estimated pose from control strategy.
        
        Returns:
            Tuple of (x, y, theta) in world frame
        """
        raise NotImplementedError

