"""Control strategy to navigate from start to target position."""

import math
from .base_strategy import BaseControlStrategy


class GoToTargetStrategy(BaseControlStrategy):
    """Control strategy that first orients toward target, then drives to it.
    
    Strategy:
    1. Phase 1: Rotate to face target (if not already facing it)
    2. Phase 2: Drive straight to target
    3. Optionally updates position estimate using cmd_vel for dead reckoning
    """

    def __init__(self, use_odometry_update: bool = True, 
                 max_linear_velocity_driving: float = 0.2,
                 max_angular_velocity: float = 1.0):
        """Initialize the strategy.
        
        Args:
            use_odometry_update: If True, update position estimate using cmd_vel
            max_linear_velocity_driving: Maximum linear velocity during driving phase (m/s)
            max_angular_velocity: Maximum angular velocity (rad/s)
        """
        super().__init__()
        self.use_odometry_update = use_odometry_update
        
        # Current estimated pose (updated via cmd_vel if enabled)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        
        # Control parameters
        self.max_linear_velocity_driving = max_linear_velocity_driving  # m/s - speed after orientation
        self.max_angular_velocity = max_angular_velocity  # rad/s
        self.orientation_tolerance = math.radians(5)  # 5 degrees
        self.position_tolerance = 0.1  # meters
        
        # Phase tracking
        self.phase = "orienting"  # "orienting" or "driving"
        self.last_update_time = None

    def initialize(self, start_x: float, start_y: float, start_theta: float,
                   target_x: float, target_y: float):
        """Initialize control with start and target positions."""
        super().initialize(start_x, start_y, start_theta, target_x, target_y)
        
        # Initialize current pose to start position
        self.current_x = start_x
        self.current_y = start_y
        self.current_theta = start_theta
        self.phase = "orienting"
        self.last_update_time = None

    def compute_control(self, current_x: float, current_y: float, 
                        current_theta: float) -> tuple:
        """Compute control command (linear_velocity, angular_velocity).
        
        Args:
            current_x: Current x position in world frame (meters)
            current_y: Current y position in world frame (meters)
            current_theta: Current orientation in world frame (radians)
            
        Returns:
            Tuple of (linear_velocity, angular_velocity) in (m/s, rad/s)
        """
        if not self.initialized:
            return (0.0, 0.0)
        
        # Update current pose estimate
        self.current_x = current_x
        self.current_y = current_y
        self.current_theta = current_theta
        
        # Check if target reached
        if self.is_target_reached(current_x, current_y, self.position_tolerance):
            return (0.0, 0.0)
        
        # Compute angle to target
        dx = self.target_x - current_x
        dy = self.target_y - current_y
        angle_to_target = math.atan2(dy, dx)
        
        # Compute angle difference
        angle_diff = angle_to_target - current_theta
        # Normalize to [-pi, pi]
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        
        # Phase 1: Orient toward target
        if abs(angle_diff) > self.orientation_tolerance:
            self.phase = "orienting"
            # Proportional angular control
            angular_vel = self.max_angular_velocity * math.tanh(angle_diff)
            return (0.0, angular_vel)
        
        # Phase 2: Drive to target
        self.phase = "driving"
        distance_to_target = math.sqrt(dx**2 + dy**2)
        
        # Proportional linear control with distance-based scaling
        # Use slower driving speed after orientation
        linear_vel = self.max_linear_velocity_driving * math.tanh(distance_to_target / 0.5)
        
        # Small angular correction to maintain heading
        angular_vel = self.max_angular_velocity * 0.3 * math.tanh(angle_diff)
        
        return (linear_vel, angular_vel)

    def update_from_cmd_vel(self, linear_vel: float, angular_vel: float, dt: float):
        """Update position estimate using cmd_vel (dead reckoning).
        
        Args:
            linear_vel: Linear velocity (m/s)
            angular_vel: Angular velocity (rad/s)
            dt: Time delta (seconds)
        """
        if not self.use_odometry_update or not self.initialized:
            return
        
        if dt <= 0 or dt > 1.0:  # Sanity check
            return
        
        # Dead reckoning motion model
        if abs(angular_vel) < 1e-6:
            # Straight line motion
            self.current_x += linear_vel * math.cos(self.current_theta) * dt
            self.current_y += linear_vel * math.sin(self.current_theta) * dt
        else:
            # Arc motion
            radius = linear_vel / angular_vel if abs(angular_vel) > 1e-6 else 0.0
            dtheta = angular_vel * dt
            self.current_x += radius * (math.sin(self.current_theta + dtheta) - 
                                       math.sin(self.current_theta))
            self.current_y += radius * (-math.cos(self.current_theta + dtheta) + 
                                       math.cos(self.current_theta))
            self.current_theta += dtheta
        
        # Normalize theta
        self.current_theta = math.atan2(math.sin(self.current_theta), 
                                        math.cos(self.current_theta))

    def get_current_pose(self) -> tuple:
        """Get current estimated pose from control strategy.
        
        Returns:
            Tuple of (x, y, theta) in world frame
        """
        return (self.current_x, self.current_y, self.current_theta)

    def get_phase(self) -> str:
        """Get current control phase.
        
        Returns:
            "orienting" or "driving"
        """
        return self.phase

