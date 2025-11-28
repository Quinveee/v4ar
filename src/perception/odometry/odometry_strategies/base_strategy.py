import numpy as np
from geometry_msgs.msg import PoseStamped


class BaseOdometryStrategy:
    """Base interface for all odometry strategies."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.initialized = False

    def initialize(self, x: float, y: float, theta: float = 0.0):
        """Initialize odometry with starting pose from triangulation."""
        self.x = x
        self.y = y
        self.theta = theta
        self.initialized = True

    def update(self, *args, **kwargs):
        """Update odometry state based on sensor data."""
        raise NotImplementedError

    def get_pose(self):
        """Return current pose as (x, y, theta)."""
        return self.x, self.y, self.theta

    def is_initialized(self):
        """Check if odometry has been initialized."""
        return self.initialized

