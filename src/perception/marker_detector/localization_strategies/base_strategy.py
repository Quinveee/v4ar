import numpy as np
from geometry_msgs.msg import PoseStamped

class BaseLocalizationStrategy:
    """Base interface for all localization strategies."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def predict(self, v, w, dt):
        """Predict next state based on cmd_vel (v, w)."""
        raise NotImplementedError

    def update(self, measurement: PoseStamped):
        """Fuse measurement into state."""
        raise NotImplementedError

    def get_pose(self):
        """Return current pose as (x, y, theta)."""
        return self.x, self.y, self.theta
