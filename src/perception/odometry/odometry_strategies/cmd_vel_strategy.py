import math
import numpy as np
from geometry_msgs.msg import Twist
from .base_strategy import BaseOdometryStrategy


class CmdVelStrategy(BaseOdometryStrategy):
    """Odometry strategy using cmd_vel for dead reckoning."""

    def __init__(self):
        super().__init__()
        self.v = 0.0  # Linear velocity
        self.w = 0.0  # Angular velocity
        self.last_update_time = None

    def update(self, cmd_vel_msg: Twist, current_time):
        """Update pose using velocity commands (dead reckoning)."""
        if not self.initialized:
            return

        # Update velocity commands
        self.v = cmd_vel_msg.linear.x
        self.w = cmd_vel_msg.angular.z

        # Calculate time delta
        if self.last_update_time is not None:
            dt = (current_time - self.last_update_time).nanoseconds / 1e9
            
            if dt > 0 and dt < 1.0:  # Sanity check
                # Dead reckoning motion model
                if abs(self.w) < 1e-6:
                    # Straight line motion
                    dx = self.v * math.cos(self.theta) * dt
                    dy = self.v * math.sin(self.theta) * dt
                    dtheta = 0.0
                else:
                    # Arc motion
                    radius = self.v / self.w
                    dtheta = self.w * dt
                    dx = radius * (math.sin(self.theta + dtheta) - math.sin(self.theta))
                    dy = radius * (-math.cos(self.theta + dtheta) + math.cos(self.theta))

                # Update pose
                self.x += dx
                self.y += dy
                self.theta += dtheta
                self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        self.last_update_time = current_time

