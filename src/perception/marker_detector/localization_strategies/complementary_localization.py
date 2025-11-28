import math
from .base_strategy import BaseLocalizationStrategy


def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion to yaw angle in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


class ComplementaryLocalization(BaseLocalizationStrategy):
    """Simpler low-pass filtered localization."""

    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha

    def predict(self, v, w, dt):
        if dt <= 0:
            return
        if abs(w) < 1e-6:
            self.x += v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt
        else:
            radius = v / w
            dtheta = w * dt
            self.x += radius * (math.sin(self.theta + dtheta) - math.sin(self.theta))
            self.y -= radius * (math.cos(self.theta + dtheta) - math.cos(self.theta))
            self.theta += dtheta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def update(self, measurement):
        mx = measurement.pose.position.x
        my = measurement.pose.position.y
        # Extract yaw from quaternion
        qx = measurement.pose.orientation.x
        qy = measurement.pose.orientation.y
        qz = measurement.pose.orientation.z
        qw = measurement.pose.orientation.w
        mtheta = quaternion_to_yaw(qx, qy, qz, qw)

        self.x = (1 - self.alpha) * self.x + self.alpha * mx
        self.y = (1 - self.alpha) * self.y + self.alpha * my
        # Fuse theta with angle wrapping
        theta_diff = mtheta - self.theta
        theta_diff = math.atan2(math.sin(theta_diff), math.cos(theta_diff))
        self.theta = self.theta + self.alpha * theta_diff
