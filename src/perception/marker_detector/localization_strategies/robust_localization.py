import math
from geometry_msgs.msg import PoseStamped
from .base_strategy import BaseLocalizationStrategy


def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion to yaw angle in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


class RobustLocalization(BaseLocalizationStrategy):
    """
    Robust localization strategy.
    Fuses triangulated pose measurements with dead-reckoned predictions from /cmd_vel.
    Includes outlier rejection and complementary fusion.
    """

    def __init__(self, max_jump=0.75, alpha=0.7):
        super().__init__()
        self.has_pose = False
        self.max_jump = max_jump
        self.alpha = alpha
        self.last_update_time = None
        self.last_marker_time = None
        self.v = 0.0
        self.w = 0.0

    # -----------------------------------------------------
    # Public interface (for LocalizationNode)
    # -----------------------------------------------------

    def predict(self, v, w, dt):
        """Propagate the pose estimate forward in time."""
        self.v = v
        self.w = w

        if not self.has_pose or dt <= 0.0:
            return

        if abs(w) < 1e-6:
            # Straight motion
            self.x += v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt
        else:
            # Circular arc motion
            radius = v / w
            dtheta = w * dt
            self.x += radius * (math.sin(self.theta + dtheta) - math.sin(self.theta))
            self.y -= radius * (math.cos(self.theta + dtheta) - math.cos(self.theta))
            self.theta += dtheta

        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def update(self, measurement: PoseStamped):
        """Fuse a new triangulated measurement."""
        mx = measurement.pose.position.x
        my = measurement.pose.position.y
        # Extract yaw from quaternion
        qx = measurement.pose.orientation.x
        qy = measurement.pose.orientation.y
        qz = measurement.pose.orientation.z
        qw = measurement.pose.orientation.w
        mtheta = quaternion_to_yaw(qx, qy, qz, qw)

        if not self.has_pose:
            self.x = mx
            self.y = my
            self.theta = mtheta
            self.has_pose = True
            return

        dx = mx - self.x
        dy = my - self.y
        dist = math.hypot(dx, dy)

        # Reject outlier if measurement jump is too large
        if dist > self.max_jump:
            return

        # Fuse via complementary filter
        self.x = (1.0 - self.alpha) * self.x + self.alpha * mx
        self.y = (1.0 - self.alpha) * self.y + self.alpha * my
        # Fuse theta with angle wrapping
        theta_diff = mtheta - self.theta
        theta_diff = math.atan2(math.sin(theta_diff), math.cos(theta_diff))
        self.theta = self.theta + self.alpha * theta_diff

    def get_pose(self):
        return self.x, self.y, self.theta
