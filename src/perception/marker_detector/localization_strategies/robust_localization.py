import math
from geometry_msgs.msg import PoseStamped
from .base_strategy import BaseLocalizationStrategy


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

        if not self.has_pose:
            self.x = mx
            self.y = my
            self.theta = 0.0
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

    def get_pose(self):
        return self.x, self.y, self.theta
