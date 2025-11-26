import math
from geometry_msgs.msg import PoseStamped
from .base_strategy import BaseLocalizationStrategy


def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion to yaw angle in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


class IdentityLocalization(BaseLocalizationStrategy):
    """
    A minimal localization strategy that directly outputs
    the received triangulated pose without modification.
    """

    def __init__(self):
        super().__init__()

    def predict(self, v, w, dt):
        """No prediction step — stays as last measurement."""
        pass

    def update(self, measurement: PoseStamped):
        """Directly use the incoming measurement."""
        self.x = measurement.pose.position.x
        self.y = measurement.pose.position.y
        # Extract yaw from quaternion
        qx = measurement.pose.orientation.x
        qy = measurement.pose.orientation.y
        qz = measurement.pose.orientation.z
        qw = measurement.pose.orientation.w
        self.theta = quaternion_to_yaw(qx, qy, qz, qw)

    def get_pose(self):
        return self.x, self.y, self.theta
