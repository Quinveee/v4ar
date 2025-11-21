from geometry_msgs.msg import PoseStamped
from .base_strategy import BaseLocalizationStrategy

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
        self.theta = 0.0  # no orientation data in triangulation

    def get_pose(self):
        return self.x, self.y, self.theta
