from abc import ABC, abstractmethod
from perception_msgs.msg import MarkerPoseArray

class BaseGazeStrategy(ABC):
    @abstractmethod
    def compute_angles(self, markers: MarkerPoseArray) -> tuple[float, float]:
        """
        Compute (pan_angle, tilt_angle) from all detected markers.
        """
        pass
