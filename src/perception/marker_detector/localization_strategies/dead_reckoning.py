import numpy as np
import time
from .base_strategy import BaseLocalizationStrategy


class DeadReckoningLocalization(BaseLocalizationStrategy):
    """
    Dead Reckoning + Pose Reset Strategy.

    - Waits for initial AprilTag position (/robot_pose_raw)
    - Then uses /cmd_vel to integrate position over time
    - If no cmd_vel received for a few seconds, waits for next AprilTag
    - When a new AprilTag pose arrives, it resets or corrects position

    This gives smooth continuous tracking even when marker detections drop out,
    while keeping drift under control by occasional corrections.
    """

    def __init__(self, cmd_timeout=3.0):
        super().__init__()
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.initialized = False
        self.last_cmd_time = None
        self.cmd_timeout = cmd_timeout  # seconds

    def predict(self, v, w, dt):
        """Integrate motion commands over time."""
        if not self.initialized:
            return  # Ignore motion until we have an initial pose

        if dt <= 0:
            return

        # Standard differential-drive motion model
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += w * dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))  # normalize
        self.last_cmd_time = time.time()

    def update(self, measurement):
        """
        Update pose when a new AprilTag-based measurement arrives.
        - Used if not initialized
        - Or if no cmd_vel received for some time (pose drift risk)
        """
        current_time = time.time()

        # Case 1: not initialized → take first pose as origin
        if not self.initialized:
            self.x = measurement.pose.position.x
            self.y = measurement.pose.position.y
            self.theta = 0.0  # We don’t have yaw from AprilTags
            self.initialized = True
            self.last_cmd_time = current_time
            return

        # Case 2: if no cmd_vel for a while, reset pose
        if self.last_cmd_time is None or (current_time - self.last_cmd_time > self.cmd_timeout):
            self.x = measurement.pose.position.x
            self.y = measurement.pose.position.y
            # Keep current heading estimate, since AprilTag gives no orientation
            self.last_cmd_time = current_time

    def get_pose(self):
        return (self.x, self.y, self.theta)
