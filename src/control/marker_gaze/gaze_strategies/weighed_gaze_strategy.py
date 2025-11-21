import numpy as np
from .base_strategy import BaseGazeStrategy


class WeightedGazeStrategy(BaseGazeStrategy):
    def __init__(
        self,
        img_width=640,
        img_height=480,
        center_x=None,
        center_y=None,
        kp_pan=0.002,
        kp_tilt=0.002,
    ):
        self.img_width = img_width
        self.img_height = img_height
        self.center_x = center_x or img_width / 2
        self.center_y = center_y or img_height / 2
        self.kp_pan = kp_pan
        self.kp_tilt = kp_tilt

    def compute_angles(self, msg):
        if not msg.markers:
            return (0.0, 0.0)

        distances = np.array([m.distance for m in msg.markers])
        weights = 1.0 / np.clip(distances, 0.1, None)
        weights /= np.sum(weights)

        avg_x = np.sum([w * m.center_x for w, m in zip(weights, msg.markers)])
        avg_y = np.sum([w * m.center_y for w, m in zip(weights, msg.markers)])

        pan_error = (avg_x - self.center_x) / self.center_x
        tilt_error = (avg_y - self.center_y) / self.center_y

        pan_angle = -self.kp_pan * pan_error
        tilt_angle = -self.kp_tilt * tilt_error

        return (pan_angle, tilt_angle)

    def compute_angles_from_markers(self, markers):
        if not markers:
            return (0.0, 0.0)

        distances = np.array([m.distance for m in markers])
        weights = 1.0 / np.clip(distances, 0.1, None)
        weights /= np.sum(weights)

        avg_x = np.sum([w * m.center_x for w, m in zip(weights, markers)])
        avg_y = np.sum([w * m.center_y for w, m in zip(weights, markers)])

        pan_error = (avg_x - self.center_x) / self.center_x
        tilt_error = (avg_y - self.center_y) / self.center_y

        pan_cmd = -self.kp_pan * pan_error
        tilt_cmd = -self.kp_tilt * tilt_error

        return pan_cmd, tilt_cmd
