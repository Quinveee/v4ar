import numpy as np
import time
from .base_strategy import BaseLocalizationStrategy


def quaternion_to_yaw(x, y, z, w):
    """Convert quaternion to yaw angle in radians."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class KalmanLocalization(BaseLocalizationStrategy):
    """Kalman filter–based localization strategy with full pose (x, y, theta) updates."""

    def __init__(self, measurement_interval=0.1):
        super().__init__()
        self.P = np.eye(3) * 0.1      # Covariance matrix
        self.Q = np.diag([0.02, 0.02, 0.01])  # Process noise
        self.R = np.diag([0.05, 0.05, 0.02])  # Measurement noise (x, y, theta)

        # Measurement throttling
        self.measurement_interval = measurement_interval  # seconds
        self.last_measurement_time = 0.0

    def predict(self, v, w, dt):
        if dt <= 0:
            return

        # Motion model
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += w * dt
        self.theta = normalize_angle(self.theta)

        # Jacobian of motion model wrt state
        F = np.array([
            [1, 0, -v * np.sin(self.theta) * dt],
            [0, 1,  v * np.cos(self.theta) * dt],
            [0, 0,  1]
        ])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        current_time = time.time()

        # Only update if enough time has passed
        if current_time - self.last_measurement_time < self.measurement_interval:
            return  # Skip this measurement

        self.last_measurement_time = current_time

        # Extract yaw from quaternion
        qx = measurement.pose.orientation.x
        qy = measurement.pose.orientation.y
        qz = measurement.pose.orientation.z
        qw = measurement.pose.orientation.w
        measured_theta = quaternion_to_yaw(qx, qy, qz, qw)

        # Full measurement vector (x, y, theta)
        z = np.array([
            measurement.pose.position.x,
            measurement.pose.position.y,
            measured_theta
        ])

        # Full observation matrix (observes all states directly)
        H = np.eye(3)

        # Current state
        state = np.array([self.x, self.y, self.theta])

        # Innovation (with angle wrapping for theta)
        y = z - H @ state
        y[2] = normalize_angle(y[2])  # Wrap angle difference to [-pi, pi]

        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        update_vec = K @ y
        self.x += update_vec[0]
        self.y += update_vec[1]
        self.theta += update_vec[2]
        self.theta = normalize_angle(self.theta)

        # Update covariance
        self.P = (np.eye(3) - K @ H) @ self.P
