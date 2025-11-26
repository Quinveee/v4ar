import numpy as np
import time
from .base_strategy import BaseLocalizationStrategy

class KalmanLocalization(BaseLocalizationStrategy):
    """Kalman filter–based localization strategy."""

    def __init__(self, measurement_interval=10.0):
        super().__init__()
        self.P = np.eye(3) * 0.1      # Covariance matrix
        self.Q = np.diag([0.02, 0.02, 0.01])  # Process noise
        self.R = np.diag([0.05, 0.05])        # Measurement noise
        
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
        
        z = np.array([measurement.pose.position.x, measurement.pose.position.y])
        H = np.array([[1, 0, 0],
                      [0, 1, 0]])

        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Innovation
        y = z - H @ np.array([self.x, self.y, self.theta])

        # Update state
        update_vec = K @ y
        self.x += update_vec[0]
        self.y += update_vec[1]
        self.theta += update_vec[2]
        self.P = (np.eye(3) - K @ H) @ self.P
