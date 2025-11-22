"""
Adaptive Kalman Filter Localization Strategy.

This strategy extends the standard Kalman filter by dynamically adjusting
the measurement noise covariance (R) based on innovation statistics.
This makes it robust to varying measurement quality (e.g., marker visibility).
"""

import numpy as np
from collections import deque
from .base_strategy import BaseLocalizationStrategy


class AdaptiveKalmanLocalization(BaseLocalizationStrategy):
    """
    Adaptive Kalman filter localization strategy.

    Dynamically adjusts measurement noise (R) based on:
    1. Innovation magnitude (how wrong predictions are)
    2. Innovation sequence statistics (empirical covariance)

    This makes the filter adapt to changing measurement quality automatically.
    """

    def __init__(
        self,
        process_noise_scale=1.0,
        measurement_noise_base=0.05,
        innovation_window=10,
        adaptation_rate=0.1,
        min_r_scale=0.1,
        max_r_scale=10.0
    ):
        """
        Initialize the adaptive Kalman filter.

        Args:
            process_noise_scale: Scale factor for process noise Q
            measurement_noise_base: Base measurement noise (meters)
            innovation_window: Number of recent innovations to track
            adaptation_rate: How quickly to adapt R (0=slow, 1=fast)
            min_r_scale: Minimum R scaling factor
            max_r_scale: Maximum R scaling factor
        """
        super().__init__()

        # State covariance
        self.P = np.eye(3) * 0.1

        # Base noise parameters
        self.Q_base = np.diag([0.02, 0.02, 0.01]) * process_noise_scale
        self.R_base = np.diag([measurement_noise_base, measurement_noise_base])

        # Adaptive parameters
        self.R = self.R_base.copy()
        self.R_scale = 1.0
        self.adaptation_rate = adaptation_rate
        self.min_r_scale = min_r_scale
        self.max_r_scale = max_r_scale

        # Innovation tracking
        self.innovation_window = innovation_window
        self.innovations = deque(maxlen=innovation_window)

        # Statistics
        self.innovation_count = 0
        self.outlier_count = 0

    def predict(self, v, w, dt):
        """Predict next state using motion model."""
        if dt <= 0:
            return

        # Motion model: update state
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += w * dt

        # Jacobian of motion model w.r.t. state
        F = np.array([
            [1, 0, -v * np.sin(self.theta) * dt],
            [0, 1,  v * np.cos(self.theta) * dt],
            [0, 0,  1]
        ])

        # Update covariance with process noise
        self.P = F @ self.P @ F.T + self.Q_base

    def update(self, measurement):
        """
        Update state with measurement and adapt R based on innovation.

        The innovation (residual) is monitored to detect:
        - Outliers (very large innovations)
        - Degraded measurement quality (consistent large innovations)
        """
        # Extract measurement
        z = np.array([measurement.pose.position.x, measurement.pose.position.y])

        # Measurement model: H maps state to measurement space
        H = np.array([[1, 0, 0],
                      [0, 1, 0]])

        # Predicted measurement
        z_pred = H @ np.array([self.x, self.y, self.theta])

        # Innovation (residual)
        innovation = z - z_pred

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # ============================================================
        # ADAPTIVE MECHANISM: Adjust R based on innovation statistics
        # ============================================================

        # Store innovation for statistics
        self.innovations.append(innovation)
        self.innovation_count += 1

        # Compute innovation magnitude
        innovation_norm = np.linalg.norm(innovation)

        # Expected innovation standard deviation (theoretical)
        expected_std = np.sqrt(np.trace(S) / 2)  # Average std over x,y

        # Compute Mahalanobis distance (normalized innovation)
        try:
            mahalanobis = np.sqrt(innovation.T @ np.linalg.inv(S) @ innovation)
        except np.linalg.LinAlgError:
            mahalanobis = innovation_norm / (expected_std + 1e-6)

        # Chi-squared test threshold (95% confidence for 2 DOF)
        chi_squared_threshold = 5.991

        # Detect outlier
        is_outlier = mahalanobis > chi_squared_threshold

        if is_outlier:
            self.outlier_count += 1
            # Scale up R significantly for this measurement
            R_adaptive = self.R_base * 20.0
        else:
            # Compute empirical innovation variance if we have enough samples
            if len(self.innovations) >= 3:
                # Empirical covariance of recent innovations
                innovations_array = np.array(self.innovations)
                empirical_cov = np.cov(innovations_array.T)

                # Expected theoretical covariance is S
                # If empirical > theoretical, increase R
                # If empirical < theoretical, decrease R
                empirical_var = np.mean(np.diag(empirical_cov))
                theoretical_var = np.mean(np.diag(S))

                # Adaptive scaling based on ratio
                variance_ratio = empirical_var / (theoretical_var + 1e-6)

                # Smooth update of R_scale
                target_scale = np.clip(variance_ratio, self.min_r_scale, self.max_r_scale)
                self.R_scale += self.adaptation_rate * (target_scale - self.R_scale)
            else:
                # Not enough data, use normalized innovation
                target_scale = np.clip(
                    innovation_norm / (expected_std + 1e-6),
                    self.min_r_scale,
                    self.max_r_scale
                )
                self.R_scale += self.adaptation_rate * (target_scale - self.R_scale)

            # Apply adaptive scaling
            R_adaptive = self.R_base * self.R_scale

        # ============================================================
        # KALMAN UPDATE with adaptive R
        # ============================================================

        # Recompute innovation covariance with adaptive R
        S_adaptive = H @ self.P @ H.T + R_adaptive

        # Kalman gain
        try:
            K = self.P @ H.T @ np.linalg.inv(S_adaptive)
        except np.linalg.LinAlgError:
            # Singular matrix, skip update
            return

        # Update state
        state_update = K @ innovation
        self.x += state_update[0]
        self.y += state_update[1]
        self.theta += state_update[2]

        # Update covariance
        self.P = (np.eye(3) - K @ H) @ self.P

        # Store adaptive R for next iteration
        self.R = R_adaptive

    def get_diagnostics(self):
        """
        Get diagnostic information about filter performance.

        Returns:
            dict: Diagnostic statistics
        """
        outlier_rate = self.outlier_count / max(1, self.innovation_count)

        return {
            'r_scale': self.R_scale,
            'innovation_count': self.innovation_count,
            'outlier_count': self.outlier_count,
            'outlier_rate': outlier_rate,
            'current_r': self.R.tolist(),
            'uncertainty': np.trace(self.P)
        }

    def reset_adaptation(self):
        """Reset adaptive parameters to baseline."""
        self.R = self.R_base.copy()
        self.R_scale = 1.0
        self.innovations.clear()
        self.innovation_count = 0
        self.outlier_count = 0
